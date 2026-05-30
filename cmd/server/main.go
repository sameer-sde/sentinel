// Sentinel — fraud detection serving layer.
//
// Day 7: model versioning + hot-swap.
// - Models live in models/<version>/fraud_model.onnx
// - models/current symlink points to active version
// - /admin/reload re-reads models/current and atomically swaps the session
// - In-flight requests finish with old model; new requests use new model
package main

import (
	"container/list"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	onnxRuntimeLib   = "../../onnxruntime/libonnxruntime.dylib"
	modelDir         = "../../models/current"
	modelFile        = "fraud_model.onnx"
	thresholdCfgPath = "../../models/threshold_config.json"
	numFeatures      = 30
	httpAddr         = ":8080"

	maxBatch     = 32
	maxWait      = 5 * time.Millisecond
	jobQueueSize = 256

	cacheCapacity = 10000
	cacheTTL      = 2 * time.Second
)

// -------------------- types --------------------

type ThresholdConfig struct {
	OptimalThreshold  float64 `json:"optimal_threshold"`
	CostFN            int     `json:"cost_fn"`
	CostFP            int     `json:"cost_fp"`
	ExpectedRecall    float64 `json:"expected_recall"`
	ExpectedPrecision float64 `json:"expected_precision"`
}

type PredictRequest struct {
	Features []float32 `json:"features"`
}

type PredictResponse struct {
	FraudProbability float32 `json:"fraud_probability"`
	PredictedClass   int64   `json:"predicted_class"`
	Decision         string  `json:"decision"`
	ThresholdUsed    float64 `json:"threshold_used"`
	LatencyMicros    int64   `json:"latency_us"`
	BatchSize        int     `json:"batch_size"`
	CacheHit         bool    `json:"cache_hit"`
	ModelVersion     string  `json:"model_version"`
}

type inferenceJob struct {
	features []float32
	reply    chan inferenceResult
}

type inferenceResult struct {
	fraudProba   float32
	class        int64
	batchSize    int
	modelVersion string
	err          error
}

// modelBundle holds a session and its associated tensors.
// We store *this struct atomically and replace it wholesale on reload —
// can't swap the session without also swapping its bound tensors.
type modelBundle struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	label   *ort.Tensor[int64]
	proba   *ort.Tensor[float32]
	version string // resolved symlink target, e.g. "v1"
	path    string // absolute path the session was loaded from
}

// -------------------- LRU cache (unchanged from Day 6) --------------------

type cacheEntry struct {
	key       uint64
	proba     float32
	class     int64
	expiresAt time.Time
}

type LRUCache struct {
	mu       sync.Mutex
	capacity int
	ttl      time.Duration
	items    map[uint64]*list.Element
	order    *list.List
}

func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		ttl:      ttl,
		items:    make(map[uint64]*list.Element, capacity),
		order:    list.New(),
	}
}

func (c *LRUCache) Get(key uint64) (*cacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.items[key]
	if !ok {
		return nil, false
	}
	entry := elem.Value.(*cacheEntry)
	if time.Now().After(entry.expiresAt) {
		c.order.Remove(elem)
		delete(c.items, key)
		return nil, false
	}
	c.order.MoveToFront(elem)
	return entry, true
}

func (c *LRUCache) Put(key uint64, proba float32, class int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		entry := elem.Value.(*cacheEntry)
		entry.proba = proba
		entry.class = class
		entry.expiresAt = time.Now().Add(c.ttl)
		c.order.MoveToFront(elem)
		return
	}

	entry := &cacheEntry{
		key:       key,
		proba:     proba,
		class:     class,
		expiresAt: time.Now().Add(c.ttl),
	}
	elem := c.order.PushFront(entry)
	c.items[key] = elem

	if c.order.Len() > c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			oldEntry := oldest.Value.(*cacheEntry)
			delete(c.items, oldEntry.key)
			c.order.Remove(oldest)
		}
	}
}

func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[uint64]*list.Element, c.capacity)
	c.order = list.New()
}

func (c *LRUCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

func hashFeatures(features []float32) uint64 {
	h := fnv.New64a()
	buf := make([]byte, 4)
	for _, f := range features {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(f))
		h.Write(buf)
	}
	return h.Sum64()
}

// -------------------- model loader --------------------

// loadModelBundle creates a new session + tensors from the file at modelDir/modelFile.
// Returns the resolved version (the symlink target) so we can report it.
func loadModelBundle() (*modelBundle, error) {
	// Resolve the symlink to find the real version directory.
	resolved, err := filepath.EvalSymlinks(modelDir)
	if err != nil {
		return nil, fmt.Errorf("resolve symlink %s: %w", modelDir, err)
	}
	version := filepath.Base(resolved)
	path := filepath.Join(modelDir, modelFile)

	input, err := ort.NewEmptyTensor[float32](ort.NewShape(maxBatch, numFeatures))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	label, err := ort.NewEmptyTensor[int64](ort.NewShape(maxBatch))
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("create label tensor: %w", err)
	}
	proba, err := ort.NewEmptyTensor[float32](ort.NewShape(maxBatch, 2))
	if err != nil {
		input.Destroy()
		label.Destroy()
		return nil, fmt.Errorf("create proba tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		path,
		[]string{"input"},
		[]string{"label", "probabilities"},
		[]ort.ArbitraryTensor{input},
		[]ort.ArbitraryTensor{label, proba},
		nil,
	)
	if err != nil {
		input.Destroy()
		label.Destroy()
		proba.Destroy()
		return nil, fmt.Errorf("create session from %s: %w", path, err)
	}

	return &modelBundle{
		session: session,
		input:   input,
		label:   label,
		proba:   proba,
		version: version,
		path:    path,
	}, nil
}

func (m *modelBundle) Destroy() {
	if m == nil {
		return
	}
	m.session.Destroy()
	m.input.Destroy()
	m.label.Destroy()
	m.proba.Destroy()
}

// -------------------- server --------------------

type Server struct {
	jobs      chan inferenceJob
	threshold float64
	cache     *LRUCache

	// Atomic pointer to the live model bundle. Batcher reads, /admin/reload writes.
	bundle atomic.Pointer[modelBundle]
	// reloadMu serializes reload operations (only one swap at a time).
	reloadMu sync.Mutex

	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	batchCount    atomic.Uint64
	cacheHits     atomic.Uint64
	cacheMisses   atomic.Uint64
	reloadCount   atomic.Uint64
	startedAt     time.Time
}

func main() {
	cfg, err := loadThresholdConfig(thresholdCfgPath)
	if err != nil {
		log.Fatalf("load threshold config: %v", err)
	}
	log.Printf("✓ Threshold config loaded: optimal=%.3f", cfg.OptimalThreshold)

	ort.SetSharedLibraryPath(onnxRuntimeLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()
	log.Println("✓ ONNX Runtime initialized")

	bundle, err := loadModelBundle()
	if err != nil {
		log.Fatalf("initial model load: %v", err)
	}
	log.Printf("✓ Loaded model version %s from %s", bundle.version, bundle.path)

	s := &Server{
		jobs:      make(chan inferenceJob, jobQueueSize),
		threshold: cfg.OptimalThreshold,
		cache:     NewLRUCache(cacheCapacity, cacheTTL),
		startedAt: time.Now(),
	}
	s.bundle.Store(bundle)
	log.Printf("✓ LRU cache ready: capacity=%d TTL=%v", cacheCapacity, cacheTTL)

	go s.batcher()

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/admin/version", s.handleVersion)
	mux.HandleFunc("/admin/reload", s.handleReload)

	log.Printf("✓ Listening on http://localhost%s (maxBatch=%d maxWait=%v)",
		httpAddr, maxBatch, maxWait)
	if err := http.ListenAndServe(httpAddr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}

// -------------------- batcher --------------------

func (s *Server) batcher() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("BATCHER PANIC: %v", r)
		}
	}()

	log.Println("✓ Batcher goroutine ready")

	batch := make([]inferenceJob, 0, maxBatch)

	for {
		first, ok := <-s.jobs
		if !ok {
			return
		}
		batch = batch[:0]
		batch = append(batch, first)

		deadline := time.After(maxWait)

	collect:
		for len(batch) < maxBatch {
			select {
			case j, ok := <-s.jobs:
				if !ok {
					break collect
				}
				batch = append(batch, j)
			case <-deadline:
				break collect
			}
		}

		// Read the live bundle for this batch.
		// If /admin/reload swaps mid-batch, the next batch picks up the new one.
		bundle := s.bundle.Load()
		inputData := bundle.input.GetData()

		for i, j := range batch {
			copy(inputData[i*numFeatures:(i+1)*numFeatures], j.features)
		}
		for i := len(batch); i < maxBatch; i++ {
			for k := 0; k < numFeatures; k++ {
				inputData[i*numFeatures+k] = 0
			}
		}

		if err := bundle.session.Run(); err != nil {
			result := inferenceResult{err: err}
			for _, j := range batch {
				j.reply <- result
			}
			log.Printf("batcher: inference error: %v", err)
			continue
		}

		labels := bundle.label.GetData()
		probas := bundle.proba.GetData()
		batchSize := len(batch)
		for i, j := range batch {
			j.reply <- inferenceResult{
				fraudProba:   probas[i*2+1],
				class:        labels[i],
				batchSize:    batchSize,
				modelVersion: bundle.version,
			}
		}
		s.batchCount.Add(1)
	}
}

// -------------------- HTTP handlers --------------------

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.totalErrors.Add(1)
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	if len(req.Features) != numFeatures {
		s.totalErrors.Add(1)
		http.Error(w, fmt.Sprintf("expected %d features, got %d", numFeatures, len(req.Features)),
			http.StatusBadRequest)
		return
	}

	start := time.Now()
	key := hashFeatures(req.Features)

	var fraudProba float32
	var predictedClass int64
	var batchSize int
	var cacheHit bool
	var modelVersion string

	if entry, ok := s.cache.Get(key); ok {
		fraudProba = entry.proba
		predictedClass = entry.class
		cacheHit = true
		batchSize = 0
		modelVersion = s.bundle.Load().version
		s.cacheHits.Add(1)
	} else {
		s.cacheMisses.Add(1)
		reply := make(chan inferenceResult, 1)
		s.jobs <- inferenceJob{features: req.Features, reply: reply}
		result := <-reply
		if result.err != nil {
			s.totalErrors.Add(1)
			log.Printf("inference error: %v", result.err)
			http.Error(w, "inference failed", http.StatusInternalServerError)
			return
		}
		fraudProba = result.fraudProba
		predictedClass = result.class
		batchSize = result.batchSize
		modelVersion = result.modelVersion
		s.cache.Put(key, fraudProba, predictedClass)
	}

	latencyMicros := time.Since(start).Microseconds()

	decision := "allow"
	if float64(fraudProba) >= s.threshold {
		decision = "block"
		s.totalBlock.Add(1)
	} else {
		s.totalAllow.Add(1)
	}
	s.totalRequests.Add(1)

	resp := PredictResponse{
		FraudProbability: fraudProba,
		PredictedClass:   predictedClass,
		Decision:         decision,
		ThresholdUsed:    s.threshold,
		LatencyMicros:    latencyMicros,
		BatchSize:        batchSize,
		CacheHit:         cacheHit,
		ModelVersion:     modelVersion,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
}

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	b := s.bundle.Load()
	out := map[string]interface{}{
		"version":      b.version,
		"path":         b.path,
		"reload_count": s.reloadCount.Load(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// handleReload: POST /admin/reload — load the model file at models/current
// (whatever the symlink now resolves to) and atomically swap the live bundle.
// Cache is invalidated since new model may produce different predictions.
func (s *Server) handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// Serialize concurrent reloads.
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()

	oldBundle := s.bundle.Load()
	log.Printf("Reload requested. Current version: %s", oldBundle.version)

	newBundle, err := loadModelBundle()
	if err != nil {
		log.Printf("Reload failed: %v", err)
		http.Error(w, fmt.Sprintf("reload failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Atomic swap. Batches in-flight finish on oldBundle.
	s.bundle.Store(newBundle)
	s.cache.Clear()
	s.reloadCount.Add(1)

	log.Printf("✓ Swapped %s → %s", oldBundle.version, newBundle.version)

	// Give in-flight batches ~50ms to finish before destroying old bundle.
	// Crude but safe for our throughput. Real systems use refcounting.
	go func(old *modelBundle) {
		time.Sleep(50 * time.Millisecond)
		old.Destroy()
	}(oldBundle)

	out := map[string]interface{}{
		"status":      "ok",
		"old_version": oldBundle.version,
		"new_version": newBundle.version,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(s.startedAt).Seconds()
	total := s.totalRequests.Load()
	batches := s.batchCount.Load()
	hits := s.cacheHits.Load()
	misses := s.cacheMisses.Load()

	rps := 0.0
	avgBatch := 0.0
	hitRate := 0.0
	if uptime > 0 {
		rps = float64(total) / uptime
	}
	if batches > 0 {
		avgBatch = float64(misses) / float64(batches)
	}
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}

	out := map[string]interface{}{
		"total_requests": total,
		"allow":          s.totalAllow.Load(),
		"block":          s.totalBlock.Load(),
		"errors":         s.totalErrors.Load(),
		"batches_run":    batches,
		"avg_batch_size": avgBatch,
		"cache_hits":     hits,
		"cache_misses":   misses,
		"cache_hit_rate": hitRate,
		"cache_size":     s.cache.Size(),
		"cache_capacity": cacheCapacity,
		"cache_ttl_ms":   cacheTTL.Milliseconds(),
		"reload_count":   s.reloadCount.Load(),
		"model_version":  s.bundle.Load().version,
		"uptime_seconds": uptime,
		"avg_rps":        rps,
		"threshold":      s.threshold,
		"max_batch":      maxBatch,
		"max_wait_ms":    maxWait.Milliseconds(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func loadThresholdConfig(path string) (*ThresholdConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	var cfg ThresholdConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	return &cfg, nil
}
