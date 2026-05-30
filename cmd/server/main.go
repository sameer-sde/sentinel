// Sentinel — fraud detection serving layer.
//
// Day 8: A/B traffic split between two model versions.
// - bundleA = production model (always present)
// - bundleB = candidate model (optional)
// - splitPercent = % of traffic routed to B
//
// /admin/ab/setup    — load a candidate as B
// /admin/ab/split    — set split percentage
// /admin/ab/promote  — promote B to A, clear B
// /admin/ab/abort    — set split to 0, destroy B
// /admin/ab/status   — current state + per-variant traffic
//
// Cache is bypassed when B is loaded — otherwise we'd defeat the A/B test
// by returning cached v1 predictions for traffic routed to v2.
package main

import (
	"container/list"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"math/rand"
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
	modelsRoot       = "../../models"
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
	Variant          string  `json:"variant"` // "A" or "B"
}

type inferenceJob struct {
	features []float32
	bundle   *modelBundle // which model to use for THIS request
	reply    chan inferenceResult
}

type inferenceResult struct {
	fraudProba   float32
	class        int64
	batchSize    int
	modelVersion string
	err          error
}

type modelBundle struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	label   *ort.Tensor[int64]
	proba   *ort.Tensor[float32]
	version string
	path    string

	// Per-bundle mutex so we don't run two batches on the same session at once.
	// (Different bundles run in parallel — that's the win.)
	mu sync.Mutex
}

// -------------------- LRU cache --------------------

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

// loadBundleFromPath loads a model from a specific directory path.
// version is the directory name (e.g. "v1", "v2") used for reporting.
func loadBundleFromPath(dirPath, version string) (*modelBundle, error) {
	path := filepath.Join(dirPath, modelFile)

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

// loadCurrentBundle resolves the current symlink and loads that model.
func loadCurrentBundle() (*modelBundle, error) {
	resolved, err := filepath.EvalSymlinks(modelDir)
	if err != nil {
		return nil, fmt.Errorf("resolve symlink %s: %w", modelDir, err)
	}
	return loadBundleFromPath(modelDir, filepath.Base(resolved))
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

	// A/B state
	bundleA      atomic.Pointer[modelBundle]
	bundleB      atomic.Pointer[modelBundle]
	splitPercent atomic.Int32 // 0-100; percentage routed to B
	abMu         sync.Mutex   // serializes admin ops (setup/promote/abort)

	// Counters
	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	batchCount    atomic.Uint64
	cacheHits     atomic.Uint64
	cacheMisses   atomic.Uint64

	predictionsA atomic.Uint64
	predictionsB atomic.Uint64
	blocksA      atomic.Uint64
	blocksB      atomic.Uint64

	startedAt time.Time
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

	bundle, err := loadCurrentBundle()
	if err != nil {
		log.Fatalf("initial model load: %v", err)
	}
	log.Printf("✓ Loaded model A=%s from %s", bundle.version, bundle.path)

	s := &Server{
		jobs:      make(chan inferenceJob, jobQueueSize),
		threshold: cfg.OptimalThreshold,
		cache:     NewLRUCache(cacheCapacity, cacheTTL),
		startedAt: time.Now(),
	}
	s.bundleA.Store(bundle)
	log.Printf("✓ LRU cache ready: capacity=%d TTL=%v", cacheCapacity, cacheTTL)

	go s.batcher()

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/admin/version", s.handleVersion)
	mux.HandleFunc("/admin/reload", s.handleReload)
	mux.HandleFunc("/admin/ab/setup", s.handleABSetup)
	mux.HandleFunc("/admin/ab/split", s.handleABSplit)
	mux.HandleFunc("/admin/ab/promote", s.handleABPromote)
	mux.HandleFunc("/admin/ab/abort", s.handleABAbort)
	mux.HandleFunc("/admin/ab/status", s.handleABStatus)

	log.Printf("✓ Listening on http://localhost%s (maxBatch=%d maxWait=%v)",
		httpAddr, maxBatch, maxWait)
	if err := http.ListenAndServe(httpAddr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}

// -------------------- batcher --------------------

// batcher groups jobs by bundle. Different bundles get separate batches
// (we can't mix v1 and v2 inputs into one session.Run call).
func (s *Server) batcher() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("BATCHER PANIC: %v", r)
		}
	}()

	log.Println("✓ Batcher goroutine ready")

	for {
		first, ok := <-s.jobs
		if !ok {
			return
		}

		// Collect jobs only for the same bundle as `first`.
		// Jobs for other bundles get held over to next iteration via local buffer.
		// Simpler approach: do one batch per iteration with whatever bundle 'first' has,
		// and skip jobs targeting a different bundle by handling them separately.
		//
		// Here we keep it simple: collect up to maxBatch jobs for first.bundle,
		// and if a different-bundle job shows up we run it as its own micro-batch.
		batch := []inferenceJob{first}
		var deferredOtherBundle []inferenceJob

		deadline := time.After(maxWait)

	collect:
		for len(batch) < maxBatch {
			select {
			case j, ok := <-s.jobs:
				if !ok {
					break collect
				}
				if j.bundle == first.bundle {
					batch = append(batch, j)
				} else {
					deferredOtherBundle = append(deferredOtherBundle, j)
				}
			case <-deadline:
				break collect
			}
		}

		s.runBatch(first.bundle, batch)

		// Now process the deferred jobs (different bundle) as their own batches.
		// They'll likely be a small group, but correctness > optimality.
		for len(deferredOtherBundle) > 0 {
			bundle := deferredOtherBundle[0].bundle
			var same, other []inferenceJob
			for _, j := range deferredOtherBundle {
				if j.bundle == bundle {
					same = append(same, j)
				} else {
					other = append(other, j)
				}
			}
			s.runBatch(bundle, same)
			deferredOtherBundle = other
		}
	}
}

// runBatch executes one inference call for jobs sharing the same bundle.
func (s *Server) runBatch(bundle *modelBundle, jobs []inferenceJob) {
	if len(jobs) == 0 {
		return
	}

	// Lock this bundle so two batcher invocations don't tread on its tensors.
	bundle.mu.Lock()
	defer bundle.mu.Unlock()

	inputData := bundle.input.GetData()
	for i, j := range jobs {
		copy(inputData[i*numFeatures:(i+1)*numFeatures], j.features)
	}
	for i := len(jobs); i < maxBatch; i++ {
		for k := 0; k < numFeatures; k++ {
			inputData[i*numFeatures+k] = 0
		}
	}

	if err := bundle.session.Run(); err != nil {
		result := inferenceResult{err: err}
		for _, j := range jobs {
			j.reply <- result
		}
		log.Printf("batcher: inference error: %v", err)
		return
	}

	labels := bundle.label.GetData()
	probas := bundle.proba.GetData()
	batchSize := len(jobs)
	for i, j := range jobs {
		j.reply <- inferenceResult{
			fraudProba:   probas[i*2+1],
			class:        labels[i],
			batchSize:    batchSize,
			modelVersion: bundle.version,
		}
	}
	s.batchCount.Add(1)
}

// -------------------- routing --------------------

// pickBundle returns the bundle for THIS request based on splitPercent.
// Returns the bundle and the variant label ("A" or "B").
// If B isn't loaded (nil), always returns A regardless of splitPercent.
func (s *Server) pickBundle() (*modelBundle, string) {
	a := s.bundleA.Load()
	b := s.bundleB.Load()
	if b == nil {
		return a, "A"
	}
	if rand.Intn(100) < int(s.splitPercent.Load()) {
		return b, "B"
	}
	return a, "A"
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
	bundle, variant := s.pickBundle()
	abActive := s.bundleB.Load() != nil

	var fraudProba float32
	var predictedClass int64
	var batchSize int
	var cacheHit bool

	// Bypass cache when A/B is active (would defeat the test).
	if !abActive {
		if entry, ok := s.cache.Get(key); ok {
			fraudProba = entry.proba
			predictedClass = entry.class
			cacheHit = true
			s.cacheHits.Add(1)
		}
	}

	if !cacheHit {
		if !abActive {
			s.cacheMisses.Add(1)
		}
		reply := make(chan inferenceResult, 1)
		s.jobs <- inferenceJob{features: req.Features, bundle: bundle, reply: reply}
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
		// Only cache when A/B is not active.
		if !abActive {
			s.cache.Put(key, fraudProba, predictedClass)
		}
	}

	latencyMicros := time.Since(start).Microseconds()

	decision := "allow"
	if float64(fraudProba) >= s.threshold {
		decision = "block"
		s.totalBlock.Add(1)
		if variant == "B" {
			s.blocksB.Add(1)
		} else {
			s.blocksA.Add(1)
		}
	} else {
		s.totalAllow.Add(1)
	}
	s.totalRequests.Add(1)
	if variant == "B" {
		s.predictionsB.Add(1)
	} else {
		s.predictionsA.Add(1)
	}

	resp := PredictResponse{
		FraudProbability: fraudProba,
		PredictedClass:   predictedClass,
		Decision:         decision,
		ThresholdUsed:    s.threshold,
		LatencyMicros:    latencyMicros,
		BatchSize:        batchSize,
		CacheHit:         cacheHit,
		ModelVersion:     bundle.version,
		Variant:          variant,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
}

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	a := s.bundleA.Load()
	out := map[string]interface{}{
		"version_a": a.version,
		"path_a":    a.path,
	}
	if b := s.bundleB.Load(); b != nil {
		out["version_b"] = b.version
		out["path_b"] = b.path
		out["split_percent"] = s.splitPercent.Load()
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (s *Server) handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	s.abMu.Lock()
	defer s.abMu.Unlock()

	if s.bundleB.Load() != nil {
		http.Error(w, "A/B test active — abort or promote first", http.StatusConflict)
		return
	}

	oldBundle := s.bundleA.Load()
	newBundle, err := loadCurrentBundle()
	if err != nil {
		log.Printf("Reload failed: %v", err)
		http.Error(w, fmt.Sprintf("reload failed: %v", err), http.StatusInternalServerError)
		return
	}

	s.bundleA.Store(newBundle)
	s.cache.Clear()
	log.Printf("✓ Reload: A %s → %s", oldBundle.version, newBundle.version)

	go func(old *modelBundle) {
		time.Sleep(50 * time.Millisecond)
		old.Destroy()
	}(oldBundle)

	out := map[string]string{"status": "ok", "old_version": oldBundle.version, "new_version": newBundle.version}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// -------------------- A/B admin handlers --------------------

type abSetupReq struct {
	CandidateVersion string `json:"candidate_version"` // e.g. "v2"
}

func (s *Server) handleABSetup(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req abSetupReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	if req.CandidateVersion == "" {
		http.Error(w, "candidate_version required", http.StatusBadRequest)
		return
	}

	s.abMu.Lock()
	defer s.abMu.Unlock()

	if s.bundleB.Load() != nil {
		http.Error(w, "candidate already loaded — abort first", http.StatusConflict)
		return
	}

	candidatePath := filepath.Join(modelsRoot, req.CandidateVersion)
	if _, err := os.Stat(filepath.Join(candidatePath, modelFile)); err != nil {
		http.Error(w, fmt.Sprintf("candidate not found at %s: %v", candidatePath, err),
			http.StatusBadRequest)
		return
	}

	newB, err := loadBundleFromPath(candidatePath, req.CandidateVersion)
	if err != nil {
		http.Error(w, fmt.Sprintf("load candidate: %v", err), http.StatusInternalServerError)
		return
	}
	s.bundleB.Store(newB)
	s.splitPercent.Store(0) // start safe — admin must set split explicitly
	s.cache.Clear()

	log.Printf("✓ A/B setup: B=%s loaded, split=0%%", newB.version)

	out := map[string]string{"status": "ok", "candidate_version": newB.version}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

type abSplitReq struct {
	Percent int `json:"percent"`
}

func (s *Server) handleABSplit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req abSplitReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	if req.Percent < 0 || req.Percent > 100 {
		http.Error(w, "percent must be 0-100", http.StatusBadRequest)
		return
	}
	if s.bundleB.Load() == nil {
		http.Error(w, "no candidate loaded — call /admin/ab/setup first", http.StatusBadRequest)
		return
	}

	s.splitPercent.Store(int32(req.Percent))
	log.Printf("✓ A/B split set to %d%%", req.Percent)

	out := map[string]interface{}{"status": "ok", "split_percent": req.Percent}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (s *Server) handleABPromote(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	s.abMu.Lock()
	defer s.abMu.Unlock()

	b := s.bundleB.Load()
	if b == nil {
		http.Error(w, "no candidate to promote", http.StatusBadRequest)
		return
	}
	oldA := s.bundleA.Load()

	s.bundleA.Store(b)
	s.bundleB.Store(nil)
	s.splitPercent.Store(0)
	s.cache.Clear()

	log.Printf("✓ A/B promote: A %s → %s, B cleared", oldA.version, b.version)

	go func(old *modelBundle) {
		time.Sleep(50 * time.Millisecond)
		old.Destroy()
	}(oldA)

	out := map[string]string{"status": "ok", "promoted_to_a": b.version, "old_a_version": oldA.version}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (s *Server) handleABAbort(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	s.abMu.Lock()
	defer s.abMu.Unlock()

	b := s.bundleB.Load()
	if b == nil {
		http.Error(w, "no candidate to abort", http.StatusBadRequest)
		return
	}

	s.bundleB.Store(nil)
	s.splitPercent.Store(0)
	s.cache.Clear()

	log.Printf("✓ A/B abort: B=%s destroyed, split=0%%", b.version)

	go func(old *modelBundle) {
		time.Sleep(50 * time.Millisecond)
		old.Destroy()
	}(b)

	out := map[string]string{"status": "ok", "aborted_version": b.version}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (s *Server) handleABStatus(w http.ResponseWriter, r *http.Request) {
	a := s.bundleA.Load()
	predA := s.predictionsA.Load()
	predB := s.predictionsB.Load()
	blocksA := s.blocksA.Load()
	blocksB := s.blocksB.Load()

	blockRateA := 0.0
	blockRateB := 0.0
	if predA > 0 {
		blockRateA = float64(blocksA) / float64(predA)
	}
	if predB > 0 {
		blockRateB = float64(blocksB) / float64(predB)
	}

	out := map[string]interface{}{
		"version_a":      a.version,
		"predictions_a":  predA,
		"blocks_a":       blocksA,
		"block_rate_a":   blockRateA,
		"split_percent":  s.splitPercent.Load(),
		"ab_active":      s.bundleB.Load() != nil,
		"predictions_b":  predB,
		"blocks_b":       blocksB,
		"block_rate_b":   blockRateB,
	}
	if b := s.bundleB.Load(); b != nil {
		out["version_b"] = b.version
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
		"predictions_a":  s.predictionsA.Load(),
		"predictions_b":  s.predictionsB.Load(),
		"ab_active":      s.bundleB.Load() != nil,
		"split_percent":  s.splitPercent.Load(),
		"uptime_seconds": uptime,
		"avg_rps":        rps,
		"threshold":      s.threshold,
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
