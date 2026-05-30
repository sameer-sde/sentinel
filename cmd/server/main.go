// Sentinel — fraud detection serving layer.
//
// Day 6: LRU prediction cache + TTL on top of Day 5's request batching.
// Cache key = FNV-1a hash of the 30 feature floats. Cache hits skip the
// batcher entirely — microsecond response vs milliseconds for the model.
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
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	onnxRuntimeLib   = "../../onnxruntime/libonnxruntime.dylib"
	modelPath        = "../../models/fraud_model.onnx"
	thresholdCfgPath = "../../models/threshold_config.json"
	numFeatures      = 30
	httpAddr         = ":8080"

	// Batching.
	maxBatch     = 32
	maxWait      = 5 * time.Millisecond
	jobQueueSize = 256

	// Cache.
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
}

type inferenceJob struct {
	features []float32
	reply    chan inferenceResult
}

type inferenceResult struct {
	fraudProba float32
	class      int64
	batchSize  int
	err        error
}

// -------------------- LRU cache --------------------

type cacheEntry struct {
	key       uint64
	proba     float32
	class     int64
	expiresAt time.Time
}

// LRUCache: map + doubly-linked list. O(1) get/put.
// Mutex-protected because /predict handlers run concurrently.
type LRUCache struct {
	mu       sync.Mutex
	capacity int
	ttl      time.Duration
	items    map[uint64]*list.Element
	order    *list.List // front = MRU, back = LRU
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
		// expired — evict
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

	// evict LRU if over capacity
	if c.order.Len() > c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			oldEntry := oldest.Value.(*cacheEntry)
			delete(c.items, oldEntry.key)
			c.order.Remove(oldest)
		}
	}
}

func (c *LRUCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

// hashFeatures: FNV-1a over the IEEE-754 bytes of the 30 floats.
// Fast, non-cryptographic, deterministic — identical floats produce
// identical bytes produce identical hash.
func hashFeatures(features []float32) uint64 {
	h := fnv.New64a()
	buf := make([]byte, 4)
	for _, f := range features {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(f))
		h.Write(buf)
	}
	return h.Sum64()
}

// -------------------- server --------------------

type Server struct {
	jobs      chan inferenceJob
	threshold float64
	cache     *LRUCache

	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	batchCount    atomic.Uint64
	cacheHits     atomic.Uint64
	cacheMisses   atomic.Uint64
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

	s := &Server{
		jobs:      make(chan inferenceJob, jobQueueSize),
		threshold: cfg.OptimalThreshold,
		cache:     NewLRUCache(cacheCapacity, cacheTTL),
		startedAt: time.Now(),
	}
	log.Printf("✓ LRU cache ready: capacity=%d TTL=%v", cacheCapacity, cacheTTL)

	go s.batcher()

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)

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

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(maxBatch, numFeatures))
	if err != nil {
		log.Fatalf("batcher: create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	labelTensor, err := ort.NewEmptyTensor[int64](ort.NewShape(maxBatch))
	if err != nil {
		log.Fatalf("batcher: create label tensor: %v", err)
	}
	defer labelTensor.Destroy()

	probaTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(maxBatch, 2))
	if err != nil {
		log.Fatalf("batcher: create proba tensor: %v", err)
	}
	defer probaTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"label", "probabilities"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{labelTensor, probaTensor},
		nil,
	)
	if err != nil {
		log.Fatalf("batcher: create session: %v", err)
	}
	defer session.Destroy()

	log.Println("✓ Batcher goroutine ready, ONNX session bound")

	batch := make([]inferenceJob, 0, maxBatch)
	inputData := inputTensor.GetData()

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

		for i, j := range batch {
			copy(inputData[i*numFeatures:(i+1)*numFeatures], j.features)
		}
		for i := len(batch); i < maxBatch; i++ {
			for k := 0; k < numFeatures; k++ {
				inputData[i*numFeatures+k] = 0
			}
		}

		if err := session.Run(); err != nil {
			result := inferenceResult{err: err}
			for _, j := range batch {
				j.reply <- result
			}
			log.Printf("batcher: inference error: %v", err)
			continue
		}

		labels := labelTensor.GetData()
		probas := probaTensor.GetData()
		batchSize := len(batch)
		for i, j := range batch {
			j.reply <- inferenceResult{
				fraudProba: probas[i*2+1],
				class:      labels[i],
				batchSize:  batchSize,
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

	// 1. Cache lookup.
	if entry, ok := s.cache.Get(key); ok {
		fraudProba = entry.proba
		predictedClass = entry.class
		cacheHit = true
		batchSize = 0
		s.cacheHits.Add(1)
	} else {
		// 2. Cache miss — go through batcher.
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
		// Populate cache.
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
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
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
		"uptime_seconds": uptime,
		"avg_rps":        rps,
		"threshold":      s.threshold,
		"max_batch":      maxBatch,
		"max_wait_ms":    maxWait.Milliseconds(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// -------------------- helpers --------------------

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
