// Sentinel — fraud detection serving layer.
//
// Day 9: monitoring + drift detection.
// - Drift tracker computes per-feature running mean/stddev (Welford's algo)
// - Baseline locked after first N requests
// - /admin/drift reports current vs baseline z-scores
// - /metrics/prom exposes Prometheus-format text
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
	"sort"
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

	// Drift detection
	baselineSampleSize = 500 // capture baseline from first N requests
	driftZThreshold    = 3.0 // |z| > 3.0 = drift
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
	Variant          string  `json:"variant"`
}

type inferenceJob struct {
	features []float32
	bundle   *modelBundle
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
	mu      sync.Mutex
}

// -------------------- drift tracker --------------------

// featureStats holds running mean and variance using Welford's online algorithm.
// O(1) update per sample, numerically stable.
type featureStats struct {
	count uint64
	mean  float64
	m2    float64 // sum of squared deviations from mean
}

func (fs *featureStats) Add(x float64) {
	fs.count++
	delta := x - fs.mean
	fs.mean += delta / float64(fs.count)
	delta2 := x - fs.mean
	fs.m2 += delta * delta2
}

func (fs *featureStats) Variance() float64 {
	if fs.count < 2 {
		return 0
	}
	return fs.m2 / float64(fs.count-1)
}

func (fs *featureStats) StdDev() float64 {
	return math.Sqrt(fs.Variance())
}

// DriftTracker holds baseline (locked after baselineSampleSize) and current stats.
type DriftTracker struct {
	mu       sync.Mutex
	baseline [numFeatures]featureStats
	current  [numFeatures]featureStats
	totalObserved uint64
	baselineLocked bool
}

func NewDriftTracker() *DriftTracker {
	return &DriftTracker{}
}

// Observe records one sample (a feature vector).
// During the baseline phase, samples accumulate into baseline.
// After lock, samples accumulate into current — current can be reset for a sliding window.
func (d *DriftTracker) Observe(features []float32) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.totalObserved++

	if !d.baselineLocked {
		for i := 0; i < numFeatures; i++ {
			d.baseline[i].Add(float64(features[i]))
		}
		if d.totalObserved >= baselineSampleSize {
			d.baselineLocked = true
			log.Printf("✓ Drift baseline locked after %d observations", d.totalObserved)
		}
		return
	}

	for i := 0; i < numFeatures; i++ {
		d.current[i].Add(float64(features[i]))
	}
}

// ResetCurrent clears the current window (call periodically for sliding window).
func (d *DriftTracker) ResetCurrent() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for i := range d.current {
		d.current[i] = featureStats{}
	}
}

type driftReport struct {
	FeatureIndex int     `json:"feature_index"`
	BaselineMean float64 `json:"baseline_mean"`
	BaselineStd  float64 `json:"baseline_std"`
	CurrentMean  float64 `json:"current_mean"`
	ZScore       float64 `json:"z_score"`
	Drifted      bool    `json:"drifted"`
}

type driftSummary struct {
	BaselineLocked  bool          `json:"baseline_locked"`
	BaselineCount   uint64        `json:"baseline_count"`
	CurrentCount    uint64        `json:"current_count"`
	DriftDetected   bool          `json:"drift_detected"`
	DriftedFeatures []int         `json:"drifted_features"`
	PerFeature      []driftReport `json:"per_feature"`
}

func (d *DriftTracker) Report() driftSummary {
	d.mu.Lock()
	defer d.mu.Unlock()

	var summary driftSummary
	summary.BaselineLocked = d.baselineLocked
	if d.baselineLocked {
		summary.BaselineCount = baselineSampleSize
	} else {
		summary.BaselineCount = d.totalObserved
	}
	summary.CurrentCount = d.current[0].count

	for i := 0; i < numFeatures; i++ {
		bs := d.baseline[i]
		cs := d.current[i]

		report := driftReport{
			FeatureIndex: i,
			BaselineMean: bs.mean,
			BaselineStd:  bs.StdDev(),
			CurrentMean:  cs.mean,
		}

		// z-score only meaningful when baseline is locked AND we have current samples
		if d.baselineLocked && cs.count >= 30 && bs.StdDev() > 0 {
			// z = (current_mean - baseline_mean) / (baseline_std / sqrt(N))
			se := bs.StdDev() / math.Sqrt(float64(cs.count))
			report.ZScore = (cs.mean - bs.mean) / se
			if math.Abs(report.ZScore) > driftZThreshold {
				report.Drifted = true
				summary.DriftDetected = true
				summary.DriftedFeatures = append(summary.DriftedFeatures, i)
			}
		}

		summary.PerFeature = append(summary.PerFeature, report)
	}
	return summary
}

// -------------------- LRU cache (unchanged) --------------------

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

// -------------------- latency histogram --------------------

// LatencyHistogram bins observed latencies into a few buckets.
// Used for the Prometheus exposition.
type LatencyHistogram struct {
	mu      sync.Mutex
	buckets []int64 // counts per bucket
	bounds  []int64 // upper bounds (microseconds); last is +Inf implicit
	count   int64
	sum     int64
}

func NewLatencyHistogram() *LatencyHistogram {
	// Buckets in microseconds: 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000
	bounds := []int64{50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000}
	return &LatencyHistogram{
		bounds:  bounds,
		buckets: make([]int64, len(bounds)+1), // +1 for +Inf
	}
}

func (h *LatencyHistogram) Observe(latencyMicros int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.count++
	h.sum += latencyMicros
	idx := sort.Search(len(h.bounds), func(i int) bool {
		return h.bounds[i] >= latencyMicros
	})
	h.buckets[idx]++
}

// Snapshot returns a copy for export.
func (h *LatencyHistogram) Snapshot() ([]int64, []int64, int64, int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	b := make([]int64, len(h.buckets))
	copy(b, h.buckets)
	return h.bounds, b, h.count, h.sum
}

// -------------------- model loader (unchanged) --------------------

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
	drift     *DriftTracker
	latencyHist *LatencyHistogram

	bundleA      atomic.Pointer[modelBundle]
	bundleB      atomic.Pointer[modelBundle]
	splitPercent atomic.Int32
	abMu         sync.Mutex

	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	batchCount    atomic.Uint64
	cacheHits     atomic.Uint64
	cacheMisses   atomic.Uint64
	predictionsA  atomic.Uint64
	predictionsB  atomic.Uint64
	blocksA       atomic.Uint64
	blocksB       atomic.Uint64

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
		jobs:        make(chan inferenceJob, jobQueueSize),
		threshold:   cfg.OptimalThreshold,
		cache:       NewLRUCache(cacheCapacity, cacheTTL),
		drift:       NewDriftTracker(),
		latencyHist: NewLatencyHistogram(),
		startedAt:   time.Now(),
	}
	s.bundleA.Store(bundle)
	log.Printf("✓ LRU cache ready: capacity=%d TTL=%v", cacheCapacity, cacheTTL)
	log.Printf("✓ Drift tracker ready: baseline=%d samples, z-threshold=%.1f",
		baselineSampleSize, driftZThreshold)

	go s.batcher()
	go s.driftMonitor()

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/metrics/prom", s.handlePromMetrics)
	mux.HandleFunc("/admin/version", s.handleVersion)
	mux.HandleFunc("/admin/reload", s.handleReload)
	mux.HandleFunc("/admin/drift", s.handleDrift)
	mux.HandleFunc("/admin/ab/setup", s.handleABSetup)
	mux.HandleFunc("/admin/ab/split", s.handleABSplit)
	mux.HandleFunc("/admin/ab/promote", s.handleABPromote)
	mux.HandleFunc("/admin/ab/abort", s.handleABAbort)
	mux.HandleFunc("/admin/ab/status", s.handleABStatus)

	log.Printf("✓ Listening on http://localhost%s", httpAddr)
	if err := http.ListenAndServe(httpAddr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}

// driftMonitor: every 30s, log a summary of drift status.
func (s *Server) driftMonitor() {
	tick := time.NewTicker(30 * time.Second)
	defer tick.Stop()
	for range tick.C {
		summary := s.drift.Report()
		if summary.DriftDetected {
			log.Printf("⚠️  DRIFT DETECTED on features %v", summary.DriftedFeatures)
		}
	}
}

// -------------------- batcher (same as Day 8) --------------------

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
		batch := []inferenceJob{first}
		var deferredOther []inferenceJob
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
					deferredOther = append(deferredOther, j)
				}
			case <-deadline:
				break collect
			}
		}
		s.runBatch(first.bundle, batch)
		for len(deferredOther) > 0 {
			b := deferredOther[0].bundle
			var same, other []inferenceJob
			for _, j := range deferredOther {
				if j.bundle == b {
					same = append(same, j)
				} else {
					other = append(other, j)
				}
			}
			s.runBatch(b, same)
			deferredOther = other
		}
	}
}

func (s *Server) runBatch(bundle *modelBundle, jobs []inferenceJob) {
	if len(jobs) == 0 {
		return
	}
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
	for i, j := range jobs {
		j.reply <- inferenceResult{
			fraudProba:   probas[i*2+1],
			class:        labels[i],
			batchSize:    len(jobs),
			modelVersion: bundle.version,
		}
	}
	s.batchCount.Add(1)
}

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

	// Feed drift tracker.
	s.drift.Observe(req.Features)

	start := time.Now()
	key := hashFeatures(req.Features)
	bundle, variant := s.pickBundle()
	abActive := s.bundleB.Load() != nil

	var fraudProba float32
	var predictedClass int64
	var batchSize int
	var cacheHit bool

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
			http.Error(w, "inference failed", http.StatusInternalServerError)
			return
		}
		fraudProba = result.fraudProba
		predictedClass = result.class
		batchSize = result.batchSize
		if !abActive {
			s.cache.Put(key, fraudProba, predictedClass)
		}
	}

	latencyMicros := time.Since(start).Microseconds()
	s.latencyHist.Observe(latencyMicros)

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

func (s *Server) handleDrift(w http.ResponseWriter, r *http.Request) {
	summary := s.drift.Report()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(summary)
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

type abSetupReq struct {
	CandidateVersion string `json:"candidate_version"`
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
	s.splitPercent.Store(0)
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
	log.Printf("✓ A/B abort: B=%s destroyed", b.version)
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
		"version_a":     a.version,
		"predictions_a": predA,
		"blocks_a":      blocksA,
		"block_rate_a":  blockRateA,
		"split_percent": s.splitPercent.Load(),
		"ab_active":     s.bundleB.Load() != nil,
		"predictions_b": predB,
		"blocks_b":      blocksB,
		"block_rate_b":  blockRateB,
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
	driftSum := s.drift.Report()
	out := map[string]interface{}{
		"total_requests":  total,
		"allow":           s.totalAllow.Load(),
		"block":           s.totalBlock.Load(),
		"errors":          s.totalErrors.Load(),
		"batches_run":     batches,
		"avg_batch_size":  avgBatch,
		"cache_hits":      hits,
		"cache_misses":    misses,
		"cache_hit_rate":  hitRate,
		"cache_size":      s.cache.Size(),
		"predictions_a":   s.predictionsA.Load(),
		"predictions_b":   s.predictionsB.Load(),
		"ab_active":       s.bundleB.Load() != nil,
		"split_percent":   s.splitPercent.Load(),
		"drift_detected":  driftSum.DriftDetected,
		"baseline_locked": driftSum.BaselineLocked,
		"uptime_seconds":  uptime,
		"avg_rps":         rps,
		"threshold":       s.threshold,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// handlePromMetrics: Prometheus text exposition format.
func (s *Server) handlePromMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintln(w, "# HELP sentinel_requests_total Total prediction requests")
	fmt.Fprintln(w, "# TYPE sentinel_requests_total counter")
	fmt.Fprintf(w, "sentinel_requests_total %d\n", s.totalRequests.Load())

	fmt.Fprintln(w, "# HELP sentinel_decisions_total Predictions by decision")
	fmt.Fprintln(w, "# TYPE sentinel_decisions_total counter")
	fmt.Fprintf(w, "sentinel_decisions_total{decision=\"allow\"} %d\n", s.totalAllow.Load())
	fmt.Fprintf(w, "sentinel_decisions_total{decision=\"block\"} %d\n", s.totalBlock.Load())

	fmt.Fprintln(w, "# HELP sentinel_errors_total Inference errors")
	fmt.Fprintln(w, "# TYPE sentinel_errors_total counter")
	fmt.Fprintf(w, "sentinel_errors_total %d\n", s.totalErrors.Load())

	fmt.Fprintln(w, "# HELP sentinel_cache_total Cache hit/miss counts")
	fmt.Fprintln(w, "# TYPE sentinel_cache_total counter")
	fmt.Fprintf(w, "sentinel_cache_total{result=\"hit\"} %d\n", s.cacheHits.Load())
	fmt.Fprintf(w, "sentinel_cache_total{result=\"miss\"} %d\n", s.cacheMisses.Load())

	fmt.Fprintln(w, "# HELP sentinel_variant_predictions_total Predictions per A/B variant")
	fmt.Fprintln(w, "# TYPE sentinel_variant_predictions_total counter")
	fmt.Fprintf(w, "sentinel_variant_predictions_total{variant=\"A\"} %d\n", s.predictionsA.Load())
	fmt.Fprintf(w, "sentinel_variant_predictions_total{variant=\"B\"} %d\n", s.predictionsB.Load())

	// Latency histogram in Prometheus format.
	bounds, buckets, count, sum := s.latencyHist.Snapshot()
	fmt.Fprintln(w, "# HELP sentinel_latency_microseconds Predict latency in µs")
	fmt.Fprintln(w, "# TYPE sentinel_latency_microseconds histogram")
	cumulative := int64(0)
	for i, b := range bounds {
		cumulative += buckets[i]
		fmt.Fprintf(w, "sentinel_latency_microseconds_bucket{le=\"%d\"} %d\n", b, cumulative)
	}
	cumulative += buckets[len(buckets)-1]
	fmt.Fprintf(w, "sentinel_latency_microseconds_bucket{le=\"+Inf\"} %d\n", cumulative)
	fmt.Fprintf(w, "sentinel_latency_microseconds_count %d\n", count)
	fmt.Fprintf(w, "sentinel_latency_microseconds_sum %d\n", sum)

	// Drift flag as a gauge.
	driftFlag := 0
	driftSum := s.drift.Report()
	if driftSum.DriftDetected {
		driftFlag = 1
	}
	fmt.Fprintln(w, "# HELP sentinel_drift_detected 1 if data drift detected")
	fmt.Fprintln(w, "# TYPE sentinel_drift_detected gauge")
	fmt.Fprintf(w, "sentinel_drift_detected %d\n", driftFlag)
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
