// Sentinel — fraud detection serving layer.
//
// Day 5: request batching. A single dedicated goroutine owns the ONNX
// session. HTTP handlers don't touch it directly — they enqueue jobs
// onto a channel and wait for results. The batcher collects up to
// maxBatch requests or waits maxWait, then runs them through the model
// in a single ONNX call.
//
// Trade-off: a few ms added to per-request latency in exchange for much
// higher throughput on the model.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
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

	// Batching parameters.
	maxBatch     = 32
	maxWait      = 5 * time.Millisecond
	jobQueueSize = 256
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
	BatchSize        int     `json:"batch_size"` // how many requests were in this batch
}

// inferenceJob is what HTTP handlers push onto the batcher.
type inferenceJob struct {
	features []float32
	reply    chan inferenceResult
}

// inferenceResult is what the batcher returns to each handler.
type inferenceResult struct {
	fraudProba float32
	class      int64
	batchSize  int
	err        error
}

// -------------------- server --------------------

type Server struct {
	jobs      chan inferenceJob
	threshold float64

	// metrics
	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	batchCount    atomic.Uint64 // how many batches the batcher has run
	startedAt     time.Time
}

func main() {
	// 1. Threshold config.
	cfg, err := loadThresholdConfig(thresholdCfgPath)
	if err != nil {
		log.Fatalf("load threshold config: %v", err)
	}
	log.Printf("✓ Threshold config loaded: optimal=%.3f", cfg.OptimalThreshold)

	// 2. ONNX runtime init.
	ort.SetSharedLibraryPath(onnxRuntimeLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()
	log.Println("✓ ONNX Runtime initialized")

	// 3. Server state.
	s := &Server{
		jobs:      make(chan inferenceJob, jobQueueSize),
		threshold: cfg.OptimalThreshold,
		startedAt: time.Now(),
	}

	// 4. Start the batcher goroutine. It owns the ONNX session.
	go s.batcher()

	// 5. HTTP server.
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

// batcher runs in its own goroutine. It owns the ONNX session, accumulates
// inference jobs from s.jobs, and runs them in batches.
func (s *Server) batcher() {
	// Set up the ONNX session for a *batch*. Input shape (maxBatch, 30),
	// outputs (maxBatch,) and (maxBatch, 2). We always feed maxBatch rows;
	// when the real batch is smaller, the unused rows just get computed
	// and ignored. Slightly wasteful but keeps the code simple.
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

	// Reusable scratch space.
	batch := make([]inferenceJob, 0, maxBatch)
	inputData := inputTensor.GetData() // shape (maxBatch * numFeatures,)

	for {
		// Block waiting for the first job in a new batch.
		first, ok := <-s.jobs
		if !ok {
			return // channel closed; shutdown
		}
		batch = batch[:0]
		batch = append(batch, first)

		// Now collect more jobs until we hit maxBatch or maxWait.
		deadline := time.NewTimer(maxWait)
	collect:
		for len(batch) < maxBatch {
			select {
			case j, ok := <-s.jobs:
				if !ok {
					break collect
				}
				batch = append(batch, j)
			case <-deadline.C:
				break collect
			}
		}
		deadline.Stop()

		// Pack features into the input tensor.
		// Each row is numFeatures floats, contiguous.
		for i, j := range batch {
			copy(inputData[i*numFeatures:(i+1)*numFeatures], j.features)
		}
		// Zero out any remaining rows (defensive — model still computes
		// them, but we won't read those outputs).
		for i := len(batch); i < maxBatch; i++ {
			for k := 0; k < numFeatures; k++ {
				inputData[i*numFeatures+k] = 0
			}
		}

		// Run inference on the full batch.
		if err := session.Run(); err != nil {
			result := inferenceResult{err: err}
			for _, j := range batch {
				j.reply <- result
			}
			continue
		}

		// Extract outputs and dispatch back to each waiting handler.
		labels := labelTensor.GetData() // length maxBatch
		probas := probaTensor.GetData() // length maxBatch*2
		batchSize := len(batch)
		for i, j := range batch {
			j.reply <- inferenceResult{
				fraudProba: probas[i*2+1], // column 1 = P(fraud)
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

	// Submit to the batcher and wait for result.
	reply := make(chan inferenceResult, 1)
	s.jobs <- inferenceJob{features: req.Features, reply: reply}
	result := <-reply

	if result.err != nil {
		s.totalErrors.Add(1)
		log.Printf("inference error: %v", result.err)
		http.Error(w, "inference failed", http.StatusInternalServerError)
		return
	}

	latencyMicros := time.Since(start).Microseconds()

	decision := "allow"
	if float64(result.fraudProba) >= s.threshold {
		decision = "block"
		s.totalBlock.Add(1)
	} else {
		s.totalAllow.Add(1)
	}
	s.totalRequests.Add(1)

	resp := PredictResponse{
		FraudProbability: result.fraudProba,
		PredictedClass:   result.class,
		Decision:         decision,
		ThresholdUsed:    s.threshold,
		LatencyMicros:    latencyMicros,
		BatchSize:        result.batchSize,
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

	rps := 0.0
	avgBatch := 0.0
	if uptime > 0 {
		rps = float64(total) / uptime
	}
	if batches > 0 {
		avgBatch = float64(total) / float64(batches)
	}

	out := map[string]interface{}{
		"total_requests": total,
		"allow":          s.totalAllow.Load(),
		"block":          s.totalBlock.Load(),
		"errors":         s.totalErrors.Load(),
		"batches_run":    batches,
		"avg_batch_size": avgBatch,
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
