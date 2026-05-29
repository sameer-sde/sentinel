// Sentinel — fraud detection serving layer.
//
// Day 4: HTTP server with /predict endpoint.
// One ONNX session shared across requests, mutex-protected.
// Loads the threshold config from Day 2 to make business decisions.
package main

import (
	"encoding/json"
	"fmt"
	"log"
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
)

// ThresholdConfig matches the JSON written by ml/03_threshold.py.
type ThresholdConfig struct {
	OptimalThreshold   float64 `json:"optimal_threshold"`
	CostFN             int     `json:"cost_fn"`
	CostFP             int     `json:"cost_fp"`
	ExpectedRecall     float64 `json:"expected_recall"`
	ExpectedPrecision  float64 `json:"expected_precision"`
}

// PredictRequest is what clients POST. 30 floats in feature order f0..f29.
type PredictRequest struct {
	Features []float32 `json:"features"`
}

// PredictResponse is what we return.
type PredictResponse struct {
	FraudProbability float32 `json:"fraud_probability"`
	PredictedClass   int64   `json:"predicted_class"`
	Decision         string  `json:"decision"`          // "allow" or "block"
	ThresholdUsed    float64 `json:"threshold_used"`
	LatencyMicros    int64   `json:"latency_us"`
}

// Server holds the long-lived ONNX session and the tensors it writes into.
// The mutex prevents two HTTP handlers from clobbering each other's tensors.
// (Day 5 will replace this with a pool of sessions for parallel throughput.)
type Server struct {
	mu       sync.Mutex
	session  *ort.AdvancedSession
	input    *ort.Tensor[float32]
	label    *ort.Tensor[int64]
	proba    *ort.Tensor[float32]
	threshold float64

	// metrics
	totalRequests atomic.Uint64
	totalAllow    atomic.Uint64
	totalBlock    atomic.Uint64
	totalErrors   atomic.Uint64
	startedAt     time.Time
}

func main() {
	// 1. Load the threshold config (the 0.41 cost-optimal threshold from Day 2).
	cfg, err := loadThresholdConfig(thresholdCfgPath)
	if err != nil {
		log.Fatalf("load threshold config: %v", err)
	}
	log.Printf("✓ Threshold config loaded: optimal=%.3f (recall=%.3f precision=%.3f)",
		cfg.OptimalThreshold, cfg.ExpectedRecall, cfg.ExpectedPrecision)

	// 2. Boot ONNX Runtime (same as Day 3).
	ort.SetSharedLibraryPath(onnxRuntimeLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()
	log.Println("✓ ONNX Runtime initialized")

	// 3. Create the long-lived tensors + session (also same as Day 3).
	input, err := ort.NewEmptyTensor[float32](ort.NewShape(1, numFeatures))
	if err != nil {
		log.Fatalf("create input tensor: %v", err)
	}
	defer input.Destroy()

	label, err := ort.NewEmptyTensor[int64](ort.NewShape(1))
	if err != nil {
		log.Fatalf("create label tensor: %v", err)
	}
	defer label.Destroy()

	proba, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
	if err != nil {
		log.Fatalf("create probability tensor: %v", err)
	}
	defer proba.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"label", "probabilities"},
		[]ort.ArbitraryTensor{input},
		[]ort.ArbitraryTensor{label, proba},
		nil,
	)
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer session.Destroy()
	log.Println("✓ Inference session ready")

	// 4. Wire up the server.
	s := &Server{
		session:   session,
		input:     input,
		label:     label,
		proba:     proba,
		threshold: cfg.OptimalThreshold,
		startedAt: time.Now(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)

	log.Printf("✓ Listening on http://localhost%s", httpAddr)
	if err := http.ListenAndServe(httpAddr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
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

	// Critical section: tensors aren't thread-safe. One prediction at a time.
	s.mu.Lock()
	copy(s.input.GetData(), req.Features)
	if err := s.session.Run(); err != nil {
		s.mu.Unlock()
		s.totalErrors.Add(1)
		log.Printf("inference error: %v", err)
		http.Error(w, "inference failed", http.StatusInternalServerError)
		return
	}
	predictedClass := s.label.GetData()[0]
	fraudProba := s.proba.GetData()[1]
	s.mu.Unlock()

	latencyMicros := time.Since(start).Microseconds()

	// Apply the business-cost-optimal threshold from Day 2.
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
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("response encode error: %v", err)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(s.startedAt).Seconds()
	total := s.totalRequests.Load()
	rps := 0.0
	if uptime > 0 {
		rps = float64(total) / uptime
	}
	out := map[string]interface{}{
		"total_requests":  total,
		"allow":           s.totalAllow.Load(),
		"block":           s.totalBlock.Load(),
		"errors":          s.totalErrors.Load(),
		"uptime_seconds":  uptime,
		"avg_rps":         rps,
		"threshold":       s.threshold,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}
