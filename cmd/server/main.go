// Sentinel — fraud detection serving layer.
//
// Day 3 Phase C: load the model, run two hardcoded test transactions
// through it (one known-fraud, one known-legit), print the fraud
// probabilities. These must match Python's predictions exactly.
//
// Expected output:
//   Fraud transaction:  P(fraud) = 0.9998  (Python: 0.999861)
//   Legit transaction:  P(fraud) = 0.0001  (Python: 0.000085)
package main

import (
	"fmt"
	"log"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	onnxRuntimeLib = "../../onnxruntime/libonnxruntime.dylib"
	modelPath      = "../../models/fraud_model.onnx"
)

// One real fraud transaction from our test set, in feature order f0..f29.
// (Same row Python predicted 0.999861 on.)
var fraudTransaction = []float32{
	57007.000000, -1.271244, 2.462675, -2.851395, 2.324480,
	-1.372245, -0.948196, -3.065234, 1.166927, -2.268771,
	-4.881143, 2.255147, -4.686387, 0.652375, -6.174288,
	0.594380, -4.849692, -6.536521, -3.119094, 1.715494,
	0.560478, 0.652941, 0.081931, -0.221348, -0.523582,
	0.224228, 0.756335, 0.632800, 0.250187, 0.010000,
}

// One real legit transaction. (Python predicted 0.000085.)
var legitTransaction = []float32{
	160760.000000, -0.674466, 1.408105, -1.110622, -1.328366,
	1.388996, -1.308439, 1.885879, -0.614233, 0.311652,
	0.650757, -0.857785, -0.229961, -0.199817, 0.266371,
	-0.046544, -0.741398, -0.605617, -0.392568, -0.162648,
	0.394322, 0.080084, 0.810034, -0.224327, 0.707899,
	-0.135837, 0.045102, 0.533837, 0.291319, 23.000000,
}

const numFeatures = 30

func main() {
	// 1. Boot the runtime (same as Phase B).
	ort.SetSharedLibraryPath(onnxRuntimeLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()

	log.Println("✓ ONNX Runtime initialized")

	// 2. Create one shared input tensor of shape (1, 30) — a batch of 1.
	//    We'll reuse this tensor for both predictions by overwriting its data.
	inputShape := ort.NewShape(1, numFeatures)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		log.Fatalf("create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// 3. The two model outputs we saw in Phase B:
	//    output[0] = "label"          shape (1,)   — predicted class (int64)
	//    output[1] = "probabilities"  shape (1, 2) — [P(legit), P(fraud)]
	labelTensor, err := ort.NewEmptyTensor[int64](ort.NewShape(1))
	if err != nil {
		log.Fatalf("create label tensor: %v", err)
	}
	defer labelTensor.Destroy()

	probaTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
	if err != nil {
		log.Fatalf("create probability tensor: %v", err)
	}
	defer probaTensor.Destroy()

	// 4. Create an inference session bound to those tensors.
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},                       // input tensor names (must match ONNX schema)
		[]string{"label", "probabilities"},      // output tensor names
		[]ort.ArbitraryTensor{inputTensor},      // input tensors
		[]ort.ArbitraryTensor{labelTensor, probaTensor}, // output tensors
		nil, // session options — defaults are fine for now
	)
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer session.Destroy()

	log.Println("✓ Inference session ready")

	// 5. Predict on the fraud transaction.
	runPrediction("FRAUD transaction", fraudTransaction,
		inputTensor, labelTensor, probaTensor, session, 0.999861)

	// 6. Predict on the legit transaction.
	runPrediction("LEGIT transaction", legitTransaction,
		inputTensor, labelTensor, probaTensor, session, 0.000085)

	fmt.Println("\nDone.")
}

// runPrediction copies features into the input tensor, runs the model,
// reads class + fraud probability, prints both and compares to Python.
func runPrediction(
	label string,
	features []float32,
	inputTensor *ort.Tensor[float32],
	labelTensor *ort.Tensor[int64],
	probaTensor *ort.Tensor[float32],
	session *ort.AdvancedSession,
	pythonProba float32,
) {
	if len(features) != numFeatures {
		log.Fatalf("%s: expected %d features, got %d", label, numFeatures, len(features))
	}

	// Copy features into the existing input tensor's backing slice.
	copy(inputTensor.GetData(), features)

	// Run the model. Outputs are written into labelTensor and probaTensor.
	if err := session.Run(); err != nil {
		log.Fatalf("%s: run failed: %v", label, err)
	}

	predictedClass := labelTensor.GetData()[0]
	probaData := probaTensor.GetData() // [P(legit), P(fraud)]
	fraudProba := probaData[1]

	fmt.Println("\n" + strRepeat("=", 60))
	fmt.Println(label)
	fmt.Println(strRepeat("=", 60))
	fmt.Printf("Predicted class:    %d\n", predictedClass)
	fmt.Printf("P(legit):           %.6f\n", probaData[0])
	fmt.Printf("P(fraud):           %.6f\n", fraudProba)
	fmt.Printf("Python P(fraud):    %.6f\n", pythonProba)
	fmt.Printf("Absolute diff:      %.2e\n", absFloat32(fraudProba-pythonProba))

	if absFloat32(fraudProba-pythonProba) < 1e-4 {
		fmt.Println("✓ MATCHES Python within tolerance")
	} else {
		fmt.Println("✗ DIVERGES from Python — investigate")
	}
}

func absFloat32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func strRepeat(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return out
}
