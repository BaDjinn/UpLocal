import * as ort from "onnxruntime-web";

const MODEL_URL =
	"https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/RealESRGAN_x4plus.onnx";

let session: ort.InferenceSession | null = null;

export async function loadModel() {
	if (session) {
		return session;
	}

	// Usa i file WASM da CDN
	ort.env.wasm.wasmPaths =
		"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
	ort.env.wasm.numThreads = 1;

	session = await ort.InferenceSession.create(MODEL_URL, {
		executionProviders: ["webgpu", "wasm"],
	});

	return session;
}
