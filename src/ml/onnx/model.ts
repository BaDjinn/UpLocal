import * as ort from "onnxruntime-web";
//import * as dotenv from "dotenv";

//dotenv.config();

// Vite exposes env vars prefixed with VITE_ via import.meta.env

const MODEL_URL = import.meta.env.VITE_ONNX_MODEL_URL; //https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx

if (!MODEL_URL) {
  throw new Error("VITE_ONNX_MODEL_URL is not defined");
}


let session: ort.InferenceSession | null = null;

export async function loadModel() {
	if (session) {
		return session;
	}

	// Usa i file WASM da CDN
	ort.env.wasm.wasmPaths =
		"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";
	ort.env.wasm.numThreads = 1;

	session = await ort.InferenceSession.create(MODEL_URL, {
		executionProviders: ["webgpu", "wasm"],
	});

	return session;
}
