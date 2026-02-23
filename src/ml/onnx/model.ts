import * as ort from "onnxruntime-web";

const MODEL_URL =
	"https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/RealESRGAN_x4plus.onnx";

let session: ort.InferenceSession | null = null;

export async function loadModel() {
	if (session) {
		return session;
	}

	ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;

	session = await ort.InferenceSession.create(MODEL_URL, {
		executionProviders: ["webgpu", "wasm"],
	});

	return session;
}
