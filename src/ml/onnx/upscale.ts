import * as ort from "onnxruntime-web";
import { loadModel } from "./model";

export async function upscaleImage(imageData: ImageData): Promise<ImageData> {
	const session = await loadModel();

	const { width, height, data } = imageData;

	// -----------------------------
	// 1. PREPROCESS (HWC -> NCHW)
	// -----------------------------
	const input = new Float32Array(width * height * 3);

	let p = 0;
	for (let i = 0; i < data.length; i += 4) {
		input[p++] = data[i] / 255; // R
		input[p++] = data[i + 1] / 255; // G
		input[p++] = data[i + 2] / 255; // B
	}

	const tensor = new ort.Tensor("float32", input, [1, 3, height, width]);

	// -----------------------------
	// 2. INFERENCE
	// -----------------------------
	const feeds: Record<string, ort.Tensor> = {
		input: tensor, // ⚠️ nome input: vedi nota sotto
	};

	const results = await session.run(feeds);
	const outputTensor = results[Object.keys(results)[0]];

	// -----------------------------
	// 3. POSTPROCESS (NCHW -> HWC)
	// -----------------------------
	const scale = 4; // Real-ESRGAN x4
	const outWidth = width * scale;
	const outHeight = height * scale;

	const outData = new Uint8ClampedArray(outWidth * outHeight * 4);
	const out = outputTensor.data as Float32Array;

	let o = 0;
	let q = 0;
	while (o < out.length) {
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // R
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // G
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // B
		outData[q++] = 255; // A
	}

	// ✅ QUI viene finalmente definito correttamente
	const outputImageData = new ImageData(outData, outWidth, outHeight);

	return outputImageData;
}
