import * as ort from "onnxruntime-web";
import { loadModel } from "./model";

export async function upscaleImage(imageData: ImageData): Promise<ImageData> {
	const session = await loadModel();

	const { width, height, data } = imageData;
	const scale = 4; // Real-ESRGAN x4

	// Estrai il canale Alpha (ogni 4 pixel: R, G, B, A)
	const alphaChannel = new Uint8ClampedArray(width * height);
	for (let i = 0; i < data.length; i += 4) {
		alphaChannel[i / 4] = data[i + 3];
	}

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
	const inputName = session.inputNames[0];
	const feeds: Record<string, ort.Tensor> = {
		[inputName]: tensor,
	};

	const results = await session.run(feeds);
	const outputTensor = results[session.outputNames[0]];

	// -----------------------------
	// 3. UPSCALE ALPHA CHANNEL
	// -----------------------------
	const outWidth = width * scale;
	const outHeight = height * scale;

	const upscaledAlpha = new Uint8ClampedArray(outWidth * outHeight);

	// Usa interpolazione bilineare per l'upscaling dell'Alpha
	for (let y = 0; y < outHeight; y++) {
		for (let x = 0; x < outWidth; x++) {
			const srcX = x / scale;
			const srcY = y / scale;

			const x0 = Math.floor(srcX);
			const x1 = Math.min(x0 + 1, width - 1);
			const y0 = Math.floor(srcY);
			const y1 = Math.min(y0 + 1, height - 1);

			const fx = srcX - x0;
			const fy = srcY - y0;

			const a00 = alphaChannel[y0 * width + x0];
			const a10 = alphaChannel[y0 * width + x1];
			const a01 = alphaChannel[y1 * width + x0];
			const a11 = alphaChannel[y1 * width + x1];

			const a0 = a00 * (1 - fx) + a10 * fx;
			const a1 = a01 * (1 - fx) + a11 * fx;
			const alpha = a0 * (1 - fy) + a1 * fy;

			upscaledAlpha[y * outWidth + x] = Math.round(alpha);
		}
	}

	// -----------------------------
	// 4. POSTPROCESS (NCHW -> HWC)
	// -----------------------------
	const outData = new Uint8ClampedArray(outWidth * outHeight * 4);
	const out = outputTensor.data as Float32Array;

	let o = 0;
	let q = 0;
	let alphaIdx = 0;
	while (o < out.length) {
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // R
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // G
		outData[q++] = Math.min(255, Math.max(0, out[o++] * 255)); // B
		outData[q++] = upscaledAlpha[alphaIdx++]; // A (upscalato)
	}

	const outputImageData = new ImageData(outData, outWidth, outHeight);

	return outputImageData;
}
