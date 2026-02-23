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

	// Fill as NCHW: all R, then all G, then all B
	const hw = width * height;
	for (let i = 0; i < hw; i++) {
		const px = i * 4;
		input[0 * hw + i] = data[px] / 255; // R channel plane
		input[1 * hw + i] = data[px + 1] / 255; // G channel plane
		input[2 * hw + i] = data[px + 2] / 255; // B channel plane
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
	// Prefer reading the output tensor dimensions if available
	let outWidth = width * scale;
	let outHeight = height * scale;
	if (Array.isArray(outputTensor.dims) && outputTensor.dims.length >= 4) {
		// ONNX NCHW: [N, C, H, W]
		outHeight = outputTensor.dims[2];
		outWidth = outputTensor.dims[3];
	}

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

	const channelSize = outWidth * outHeight;
	const rOffset = 0;
	const gOffset = channelSize;
	const bOffset = channelSize * 2;

	let q = 0;
	for (let i = 0; i < channelSize; i++) {
		const rv = Math.min(255, Math.max(0, out[rOffset + i] * 255));
		const gv = Math.min(255, Math.max(0, out[gOffset + i] * 255));
		const bv = Math.min(255, Math.max(0, out[bOffset + i] * 255));
		outData[q++] = Math.round(rv);
		outData[q++] = Math.round(gv);
		outData[q++] = Math.round(bv);
		outData[q++] = upscaledAlpha[i];
	}

	const outputImageData = new ImageData(outData, outWidth, outHeight);

	return outputImageData;
}
