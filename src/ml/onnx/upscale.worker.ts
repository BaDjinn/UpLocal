// src/ml/onnx/upscale.worker.ts
import * as ort from "onnxruntime-web";

type UpscaleRequest =
	| { type: "init" }
	| {
			type: "upscale";
			id: string;
			width: number;
			height: number;
			rgba: ArrayBuffer;
	  };

type UpscaleResponse =
	| { type: "ready" }
	| { type: "progress"; stage: string; id?: string }
	| {
			type: "result";
			id: string;
			outWidth: number;
			outHeight: number;
			rgba: ArrayBuffer;
	  }
	| { type: "error"; id?: string; message: string };

const MODEL_URL = (import.meta as any).env?.VITE_ONNX_MODEL_URL as
	| string
	| undefined;
if (!MODEL_URL) {
	throw new Error("VITE_ONNX_MODEL_URL is not defined");
}

let session: ort.InferenceSession | null = null;

async function getSession() {
	if (session) return session;

	// Meglio NON usare CDN: nel tuo Vite config copi i wasm in /wasm (viteStaticCopy).
	// Quindi usa path locale.
	ort.env.wasm.wasmPaths = "/wasm/";
	ort.env.wasm.numThreads = 1;

	// WebGPU in worker: funziona sui browser moderni.
	session = await ort.InferenceSession.create(MODEL_URL!, {
		executionProviders: ["webgpu", "wasm"],
	});

	return session;
}

function upscaleAlphaBilinear(
	alpha: Uint8ClampedArray,
	w: number,
	h: number,
	scale: number,
	outW: number,
	outH: number,
) {
	const out = new Uint8ClampedArray(outW * outH);

	for (let y = 0; y < outH; y++) {
		for (let x = 0; x < outW; x++) {
			const srcX = x / scale;
			const srcY = y / scale;

			const x0 = Math.floor(srcX);
			const x1 = Math.min(x0 + 1, w - 1);
			const y0 = Math.floor(srcY);
			const y1 = Math.min(y0 + 1, h - 1);

			const fx = srcX - x0;
			const fy = srcY - y0;

			const a00 = alpha[y0 * w + x0];
			const a10 = alpha[y0 * w + x1];
			const a01 = alpha[y1 * w + x0];
			const a11 = alpha[y1 * w + x1];

			const a0 = a00 * (1 - fx) + a10 * fx;
			const a1 = a01 * (1 - fx) + a11 * fx;
			out[y * outW + x] = Math.round(a0 * (1 - fy) + a1 * fy);
		}
	}

	return out;
}

async function upscaleRGBA(
	width: number,
	height: number,
	rgbaBuffer: ArrayBuffer,
): Promise<{ outW: number; outH: number; outRGBA: ArrayBuffer }> {
	const sess = await getSession();

	const data = new Uint8ClampedArray(rgbaBuffer);
	const scale = 4;

	// Alpha channel
	const alpha = new Uint8ClampedArray(width * height);
	for (let i = 0, p = 0; i < data.length; i += 4, p++) alpha[p] = data[i + 3];

	// PREPROCESS HWC->NCHW float32
	const hw = width * height;
	const input = new Float32Array(hw * 3);
	for (let i = 0; i < hw; i++) {
		const px = i * 4;
		input[0 * hw + i] = data[px] / 255;
		input[1 * hw + i] = data[px + 1] / 255;
		input[2 * hw + i] = data[px + 2] / 255;
	}

	const tensor = new ort.Tensor("float32", input, [1, 3, height, width]);
	const inputName = sess.inputNames[0];

	const results = await sess.run({ [inputName]: tensor });
	const outTensor = results[sess.outputNames[0]];
	const out = outTensor.data as Float32Array;

	// Dimensioni output
	let outW = width * scale;
	let outH = height * scale;
	if (Array.isArray(outTensor.dims) && outTensor.dims.length >= 4) {
		outH = outTensor.dims[2];
		outW = outTensor.dims[3];
	}

	// Upscale alpha e ricomposizione
	const upA = upscaleAlphaBilinear(alpha, width, height, scale, outW, outH);

	const channelSize = outW * outH;
	const rOff = 0;
	const gOff = channelSize;
	const bOff = channelSize * 2;

	const outData = new Uint8ClampedArray(outW * outH * 4);
	let q = 0;
	for (let i = 0; i < channelSize; i++) {
		const rv = Math.min(255, Math.max(0, out[rOff + i] * 255));
		const gv = Math.min(255, Math.max(0, out[gOff + i] * 255));
		const bv = Math.min(255, Math.max(0, out[bOff + i] * 255));
		outData[q++] = Math.round(rv);
		outData[q++] = Math.round(gv);
		outData[q++] = Math.round(bv);
		outData[q++] = upA[i];
	}

	return { outW, outH, outRGBA: outData.buffer };
}

self.onmessage = async (ev: MessageEvent<UpscaleRequest>) => {
	const msg = ev.data;

	try {
		if (msg.type === "init") {
			(self as any).postMessage({
				type: "progress",
				stage: "loading-model",
			} satisfies UpscaleResponse);
			await getSession();
			(self as any).postMessage({ type: "ready" } satisfies UpscaleResponse);
			return;
		}

		if (msg.type === "upscale") {
			(self as any).postMessage({
				type: "progress",
				stage: "inference",
				id: msg.id,
			} satisfies UpscaleResponse);

			const { outW, outH, outRGBA } = await upscaleRGBA(
				msg.width,
				msg.height,
				msg.rgba,
			);

			// Transfer buffer back (zero-copy)
			(self as any).postMessage(
				{
					type: "result",
					id: msg.id,
					outWidth: outW,
					outHeight: outH,
					rgba: outRGBA,
				} satisfies UpscaleResponse,
				[outRGBA],
			);
			return;
		}
	} catch (e: any) {
		(self as any).postMessage({
			type: "error",
			id: (msg as any).id,
			message: e?.message ?? String(e),
		} satisfies UpscaleResponse);
	}
};
