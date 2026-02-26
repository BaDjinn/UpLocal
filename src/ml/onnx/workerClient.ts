// src/ml/onnx/workerClient.ts
type WorkerResult = { width: number; height: number; rgba: Uint8ClampedArray };

type Pending = {
	resolve: (r: WorkerResult) => void;
	reject: (e: Error) => void;
};

const worker = new Worker(new URL("./upscale.worker.ts", import.meta.url), {
	type: "module",
});

let ready = false;
const pending = new Map<string, Pending>();

worker.onmessage = (ev: MessageEvent<any>) => {
	const msg = ev.data;

	if (msg.type === "ready") {
		ready = true;
		return;
	}

	if (msg.type === "result") {
		const p = pending.get(msg.id);
		if (!p) return;
		pending.delete(msg.id);

		p.resolve({
			width: msg.outWidth,
			height: msg.outHeight,
			rgba: new Uint8ClampedArray(msg.rgba),
		});
		return;
	}

	if (msg.type === "error") {
		const p = msg.id ? pending.get(msg.id) : undefined;
		if (p) pending.delete(msg.id);
		(p?.reject ?? (() => {}))(new Error(msg.message));
		return;
	}

	// msg.type === "progress" -> qui puoi agganciare UI (spinner / progress bar)
	// console.log("[worker]", msg.stage, msg.id ?? "");
};

function uid() {
	return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export async function initUpscaler() {
	if (ready) return;
	worker.postMessage({ type: "init" });
	// opzionale: attendi davvero "ready" con una promise, se vuoi bloccare UI finché non carica
}

export function upscaleImageInWorker(imageData: ImageData): Promise<ImageData> {
	const id = uid();
	const rgba = imageData.data.buffer.slice(0); // buffer “owned” da trasferire

	return new Promise((resolve, reject) => {
		pending.set(id, {
			resolve: (r) =>
				resolve(
					new ImageData(new Uint8ClampedArray(r.rgba), r.width, r.height),
				),
			reject,
		});
		worker.postMessage(
			{
				type: "upscale",
				id,
				width: imageData.width,
				height: imageData.height,
				rgba,
			},
			[rgba],
		);
	});
}
