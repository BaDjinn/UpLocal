// src/main.ts
import { initUpscaler, upscaleImageInWorker } from "./ml/onnx/workerClient";

const input = document.createElement("input");
input.type = "file";
input.accept = "image/*";
document.body.appendChild(input);

// opzionale: pre-load modello al load della pagina (così il primo upscale non “stalla”)
initUpscaler();

input.onchange = async () => {
	try {
		const file = input.files?.[0];
		if (!file) return;

		const img = await createImageBitmap(file);

		const canvas = document.createElement("canvas");
		canvas.width = img.width;
		canvas.height = img.height;

		const ctx = canvas.getContext("2d")!;
		ctx.drawImage(img, 0, 0);

		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

		const out = await upscaleImageInWorker(imageData);

		canvas.width = out.width;
		canvas.height = out.height;
		ctx.putImageData(out, 0, 0);

		document.body.appendChild(canvas);
	} catch (error) {
		console.error("Upscaling failed:", error);
		alert("Errore durante l'upscaling dell'immagine");
	}
};
