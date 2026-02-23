import { upscaleImage } from "./ml/onnx/upscale";

const input = document.createElement("input");
input.type = "file";
input.accept = "image/*";
document.body.appendChild(input);

input.onchange = async () => {
	const file = input.files?.[0];
	if (!file) return;

	const img = await createImageBitmap(file);

	const canvas = document.createElement("canvas");
	canvas.width = img.width;
	canvas.height = img.height;

	const ctx = canvas.getContext("2d")!;
	ctx.drawImage(img, 0, 0);

	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

	const out = await upscaleImage(imageData);

	canvas.width = out.width;
	canvas.height = out.height;
	ctx.putImageData(out, 0, 0);

	document.body.appendChild(canvas);
};
