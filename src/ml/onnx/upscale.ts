import { loadModel } from "./model";

export async function upscaleImage(
    imageData: ImageData): Promise<ImageData> {
    const session = await loadModel();

    return outputImageData;
}