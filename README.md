# UpLocal

Local-first image upscaling in the browser.

# Real-ESRGAN Web (ONNX)

Image upscaling **entirely in the browser** using  
**ONNX Runtime Web + WebGPU**.

- No uploads
- No backend
- Privacy-first

> 🚧 Early PoC – UI borrowed from `web-realesrgan`, runtime replaced with ONNX.

## Goals

- Real-ESRGAN inference via ONNX Runtime Web
- WebGPU with WASM fallback
- Static deployment (Netlify / Pages)

## Status

Work in progress.

## .ENV

```
VITE_ONNX_MODEL_URL="https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx"
```
