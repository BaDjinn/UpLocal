import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
	plugins: [
		viteStaticCopy({
			targets: [
				// WASM
				{
					src: "node_modules/onnxruntime-web/dist/ort-wasm*.wasm",
					dest: "wasm",
				},
				// JS modules (inclusi jsep)
				{
					src: "node_modules/onnxruntime-web/dist/ort-wasm*.mjs",
					dest: "wasm",
				},
				// (opzionale ma spesso utile: alcuni pacchetti hanno anche .js “shim”)
				{
					src: "node_modules/onnxruntime-web/dist/ort-wasm*.js",
					dest: "wasm",
				},
			],
		}),
	],
	server: {
		open: true,
		headers: {
			"Cross-Origin-Opener-Policy": "same-origin",
			"Cross-Origin-Embedder-Policy": "require-corp",
		},
	},
	build: {
		target: "esnext",
		minify: "esbuild",
	},
});
