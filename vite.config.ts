import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
	plugins: [
		viteStaticCopy({
			targets: [
				{
					src: "node_modules/onnxruntime-web/dist/ort-wasm*.wasm",
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
	optimizeDeps: {
		esbuildOptions: {
			target: "esnext",
		},
	},
});
