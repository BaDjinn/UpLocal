// src/log.ts
export type LogLevel = "error" | "warn" | "info" | "debug" | "trace";

function now() {
	const d = new Date();
	return d.toISOString().slice(11, 23); // HH:MM:SS.mmm
}

function isVerboseEnabled() {
	// 1) env build-time
	const envVerbose = (import.meta as any).env?.VITE_VERBOSE;
	if (envVerbose === "1" || envVerbose === "true") return true;

	// 2) runtime switch
	try {
		return localStorage.getItem("verbose") === "1";
	} catch {
		return false;
	}
}

const VERBOSE = isVerboseEnabled();

const levelOrder: Record<LogLevel, number> = {
	error: 0,
	warn: 1,
	info: 2,
	debug: 3,
	trace: 4,
};

export function makeLogger(scope: string, minLevel: LogLevel = "debug") {
	const enabled = VERBOSE;

	function log(level: LogLevel, ...args: any[]) {
		if (!enabled) return;
		if (levelOrder[level] > levelOrder[minLevel]) return;

		const prefix = `[${now()}][${scope}][${level}]`;
		// console[level] non esiste per trace in alcuni browser
		const fn =
			level === "error"
				? console.error
				: level === "warn"
					? console.warn
					: level === "info"
						? console.info
						: level === "debug"
							? console.debug
							: console.log;

		fn(prefix, ...args);
	}

	return {
		error: (...a: any[]) => log("error", ...a),
		warn: (...a: any[]) => log("warn", ...a),
		info: (...a: any[]) => log("info", ...a),
		debug: (...a: any[]) => log("debug", ...a),
		trace: (...a: any[]) => log("trace", ...a),
		enabled,
	};
}
