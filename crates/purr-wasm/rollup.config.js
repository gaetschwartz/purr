import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";

export default {
    input: "js/worker.js",
    output: {
        file: "dist/worker.js",
        format: "es",
        sourcemap: false,
    },
    plugins: [resolve({ browser: true, preferBuiltins: false }), commonjs(), terser()],
    onwarn(warning, warn) {
        if (warning.code === "CIRCULAR_DEPENDENCY" || warning.code === "INVALID_ANNOTATION") {
            return;
        }
        warn(warning);
    },
};
