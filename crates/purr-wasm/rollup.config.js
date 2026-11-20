import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";

function bundle(input, file) {
    return {
        input,
        output: {
            file,
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
}

export default [
    bundle("js/worker.js", "dist/worker.js"),
    bundle("js/convert_worker.js", "dist/convert_worker.js"),
];
