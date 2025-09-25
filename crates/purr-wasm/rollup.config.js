import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';

export default {
  input: 'src/worker.js',
  output: {
    file: 'dist/worker.js',
    format: 'iife',
    name: 'PurrWorker',
    sourcemap: true
  },
  plugins: [
    resolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs(),
    terser({
      compress: {
        drop_console: false // Keep console logs for debugging
      }
    })
  ],
  external: []
};