# Development

`purr-app` is the Dioxus interface shared by the desktop app and the browser build.

```
crates/purr-app/
├─ assets/
│  ├─ styling/design.css   # design tokens, glass materials, primitives, shell and layers
│  ├─ styling/screens.css  # screen-level styles (Transcribe, Result, Models, History, Settings, Logs)
│  ├─ fonts/               # bundled Inter
│  └─ icons/               # app mark and favicon
├─ index.html              # the web page template (language, description, theme colours)
├─ src/
│  ├─ main.rs              # entry point: desktop window or web launch
│  ├─ lib.rs               # root component, providers and stylesheet links
│  ├─ components/          # design-system components, one per file
│  ├─ views/               # routed screens and their parts
│  ├─ platform/            # desktop and browser implementations of the Platform trait
│  └─ …                    # settings store, history, transcription session, model catalog
└─ Cargo.toml
```

The visual design is specified by `docs/design/purr-redesign-opus-v2.html`; the stylesheets are plain
CSS with no build step.

### Serving the app

The `dx` CLI must match the Dioxus version of the workspace (see `DIOXUS_VERSION` in
`.github/workflows/build.yml`).

```bash
dx serve -p purr-app --desktop   # desktop
dx serve -p purr-app --web       # browser
```

A plain `cargo build` of the desktop app runs without its stylesheets; use `dx` to get the assets.

### Web build

The browser build needs Node.js and npm: building `purr-wasm` runs `npm ci` and rollup to bundle the
transcription worker (`crates/purr-wasm/js`) and ships it, together with the ONNX runtime files, as dx assets.

```bash
dx serve -p purr-app --web --cross-origin-policy   # development, cross-origin isolated like production
dx bundle -p purr-app --release --web --out-dir dist-web   # production, deploy dist-web/public
```

`res/vercel-config.json` serves `index.html` for every route that is not a file, so deep links such as
`/settings` reload, and answers 404 for a missing file under `/assets/`. It also sends
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`: without
cross-origin isolation the CPU runtime runs on one thread, about half as fast.
