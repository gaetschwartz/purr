use dioxus::prelude::*;

/// The overlay stack: popovers and sheets render here, inside `.app`, whatever component opened them.
#[derive(Clone, Copy)]
pub struct LayerHost {
    layers: Signal<Vec<(String, Element)>>,
}

impl LayerHost {
    fn put(mut self, id: &str, view: Element) {
        let mut layers = self.layers.write();
        match layers.iter_mut().find(|(existing, _)| existing == id) {
            Some(entry) => entry.1 = view,
            None => layers.push((id.to_owned(), view)),
        }
    }

    fn remove(mut self, id: &str) {
        self.layers.write().retain(|(existing, _)| existing != id);
    }
}

/// Provides the [`LayerHost`]; call once at the root, above [`LayerOutlet`].
pub fn use_layer_host_provider() -> LayerHost {
    use_context_provider(|| LayerHost {
        layers: Signal::new(Vec::new()),
    })
}

/// Renders the open overlays; place it as the last child of `.app`.
#[component]
pub fn LayerOutlet() -> Element {
    let host = use_context::<LayerHost>();
    let layers = host.layers.read().clone();
    rsx! {
        for (id , view) in layers {
            Fragment { key: "{id}", {view} }
        }
    }
}

/// Shows `view` in the overlay stack for as long as the calling component is mounted.
pub fn use_layer(id: &str, view: Element) {
    let host = use_context::<LayerHost>();
    host.put(id, view);
    let owned = use_hook(|| id.to_owned());
    use_drop(move || host.remove(&owned));
}

/// Which surface a layer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Popover,
    Sheet,
}

/// Where an anchored popover lines up with its anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PopoverAlign {
    #[default]
    Start,
    End,
}

impl PopoverAlign {
    const fn script_flag(self) -> &'static str {
        match self {
            Self::Start => "false",
            Self::End => "true",
        }
    }
}

fn behaviour_script(dom_id: &str, anchor: Option<&str>, align: PopoverAlign) -> String {
    let anchor = anchor.map_or_else(|| "null".to_owned(), |anchor| format!("{anchor:?}"));
    let align_end = align.script_flag();
    format!(
        r#"
const id = {dom_id:?};
const anchorId = {anchor};
const alignEnd = {align_end};
const root = document.getElementById(id);
if (root) {{
  const openers = window.__purrLayerOpeners || (window.__purrLayerOpeners = new Map());
  const active = document.activeElement;
  if (active && !root.contains(active)) openers.set(id, active);
  const pop = root.querySelector('.popover');
  const frame = root.closest('.app');
  const anchor = anchorId ? document.getElementById(anchorId) : null;
  if (pop) {{
    if (anchor && frame) {{
      const app = frame.getBoundingClientRect();
      const r = anchor.getBoundingClientRect();
      const pw = pop.offsetWidth, ph = pop.offsetHeight;
      let left = alignEnd ? r.right - app.left - pw : r.left - app.left;
      left = Math.max(12, Math.min(left, app.width - pw - 12));
      const below = app.height - (r.bottom - app.top) - 20;
      const above = r.top - app.top - 20;
      const placeBelow = ph <= below || below >= above;
      const room = Math.max(120, placeBelow ? below : above);
      const height = Math.min(ph, room);
      pop.style.maxHeight = room + 'px';
      const top = placeBelow ? r.bottom - app.top + 8 : r.top - app.top - 8 - height;
      pop.style.left = left + 'px';
      pop.style.top = top + 'px';
    }}
    pop.setAttribute('data-placed', '');
  }}
  const selector = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  const focusables = () => [...root.querySelectorAll(selector)].filter((el) => el.getClientRects().length > 0);
  const first = root.querySelector('[autofocus]') || focusables()[0] || root.querySelector('[role=dialog]');
  if (first) first.focus({{ preventScroll: true }});
  root.addEventListener('keydown', (event) => {{
    if (event.key === 'Escape') {{
      event.preventDefault();
      event.stopPropagation();
      dioxus.send(true);
      return;
    }}
    if (event.key === 'Tab') {{
      const list = focusables();
      if (!list.length) {{ event.preventDefault(); return; }}
      const index = list.indexOf(document.activeElement);
      if (event.shiftKey && index <= 0) {{ event.preventDefault(); list[list.length - 1].focus(); }}
      else if (!event.shiftKey && index === list.length - 1) {{ event.preventDefault(); list[0].focus(); }}
      return;
    }}
    if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && event.target.matches('[role=menuitem],[role=option]')) {{
      const items = [...root.querySelectorAll('[role=menuitem],[role=option]')].filter((el) => el.getAttribute('aria-disabled') !== 'true');
      const index = items.indexOf(event.target);
      if (index > -1) {{
        event.preventDefault();
        items[(index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length].focus();
      }}
    }}
  }});
  const parent = root.parentNode;
  if (parent) {{
    const watcher = new MutationObserver(() => {{
      if (root.isConnected) return;
      watcher.disconnect();
      const opener = openers.get(id);
      openers.delete(id);
      const focus = document.activeElement;
      const lost = !focus || focus === document.body || !focus.isConnected;
      if (opener && opener.isConnected && lost) opener.focus({{ preventScroll: true }});
    }});
    watcher.observe(parent, {{ childList: true }});
  }}
}}
"#
    )
}

/// The scrim, focus trap, Escape handling and anchoring shared by popovers and sheets.
#[component]
pub fn LayerFrame(
    dom_id: String,
    kind: LayerKind,
    anchor: Option<String>,
    #[props(default)] align: PopoverAlign,
    on_close: EventHandler<()>,
    children: Element,
) -> Element {
    let class = match kind {
        LayerKind::Popover => "layer pop",
        LayerKind::Sheet => "layer",
    };
    let script = behaviour_script(&dom_id, anchor.as_deref(), align);
    rsx! {
        div {
            id: dom_id,
            class,
            onmounted: move |_| {
                let script = script.clone();
                spawn(async move {
                    let mut behaviour = document::eval(&script);
                    while behaviour.recv::<bool>().await.is_ok() {
                        on_close.call(());
                    }
                });
            },
            div { class: "scrim", onclick: move |_| on_close.call(()) }
            {children}
        }
    }
}
