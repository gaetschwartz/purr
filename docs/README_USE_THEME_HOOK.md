# use_theme Hook

The `use_theme` hook provides a simplified interface for theme management functionality with reactive updates and proper integration with the ThemeProvider component.

## Features

- **System Theme Detection**: Automatically detects system theme preferences
- **Theme Persistence**: Uses the storage system to persist theme choices
- **Reactive Updates**: Provides reactive theme changes throughout the application
- **Theme Resolution**: Handles custom themes and inheritance properly
- **Simple API**: Provides the exact interface requested with `resolved_theme()` and `set_theme`

## Usage

### Basic Setup

First, make sure your app is wrapped with the `ThemeProvider`:

```rust
use purr_app::theme::{ThemeProvider, Theme, StorageType, CustomTheme, BaseTheme, ColorTokens};

#[component]
pub fn App() -> Element {
    let custom_themes = vec![
        CustomTheme {
            name: "solarized".to_string(),
            base: BaseTheme::Light,
            tokens: ColorTokens {
                primary: "#268bd2".to_string(),
                secondary: "#2aa198".to_string(),
                background: "#fdf6e3".to_string(),
                text: "#657b83".to_string(),
                error: Some("#dc322f".to_string()),
                warning: Some("#cb4b16".to_string()),
                success: Some("#859900".to_string()),
            },
        }
    ];

    rsx! {
        ThemeProvider {
            default_theme: Theme::System,
            storage_type: StorageType::LocalStorage,
            storage_name: "theme".to_string(),
            custom_themes: Some(custom_themes),
            // Your app content here
            MainContent {}
        }
    }
}
```

### Using the Hook

```rust
use purr_app::hooks::use_theme;
use purr_common::settings::Theme;

#[component]
pub fn ThemeToggle() -> Element {
    let ctx = use_theme();

    // Get the current resolved theme as a string
    let resolved_theme = (ctx.resolved_theme)();

    // Create onclick handlers for theme switching
    let onclick_dark = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::Dark)
    };

    let onclick_light = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::Light)
    };

    let onclick_system = {
        let set_theme = ctx.set_theme.clone();
        move |_| (set_theme)(Theme::System)
    };

    rsx! {
        div { class: "theme-controls",
            p { "Current theme: {resolved_theme}" }

            button { onclick: onclick_light, "Light" }
            button { onclick: onclick_dark, "Dark" }
            button { onclick: onclick_system, "System" }
        }
    }
}
```

## API Reference

### `use_theme() -> UseThemeResult`

Returns a `UseThemeResult` struct with the following fields:

#### `resolved_theme: Rc<dyn Fn() -> String>`

A function that returns the current resolved theme as a string. Possible values:
- `"light"` - Light theme is active
- `"dark"` - Dark theme is active
- `"custom:theme_name"` - Custom theme is active (e.g., "custom:solarized")

For `Theme::System`, this function will resolve to either "light" or "dark" based on:
1. System preference detection (on web platforms)
2. Color token analysis as fallback

#### `set_theme: Rc<dyn Fn(BaseTheme)>`

A function to change the current theme. Accepts values from `purr_common::settings::Theme`:
- `Theme::Light` - Switch to light theme
- `Theme::Dark` - Switch to dark theme
- `Theme::System` - Follow system preference

**Note**: Custom themes are handled through the main theme system's API, not through this simplified hook.

## Integration with Existing Systems

The hook integrates seamlessly with:

1. **Settings System**: Uses `purr_common::settings::Theme` for compatibility
2. **Storage System**: Themes are persisted automatically via the ThemeProvider
3. **Platform Abstraction**: Works with both web and desktop platforms
4. **Existing Components**: All existing UI components continue to work unchanged

## System Theme Detection

The hook provides intelligent system theme detection:

### Web Platforms
- Uses `prefers-color-scheme` media query
- Automatically updates when system preference changes
- Provides real-time reactive updates

### Desktop Platforms
- Falls back to color token analysis
- Can be extended with platform-specific detection

### Fallback Logic
When system theme cannot be detected:
1. Analyzes current color tokens
2. Compares background vs text brightness
3. Returns appropriate "light" or "dark" string

## Theme Resolution Logic

The `resolved_theme()` function handles different theme types:

```rust
match theme {
    Theme::Light => "light",
    Theme::Dark => "dark",
    Theme::System => /* detect system preference or analyze tokens */,
    Theme::Custom(name) => "custom:{name}",
}
```

## Error Handling

The hook is designed to be robust:
- Always returns valid theme strings
- Gracefully handles missing custom themes
- Provides sensible fallbacks for system detection failures
- No panics or unwraps in normal operation

## Performance Considerations

- Functions are wrapped in `Rc` for efficient cloning
- Theme resolution is cached and only updates when needed
- Storage operations are asynchronous and don't block the UI
- System theme listeners are cleaned up automatically

## Examples

See `/examples/theme_usage_example.rs` for a complete working example demonstrating:
- Theme switching buttons
- Real-time theme display
- Proper event handling
- Integration patterns

## Testing

The hook includes comprehensive tests for:
- Color brightness calculation
- Dark theme detection logic
- Theme resolution accuracy
- Edge case handling

Run tests with:
```bash
cargo test use_theme
```