# GUI DPI Scaling Fix Plan

## Problem Summary
Both GUI applications (pro-data-loader-gui and pro-project) show severely truncated labels and buttons. Screenshots show:
- "Organizations:" truncated
- "Regions" truncated (missing "(Optional):")
- Buttons show "ad from Director", "enerate Template", "/alidate & Impor"

## Root Cause Analysis

### The DPI Problem
1. **Manifest declares Per-Monitor V2 DPI awareness** - This tells Windows NOT to bitmap-scale the app
2. **NWG uses fixed pixel values** - All control positions and sizes are hardcoded in pixels
3. **Pixels are physical at high DPI** - At 125% scaling (120 DPI), controls designed for 96 DPI are 20% smaller than expected
4. **User's display is high DPI** - The visible truncation indicates 125% or higher scaling

### DPI/Scale Factor Reference
| Scale Factor | DPI Value | Effect on 100px control |
|--------------|-----------|-------------------------|
| 100% | 96 | 100px |
| 125% | 120 | 80px (truncated) |
| 150% | 144 | 67px (severely truncated) |
| 200% | 192 | 50px (half size) |

## Solution: Runtime DPI Scaling

### Approach
NWG provides `nwg::scale_factor()` which returns the current display scale factor (e.g., 1.25 for 125% DPI). We will:

1. Query `nwg::scale_factor()` at runtime
2. Calculate scaled dimensions: `scaled = (base * scale_factor) as i32`
3. Apply scaled dimensions to window and all controls in `on_init()`

### Why This Works
- Manifest declares DPI awareness so Windows sends true pixel values
- We query the actual scale factor at runtime
- We resize controls to match the scale factor
- Controls render at correct size on any DPI setting

### Alternative Considered: Remove DPI Awareness
- **Pros**: Simpler, Windows handles scaling automatically
- **Cons**: Bitmap scaling looks blurry, especially text
- **Decision**: Runtime scaling is the correct approach per Microsoft best practices

## Implementation

### Changes to pro-data-loader-gui/src/main.rs

1. Store base dimensions as constants
2. In `on_init()`, query `nwg::scale_factor()`
3. Resize window and reposition all controls using scaled values

### Changes to pro-project/src/gui/app.rs

Same approach as above.

## Files Modified
- [x] `crates/pro-data-loader-gui/src/main.rs` - Added `apply_dpi_scaling()` function
- [x] `crates/pro-project/src/gui/app.rs` - Added `apply_dpi_scaling()` function

## Testing Checklist
- [x] Build: `cargo build --release -p pro-project -p pro-data-loader-gui`
- [ ] Test at 100% scaling - controls should look normal
- [ ] Test at 125% scaling - controls should scale proportionally
- [ ] Test at 150% scaling - controls should scale proportionally
- [x] Build MSI installer v2.12.63.0
- [ ] Test installed application

## Version
2.12.63.0

## Completed
2025-12-30
