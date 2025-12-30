# NWG GUI Layout Fix Plan

## Problem Summary
Both GUI applications (pro-project and pro-data-loader-gui) have layout issues after migrating from egui/eframe to Native Windows GUI (NWG):
1. **Project Database Manager**: ListView shows black screen, buttons/labels truncated
2. **Data Loader GUI**: Labels truncated ("Organizati", "Regions", etc.)

## Root Cause Analysis

### ListView Black Screen
Based on official NWG documentation (https://docs.rs/native-windows-gui/1.0.13/native_windows_gui/struct.ListView.html):
- ListView needs proper flags for visibility: `VISIBLE | TAB_STOP`
- Adding `focus: true` can help ensure the control renders
- The `double_buffer` feature (enabled by default) helps prevent flickering

### Label Truncation
- Label widths are too narrow for their text content
- Need to increase width values in the `size` attribute

## Implementation Plan

### 1. Fix pro-project ListView (app.rs)

**Current ListView definition:**
```rust
#[nwg_control(parent: window, position: (10, 90), size: (875, 280),
    list_style: nwg::ListViewStyle::Detailed,
    ex_flags: nwg::ListViewExFlags::FULL_ROW_SELECT | nwg::ListViewExFlags::GRID)]
```

**Fix:**
- Add `flags: "VISIBLE|TAB_STOP"` to ensure proper visibility
- Add `focus: true` for initial focus
- Verify column setup in `on_init`

### 2. Fix pro-project Label Widths

Labels to widen:
- `lbl_password`: 65 → 70 pixels
- `lbl_count`: 180 → 200 pixels
- Button widths are adequate

### 3. Fix pro-data-loader-gui Label Widths

Labels to widen:
- `lbl_org`: "Organizations:" 120 → 140 pixels
- `lbl_region`: "Regions (Optional):" 120 → 140 pixels
- `lbl_provider`: "Providers (Optional):" 130 → 145 pixels
- Shift text inputs left edge from 140 → 155 to accommodate wider labels

## Files to Modify
- [x] `crates/pro-project/src/gui/app.rs`
- [x] `crates/pro-data-loader-gui/src/main.rs`

## Testing
1. Build both crates: `cargo build --release -p pro-project -p pro-data-loader-gui`
2. Run pro-project GUI and verify ListView renders with columns
3. Run pro-data-loader-gui and verify all labels display fully
4. Build MSI installer
5. Install and test on Windows 11

## Version
2.12.58.0
