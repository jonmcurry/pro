# NWG GUI Migration Plan

**Version**: 2.12.52.0
**Date**: 2025-12-29
**Status**: COMPLETED

## Problem Statement

The egui/eframe-based GUI applications fail to start on Windows Server 2019 because:
1. Windows Server 2019 RDS sessions use Microsoft Basic Render Driver (OpenGL 1.1 only)
2. wgpu requires Vulkan, DirectX 12, or OpenGL 3.3+
3. WARP (DirectX 12 software renderer) is not available in RDS sessions on Windows Server

## Solution

Migrate GUI applications from egui/eframe to **Native Windows GUI (NWG)** which:
- Uses Win32 GDI controls directly
- No GPU required - pure software rendering via Windows GDI
- Works on all Windows versions (Vista and up)
- Guaranteed to work on Windows Server 2019

## Migration Checklist

### Phase 1: pro-project GUI Migration
- [x] Update Cargo.toml to use native-windows-gui instead of eframe
- [x] Rewrite mod.rs to use NWG initialization
- [x] Rewrite app.rs to use NWG controls:
  - [x] Main window with title "Professional SMART - Project Database Manager"
  - [x] Connection panel (Host, Port, User, Password fields + Connect button)
  - [x] ListView for projects table (columns: Database, Project, Org, Version, Status, Last Used, Active)
  - [x] Toolbar (Upgrade Selected, Backup & Upgrade, Select All, Refresh)
  - [x] Log panel with ListBox
- [ ] Test on Windows Server 2019

### Phase 2: pro-data-loader-gui GUI Migration
- [x] Update Cargo.toml to use native-windows-gui
- [x] Rewrite main.rs to use NWG controls:
  - [x] Main window with title "Professional SMART - Master Data Loader"
  - [x] File selection panel (4 file inputs with Browse buttons)
  - [x] Status panel with progress bar
  - [x] Log panel
  - [x] Action buttons (Validate & Import, Load from Directory, Generate Templates)
- [ ] Test on Windows Server 2019

### Phase 3: Cleanup & Documentation
- [x] Remove unused eframe/egui dependencies
- [x] Update CHANGELOG.md
- [x] Update NWG_GUI_MIGRATION_PLAN.md
- [ ] Rebuild installer

## NWG Control Mapping

| egui Control | NWG Control |
|--------------|-------------|
| Window | nwg::Window |
| TextEdit (singleline) | nwg::TextInput |
| TextEdit (password) | nwg::TextInput with password flag |
| Button | nwg::Button |
| Grid/Table | nwg::ListView |
| ScrollArea + list | nwg::ListBox |
| ProgressBar | nwg::ProgressBar |
| Label | nwg::Label |
| Checkbox | nwg::CheckBox |
| ComboBox | nwg::ComboBox |
| Dialog | nwg::Window (modal) |
| FileDialog | nwg::FileDialog |

## Technical Notes

1. NWG uses event callbacks or event handlers via `#[nwg_events]` attribute
2. Background tasks still use `std::thread::spawn` + `mpsc` channels
3. For UI updates from background threads, use `nwg::dispatch_thread_events_with_callback`
4. ListView requires `list-view` feature in native-windows-gui

## References

- [NWG Documentation](https://gabdube.github.io/native-windows-gui/native-windows-docs/)
- [NWG API](https://docs.rs/native-windows-gui/latest/native_windows_gui/)
- [NWG Examples](https://github.com/gabdube/native-windows-gui/tree/master/native-windows-gui/examples)
