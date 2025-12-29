# Windows Server 2019 GUI Compatibility - Solution

**Status**: RESOLVED in v2.12.50.0

## Problem Statement

The Project Database Manager (`pro-project.exe`) and Master Data Loader (`pro-data-loader-gui.exe`) GUI applications fail to start on Windows Server 2019 with error:

```
Failed to start GUI: WGPU error: Failed to create wgpu adapter, no suitable adapter found.
```

## Root Cause

1. **Windows Server 2019 RDS sessions use Microsoft Basic Render Driver** - Only provides OpenGL 1.1
2. **No GPU available** - Headless/VM environments don't have GPU hardware
3. **wgpu requires**: Vulkan, DirectX 12, or OpenGL 3.3+ (none available on headless Windows Server)
4. **DirectX 12 WARP** - Only available on Windows 10/11 with Desktop Experience

## Solution: CLI-First Architecture

The GUI tools are **administrative utilities**, not runtime requirements. The solution is to use CLI commands on Windows Server:

### Project Database Manager

```powershell
# List all projects
pro-project.exe list

# Create a new project
pro-project.exe create --name MyProject

# Switch active project
pro-project.exe switch --name MyProject

# Show upgrade status
pro-project.exe status

# Upgrade databases
pro-project.exe upgrade --all
```

### Master Data Loader

```powershell
# Generate CSV templates
pro-data-loader.exe generate-templates C:\data

# Load master data
pro-data-loader.exe --csv-dir C:\data\master

# Or load individual files
pro-data-loader.exe --organizations orgs.csv --regions regions.csv --facilities fac.csv --providers prov.csv
```

## For GUI Access

Run GUI from a Windows 10/11 workstation with RDP to the server for database operations:

1. Copy `pro-project.exe` to your Windows 10/11 workstation
2. Configure database connection: `--db-host <server> --db-port 5432`
3. Run GUI: `pro-project.exe --gui`

## Implementation (v2.12.50.0)

- [x] `pro-project.exe` is CLI-first (no `windows_subsystem = "windows"`)
- [x] `pro-project.exe --gui` or `pro-project.exe gui` launches GUI
- [x] `pro-data-loader.exe` is pure CLI (no GUI)
- [x] `pro-data-loader-gui.exe` shows helpful error with CLI instructions if GUI fails
- [x] Clear error messages direct users to CLI alternatives
