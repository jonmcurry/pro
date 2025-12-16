# Add Configurable Log Level for Service

## Issue
Currently, the log level is hardcoded in `service.rs` and requires rebuilding the MSI to change between debug/info/warn/error levels.

## Solution
Add a `LOG_LEVEL` configuration option in the `.env` file that the service reads on startup.

## Changes Required

### 1. Update service.rs - init_service_logging function
- [x] Read LOG_LEVEL from environment/config file
- [x] Default to "info" if not specified
- [x] Support levels: trace, debug, info, warn, error

### 2. Update .env configuration
- [x] Add LOG_LEVEL option with documentation
- [x] Example: LOG_LEVEL=info

### 3. Documentation
- [x] Update .env.example with LOG_LEVEL option

## Implementation Details

### service.rs Changes
1. Added `load_env_config()` function - loads .env file BEFORE logging initialization
2. Added `get_log_level()` function - reads LOG_LEVEL, falls back to RUST_LOG, defaults to "info"
3. Reordered `run_service()` to call `load_env_config()` before `init_service_logging()`
4. Updated `init_service_logging()` to use `get_log_level()` for EnvFilter

### Priority Order
1. `LOG_LEVEL` environment variable (simple, user-friendly)
2. `RUST_LOG` environment variable (advanced, per-module control)
3. Default: "info"

## Configuration Options
| Value | Description |
|-------|-------------|
| trace | Most verbose - all trace logs |
| debug | Debug and above |
| info  | Info, warn, error (recommended for production) |
| warn  | Warnings and errors only |
| error | Errors only |

## Testing Checklist
- [ ] Set LOG_LEVEL=debug in .env, verify debug logs appear
- [ ] Set LOG_LEVEL=error in .env, verify only error logs appear
- [ ] Restart service without changing MSI, verify log level changes
- [ ] Test with missing LOG_LEVEL, verify defaults to info

## Version
- Previous: 2.9.3.0
- New: 2.9.4.0 (minor - configuration enhancement)
