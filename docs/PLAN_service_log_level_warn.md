# Plan: Change default service.log level from "info" to "warn"

## Problem

The service.log file at `C:\ProgramData\Professional SMART\logs\service.log`
grows over 300MB because the default log level is "info", capturing every claim
processing detail, batch count, and API call trace.

## Resolution Checklist

- [x] Change default in `get_log_level()` from "info" to "warn"
- [x] Update invalid-level fallback message to reference "warn"
- [x] Update `EnvFilter` fallback from "info" to "warn"
- [x] Update `.env.example` default and comments
- [x] Update CHANGELOG.md with version 2.17.0.3
- [x] Commit and push

## Verification

After deployment:
1. Restart the service
2. Confirm new log entries are only WARN and ERROR level
3. To temporarily get verbose logging, set `LOG_LEVEL=info` in `.env` and restart
