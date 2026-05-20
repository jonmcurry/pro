# Plan: Remove PHI from claims importer logs

## Problem

Security review identified that subscriber first name, last name, patient
control number, and date of service were logged at `info!` level in
`claims_importer.rs` lines 788-803. This is leftover debug code that exposes
Protected Health Information in production log files, violating HIPAA.

## Resolution Checklist

- [x] Remove `info!` statements logging subscriber names and patient control number
- [x] Replace with `trace!()` containing only non-PHI fields (payer name, DOS, delay reason)
- [x] Add `trace` to tracing import
- [x] Update CHANGELOG.md with version 2.17.0.5
- [x] Commit and push

## Verification

After deployment:
- Set `LOG_LEVEL=info` and process a claim — no patient names should appear in logs
- Set `LOG_LEVEL=trace` — only payer name and operational codes appear (no subscriber PII)
