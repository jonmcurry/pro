# Plan: Warn when 837 claim is missing NM1*PR (payer identification)

## Problem

Claim PAAA0066925021 in sample2.837p.txt has `payer_id` and `payer_name` blank in
`claims.encounter_payer` despite having `payer_responsibility_code='P'`. The source
837 file is missing Loop 2010BB (NM1*PR segment) entirely for that subscriber loop.

The system correctly stores what's available (SBR responsibility code and filing
indicator) but the missing payer data was invisible — no log, no warning (Rule 3
violation).

## Root Cause

The source 837 file omits Loop 2010BB (NM1*PR - Payer Name) for subscriber loop
HL*580. This is valid-but-incomplete EDI: the X12 spec makes Loop 2010BB required
for 837P, but real-world clearinghouses occasionally emit files without it.

The parser faithfully processes what's there, but never alerts that mandatory data
is missing.

## Fix (v2.17.0.1 - build)

Add a WARN-level log in `parse_claim_info` (loops.rs) when `subscriber_sbr_seen`
is true but `primary_payer_captured` is false after all segments are processed.
This makes source data quality issues visible without rejecting the claim.

## Checklist

- [x] Add warn! log in parse_claim_info when SBR present but NM1*PR missing
- [x] Update CHANGELOG.md
- [x] Update version to 2.17.0.1
- [ ] Git commit and push
- [ ] Rebuild installer (Windows-only, skip on Mac dev)
