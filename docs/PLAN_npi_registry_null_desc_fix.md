# Plan: Fix NPI Registry API deserialization failure on null taxonomy desc

## Problem

NPI enrichment fails for providers whose taxonomy codes have `"desc": null` in
the CMS NPI Registry API response. Example: NPI 1760473185 (taxonomy code
`2085R0001X`). The error message is generic: "NPI Registry API call failed"
with no indication of the root cause (JSON deserialization failure).

## Root Cause

The `Taxonomy` struct in `client.rs` defines `desc: String`, but the CMS API
returns `null` for this field on certain taxonomy codes. Serde cannot
deserialize `null` into a non-optional `String`.

## Resolution Checklist

- [x] Change `Taxonomy.desc` from `String` to `Option<String>` in `client.rs`
- [x] Update `worker.rs` to handle `Option<String>` with fallback `"Unknown"`
- [x] Update CHANGELOG.md with version 2.17.0.2
- [x] Commit and push

## Verification

After deployment, reset the failed queue item:

```sql
UPDATE claims.provider_enrichment_queue
SET status = 'PENDING', retry_count = 0, next_retry_at = NOW()
WHERE npi = '1760473185';
```

The enrichment worker will re-process and succeed on the next poll cycle.
