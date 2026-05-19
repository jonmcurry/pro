# Plan: Compute Encounter date_of_service_from/to as MIN/MAX of Service Lines

## Problem

Today, `Transaction837p.date_of_service_from` / `..._to` are set by whichever
mechanism happens to fire first:

1. The most recent claim-level `DTP*472` segment overwrites the prior value
   (last-claim-level-wins). No span aggregation.
2. If no claim-level `DTP*472` appeared at all, a post-loop fallback at
   `loops.rs:1138-1143` copies dates from `service_lines[0]` only.

In real prod 837P data, claims commonly have one `DTP*472*RD8` per service
line (Loop 2400) and none at claim level. Today's first-line fallback then
loads only the FIRST line's range onto the encounter, silently dropping the
end dates of every other service line. Reviewers looking at
`encounter.date_of_service_to` think the encounter ended earlier than it did.

This is a Rule 3 violation: data is loaded but wrong, with no log indicating
why.

## Option A semantics (chosen)

After the service-line parse loop finishes, ALWAYS recompute the encounter
span from the lines (when any service line is present):

```text
date_of_service_from = MIN(line.service_date_from for each line)
date_of_service_to   = MAX(line.service_date_to.unwrap_or(line.service_date_from) for each line)
```

Notes:

* `D8` (single-date) lines have `service_date_to == None`; the MAX falls back
  to `service_date_from` so they participate correctly.
* If a claim-level `DTP*472` ALSO existed, its value is OVERWRITTEN by the
  line-derived span. A `debug!` log emits when this happens so the diff is
  visible if anyone audits.
* Empty `service_lines` is left alone - validator will reject the claim
  anyway (a claim with zero lines fails downstream).
* The `chk_dos_range` DB constraint (`date_of_service_to >= date_of_service_from`)
  is automatically satisfied by construction (MIN <= MAX).

Rejected alternative (Option B): include the claim-level `DTP*472` value as a
sample point in the MIN/MAX. Would defensively widen the encounter span to
match a stated statement period, but 837P doesn't really have that concept,
and "widen because the submitter said wider" risks over-trusting bad data.

## Why this is not a "silent fallback" (Rule 3)

* The line-level dates ARE the authoritative billing record (`SV1` lines are
  what the claim is asking to be paid for).
* The validator (`validator.rs:333-334`) already rejects any service line
  without a real `service_date_from`, so MIN/MAX never sees a default sentinel.
* A `debug!` log emits when the line-derived span differs from a pre-existing
  claim-level DTP value, so the override is auditable.

## Affected files

* `crates/pro-parser-edi/src/loops.rs:1138-1143` - replace the first-line
  fallback with the MIN/MAX computation.
* `crates/pro-parser-edi/src/loops.rs` - new unit test exercising the
  span across single-line, mixed D8/RD8, and pre-existing claim-DTP cases.

## Checklist

- [x] Replace fallback block with MIN/MAX span computation.
- [x] `debug!` log when line-derived span differs from prior claim-level DTP.
- [x] Unit test covering: single line; multiple lines all RD8; mixed D8/RD8;
      claim-level DTP overwritten by line span.
- [x] CHANGELOG entry for 2.14.3.0 (Z bump - parser correctness fix, no
      schema change).
- [x] Rebuild MSI (Rule 10).
- [x] Commit + push.
