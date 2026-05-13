# Plan: Fix Hard-Coded Component Element Separator in EDI 837P Parser

## Problem

The 837P EDI parser reads the **component element separator** from ISA segment position 104
(stored on `EdiEnvelope.component_element_separator`), but every composite-element parser in
`crates/pro-parser-edi/src/segments.rs` is hard-coded to `split(':')`.

When a production 837 file declares a different component separator in ISA16 (e.g. `>` instead
of `:`), composites like `SV1*HC>99213>25` fail to split. The qualifier field receives the whole
`HC>99213>25` string, and procedure code, modifiers, and diagnosis pointers come back empty.

User-reported symptom: "prod file shows `P*>~` and the service units/line items are not being
processed correctly."

This also violates CLAUDE.md Rule 3 (no silent failures): the parser silently produces
empty/incorrect fields instead of either honoring the declared separator or failing loudly.

## Affected sites in `segments.rs`

- `Sv1Segment::parse` — line 404 (procedure composite: qualifier:code:mod1-4)
- `Sv1Segment::parse` — line 420 (diagnosis code pointer composite)
- `HiSegment::parse` — line 460 (diagnosis qualifier:code)
- `SvdSegment::parse` — line 824 (payer-adjudicated procedure composite)
- `ClmSegment::parse` — line 295 (CLM05: POS:facility:freq)
- `ClmSegment::parse` — line 313 (CLM10: related causes)

## Approach: Carry the separator on EdiSegment

Two options were considered:

1. Pass `component_separator: char` as a parameter to every composite parser.
2. Carry the separator on `EdiSegment` itself.

Chose Option 2 because it removes the "easy to forget when adding a new composite site"
failure mode that produced this bug. The parser already knows the separator when it builds
segments — wiring it onto each segment is a one-line change in `split_segments`.

## Checklist

- [x] Add `component_separator: char` field to `EdiSegment` in `types.rs` (default `':'`)
- [x] Add `EdiSegment::split_composite(index)` helper that uses the segment's separator
- [x] In `parser.rs::split_segments`, propagate `self.component_separator` onto each segment
- [x] Replace every `composite.split(':')` in `segments.rs` with the new helper
- [x] Update the remaining `EdiSegment { ... }` struct literals in test code
      (`validator.rs:402`, `loops.rs:1306`, `segments.rs:954/982/1013`) to include the new field
- [x] Add a regression test that parses an ISA declaring `>` as component separator and asserts
      SV1 composites are split correctly
- [x] Fix pre-existing broken `test_split_segments` (ISA was too short)
- [x] Fix latent bug in `SvdSegment::parse` (mod1 read index 3 instead of 2; index 3 duplicated)
- [x] `cargo test -p pro-parser-edi` passes (12/12)
- [x] `cargo build --workspace` succeeds
- [x] Update `CHANGELOG.md` with `[2.12.75.0]` entry
- [x] Bump `installer/version.txt` to `2.12.75.0`
- [ ] Rebuild installer: `.\build-msi.ps1 -Version "2.12.75.0"` (Rule 10/11)

## Version bump rationale

Z-bump (patch) per Rule 11: this is a bug fix to correctness of EDI parsing, no new
feature, no schema/migration change. Current `2.12.74.0` → `2.12.75.0`.

## Out of scope

- Repetition separator (ISA11) handling — not used by current composites and no symptom reported.
- Adding stricter validation that the separator chosen in ISA16 is unique vs. element/segment delimiters.
