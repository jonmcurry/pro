# Fix: Missing Claims During HL Segment Transitions

## Problem
Not all patient control numbers from CLM segments were making it into raw_claims/encounter tables. Claims were being lost when:
1. A new HL*22 (subscriber level) was encountered while a claim was in progress
2. A new HL*23 (patient level) was encountered while a claim was in progress

## Root Cause
In `parser.rs`, when a new HL segment was encountered, the parser would reset state flags without first finalizing any in-progress claim.

**Before (buggy code):**
```rust
"22" => {
    subscriber_segments.clear();
    in_subscriber_loop = true;
    in_patient_loop = false;
    in_claim = false;  // BUG: Claim not finalized before this!
}
```

**What happened:**
1. HL*2*1*22*0 starts subscriber loop
2. CLM*123 starts processing, `in_claim = true`
3. Claim segments collected
4. HL*3*2*22*0 starts NEW subscriber loop
5. `in_claim = false` set immediately - **CLM*123 never pushed to claims vector!**

## Fix
Added claim finalization logic before resetting state in both HL*22 and HL*23 handlers:

**After (fixed code):**
```rust
"22" => {
    // IMPORTANT: Finalize any in-progress claim before starting new subscriber
    if in_claim && !current_claim_segments.is_empty() {
        let mut claim = parse_claim_info(&current_claim_segments)?;
        claim.billing_date = billing_date;
        claims.push(claim);
        current_claim_segments.clear();
    }
    // Reset subscriber segments for new subscriber
    subscriber_segments.clear();
    in_subscriber_loop = true;
    in_patient_loop = false;
    in_claim = false;
}
"23" => {
    // IMPORTANT: Finalize any in-progress claim before starting patient loop
    if in_claim && !current_claim_segments.is_empty() {
        let mut claim = parse_claim_info(&current_claim_segments)?;
        claim.billing_date = billing_date;
        claims.push(claim);
        current_claim_segments.clear();
        in_claim = false;
    }
    in_subscriber_loop = false;
    in_patient_loop = true;
}
```

## Files Modified
- `crates/pro-parser-edi/src/parser.rs` - Lines 305-332

## Testing
Created comprehensive test file `test_data/test_multi_claim_hierarchy.edi` with:
- 10 total claims
- Multiple HL*22 transitions
- HL*23 patient loops with multiple claims under same patient
- Newborn claims with patient DOB different from subscriber DOB

Test results:
- All 10 claims now parsed correctly
- Newborn claims (0011, 7788, 9900) have correct patient DOB
- Multiple claims under same HL*23 (7788, 9900) both captured with patient info

## Version
2.11.8.0 -> 2.11.9.0 (minor version - bug fix)
