# FDW Password Authentication Support Plan

## Overview
Update the Foreign Data Wrapper setup to use password authentication instead of peer authentication.

## Changes Required

- [ ] Update `migrations/069_setup_smartproaudit_fdw.sql` - Add password option to USER MAPPING
- [ ] Update `migrations/000_baseline_v2.12.sql` - Sync the same change
- [ ] Update `docs/FDW_HOWTO.md` - Document password authentication
- [ ] Update CHANGELOG.md
- [ ] Rebuild installer with new version

## Implementation Details

The USER MAPPING needs to change from:
```sql
OPTIONS (user 'postgres')
```

To:
```sql
OPTIONS (user 'postgres', password 'postgres')
```

Note: Using 'postgres' as default password. In production, users should change this or use environment-based configuration.

## Version
This is a patch change: 2.12.70.0 -> 2.12.70.1
