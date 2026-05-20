# Plan: Fix sqlx connect parameter warnings and AHRQOP001A rule encryption

## Problem

Two warnings on every service start:
1. sqlx logs "ignoring unrecognized connect parameter" for `statement_timeout`
   and `statement_cache_size` because they were passed as URL query params
2. Rule AHRQOP001A has NULL encrypted parameters because the encryption key
   SET statement came after the INSERT in the baseline migration

## Resolution Checklist

- [x] Rewrite `create_pool()` to use `PgConnectOptions` with proper API methods
- [x] Set `statement_timeout` via `after_connect` hook instead of URL param
- [x] Move `SET app.rule_encryption_key` before rule INSERT in baseline
- [x] Add self-healing UPDATE for existing deployments with NULL parameters
- [x] Update CHANGELOG.md with version 2.17.0.4
- [x] Commit and push

## Verification

After deployment:
1. Restart service - no more sqlx "ignoring unrecognized connect parameter" warnings
2. No more "Rule AHRQOP001A has COMPOSITE template but NULL parameters" warning
3. Verify: `SELECT rule_code, rule_parameters_encrypted IS NOT NULL FROM claims.rule_definition WHERE rule_code = 'AHRQOP001A';`

## Note for existing deployments

Run the baseline UPDATE manually if the upgrade manager doesn't re-run it:
```sql
SET app.rule_encryption_key = 'ProfessionalSmartRulesKey2024';
UPDATE claims.rule_definition
SET rule_parameters_encrypted = pgp_sym_encrypt(
    '{"operator": "AND", "conditions": [{"type": "date_gte", "min_date": "2012-07-01"}, {"type": "cpt_in", "codes": ["99281", "99282", "99283", "99284", "99285", "99291"]}, {"type": "dx_pattern_exclude", "include": "^(F11|T40)", "exclude": "^F1121"}]}',
    current_setting('app.rule_encryption_key', true)
)
WHERE rule_code = 'AHRQOP001A'
  AND rule_parameters_encrypted IS NULL;
```
