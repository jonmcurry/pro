# Professional SMART MSI Installation Instructions

## Fresh Installation with Logging

To install with full logging to diagnose any issues:

```powershell
# Navigate to installer directory
cd C:\Users\jonmc\dev\pro\installer

# Run MSI with verbose logging
msiexec /i ProfessionalSMART.msi /l*v install.log
```

This will create `install.log` which contains all installation actions including:
- CreateDatabaseAction execution
- pro-upgrade.exe apply-migrations output
- Any errors or warnings

## Expected Installation Flow

### Fresh Install (NOT Installed AND NOT WIX_UPGRADE_DETECTED)
1. Files copied to `C:\Program Files\Professional SMART\`
2. **CreateDatabaseAction** runs:
   - Creates database (if doesn't exist)
   - Runs: `pro-upgrade.exe apply-migrations`
   - Applies all 49 embedded migrations (001-051 except 040)
3. Service starts

### Upgrade Install (WIX_UPGRADE_DETECTED)
1. Service stopped
2. Old files removed
3. New files copied
4. **UpgradeDatabaseAction** runs:
   - Optional backup
   - Runs: `pro-upgrade.exe apply-migrations`
   - Applies pending migrations only
5. Service starts

## Current MSI Details

**File**: `installer/ProfessionalSMART.msi`
**Version**: 1.7.1.0
**Build Time**: 10:11 AM (Nov 6, 2025)
**SHA256**: `515378756B9F9A882109954E71551418B966073FA945D67DD36C723FD1FE9794`

**Embedded Migrations**: 49 total
- 001-039: Core schema
- 041-045: Provider taxonomy and indexes
- 046-049: Rule configuration system
- 050: Performance indexes (Phase 6-8)
- 051: Rule execution statistics (Phase 8)

## Post-Installation Verification

Check that all migrations were applied:

```powershell
# Connect to database
$env:PGPASSWORD="ClearToFly1"
psql -U postgres -d professional_smart

# Check migration count (should be 50)
SELECT COUNT(*) FROM staging.schema_migrations;

# Verify rule tables exist
\dt claims.rule_*

# Check pending migrations (should be 0)
& "C:\Program Files\Professional SMART\bin\pro-upgrade.exe" --db-password ClearToFly1 list-pending-migrations
```

## Troubleshooting

### Migrations Not Applied

If migrations weren't applied during installation:

1. Check the install.log for:
   ```
   CreateDatabase: Executing pro-upgrade apply-migrations
   CreateDatabase: SUCCESS - All migrations applied
   ```

2. If pro-upgrade.exe wasn't found or failed, run manually:
   ```powershell
   & "C:\Program Files\Professional SMART\bin\pro-upgrade.exe" --db-password ClearToFly1 apply-migrations
   ```

3. Verify migrations applied:
   ```sql
   SELECT COUNT(*) FROM staging.schema_migrations;
   -- Should return 50
   ```

### Service Won't Start

1. Check service log:
   ```powershell
   Get-Content "C:\ProgramData\Professional SMART\logs\service.log" -Tail 50
   ```

2. Check service status:
   ```powershell
   Get-Service ProfessionalSMART
   ```

3. Start service manually:
   ```powershell
   Start-Service ProfessionalSMART
   ```
