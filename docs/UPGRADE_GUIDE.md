# Professional SMART Upgrade Guide

This guide explains how to upgrade Professional SMART from version 1.0.0 to 1.1.0 or later versions.

## Overview

Starting with version 1.1.0, Professional SMART supports in-place upgrades without requiring complete uninstallation and data loss. The upgrade process:

- Preserves all existing data (claims, providers, organizations, etc.)
- Maintains your configuration settings
- Automatically backs up the database before upgrade (recommended)
- Applies only new database migrations
- Minimal downtime (typically 1-2 minutes)

## Pre-Upgrade Checklist

Before upgrading, ensure:

- [ ] You have the new MSI installer (version 1.1.0 or later)
- [ ] PostgreSQL service is running
- [ ] You have sufficient disk space for a database backup (typically 10x database size)
- [ ] No claims are currently being processed
- [ ] You have database credentials handy
- [ ] Optional: Create a manual backup for extra safety

## Upgrade Steps

### Step 1: Stop the Service (Recommended)

While not required, stopping the service ensures no data is being processed during upgrade:

```cmd
sc stop ProfessionalSMART
```

Or use Services.msc to stop the "Professional SMART Claims Processing Service".

### Step 2: Run the Installer

1. Double-click the new `ProfessionalSMART.msi` installer
2. Click "Next" on the Welcome screen
3. The installer will detect your existing installation
4. Configure backup options (recommended: enabled)
5. Follow the prompts to complete the upgrade

### Step 3: Verify the Upgrade

After installation completes:

1. Check the Windows Event Log for any errors
2. Verify the service started successfully:
   ```cmd
   sc query ProfessionalSMART
   ```

3. Check the database version:
   ```cmd
   cd "C:\Program Files\Professional SMART\bin"
   pro-upgrade.exe check-version
   ```

4. Verify new migrations were applied:
   ```cmd
   pro-upgrade.exe list-pending-migrations
   ```
   Should show: "No pending migrations"

### Step 4: Test the System

1. Open the Configuration Wizard from Start Menu
2. Verify settings are preserved
3. Test processing a sample claim file
4. Check the web dashboard if applicable

## Backup and Restore

### Automatic Backups

The installer automatically creates database backups during upgrades (if enabled). Backups are stored in:

```
C:\ProgramData\Professional SMART\backups\
```

Backup filename format:
```
professional_smart_backup_YYYYMMDD_HHMMSS.sql.gz
```

### Manual Backup

To create a manual backup anytime:

```cmd
cd "C:\Program Files\Professional SMART\bin"
set DB_PASSWORD=your_password
pro-upgrade.exe backup-database
```

### List Available Backups

```cmd
pro-upgrade.exe list-backups
```

### Restore from Backup

If the upgrade fails or you need to roll back:

```cmd
cd "C:\Program Files\Professional SMART\bin"
set DB_PASSWORD=your_password
pro-upgrade.exe restore-database "C:\ProgramData\Professional SMART\backups\professional_smart_backup_20251020_143052.sql.gz"
```

**Warning:** Restoring will overwrite the current database. Any data added after the backup was created will be lost.

## Troubleshooting

### Upgrade Fails with "Migration Error"

**Problem:** One or more database migrations failed to apply.

**Solution:**
1. Check the MSI installation log (usually in `%TEMP%`)
2. Look for specific migration errors
3. Restore from backup if needed:
   ```cmd
   pro-upgrade.exe restore-database <backup-file>
   ```
4. Contact support with the error details

### Service Won't Start After Upgrade

**Problem:** Service fails to start after upgrade completes.

**Solution:**
1. Check Windows Event Viewer for errors
2. Verify database connectivity:
   ```cmd
   psql -h localhost -U postgres -d professional_smart -c "SELECT version FROM staging.application_version ORDER BY installed_at DESC LIMIT 1;"
   ```
3. Check the service log files in `C:\ProgramData\Professional SMART\logs\`
4. Try restarting the service manually:
   ```cmd
   sc start ProfessionalSMART
   ```

### Configuration Lost After Upgrade

**Problem:** Settings don't seem to be preserved.

**Solution:**
Configuration backups are created automatically. Check for backup files:
```
C:\ProgramData\Professional SMART\config\.env.backup_*
```

Copy the backup over the current .env file if needed.

### "Version Tracking Not Found" Warning

**Problem:** Upgrading from pre-1.1.0 version shows this warning.

**Solution:**
This is normal for first upgrade from 1.0.0. The installer will:
1. Detect this is a legacy installation
2. Create version tracking tables
3. Backfill migration history
4. Continue with upgrade normally

No action required - this is expected behavior.

## Version History

### Version 1.1.0
- **New:** In-place upgrade support
- **New:** Database version tracking
- **New:** Automatic backup before upgrade
- **New:** Migration management tool (`pro-upgrade.exe`)
- **New:** Configuration preservation during upgrades
- **Improved:** Installer now detects existing installations

### Version 1.0.0
- Initial release
- Required full uninstall for upgrades (no longer necessary)

## Advanced Topics

### Manual Migration Management

Check which migrations have been applied:
```cmd
cd "C:\Program Files\Professional SMART\bin"
set DB_PASSWORD=your_password
pro-upgrade.exe verify-checksums
```

List pending migrations:
```cmd
pro-upgrade.exe list-pending-migrations --migrations-dir "C:\Program Files\Professional SMART\migrations"
```

Apply pending migrations manually:
```cmd
pro-upgrade.exe apply-migrations --migrations-dir "C:\Program Files\Professional SMART\migrations"
```

### Database Version Tracking

The upgrade system tracks versions in two tables:

1. `staging.application_version` - Application version history
2. `staging.schema_migrations` - Individual migration history

Query current version:
```sql
SELECT * FROM staging.application_version ORDER BY installed_at DESC LIMIT 1;
```

Query migration history:
```sql
SELECT migration_name, applied_at FROM staging.schema_migrations ORDER BY migration_name;
```

### Upgrading Multiple Versions

If you're several versions behind (e.g., upgrading from 1.0.0 to 1.3.0), the installer will:

1. Detect the current version
2. Apply all intermediate migrations in order
3. Update to the target version

All migrations between versions are applied sequentially and automatically.

### Backup Retention

By default, the system keeps the last 5 backups. To clean up old backups manually:

1. Navigate to `C:\ProgramData\Professional SMART\backups\`
2. Delete old backup files (keep at least one recent backup)
3. Or let the system auto-clean during future upgrades

## Getting Help

If you encounter issues during upgrade:

1. **Check the logs:**
   - MSI installation log: `%TEMP%\MSI*.log`
   - Application logs: `C:\ProgramData\Professional SMART\logs\`
   - Windows Event Viewer: Application and System logs

2. **Gather information:**
   - Current database version
   - MSI installer version
   - Error messages from logs
   - Recent changes to the system

3. **Contact support:**
   - Email: support@professional-smart.com
   - Include: logs, error messages, and version information

## Rollback Procedure

If you need to rollback to a previous version:

1. **Stop the service:**
   ```cmd
   sc stop ProfessionalSMART
   ```

2. **Restore database from backup:**
   ```cmd
   cd "C:\Program Files\Professional SMART\bin"
   set DB_PASSWORD=your_password
   pro-upgrade.exe restore-database <backup-file>
   ```

3. **Uninstall current version:**
   - Use "Add or Remove Programs"
   - Or run: `msiexec /x {PRODUCT-CODE}`

4. **Install previous version:**
   - Run the old MSI installer
   - Use the same database credentials

5. **Verify rollback:**
   - Check service starts successfully
   - Verify data integrity
   - Test basic functionality

**Note:** Rollback will lose any data added between the backup and rollback time.

## Best Practices

1. **Always enable backup during upgrade** - The few extra minutes are worth the safety
2. **Test upgrades in a development environment first** - If you have a test system, upgrade it first
3. **Schedule upgrades during maintenance windows** - Minimize impact on users
4. **Keep at least 2 recent backups** - Don't delete all backups after successful upgrade
5. **Document any customizations** - Note any manual database changes before upgrading
6. **Monitor the first few claims after upgrade** - Ensure processing works correctly
7. **Review release notes** - Check for any version-specific considerations

## FAQ

**Q: Do I need to uninstall the old version first?**
A: No! Starting with version 1.1.0, the installer handles upgrades automatically. Simply run the new MSI.

**Q: Will I lose my data during upgrade?**
A: No. All data is preserved. The upgrade process only adds new features and database structures.

**Q: How long does an upgrade take?**
A: Typically 5-10 minutes, including backup creation. Most of that time is the backup process.

**Q: Can I skip versions (e.g., upgrade from 1.0.0 to 1.3.0)?**
A: Yes. The installer applies all intermediate migrations automatically.

**Q: What happens if the upgrade fails?**
A: The installer will stop and rollback any incomplete migrations. Your database will remain in its pre-upgrade state (or can be restored from automatic backup).

**Q: Can I upgrade while the service is running?**
A: The installer will stop the service automatically before making changes. However, we recommend stopping it manually first.

**Q: Where are backups stored?**
A: `C:\ProgramData\Professional SMART\backups\`

**Q: How do I know if the upgrade was successful?**
A: Check that: (1) Service is running, (2) No pending migrations, (3) Version number updated in registry and database.

**Q: Can I automate upgrades?**
A: Yes, for silent installations use: `msiexec /i ProfessionalSMART.msi /quiet /l*v upgrade.log`

**Q: What if I modified the database schema manually?**
A: Document your changes first. The upgrade's checksum verification will detect modifications. You may need to merge changes manually or restore from a clean backup.
