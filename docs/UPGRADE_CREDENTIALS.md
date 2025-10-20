# Database Credential Handling During Upgrades

## Overview

Professional SMART implements **zero-touch upgrade credential management** to provide a seamless and secure upgrade experience. This document explains how database credentials are handled during upgrades.

## Implementation: Option 1 - Full Automation

### Design Principles

1. **Zero-Touch Upgrades** - No credential re-entry during upgrades
2. **Security First** - Credentials never re-exposed in UI or logs
3. **Enterprise Ready** - Supports silent/unattended installations
4. **Consistency** - Uses the exact same credentials as the running system
5. **No Credential Drift** - Prevents accidental changes during upgrades

### How It Works

#### Fresh Installation Flow

```
User runs MSI
  ↓
Welcome Dialog
  ↓
Prerequisite Check
  ↓
Customize Installation
  ↓
DATABASE DIALOG SHOWN ← User enters credentials
  ↓
Credentials written to .env
  ↓
Database created
  ↓
Installation complete
```

#### Upgrade Flow (Existing Installation)

```
User runs MSI
  ↓
Welcome Dialog
  ↓
LoadEnvCredentials.vbs executed
  ├─ Reads C:\ProgramData\Professional SMART\config\.env
  ├─ Extracts DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  ├─ Sets MSI properties: DB_HOST, DB_PORT, etc.
  └─ Sets ENV_CREDENTIALS_LOADED=1
  ↓
Prerequisite Check
  ↓
Customize Installation
  ↓
DATABASE DIALOG SKIPPED ← Credentials loaded from .env
  ↓
Existing .env preserved (WriteConfig.vbs detects existing file)
  ↓
UpgradeDatabase.vbs uses loaded credentials
  ├─ Creates backup (if enabled)
  └─ Applies migrations
  ↓
Upgrade complete
```

### Technical Implementation

#### 1. LoadEnvCredentials.vbs

**Location:** `installer\LoadEnvCredentials.vbs`

**Purpose:** Reads database credentials from existing `.env` file and sets MSI properties

**Key Features:**
- Parses `.env` file format
- Handles quoted and unquoted values
- Validates all required credentials are present
- Sets `ENV_CREDENTIALS_LOADED=1` on success
- Logs actions to MSI log file

**Execution:** Runs immediately after Welcome dialog, before any user prompts

#### 2. Product.wxs Updates

**Added Property:**
```xml
<Property Id="ENV_CREDENTIALS_LOADED" Value="0" />
```

**Custom Action:**
```xml
<Binary Id="LoadEnvCredentialsScript" SourceFile="LoadEnvCredentials.vbs" />
<CustomAction Id="LoadEnvCredentialsAction"
              BinaryKey="LoadEnvCredentialsScript"
              VBScriptCall="LoadEnvCredentials"
              Execute="immediate"
              Return="check" />
```

**UI Flow Logic:**
```xml
<!-- Skip DatabaseConfigDlg if credentials loaded from .env -->
<Publish Dialog="CustomizeDlg" Control="Next" Event="NewDialog" Value="VerifyReadyDlg" Order="2">
  ENV_CREDENTIALS_LOADED = "1"
</Publish>
<!-- Otherwise show DatabaseConfigDlg -->
<Publish Dialog="CustomizeDlg" Control="Next" Event="NewDialog" Value="DatabaseConfigDlg" Order="3">
  ENV_CREDENTIALS_LOADED = "0"
</Publish>
```

#### 3. UpgradeDatabase.vbs Enhancement

**Enhancement:** Added logging to indicate credentials source

```vbscript
LogMessage "UpgradeDatabase: Using credentials from existing .env configuration"
```

The script already receives credentials via `CustomActionData`, which now come from the `.env` file (via LoadEnvCredentials.vbs) instead of the UI dialog.

#### 4. WriteConfig.vbs Preservation

**Existing Logic:** Already preserves `.env` during upgrades

```vbscript
If fso.FileExists(configPath) Then
    ' Create backup
    fso.CopyFile configPath, backupPath
    ' Exit without overwriting
    Exit Function
End If
```

This ensures credentials (and all other settings) remain unchanged during upgrades.

## Manual Credential Changes

### Using the Reconfiguration Tool

For scenarios where database credentials need to be changed (password rotation, database migration, etc.), use the `pro-upgrade reconfigure-database` command.

**Command:**
```cmd
"C:\Program Files\Professional SMART\bin\pro-upgrade.exe" reconfigure-database
```

**Features:**
1. **Interactive Wizard** - Prompts for each credential with current values shown
2. **Connection Testing** - Tests credentials before saving
3. **Automatic Backup** - Creates timestamped backup of `.env`
4. **Config Preservation** - Keeps all non-database settings unchanged
5. **Service Restart Reminder** - Prompts user to restart service

**Example Usage:**
```
C:\> "C:\Program Files\Professional SMART\bin\pro-upgrade.exe" reconfigure-database

Database Reconfiguration Wizard
================================

This will update the database credentials in: C:\ProgramData\Professional SMART\config\.env

Reading current configuration...

Database Host [localhost]:
Database Port [5432]:
Database Name [professional_smart]:
Database User [postgres]:
Database Password: ********

Testing database connection...
✓ Connection successful
✓ Backup created: C:\ProgramData\Professional SMART\config\.env.backup_20251020_143022
✓ Configuration updated successfully

IMPORTANT: You must restart the Professional SMART service for changes to take effect:
  net stop ProfessionalSMART
  net start ProfessionalSMART
```

## Security Considerations

### Credential Storage

**Location:** `C:\ProgramData\Professional SMART\config\.env`

**Permissions:** Should be restricted to Administrators and SYSTEM

**Format:** Plain text (standard for `.env` files)
```
DB_PASSWORD=MySecurePassword123
```

### MSI Log Files

**Credentials in Logs:** Passwords are masked in log output
```
DB_PASSWORD = ********
```

**Implementation:**
```vbscript
LogMessage "DB_PASSWORD = " & String(Len(dbPassword), "*")
```

### Best Practices

1. **File System Permissions**
   - Restrict `.env` file access to Administrators only
   - Use NTFS permissions to prevent unauthorized access

2. **Network Security**
   - Use PostgreSQL SSL/TLS connections
   - Restrict database access to localhost when possible

3. **Password Management**
   - Use strong passwords (minimum 16 characters)
   - Rotate passwords regularly
   - Use `pro-upgrade reconfigure-database` for updates

4. **Backup Security**
   - Database backups contain full data including credentials
   - Secure backup directory: `C:\ProgramData\Professional SMART\backups`
   - Consider encrypting backups at rest

## Troubleshooting

### Upgrade Shows Database Dialog (Should Skip)

**Symptoms:** During upgrade from 1.2.0 → 1.2.1, database dialog appears

**Causes:**
1. `.env` file missing at `C:\ProgramData\Professional SMART\config\.env`
2. `.env` file exists but missing required credentials
3. `LoadEnvCredentialsAction` failed to execute

**Resolution:**
1. Check MSI log for LoadEnvCredentials messages:
   ```
   Professional SMART Installer: LoadEnvCredentials: Config file found, loading credentials...
   Professional SMART Installer: LoadEnvCredentials: Successfully loaded all credentials
   ```

2. Verify `.env` file exists and contains:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=professional_smart
   DB_USER=postgres
   DB_PASSWORD=YourPassword
   ```

3. If file is corrupt, re-enter credentials in dialog (this time they will be saved)

### Wrong Credentials After Upgrade

**Symptoms:** Service fails to start after upgrade, database connection errors

**Causes:**
1. Database credentials were changed outside of Professional SMART
2. `.env` file was manually edited with incorrect values

**Resolution:**
```cmd
"C:\Program Files\Professional SMART\bin\pro-upgrade.exe" reconfigure-database
```

### Need to Change Database Server

**Scenario:** Migrating from local PostgreSQL to remote server

**Steps:**
1. Run reconfiguration tool:
   ```cmd
   "C:\Program Files\Professional SMART\bin\pro-upgrade.exe" reconfigure-database
   ```

2. Enter new database details:
   - Host: `db-server.company.com`
   - Port: `5432`
   - Name: `professional_smart`
   - User: `app_user`
   - Password: `NewPassword123`

3. Tool tests connection before saving

4. Restart service:
   ```cmd
   net stop ProfessionalSMART
   net start ProfessionalSMART
   ```

## Silent/Unattended Installations

### Upgrade Example

For automated deployments, upgrades can be performed silently:

```cmd
msiexec /i ProfessionalSMART.msi /quiet /l*v upgrade.log
```

**Key Points:**
- No user interaction required
- Credentials automatically loaded from `.env`
- Upgrade completes without prompts
- Database backup created automatically
- Migrations applied automatically

### Fresh Installation (Silent)

For fresh installations, you must provide credentials:

```cmd
msiexec /i ProfessionalSMART.msi /quiet /l*v install.log ^
  DB_HOST=localhost ^
  DB_PORT=5432 ^
  DB_NAME=professional_smart ^
  DB_USER=postgres ^
  DB_PASSWORD=SecurePassword123 ^
  INSTALLFOLDER="C:\Program Files\Professional SMART"
```

## Version History

| Version | Change |
|---------|--------|
| 1.2.1.0 | Implemented automatic credential loading from .env during upgrades |
| 1.2.1.0 | Added `pro-upgrade reconfigure-database` command |
| 1.2.1.0 | Database dialog now skipped during upgrades when .env exists |
| 1.2.0.0 | Added upgrade infrastructure with version tracking |
| 1.1.0.0 | Initial installer with manual credential entry |

## Related Documentation

- [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) - Complete upgrade procedures
- [INSTALLATION.md](INSTALLATION.md) - Fresh installation guide
- [TESTING_UPGRADE.md](TESTING_UPGRADE.md) - Testing upgrade paths
- [VERSIONING_GUIDE.md](VERSIONING_GUIDE.md) - Version management
