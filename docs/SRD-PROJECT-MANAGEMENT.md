# Software Requirements Document: Project Database Management System

**Document Version:** 1.3
**Date:** 2025-12-27
**Author:** Senior Software Engineer
**Status:** Draft

**Revision History:**
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-24 | Initial draft with create, switch, list, info, delete, backup commands |
| 1.1 | 2025-12-26 | Added upgrade and status commands for multi-database schema management |
| 1.2 | 2025-12-26 | Changed registry from file-based (projects.json) to PostgreSQL-based (SmartProAudit database) |
| 1.3 | 2025-12-27 | Added GUI requirements for pro-project tool with selectable database upgrades |

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies the requirements for a **Project Database Management System** that enables users to create, switch between, and manage multiple isolated project databases on a single production server without requiring application reinstallation.

### 1.2 Problem Statement

Currently, Professional SMART is configured to connect to a single PostgreSQL database specified at installation time. When users need to work on multiple projects (each requiring isolated data), they must:

1. Uninstall the application
2. Reinstall pointing to a new database
3. Lose easy access to previous project data

This workflow is:
- **Time-consuming** - Full reinstall cycle for each project
- **Error-prone** - Risk of configuration mistakes during reinstall
- **Inefficient** - No ability to quickly reference or switch between projects
- **Unprofessional** - Does not meet enterprise software expectations

### 1.3 Proposed Solution

Implement a new tool (`pro-project.exe`) with both CLI and GUI interfaces that provides:

1. **Database Creation** - Create new project databases with full schema
2. **Database Switching** - Update configuration and restart service to use different database
3. **Project Listing** - View all available project databases
4. **Project Information** - Display details about specific projects
5. **Project Deletion** - Remove project databases (with safeguards)
6. **Schema Upgrades** - Apply pending migrations to one or all project databases
7. **Status Monitoring** - Quick view of which projects need schema upgrades
8. **GUI Mode** - Visual interface for managing and upgrading databases (NEW)

---

## 2. Functional Requirements

### 2.1 Command: `create`

**Purpose:** Create a new project database with the complete Professional SMART schema.

**Usage:**
```
pro-project create --name <PROJECT_NAME> [--switch]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | Yes | N/A | Project database name (alphanumeric, underscores, max 63 chars) |
| `--switch` | No | false | Automatically switch to the new database after creation |
| `--db-host` | No | localhost | PostgreSQL host |
| `--db-port` | No | 5432 | PostgreSQL port |
| `--db-user` | No | postgres | PostgreSQL user |
| `--db-password` | No | From .env | PostgreSQL password |

**Behavior:**

1. Validate project name (alphanumeric + underscores, 1-63 characters, not a reserved name)
2. Connect to PostgreSQL server using `postgres` database
3. Check if database already exists; abort if it does
4. Execute `CREATE DATABASE <project_name>`
5. Connect to new database
6. Apply baseline migration (000_baseline_v2.12.sql) via embedded migrations
7. Record creation metadata in `staging.application_version`
8. Register project in the project registry (see Section 2.7)
9. If `--switch` flag provided, execute switch command
10. Display success message with connection details

**Output (Success):**
```
Project database created successfully.

  Database:    ClientA_2025
  Host:        localhost
  Port:        5432
  Created:     2025-12-24 10:30:00
  Schema:      v2.12.23.0 (67 migrations)

To switch to this project:
  pro-project switch --name ClientA_2025
```

**Output (Error - Already Exists):**
```
Error: Database 'ClientA_2025' already exists.

Use 'pro-project list' to see existing projects.
Use 'pro-project switch --name ClientA_2025' to switch to it.
```

**Exit Codes:**
- 0: Success
- 1: Database already exists
- 2: Connection failure
- 3: Migration failure
- 4: Invalid project name

---

### 2.2 Command: `switch`

**Purpose:** Switch the application to use a different project database.

**Usage:**
```
pro-project switch --name <PROJECT_NAME> [--no-restart]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | Yes | N/A | Target project database name |
| `--no-restart` | No | false | Update config only, do not restart service |

**Behavior:**

1. Validate that the target database exists and is accessible
2. Verify the database has Professional SMART schema (check `staging.application_version`)
3. Stop the `ProfessionalSMART` Windows service (if running)
4. Create backup of current `.env` file with timestamp
5. Update `DB_NAME` in `.env` file at `C:\ProgramData\Professional SMART\config\.env`
6. Update `DATABASE_URL` if present (recalculate from components)
7. Update project registry with "last used" timestamp
8. Start the `ProfessionalSMART` Windows service (unless `--no-restart`)
9. Wait for service to reach Running state (max 30 seconds)
10. Verify database connection from service logs

**Output (Success):**
```
Switching project database...

  Previous:    professional_smart
  New:         ClientA_2025
  Config:      C:\ProgramData\Professional SMART\config\.env
  Backup:      C:\ProgramData\Professional SMART\config\.env.20251224_103000.bak

Service status:
  Stopped:     ProfessionalSMART
  Updated:     .env configuration
  Started:     ProfessionalSMART
  Status:      Running

Successfully switched to project 'ClientA_2025'.
```

**Output (Error - Database Not Found):**
```
Error: Database 'NonExistent' does not exist or is not accessible.

Available projects:
  - professional_smart (current)
  - ClientA_2025
  - ClientB_2025

Use 'pro-project create --name NonExistent' to create a new project.
```

**Exit Codes:**
- 0: Success
- 1: Database not found
- 2: Database exists but missing schema
- 3: Service stop failed
- 4: Config update failed
- 5: Service start failed

---

### 2.3 Command: `list`

**Purpose:** Display all available project databases.

**Usage:**
```
pro-project list [--format <FORMAT>]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--format` | No | table | Output format: `table`, `json`, `csv` |

**Behavior:**

1. Read project registry from `C:\ProgramData\Professional SMART\config\projects.json`
2. For each registered project, verify database accessibility
3. Query `staging.application_version` for schema version
4. Mark current active database (from .env)
5. Sort by last_used timestamp (most recent first)

**Output (Table Format):**
```
PROJECT DATABASES
=================

  NAME                  STATUS    SCHEMA VERSION   LAST USED             SIZE
  --------------------  --------  ---------------  --------------------  --------
* professional_smart    Active    v2.12.23.0       2025-12-24 10:30:00   1.2 GB
  ClientA_2025          Ready     v2.12.23.0       2025-12-23 15:45:00   856 MB
  ClientB_2025          Ready     v2.12.22.0       2025-12-20 09:00:00   2.1 GB
  OldProject_2024       Offline   v2.12.15.0       2025-11-01 12:00:00   --

* = Currently active database
Offline = Database exists in registry but connection failed

Total: 4 projects (3 accessible, 1 offline)
```

**Output (JSON Format):**
```json
{
  "projects": [
    {
      "name": "professional_smart",
      "status": "active",
      "schema_version": "v2.12.23.0",
      "last_used": "2025-12-24T10:30:00Z",
      "size_bytes": 1288490188,
      "host": "localhost",
      "port": 5432
    }
  ],
  "total": 4,
  "accessible": 3,
  "offline": 1
}
```

**Exit Codes:**
- 0: Success
- 1: Registry file not found (first run - display help)
- 2: No accessible databases

---

### 2.4 Command: `info`

**Purpose:** Display detailed information about a specific project database.

**Usage:**
```
pro-project info [--name <PROJECT_NAME>]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | No | Current active | Project database name |

**Behavior:**

1. Connect to specified (or current) database
2. Query `staging.application_version` for version history
3. Query `staging.schema_migrations` for migration status
4. Query table counts for key entities
5. Calculate database size

**Output:**
```
PROJECT INFORMATION: ClientA_2025
=================================

Connection:
  Host:              localhost
  Port:              5432
  Database:          ClientA_2025
  User:              postgres

Schema:
  Current Version:   v2.12.23.0
  Installed:         2025-12-23 14:00:00
  Migrations:        67 applied
  Baseline:          000_baseline_v2.12.sql

Statistics:
  Organizations:     3
  Facilities:        12
  Providers:         45
  Encounters:        125,430
  Service Lines:     387,291
  Raw Claims:        0 (pending)

Storage:
  Database Size:     856 MB
  Indexes:           124 MB
  Tables:            698 MB
  Other:             34 MB

Last Activity:
  Last Import:       2025-12-23 16:45:00
  Last Switch:       2025-12-23 14:05:00
  Created:           2025-12-23 14:00:00
```

**Exit Codes:**
- 0: Success
- 1: Database not found
- 2: Connection failure

---

### 2.5 Command: `delete`

**Purpose:** Remove a project database (with safeguards).

**Usage:**
```
pro-project delete --name <PROJECT_NAME> [--force] [--backup]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | Yes | N/A | Project database to delete |
| `--force` | No | false | Skip confirmation prompt |
| `--backup` | No | false | Create backup before deletion |

**Behavior:**

1. Verify database exists
2. Prevent deletion of currently active database
3. If `--backup`, create pg_dump backup to `C:\ProgramData\Professional SMART\backups\`
4. Display warning with database statistics
5. Prompt for confirmation (unless `--force`)
6. Require typing database name to confirm (unless `--force`)
7. Execute `DROP DATABASE <project_name>`
8. Remove from project registry
9. Display success message

**Output (Confirmation):**
```
WARNING: You are about to delete project database 'ClientA_2025'

This database contains:
  - 3 organizations
  - 12 facilities
  - 125,430 encounters
  - 387,291 service lines

This action is IRREVERSIBLE.

To confirm deletion, type the database name: ClientA_2025
> _
```

**Output (Success):**
```
Project database 'ClientA_2025' has been deleted.

Backup saved to:
  C:\ProgramData\Professional SMART\backups\ClientA_2025_20251224_110000.backup

Registry updated: 3 projects remaining.
```

**Exit Codes:**
- 0: Success
- 1: Database not found
- 2: Cannot delete active database
- 3: Backup failed
- 4: User cancelled
- 5: Drop failed

---

### 2.6 Command: `backup`

**Purpose:** Create a backup of a project database.

**Usage:**
```
pro-project backup [--name <PROJECT_NAME>] [--output <PATH>]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | No | Current active | Project database to backup |
| `--output` | No | Auto-generated | Output file path |

**Behavior:**

1. Connect to specified (or current) database
2. Generate filename: `<name>_<timestamp>.backup`
3. Execute `pg_dump` with custom format (-Fc)
4. Verify backup integrity
5. Record backup in registry

**Output:**
```
Creating backup of 'ClientA_2025'...

  Source:      ClientA_2025 (856 MB)
  Output:      C:\ProgramData\Professional SMART\backups\ClientA_2025_20251224_110000.backup
  Format:      PostgreSQL custom (compressed)
  Duration:    45 seconds
  Size:        234 MB (73% compression)

Backup completed successfully.
```

---

### 2.7 Command: `upgrade`

**Purpose:** Apply pending schema migrations to one or all project databases.

**Usage:**
```
pro-project upgrade [--name <PROJECT_NAME>] [--all] [--backup] [--dry-run]
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | No | N/A | Specific project database to upgrade |
| `--all` | No | false | Upgrade all registered project databases |
| `--backup` | No | false | Create backup before each upgrade |
| `--dry-run` | No | false | Show what would be upgraded without applying |
| `--continue-on-error` | No | false | Continue upgrading other databases if one fails |

**Note:** Either `--name` or `--all` must be specified (mutually exclusive).

**Behavior:**

1. Stop the `ProfessionalSMART` Windows service (prevents conflicts during migration)
2. If `--all`, retrieve all projects from registry; otherwise use specified project
3. For each project database:
   a. Connect to database
   b. Query `staging.schema_migrations` for current migration state
   c. Determine pending migrations from embedded migrations
   d. If `--dry-run`, display pending migrations and skip to next database
   e. If `--backup`, create pg_dump backup before applying
   f. Apply pending migrations in order using `pro-upgrade-manager`
   g. Update `staging.application_version` with new version
   h. Update project registry with new schema_version
   i. Log success/failure
4. Display summary of all upgrades
5. Restart the `ProfessionalSMART` Windows service

**Output (Dry Run):**
```
Schema Upgrade Analysis (Dry Run)
=================================

Installed Version:  v2.12.25.0
Embedded Baseline:  v2.12.25.0 (69 migrations)

Project: professional_smart
  Current:     v2.12.23.0 (67 migrations)
  Pending:     2 migrations
    - 068_new_feature.sql
    - 069_performance_index.sql
  Status:      NEEDS UPGRADE

Project: ClientA_2025
  Current:     v2.12.23.0 (67 migrations)
  Pending:     2 migrations
    - 068_new_feature.sql
    - 069_performance_index.sql
  Status:      NEEDS UPGRADE

Project: ClientB_2025
  Current:     v2.12.25.0 (69 migrations)
  Pending:     0 migrations
  Status:      UP TO DATE

Summary: 2 of 3 projects need upgrade

To apply upgrades, run:
  pro-project upgrade --all
  pro-project upgrade --all --backup  (recommended)
```

**Output (Upgrade Execution):**
```
Schema Upgrade
==============

Installed Version:  v2.12.25.0

Stopping service... Done

[1/2] Upgrading 'professional_smart'...
  Backup:      C:\ProgramData\Professional SMART\backups\professional_smart_20251224_120000.backup
  Migrations:  Applying 068_new_feature.sql... OK
               Applying 069_performance_index.sql... OK
  Result:      SUCCESS (v2.12.23.0 -> v2.12.25.0)

[2/2] Upgrading 'ClientA_2025'...
  Backup:      C:\ProgramData\Professional SMART\backups\ClientA_2025_20251224_120015.backup
  Migrations:  Applying 068_new_feature.sql... OK
               Applying 069_performance_index.sql... OK
  Result:      SUCCESS (v2.12.23.0 -> v2.12.25.0)

Starting service... Done

UPGRADE SUMMARY
===============
  Total:       2 projects
  Succeeded:   2
  Failed:      0
  Skipped:     0

All project databases are now at v2.12.25.0
```

**Output (Partial Failure with --continue-on-error):**
```
Schema Upgrade
==============

Installed Version:  v2.12.25.0

Stopping service... Done

[1/3] Upgrading 'professional_smart'...
  Migrations:  Applying 068_new_feature.sql... OK
               Applying 069_performance_index.sql... OK
  Result:      SUCCESS (v2.12.23.0 -> v2.12.25.0)

[2/3] Upgrading 'ClientA_2025'...
  Migrations:  Applying 068_new_feature.sql... FAILED
               ERROR: relation "claims.some_table" does not exist
  Result:      FAILED (rolled back to v2.12.23.0)
  Continuing due to --continue-on-error flag...

[3/3] Upgrading 'ClientB_2025'...
  Migrations:  Applying 068_new_feature.sql... OK
               Applying 069_performance_index.sql... OK
  Result:      SUCCESS (v2.12.23.0 -> v2.12.25.0)

Starting service... Done

UPGRADE SUMMARY
===============
  Total:       3 projects
  Succeeded:   2
  Failed:      1
  Skipped:     0

FAILED DATABASES:
  - ClientA_2025: relation "claims.some_table" does not exist

WARNING: Not all databases were upgraded. Review errors above.
Run 'pro-project info --name ClientA_2025' for details.
```

**Exit Codes:**
- 0: All upgrades succeeded
- 1: No projects found / invalid arguments
- 2: Some upgrades failed (with --continue-on-error)
- 3: Upgrade failed (without --continue-on-error, stops on first failure)
- 4: Service control failed
- 5: Backup failed

**Important Considerations:**

1. **Transactional Safety:** Each database upgrade runs in a transaction. If a migration fails, that database is rolled back to its previous state.

2. **Version Compatibility:** The tool uses the same embedded migrations as `pro-upgrade.exe`, ensuring consistency.

3. **Service Downtime:** The service is stopped during upgrades to prevent data corruption. Plan upgrades during maintenance windows.

4. **Backup Recommendation:** Always use `--backup` for production upgrades. Backups are created before each database upgrade, not once at the start.

5. **Order Independence:** Databases are upgraded independently. A failure in one does not affect others when using `--continue-on-error`.

---

### 2.8 Command: `status`

**Purpose:** Quick health check showing upgrade status of all projects.

**Usage:**
```
pro-project status
```

**Behavior:**

1. Query installed application version from executable or embedded migrations
2. For each registered project, compare schema version to installed version
3. Display color-coded status

**Output:**
```
PROJECT STATUS
==============

Installed Version: v2.12.25.0

  DATABASE              SCHEMA VERSION   STATUS
  --------------------  ---------------  ----------------
* professional_smart    v2.12.25.0       Up to date
  ClientA_2025          v2.12.23.0       Needs upgrade (2 pending)
  ClientB_2025          v2.12.25.0       Up to date
  OldProject_2024       v2.12.15.0       Needs upgrade (10 pending)

1 of 4 projects need schema upgrades.

Run 'pro-project upgrade --all --dry-run' to see pending migrations.
Run 'pro-project upgrade --all --backup' to apply upgrades.
```

**Exit Codes:**
- 0: All projects up to date
- 1: Some projects need upgrades (informational, not an error)
- 2: Registry not found

---

### 2.9 Command: `gui`

**Purpose:** Launch a graphical user interface for managing project databases with visual selection and upgrade capabilities.

**Usage:**
```
pro-project gui
pro-project --gui
```

**Behavior:**

1. Connect to SmartProAudit database and retrieve all registered projects
2. Display a window with a data grid showing all project columns from `projects.project`
3. Allow user to select which databases to upgrade via checkboxes
4. Provide upgrade action with progress feedback
5. Display real-time status updates during operations

**GUI Layout:**

```
+------------------------------------------------------------------+
|  Professional SMART - Project Database Manager                    |
+------------------------------------------------------------------+
|  SmartProAudit Registry: localhost:5432/smartproaudit   [Refresh] |
+------------------------------------------------------------------+
|  [ ] Select All                                                   |
+------------------------------------------------------------------+
|  [x] | Database Name          | Organization | Schema Ver | Status      |
|  [ ] | professional_smart     | Acme Corp    | 2.12.32.0  | Up to date  |
|  [x] | professional_smart_A   | Client A     | 2.12.30.0  | 2 pending   |
|  [x] | professional_smart_B   | Client B     | 2.12.28.0  | 4 pending   |
|  [ ] | professional_smart_C   | Client C     | 2.12.32.0  | Up to date  |
+------------------------------------------------------------------+
|  Selected: 2 databases                                            |
|                                                                   |
|  [Upgrade Selected]  [Backup & Upgrade]  [View Details]  [Cancel] |
+------------------------------------------------------------------+
|  Status: Ready                                                    |
+------------------------------------------------------------------+
```

**Grid Columns (from `projects.project` table):**

| Column | Header | Width | Description |
|--------|--------|-------|-------------|
| (checkbox) | Select | 30px | Selectable for batch operations |
| `project_name` | Project Name | 150px | User-friendly project name |
| `database_name` | Database | 200px | PostgreSQL database name |
| `organization` | Organization | 150px | Client/organization name |
| `database_version` | Schema Version | 100px | Current schema version |
| `application_version` | App Version | 100px | Application version at creation |
| `is_active` | Active | 60px | Currently active database (indicator) |
| `last_used_at` | Last Used | 120px | Last access timestamp |
| `created_at` | Created | 120px | Creation timestamp |
| `notes` | Notes | 200px | User notes/comments |

**Column Visibility:**
- All columns are visible by default
- User can right-click header to show/hide columns
- Column preferences saved to user settings

**Status Indicators:**
- Green checkmark: Up to date
- Yellow warning: Pending migrations (shows count)
- Red X: Error/unreachable
- Star icon: Currently active database

**GUI Actions:**

1. **Select/Deselect Databases**
   - Click checkbox to select individual databases
   - "Select All" checkbox for bulk selection
   - Shift+click for range selection
   - Cannot deselect currently active database from destructive operations

2. **Upgrade Selected**
   - Enabled when 1+ databases with pending migrations are selected
   - Shows confirmation dialog with list of databases and pending migrations
   - Progress bar during upgrade
   - Real-time log output in status area

3. **Backup & Upgrade**
   - Same as Upgrade Selected but creates backup first
   - Shows backup location in confirmation dialog
   - Backup created before each database upgrade (not once at start)

4. **View Details**
   - Opens detail panel/dialog for selected database
   - Shows full project information
   - Lists all applied migrations
   - Shows pending migrations with SQL preview

5. **Refresh**
   - Re-queries SmartProAudit for current project list
   - Updates status indicators

**Upgrade Progress Dialog:**

```
+--------------------------------------------------+
|  Upgrading Databases                              |
+--------------------------------------------------+
|  Overall Progress: [=========>          ] 2/3    |
|                                                   |
|  Current: professional_smart_B                    |
|  Migration: 069_setup_smartproaudit_fdw.sql       |
|  [====================>      ] 75%                |
|                                                   |
|  Log:                                             |
|  [10:30:01] Starting upgrade of professional_A    |
|  [10:30:05] Applied 068_create_encounter_view.sql |
|  [10:30:08] Applied 069_setup_smartproaudit_fdw   |
|  [10:30:08] SUCCESS: professional_smart_A         |
|  [10:30:09] Starting upgrade of professional_B    |
|  [10:30:12] Applied 068_create_encounter_view.sql |
|                                                   |
|  [Cancel]                               [Close]   |
+--------------------------------------------------+
```

**Error Handling:**

- Connection errors display reconnect dialog
- Failed upgrades highlight in red with error message
- Option to continue or abort on error
- Rollback status shown for failed databases

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| F5 | Refresh list |
| Ctrl+A | Select all |
| Ctrl+U | Upgrade selected |
| Ctrl+B | Backup & upgrade selected |
| Enter | View details of selected |
| Escape | Cancel/close dialog |
| Space | Toggle selection of focused row |

**Technology:**

- Built with `egui` or `iced` (Rust native GUI frameworks)
- Single executable (no additional runtime dependencies)
- Windows native look and feel
- High DPI aware

---

### 2.10 Project Registry (SmartProAudit Database)

**Purpose:** Track all project databases created/used by this installation.

**Location:** PostgreSQL database `SmartProAudit` on the same server as project databases.

**Database Structure:**

The `SmartProAudit` database contains two schemas:

#### Schema: `projects`

Stores the project registry (replaces file-based `projects.json`).

**Table: `projects.project`**
```sql
CREATE TABLE projects.project (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    application_version VARCHAR(50),
    backend_version VARCHAR(50),
    database_version VARCHAR(50),
    connection_information VARCHAR(500),  -- host:port format
    database_name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT FALSE,
    notes TEXT
);
```

**Table: `projects.schema_migrations`**
Tracks migrations applied to SmartProAudit itself.

#### Schema: `fields`

Stores field metadata for claims data display and export.

**Table: `fields.lookup_field_definitions`**
```sql
CREATE TABLE fields.lookup_field_definitions (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    column_sort_order INT DEFAULT 0,
    data_type VARCHAR(255),
    friendly_name VARCHAR(255),
    source VARCHAR(10) DEFAULT 'system'  -- 'system' or 'user'
);
```

Pre-populated with field definitions for:
- `encounter` table
- `encounter_view` view
- `encounter_diagnosis` table
- `encounter_payer` table
- `service_line` table

**Registry Operations:**
- **Auto-creation:** Installer creates `SmartProAudit` database if it doesn't exist
- **Registration:** Automatically register databases created by `pro-project create`
- **Sync:** Validate registry entries against actual databases on `list` command
- **Active tracking:** Only one project can have `is_active = true` at a time
- **Version tracking:** `application_version` and `database_version` updated on upgrades

---

## 3. Non-Functional Requirements

### 3.1 Performance

| Requirement | Target |
|-------------|--------|
| Database creation | < 30 seconds |
| Database switch | < 15 seconds (including service restart) |
| Project list | < 5 seconds |
| Project info | < 3 seconds |
| Status check | < 5 seconds (all projects) |
| Single database upgrade | < 60 seconds per migration |
| Batch upgrade (10 projects) | < 10 minutes |

### 3.2 Reliability

- **Atomic operations:** Config updates must be atomic (write to temp, rename)
- **Rollback capability:** Failed switches must restore previous configuration
- **Backup verification:** All backups must be verified after creation
- **Service recovery:** If service fails to start, provide clear diagnostics

### 3.3 Security

- **Password handling:** Never log or display passwords
- **File permissions:** Registry and config files inherit ProgramData permissions
- **Credential storage:** Use existing .env credential storage pattern
- **Backup security:** Backups contain sensitive data; warn user about storage

### 3.4 Compatibility

- **PostgreSQL versions:** 12, 13, 14, 15, 16, 17
- **Windows versions:** Windows 10, Windows 11, Windows Server 2019/2022
- **Existing installations:** Must work with databases created by installer

---

## 4. Technical Design

### 4.1 Architecture

```
pro-project.exe
    |
    +-- Entry Point
    |       |-- CLI Mode (default, via clap)
    |       +-- GUI Mode (--gui flag or `gui` subcommand)
    |
    +-- Commands (clap)
    |       |-- CreateCommand
    |       |-- SwitchCommand
    |       |-- ListCommand
    |       |-- InfoCommand
    |       |-- DeleteCommand
    |       |-- BackupCommand
    |       |-- UpgradeCommand
    |       |-- StatusCommand
    |       +-- GuiCommand (launches GUI mode)
    |
    +-- GUI (egui/eframe)
    |       |-- MainWindow
    |       |       |-- ProjectGrid (data table with checkboxes)
    |       |       |-- ActionBar (Upgrade, Backup, View Details)
    |       |       +-- StatusBar (connection info, messages)
    |       |-- UpgradeDialog (progress tracking)
    |       |-- DetailsDialog (project info, migration list)
    |       +-- SettingsDialog (column visibility, preferences)
    |
    +-- Services
    |       |-- DatabaseService (create, drop, connect, query)
    |       |-- ConfigService (read/write .env, backup config)
    |       |-- RegistryService (query SmartProAudit projects.project)
    |       |-- WindowsServiceManager (start, stop, status)
    |       |-- MigrationService (apply baseline, check version, apply pending)
    |       +-- BackupService (pg_dump, pg_restore, verify)
    |
    +-- Shared (from existing crates)
            |-- pro-db (connection pool, queries)
            |-- pro-upgrade-manager (embedded migrations)
            +-- common utilities
```

### 4.2 New Crate Structure

```
crates/
  pro-project/
    Cargo.toml
    src/
      main.rs           # Entry point - CLI or GUI based on args
      cli/
        mod.rs          # CLI module
        commands/
          mod.rs
          create.rs     # CreateCommand implementation
          switch.rs     # SwitchCommand implementation
          list.rs       # ListCommand implementation
          info.rs       # InfoCommand implementation
          delete.rs     # DeleteCommand implementation
          backup.rs     # BackupCommand implementation
          upgrade.rs    # UpgradeCommand implementation
          status.rs     # StatusCommand implementation
      gui/
        mod.rs          # GUI module entry point
        app.rs          # Main application state
        main_window.rs  # Main window layout
        project_grid.rs # Data grid with project list
        upgrade_dialog.rs # Progress dialog for upgrades
        details_dialog.rs # Project details view
        settings.rs     # User preferences
        styles.rs       # Visual styling
      services/
        mod.rs
        database.rs     # PostgreSQL operations
        config.rs       # .env file operations
        registry.rs     # SmartProAudit registry operations
        windows.rs      # Windows service control
        migration.rs    # Migration detection and application
        backup.rs       # Backup/restore operations
```

### 4.3 Dependencies

```toml
[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Async runtime
tokio = { version = "1", features = ["full"] }

# Database
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }

# Error handling & logging
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"

# Windows
windows-service = "0.6"         # For service control

# Configuration
dotenvy = "0.15"                # For .env parsing

# GUI (NEW)
eframe = "0.27"                 # egui framework for native apps
egui = "0.27"                   # Immediate mode GUI
egui_extras = "0.27"            # Table/grid widget
rfd = "0.14"                    # Native file dialogs

# Internal dependencies
pro-db = { path = "../pro-db" }
pro-upgrade-manager = { path = "../pro-upgrade-manager" }
```

### 4.4 GUI Data Model

```rust
/// Project data loaded from SmartProAudit
#[derive(Clone, Debug)]
pub struct ProjectRow {
    pub id: i32,
    pub project_name: String,
    pub database_name: String,
    pub organization: Option<String>,
    pub application_version: Option<String>,
    pub backend_version: Option<String>,
    pub database_version: Option<String>,
    pub connection_information: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub notes: Option<String>,

    // Computed fields
    pub selected: bool,           // Checkbox state
    pub pending_migrations: u32,  // Count of pending migrations
    pub status: ProjectStatus,    // Up to date, Pending, Error
    pub reachable: bool,          // Can connect to database
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProjectStatus {
    UpToDate,
    PendingUpgrade(u32),  // Number of pending migrations
    Error(String),
    Checking,
}

/// Application state
pub struct ProjectManagerApp {
    pub projects: Vec<ProjectRow>,
    pub selected_count: usize,
    pub connection_string: String,
    pub status_message: String,
    pub show_upgrade_dialog: bool,
    pub upgrade_progress: Option<UpgradeProgress>,
    pub column_visibility: ColumnVisibility,
}

/// Column visibility settings
pub struct ColumnVisibility {
    pub project_name: bool,
    pub database_name: bool,
    pub organization: bool,
    pub database_version: bool,
    pub application_version: bool,
    pub is_active: bool,
    pub last_used_at: bool,
    pub created_at: bool,
    pub notes: bool,
}
```

### 4.4 Configuration File Updates

**Updated .env structure** (no changes required - existing fields sufficient):
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=professional_smart    # <-- This field changes on switch
DB_USER=postgres
DB_PASSWORD=your_password

# Optional: Full URL (recalculated on switch if present)
DATABASE_URL=postgres://postgres:password@localhost:5432/professional_smart
```

### 4.5 Service Control Implementation

```rust
use windows_service::{
    service::ServiceAccess,
    service_manager::{ServiceManager, ServiceManagerAccess},
};

pub fn stop_service() -> Result<()> {
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT,
    )?;

    let service = manager.open_service(
        "ProfessionalSMART",
        ServiceAccess::STOP | ServiceAccess::QUERY_STATUS,
    )?;

    service.stop()?;

    // Wait for stopped state
    wait_for_state(&service, ServiceState::Stopped, Duration::from_secs(30))
}

pub fn start_service() -> Result<()> {
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT,
    )?;

    let service = manager.open_service(
        "ProfessionalSMART",
        ServiceAccess::START | ServiceAccess::QUERY_STATUS,
    )?;

    service.start(&[] as &[&OsStr])?;

    // Wait for running state
    wait_for_state(&service, ServiceState::Running, Duration::from_secs(30))
}
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
- [ ] Create `crates/pro-project` crate structure
- [ ] Implement ConfigService (.env read/write with atomic updates)
- [ ] Implement RegistryService (projects.json CRUD)
- [ ] Implement WindowsServiceManager (start/stop/status)
- [ ] Add CLI scaffolding with clap

### Phase 2: Create Command
- [ ] Implement DatabaseService.create_database()
- [ ] Implement DatabaseService.apply_baseline()
- [ ] Integrate with pro-upgrade-manager for embedded migrations
- [ ] Implement `pro-project create` command
- [ ] Add validation for project names
- [ ] Register new projects in registry

### Phase 3: Switch Command
- [ ] Implement config backup mechanism
- [ ] Implement .env update logic
- [ ] Implement service stop/start cycle
- [ ] Implement `pro-project switch` command
- [ ] Add rollback on failure
- [ ] Update registry last_used timestamp

### Phase 4: List and Info Commands
- [ ] Implement database discovery (scan for PS schema)
- [ ] Implement `pro-project list` command
- [ ] Implement `pro-project info` command
- [ ] Add JSON/CSV output formats
- [ ] Calculate database sizes

### Phase 5: Delete and Backup Commands
- [ ] Implement pg_dump integration
- [ ] Implement `pro-project backup` command
- [ ] Implement `pro-project delete` command
- [ ] Add safety confirmations
- [ ] Registry cleanup

### Phase 6: Upgrade and Status Commands
- [ ] Implement MigrationService.get_pending_migrations()
- [ ] Implement MigrationService.apply_migrations_to_database()
- [ ] Implement `pro-project status` command
- [ ] Implement `pro-project upgrade` command with --dry-run
- [ ] Implement `pro-project upgrade --all` for batch upgrades
- [ ] Add --backup flag integration
- [ ] Add --continue-on-error flag for resilient batch upgrades
- [ ] Update registry schema_version after successful upgrades

### Phase 7: GUI Implementation
- [ ] Set up egui/eframe project structure
- [ ] Implement main window layout with project grid
- [ ] Add checkbox selection for databases
- [ ] Display all SmartProAudit columns in grid
- [ ] Implement column show/hide functionality
- [ ] Add status indicators (up to date, pending, error)
- [ ] Implement "Upgrade Selected" action
- [ ] Implement "Backup & Upgrade" action
- [ ] Add upgrade progress dialog with real-time log
- [ ] Implement "View Details" dialog
- [ ] Add keyboard shortcuts
- [ ] Save/restore user preferences (column visibility, window size)
- [ ] Error handling and reconnection dialogs

### Phase 8: Integration and Polish
- [ ] Add to installer build process
- [ ] Add Start Menu shortcut for GUI mode
- [ ] Update CLAUDE.md with new tool documentation
- [ ] Create user documentation
- [ ] Error message improvements
- [ ] Logging and diagnostics
- [ ] Windows high DPI support testing

---

## 6. Installer Integration

### 6.1 MSI Changes

Add `pro-project.exe` to the installer:

**Product.wxs additions:**
```xml
<Component Id="ProProjectExe" Guid="NEW-GUID-HERE">
  <File Id="ProProjectExe"
        Source="$(var.SolutionDir)\target\release\pro-project.exe"
        KeyPath="yes" />
</Component>
```

### 6.2 Initial Project Registration

During installation, register the initial database in `projects.json`:

**WriteConfig.vbs addition:**
```vbscript
' Create initial projects.json
Set projectsJson = CreateObject("Scripting.Dictionary")
projectsJson.Add "version", 1
projectsJson.Add "default_host", dbHost
projectsJson.Add "default_port", dbPort
projectsJson.Add "default_user", dbUser

Set initialProject = CreateObject("Scripting.Dictionary")
initialProject.Add "name", dbName
initialProject.Add "host", dbHost
initialProject.Add "port", dbPort
initialProject.Add "created_at", Now()
initialProject.Add "created_by", "installer"
initialProject.Add "schema_version", installerVersion

' Write to projects.json
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

| Test Case | Description |
|-----------|-------------|
| `test_validate_project_name` | Valid/invalid project names |
| `test_config_atomic_write` | Config update atomicity |
| `test_registry_crud` | Registry operations |
| `test_service_state_detection` | Service status checking |

### 7.2 Integration Tests

| Test Case | Description |
|-----------|-------------|
| `test_create_switch_cycle` | Create project, switch to it, switch back |
| `test_create_duplicate_fails` | Creating existing database fails gracefully |
| `test_switch_nonexistent_fails` | Switching to missing database fails gracefully |
| `test_delete_active_fails` | Cannot delete currently active database |
| `test_backup_restore_cycle` | Backup and verify integrity |

### 7.3 Manual Testing Checklist

- [ ] Create new project on fresh installation
- [ ] Switch between projects multiple times
- [ ] List shows correct status for all projects
- [ ] Delete project with backup
- [ ] Verify service restarts correctly after switch
- [ ] Verify data isolation between projects
- [ ] Test with PostgreSQL 12, 14, 16

---

## 8. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Service fails to restart | Medium | High | Automatic rollback, manual recovery docs |
| Config corruption | Low | High | Atomic writes, automatic backups |
| Wrong database deleted | Low | Critical | Confirmation prompts, backup option |
| Registry out of sync | Medium | Low | Auto-sync on list, manual refresh |
| Migration version mismatch | Low | Medium | Version check before switch, upgrade prompt |

---

## 9. Future Considerations

### 9.1 Potential Enhancements (Not in Scope)

1. **GUI Integration** - Add project selector to future web interface
2. **Remote Databases** - Support databases on different servers
3. **Project Templates** - Pre-configured projects with sample data
4. **Data Migration** - Copy data between project databases
5. **Scheduled Backups** - Automatic backup scheduling
6. **Project Archival** - Compress and archive inactive projects

### 9.2 Multi-Server Architecture

Future versions may support:
- Central project registry server
- Database server pooling
- Load balancing across project databases

---

## 10. Glossary

| Term | Definition |
|------|------------|
| Project Database | An isolated PostgreSQL database containing complete Professional SMART schema |
| Project Registry | JSON file tracking all known project databases |
| Active Database | The database currently configured in .env and used by the service |
| Baseline Migration | The consolidated schema file (000_baseline_v2.12.sql) containing all migrations |
| Switch | The process of updating configuration and restarting service to use a different database |
| Pending Migrations | Migrations embedded in the application but not yet applied to a database |
| Schema Version | The version recorded in `staging.application_version` indicating migration state |

---

## 11. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Senior Software Engineer | 2025-12-24 | |
| Technical Review | | | |
| Product Owner | | | |

---

## Appendix A: Command Quick Reference

```
pro-project create --name <NAME> [--switch]
    Create a new project database

pro-project switch --name <NAME> [--no-restart]
    Switch to a different project database

pro-project list [--format table|json|csv]
    List all project databases

pro-project info [--name <NAME>]
    Show detailed information about a project

pro-project delete --name <NAME> [--force] [--backup]
    Delete a project database

pro-project backup [--name <NAME>] [--output <PATH>]
    Create a backup of a project database

pro-project status
    Show upgrade status of all project databases

pro-project upgrade --name <NAME> [--backup] [--dry-run]
    Upgrade a specific project database

pro-project upgrade --all [--backup] [--dry-run] [--continue-on-error]
    Upgrade all project databases to latest schema

pro-project gui
pro-project --gui
    Launch graphical user interface for database management
```

## Appendix B: File and Database Locations

| Resource | Location |
|----------|----------|
| Executable | `C:\Program Files\Professional SMART\bin\pro-project.exe` |
| Configuration | `C:\ProgramData\Professional SMART\config\.env` |
| Project Registry | PostgreSQL database: `SmartProAudit` (table: `projects.project`) |
| Field Definitions | PostgreSQL database: `SmartProAudit` (table: `fields.lookup_field_definitions`) |
| SmartProAudit Schema | `C:\Program Files\Professional SMART\migrations\smartproaudit\000_baseline.sql` |
| Backups | `C:\ProgramData\Professional SMART\backups\` |
| Logs | `C:\ProgramData\Professional SMART\logs\` |
