# Foreign Data Wrapper (FDW) How-To Guide

**Version:** 2.12.70.1
**Last Updated:** 2025-12-30

---

## Overview

Professional SMART uses PostgreSQL's Foreign Data Wrapper (FDW) to connect project databases to the central `smartproaudit` master database. This allows each project database to query security users/roles and field definitions without duplicating data.

### Architecture

```
+---------------------------+       +---------------------------+
|   Project Database        |       |  SmartProAudit Database   |
|   (professional_smart)    |       |  (smartproaudit)          |
+---------------------------+       +---------------------------+
|                           |       |                           |
|  claims schema            |       |  security schema          |
|  staging schema           |  FDW  |    - security_user        |
|  ml schema                | ----> |    - security_role        |
|                           |       |    - security_user_role   |
|  smartproaudit schema     |       |                           |
|  (foreign tables)         |       |  fields schema            |
|    - security_user        |       |    - lookup_field_defs    |
|    - security_role        |       |                           |
|    - lookup_field_defs    |       |  projects schema          |
|    - project              |       |    - project              |
+---------------------------+       +---------------------------+
```

---

## Prerequisites

1. **SmartProAudit database exists** - The master database must be created first
2. **postgres_fdw extension available** - Included in standard PostgreSQL installations
3. **Network access** - Project database must be able to connect to SmartProAudit (usually same server)

---

## Setup (Automatic)

The FDW is automatically configured when you:

1. **Create a new project database** using the Project Database Manager GUI
2. **Run migrations** via `pro-upgrade.exe` (migration 069 sets up FDW)

---

## Setup (Manual)

If you need to set up FDW manually:

### Step 1: Enable the Extension

```sql
-- Connect to your project database
\c professional_smart

-- Enable postgres_fdw
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
```

### Step 2: Create the Foreign Server

```sql
-- Create server connection to SmartProAudit
CREATE SERVER smartproaudit_server
    FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'localhost', port '5432', dbname 'smartproaudit');
```

### Step 3: Create User Mapping

```sql
-- Map current user to postgres on the foreign server with password authentication
CREATE USER MAPPING FOR postgres
    SERVER smartproaudit_server
    OPTIONS (user 'postgres', password 'postgres');
```

**Note:** The default password is `postgres`. Change this in production environments.

### Step 4: Create Schema and Foreign Tables

```sql
-- Create schema to hold foreign tables
CREATE SCHEMA IF NOT EXISTS smartproaudit;

-- Create foreign table for security users
CREATE FOREIGN TABLE smartproaudit.security_user (
    id BIGINT,
    user_name VARCHAR(100),
    active BOOLEAN
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_user');

-- Create foreign table for security roles
CREATE FOREIGN TABLE smartproaudit.security_role (
    id BIGINT,
    role_name VARCHAR(50),
    role_description VARCHAR(100)
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_role');

-- Create foreign table for user-role assignments
CREATE FOREIGN TABLE smartproaudit.security_user_role (
    id BIGINT,
    user_id BIGINT,
    role_id BIGINT
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_user_role');

-- Create foreign table for field definitions
CREATE FOREIGN TABLE smartproaudit.lookup_field_definitions (
    id INTEGER,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    column_sort_order INTEGER,
    data_type VARCHAR(255),
    friendly_name VARCHAR(255),
    source VARCHAR(10)
)
SERVER smartproaudit_server
OPTIONS (schema_name 'fields', table_name 'lookup_field_definitions');
```

---

## Using the FDW

### Query Security Users

```sql
-- List all active users
SELECT * FROM smartproaudit.security_user WHERE active = true;

-- List all roles
SELECT * FROM smartproaudit.security_role;

-- Get users with their roles (using convenience view)
SELECT * FROM smartproaudit.user_roles;
```

### Check User Permissions

```sql
-- Check if a user has a specific role
SELECT smartproaudit.user_has_role('john.doe', 'Admin');

-- Example: Check if current user is an auditor
SELECT smartproaudit.user_has_role(current_user, 'Auditor');
```

### Query Field Definitions

```sql
-- Get all field definitions for encounter table
SELECT * FROM smartproaudit.get_field_definitions('encounter');

-- Get field definitions for service_line table
SELECT * FROM smartproaudit.get_field_definitions('service_line');

-- Direct query to field definitions
SELECT column_name, friendly_name, data_type
FROM smartproaudit.lookup_field_definitions
WHERE table_name = 'encounter'
ORDER BY column_sort_order;
```

### Query Project Registry

```sql
-- List all registered projects
SELECT project_name, database_name, is_active, last_used_at
FROM smartproaudit.project
ORDER BY last_used_at DESC;

-- Find active projects
SELECT * FROM smartproaudit.project WHERE is_active = true;
```

### Join Foreign Tables with Local Tables

```sql
-- Example: Get encounters with auditor information
SELECT
    e.encounter_id,
    e.patient_control_number,
    e.created_at,
    u.user_name AS created_by_user
FROM claims.encounter e
LEFT JOIN smartproaudit.security_user u ON e.created_by = u.id
WHERE e.created_at > CURRENT_DATE - INTERVAL '7 days';
```

---

## Troubleshooting

### Error: "could not connect to server"

**Cause:** SmartProAudit database is not accessible.

**Solution:**
```sql
-- Verify SmartProAudit database exists
\l smartproaudit

-- Check server connection settings
SELECT * FROM pg_foreign_server WHERE srvname = 'smartproaudit_server';
```

### Error: "relation does not exist"

**Cause:** The source table in SmartProAudit doesn't exist or has different schema.

**Solution:**
```sql
-- Connect to SmartProAudit and verify table exists
\c smartproaudit
\dt security.*
\dt fields.*
```

### Error: "permission denied"

**Cause:** User mapping doesn't have access to the foreign tables.

**Solution:**
```sql
-- Check user mappings
SELECT * FROM pg_user_mappings;

-- Recreate user mapping if needed (include password for authentication)
DROP USER MAPPING IF EXISTS FOR postgres SERVER smartproaudit_server;
CREATE USER MAPPING FOR postgres SERVER smartproaudit_server OPTIONS (user 'postgres', password 'postgres');
```

### Slow Queries on Foreign Tables

**Cause:** PostgreSQL may not push down all query conditions to the foreign server.

**Solutions:**
1. Use `EXPLAIN VERBOSE` to see what's being sent to the foreign server
2. Add indexes on the SmartProAudit tables
3. Consider materializing frequently-accessed data locally

```sql
-- Analyze query execution
EXPLAIN VERBOSE SELECT * FROM smartproaudit.security_user WHERE active = true;
```

---

## Verifying FDW Setup

Run these commands to verify your FDW is working:

```sql
-- Connect to project database
\c professional_smart

-- 1. Check extension is installed
SELECT * FROM pg_extension WHERE extname = 'postgres_fdw';

-- 2. Check foreign server exists
SELECT srvname, srvowner::regrole, srvoptions
FROM pg_foreign_server
WHERE srvname = 'smartproaudit_server';

-- 3. Check user mapping exists
SELECT * FROM pg_user_mappings WHERE srvname = 'smartproaudit_server';

-- 4. List foreign tables
\det smartproaudit.*

-- 5. Test query (should return results if SmartProAudit has data)
SELECT COUNT(*) FROM smartproaudit.security_role;
SELECT COUNT(*) FROM smartproaudit.security_user;
```

---

## Security Considerations

1. **User Mappings**: The FDW user mapping determines what credentials are used to connect. Ensure the mapped user has minimal required permissions.

2. **Network Security**: If SmartProAudit is on a different server, ensure the connection uses SSL/TLS.

3. **Read-Only Access**: Foreign tables are read-only by default, which is the intended behavior for security data.

4. **Password Storage**: User mapping passwords are stored in the database. Consider using `password_required=false` with peer authentication for local connections.

---

## Reference

- **Migration File:** `migrations/069_setup_smartproaudit_fdw.sql`
- **PostgreSQL FDW Documentation:** https://www.postgresql.org/docs/current/postgres-fdw.html
- **Related Docs:** [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md), [DATABASE_SETUP.md](DATABASE_SETUP.md)
