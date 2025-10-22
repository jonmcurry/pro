# Migration Deployment Strategy

## Current Problem
- 24+ individual .sql files deployed to migrations folder
- WiX Heat.exe generates file list dynamically
- Doesn't handle schema updates vs fresh installs elegantly
- File count will continue to grow

## Requirements
1. **Fresh Installs**: Apply all migrations to create complete schema
2. **Upgrades**: Only apply new migrations since last installed version
3. **Single Source of Truth**: Migrations should be versioned and tracked
4. **Minimal Deployment**: Reduce number of files in MSI

## Solution Options

### Option 1: Embedded SQL Resources (Recommended)
**Approach**: Embed all .sql files as resources in pro-upgrade.exe binary

**Pros**:
- Zero migration files deployed to disk
- pro-upgrade.exe contains all migrations
- Clean installation folder
- Migrations are versioned with the executable
- Reduces MSI complexity

**Cons**:
- Migrations embedded at compile time
- Larger executable size (minimal - SQL is small)

**Implementation**:
1. Use Rust `include_str!()` macro to embed migrations
2. Create migration registry in pro-upgrade with version tracking
3. On fresh install: Run all migrations
4. On upgrade: Query schema_migrations table, run only new ones

```rust
const MIGRATIONS: &[(&str, &str)] = &[
    ("001_create_schemas.sql", include_str!("../../migrations/001_create_schemas.sql")),
    ("002_create_organization_tables.sql", include_str!("../../migrations/002_create_organization_tables.sql")),
    // ...
];
```

### Option 2: Single Consolidated SQL File
**Approach**: Concatenate all migrations into baseline.sql for fresh installs, keep individual files for upgrades

**Pros**:
- Fresh installs only need one file
- Faster fresh install execution
- Still have granular upgrade path

**Cons**:
- Still need to deploy individual migration files for upgrades
- Maintenance overhead (keep baseline in sync)
- Migrations folder still grows

### Option 3: ZIP Archive of Migrations
**Approach**: Deploy migrations.zip, extract on install

**Pros**:
- Single file to deploy
- Can add new migrations easily

**Cons**:
- Still creates 24+ files on disk
- Adds extraction step complexity

## Recommended Approach: Option 1 (Embedded Resources)

### Implementation Plan

#### Phase 1: Embed Migrations in pro-upgrade
- [ ] Create `crates/pro-upgrade/src/embedded_migrations.rs`
- [ ] Use `include_str!()` to embed all .sql files
- [ ] Create migration registry with version numbers
- [ ] Update pro-upgrade to use embedded migrations instead of reading from disk

#### Phase 2: Remove Migrations from MSI
- [ ] Remove MigrationsFragment.wxs generation from build.bat
- [ ] Remove migrations folder from Product.wxs
- [ ] Test fresh install with embedded migrations
- [ ] Test upgrade scenario

#### Phase 3: Schema Version Tracking
- [ ] Ensure schema_migrations table tracks applied migrations
- [ ] Add migration checksum verification
- [ ] Handle rollback scenarios

### Migration Loading Logic

```rust
pub struct EmbeddedMigration {
    pub version: &'static str,
    pub name: &'static str,
    pub sql: &'static str,
}

pub fn get_all_migrations() -> Vec<EmbeddedMigration> {
    vec![
        EmbeddedMigration {
            version: "001",
            name: "create_schemas",
            sql: include_str!("../../migrations/001_create_schemas.sql"),
        },
        // ...
    ]
}

pub async fn apply_migrations(pool: &PgPool, target_version: Option<&str>) -> Result<()> {
    // Fresh install: apply all migrations
    // Upgrade: query schema_migrations, apply only new ones
}
```

### Benefits
1. **Zero migration files on disk** - clean installation
2. **Versioned with executable** - migrations match code version
3. **Faster deploys** - no file copying during install
4. **Simpler MSI** - no Heat.exe, no file tracking
5. **Atomic** - migrations and code are always in sync

## Migration Versioning Strategy

Track in `staging.schema_migrations`:
```sql
CREATE TABLE staging.schema_migrations (
    version VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64)
);
```

On install/upgrade:
1. Read current schema version
2. Apply all migrations > current version
3. Record each migration in schema_migrations
