use sha2::{Sha256, Digest};

#[derive(Debug, Clone)]
pub struct EmbeddedMigration {
    pub version: &'static str,
    pub name: &'static str,
    pub sql: &'static str,
}

impl EmbeddedMigration {
    pub fn file_name(&self) -> String {
        format!("{}_{}.sql", self.version, self.name)
    }

    pub fn checksum(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.sql.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

pub fn get_all_migrations() -> Vec<EmbeddedMigration> {
    vec![
        EmbeddedMigration {
            version: "001",
            name: "create_schemas",
            sql: include_str!("../../../migrations/001_create_schemas.sql"),
        },
        EmbeddedMigration {
            version: "002",
            name: "create_organization_tables",
            sql: include_str!("../../../migrations/002_create_organization_tables.sql"),
        },
        EmbeddedMigration {
            version: "003",
            name: "create_provider_tables",
            sql: include_str!("../../../migrations/003_create_provider_tables.sql"),
        },
        EmbeddedMigration {
            version: "004",
            name: "create_encounter_tables",
            sql: include_str!("../../../migrations/004_create_encounter_tables.sql"),
        },
        EmbeddedMigration {
            version: "005",
            name: "create_diagnosis_procedure_tables",
            sql: include_str!("../../../migrations/005_create_diagnosis_procedure_tables.sql"),
        },
        EmbeddedMigration {
            version: "006",
            name: "create_flag_tables",
            sql: include_str!("../../../migrations/006_create_flag_tables.sql"),
        },
        EmbeddedMigration {
            version: "007",
            name: "create_staging_tables",
            sql: include_str!("../../../migrations/007_create_staging_tables.sql"),
        },
        EmbeddedMigration {
            version: "008",
            name: "create_audit_tables",
            sql: include_str!("../../../migrations/008_create_audit_tables.sql"),
        },
        EmbeddedMigration {
            version: "009",
            name: "create_rvu_tables",
            sql: include_str!("../../../migrations/009_create_rvu_tables.sql"),
        },
        EmbeddedMigration {
            version: "010",
            name: "create_denial_tables",
            sql: include_str!("../../../migrations/010_create_denial_tables.sql"),
        },
        EmbeddedMigration {
            version: "011",
            name: "create_schedule_tables",
            sql: include_str!("../../../migrations/011_create_schedule_tables.sql"),
        },
        EmbeddedMigration {
            version: "012",
            name: "create_ml_tables",
            sql: include_str!("../../../migrations/012_create_ml_tables.sql"),
        },
        EmbeddedMigration {
            version: "013",
            name: "create_dashboard_views",
            sql: include_str!("../../../migrations/013_create_dashboard_views.sql"),
        },
        EmbeddedMigration {
            version: "014",
            name: "create_utility_functions",
            sql: include_str!("../../../migrations/014_create_utility_functions.sql"),
        },
        EmbeddedMigration {
            version: "015",
            name: "create_fifo_queue",
            sql: include_str!("../../../migrations/015_create_fifo_queue.sql"),
        },
        EmbeddedMigration {
            version: "016",
            name: "phase5_performance_indexes",
            sql: include_str!("../../../migrations/016_phase5_performance_indexes.sql"),
        },
        EmbeddedMigration {
            version: "017",
            name: "streaming_progress_tracking",
            sql: include_str!("../../../migrations/017_streaming_progress_tracking.sql"),
        },
        EmbeddedMigration {
            version: "018",
            name: "phase6_strategic_indexes",
            sql: include_str!("../../../migrations/018_phase6_strategic_indexes.sql"),
        },
        EmbeddedMigration {
            version: "019",
            name: "phase6_materialized_views",
            sql: include_str!("../../../migrations/019_phase6_materialized_views.sql"),
        },
        EmbeddedMigration {
            version: "020",
            name: "create_version_tracking",
            sql: include_str!("../../../migrations/020_create_version_tracking.sql"),
        },
        EmbeddedMigration {
            version: "021",
            name: "insert_initial_version",
            sql: include_str!("../../../migrations/021_insert_initial_version.sql"),
        },
        EmbeddedMigration {
            version: "022",
            name: "test_upgrade_migration",
            sql: include_str!("../../../migrations/022_test_upgrade_migration.sql"),
        },
        EmbeddedMigration {
            version: "023",
            name: "create_raw_claims_table",
            sql: include_str!("../../../migrations/023_create_raw_claims_table.sql"),
        },
        EmbeddedMigration {
            version: "024",
            name: "add_batch_sequence_tracking",
            sql: include_str!("../../../migrations/024_add_batch_sequence_tracking.sql"),
        },
        EmbeddedMigration {
            version: "025",
            name: "rename_duration_column",
            sql: include_str!("../../../migrations/025_rename_duration_column.sql"),
        },
        EmbeddedMigration {
            version: "026",
            name: "fix_timestamp_columns",
            sql: include_str!("../../../migrations/026_fix_timestamp_columns.sql"),
        },
    ]
}
