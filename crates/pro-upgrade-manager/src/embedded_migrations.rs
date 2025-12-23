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

/// Baseline schema (complete schema snapshot)
/// Used for fresh installations - faster than running 60+ migrations
pub const BASELINE: EmbeddedMigration = EmbeddedMigration {
    version: "000",
    name: "baseline_v2.12",
    sql: include_str!("../../../migrations/000_baseline_v2.12.sql"),
};

/// Version at which the baseline was created
/// All migrations up to and including this number are covered by the baseline
pub const BASELINE_COVERS_THROUGH: u32 = 64;

/// Get the baseline migration
pub fn get_baseline() -> &'static EmbeddedMigration {
    &BASELINE
}

/// Get all incremental migrations (excluding baseline)
/// These are used for upgrades from existing installations
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
        EmbeddedMigration {
            version: "027",
            name: "drop_unused_scheduling_tables",
            sql: include_str!("../../../migrations/027_drop_unused_scheduling_tables.sql"),
        },
        EmbeddedMigration {
            version: "028",
            name: "add_project_id_to_organization",
            sql: include_str!("../../../migrations/028_add_project_id_to_organization.sql"),
        },
        EmbeddedMigration {
            version: "029",
            name: "drop_charge_amount_constraints",
            sql: include_str!("../../../migrations/029_drop_charge_amount_constraints.sql"),
        },
        EmbeddedMigration {
            version: "030",
            name: "create_import_headers_table",
            sql: include_str!("../../../migrations/030_create_import_headers_table.sql"),
        },
        EmbeddedMigration {
            version: "031",
            name: "create_delete_project_procedure",
            sql: include_str!("../../../migrations/031_create_delete_project_procedure.sql"),
        },
        EmbeddedMigration {
            version: "032",
            name: "create_claims_detail_view",
            sql: include_str!("../../../migrations/032_create_claims_detail_view.sql"),
        },
        EmbeddedMigration {
            version: "033",
            name: "create_field_definitions_table",
            sql: include_str!("../../../migrations/033_create_field_definitions_table.sql"),
        },
        EmbeddedMigration {
            version: "034",
            name: "add_provider_full_name",
            sql: include_str!("../../../migrations/034_add_provider_full_name.sql"),
        },
        EmbeddedMigration {
            version: "035",
            name: "add_medical_record_number",
            sql: include_str!("../../../migrations/035_add_medical_record_number.sql"),
        },
        EmbeddedMigration {
            version: "036",
            name: "phase3_advanced_segments",
            sql: include_str!("../../../migrations/036_phase3_advanced_segments.sql"),
        },
        EmbeddedMigration {
            version: "037",
            name: "phase4_advanced_cob",
            sql: include_str!("../../../migrations/037_phase4_advanced_cob.sql"),
        },
        EmbeddedMigration {
            version: "038",
            name: "phase5_specialized_claims",
            sql: include_str!("../../../migrations/038_phase5_specialized_claims.sql"),
        },
        EmbeddedMigration {
            version: "039",
            name: "phase6_additional_loops",
            sql: include_str!("../../../migrations/039_phase6_additional_loops.sql"),
        },
        EmbeddedMigration {
            version: "041",
            name: "create_provider_taxonomy",
            sql: include_str!("../../../migrations/041_create_provider_taxonomy.sql"),
        },
        EmbeddedMigration {
            version: "042",
            name: "create_provider_enrichment_queue",
            sql: include_str!("../../../migrations/042_create_provider_enrichment_queue.sql"),
        },
        EmbeddedMigration {
            version: "043",
            name: "add_missing_foreign_key_indexes",
            sql: include_str!("../../../migrations/043_add_missing_foreign_key_indexes.sql"),
        },
        EmbeddedMigration {
            version: "044",
            name: "add_taxonomy_foreign_key",
            sql: include_str!("../../../migrations/044_add_taxonomy_foreign_key.sql"),
        },
        EmbeddedMigration {
            version: "045",
            name: "add_staging_foreign_keys",
            sql: include_str!("../../../migrations/045_add_staging_foreign_keys.sql"),
        },
        EmbeddedMigration {
            version: "046",
            name: "create_rule_configuration_system",
            sql: include_str!("../../../migrations/046_create_rule_configuration_system.sql"),
        },
        EmbeddedMigration {
            version: "047",
            name: "add_test_facility_rule_assignments",
            sql: include_str!("../../../migrations/047_add_test_facility_rule_assignments.sql"),
        },
        EmbeddedMigration {
            version: "048",
            name: "add_rule_templates",
            sql: include_str!("../../../migrations/048_add_rule_templates.sql"),
        },
        EmbeddedMigration {
            version: "049",
            name: "add_flag_issue_helpers",
            sql: include_str!("../../../migrations/049_add_flag_issue_helpers.sql"),
        },
        EmbeddedMigration {
            version: "050",
            name: "add_performance_indexes",
            sql: include_str!("../../../migrations/050_add_performance_indexes.sql"),
        },
        EmbeddedMigration {
            version: "051",
            name: "add_rule_execution_stats",
            sql: include_str!("../../../migrations/051_add_rule_execution_stats.sql"),
        },
        EmbeddedMigration {
            version: "052",
            name: "add_npi_registry_link",
            sql: include_str!("../../../migrations/052_add_npi_registry_link.sql"),
        },
        EmbeddedMigration {
            version: "053",
            name: "add_837p_v2_fields",
            sql: include_str!("../../../migrations/053_add_837p_v2_fields.sql"),
        },
        EmbeddedMigration {
            version: "054",
            name: "create_specialty_table",
            sql: include_str!("../../../migrations/054_create_specialty_table.sql"),
        },
        EmbeddedMigration {
            version: "055",
            name: "add_partial_status_to_import_batch",
            sql: include_str!("../../../migrations/055_add_partial_status_to_import_batch.sql"),
        },
        EmbeddedMigration {
            version: "056",
            name: "create_archive_system",
            sql: include_str!("../../../migrations/056_create_archive_system.sql"),
        },
        EmbeddedMigration {
            version: "057",
            name: "create_dashboard_materialized_views",
            sql: include_str!("../../../migrations/057_create_dashboard_materialized_views.sql"),
        },
        EmbeddedMigration {
            version: "058",
            name: "change_cascade_to_restrict",
            sql: include_str!("../../../migrations/058_change_cascade_to_restrict.sql"),
        },
        EmbeddedMigration {
            version: "059",
            name: "fifo_optimization_and_recovery",
            sql: include_str!("../../../migrations/059_fifo_optimization_and_recovery.sql"),
        },
        EmbeddedMigration {
            version: "060",
            name: "add_patient_fields",
            sql: include_str!("../../../migrations/060_add_patient_fields.sql"),
        },
        EmbeddedMigration {
            version: "061",
            name: "fix_patient_relationship_code_length",
            sql: include_str!("../../../migrations/061_fix_patient_relationship_code_length.sql"),
        },
        EmbeddedMigration {
            version: "062",
            name: "create_encounter_payer_table",
            sql: include_str!("../../../migrations/062_create_encounter_payer_table.sql"),
        },
        EmbeddedMigration {
            version: "063",
            name: "add_billing_date_to_encounter",
            sql: include_str!("../../../migrations/063_add_billing_date_to_encounter.sql"),
        },
        EmbeddedMigration {
            version: "064",
            name: "make_dates_nullable",
            sql: include_str!("../../../migrations/064_make_dates_nullable.sql"),
        },
        EmbeddedMigration {
            version: "065",
            name: "cte_batch_acquisition_indexes",
            sql: include_str!("../../../migrations/065_cte_batch_acquisition_indexes.sql"),
        },
        EmbeddedMigration {
            version: "066",
            name: "enforce_postgresql_settings",
            sql: include_str!("../../../migrations/066_enforce_postgresql_settings.sql"),
        },
        EmbeddedMigration {
            version: "067",
            name: "create_encounter_procedure_modifiers",
            sql: include_str!("../../../migrations/067_create_encounter_procedure_modifiers.sql"),
        },
    ]
}
