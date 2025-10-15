# Database Migrations

This directory contains all PostgreSQL migration files for the Professional SMART application. Migrations are designed to be run in order and create a comprehensive database schema for healthcare claims processing, auditing, and analytics.

## Migration Overview

### Schema Structure

The database uses three schemas:

- **staging**: Import processing, file tracking, configuration, and job scheduling
- **claims**: Main claims data, encounters, service lines, flags, audits, and denials
- **ml**: Machine learning models, predictions, features, and training datasets

### Migration Files

Run migrations in numeric order:

1. **001_create_schemas.sql**
   - Creates three schemas (staging, claims, ml)
   - Enables PostgreSQL extensions (uuid-ossp, citext, pg_trgm, pgcrypto)

2. **002_create_organization_tables.sql**
   - Organization hierarchy: organization -> region -> facility
   - Proper constraints to enforce hierarchy rules
   - Audit trail fields and updated_at triggers

3. **003_create_provider_tables.sql**
   - Provider table (NPI, taxonomy, specialty)
   - Coder table (medical coders)
   - Reviewer table (audit reviewers)
   - All with certifications and organizational links

4. **004_create_encounter_tables.sql**
   - Main encounter/claim table with ALL 837p data elements
   - Includes all loops: submitter, subscriber, payer, billing/rendering/referring/supervising providers
   - Service facility, dates, ambulance info, COB
   - Comprehensive indexing for 10K claims/15sec performance target
   - Encounter notes table

5. **005_create_diagnosis_procedure_tables.sql**
   - encounter_diagnosis: up to 12 diagnoses with HCC indicators
   - service_line: comprehensive procedure/service line data (Loop 2400)
   - service_line_adjustment: other payer adjustments
   - service_line_diagnosis_pointer: explicit diagnosis mappings
   - Includes NDC, DME, anesthesia, ambulance, test results

6. **006_create_flag_tables.sql**
   - flag_category: predefined categories (COD, DOC, EMO, EMU, EMI, EMT, MOD, OTH, QTY, SUP, DX)
   - flag_issue: 24 specific issue definitions with severity
   - encounter_flag: flags at encounter level
   - service_line_flag: flags at service line level
   - Pre-populated with all SRD issue types

7. **007_create_staging_tables.sql**
   - import_batch: file import tracking and metrics
   - file_upload: multi-part upload tracking
   - import_configuration: CSV/EDI import profiles with header mappings
   - rules_configuration: rules engine configuration
   - processing_metrics: performance tracking
   - import_error_log: detailed error logging

8. **008_create_audit_tables.sql**
   - audit_assignment: retrospective audit assignments
   - audit_encounter: encounters selected for review
   - service_line_evaluation: detailed audit findings
   - diagnosis_evaluation: diagnosis code audit findings
   - coder_accuracy: coder accuracy metrics over time
   - provider_accuracy: provider documentation accuracy

9. **009_create_rvu_tables.sql**
   - rvu_reference: CMS Physician Fee Schedule data
   - conversion_factor: annual Medicare conversion factors (2024 = $33.2875)
   - gpci_reference: Geographic Practice Cost Indexes
   - modifier_adjustment: modifier reimbursement rules
   - service_line_reimbursement: estimated payments
   - Pre-populated with 2024 conversion factor and common modifiers

10. **010_create_denial_tables.sql**
    - denial_event: denial tracking with root cause analysis
    - denial_appeal: appeal actions and correspondence
    - denial_reason_code: CARC/RARC reference
    - denial_statistics: aggregated denial metrics
    - Tracks preventability, appeals, and financial recovery

11. **011_create_schedule_tables.sql**
    - scheduled_job: automated job configuration (CRON, interval, one-time)
    - job_execution_log: execution history with performance metrics
    - report_subscription: scheduled report delivery
    - report_generation_log: report generation tracking
    - data_refresh_schedule: materialized view refresh scheduling

12. **012_create_ml_tables.sql**
    - model_registry: ML model versioning and metadata
    - model_prediction: predictions with explanations (SHAP values)
    - feature_definition: feature engineering definitions
    - training_dataset: training data versioning
    - model_performance_log: drift detection and monitoring
    - ab_test_experiment: A/B testing for model comparison

13. **013_create_dashboard_views.sql**
    - v_management_overview: high-level executive metrics
    - v_claim_status_summary: claims by status
    - v_coder_performance: coder productivity and accuracy
    - v_provider_documentation_accuracy: provider quality metrics
    - v_flags_by_category: flag statistics and resolution
    - v_service_line_flags_detail: detailed flag information
    - v_denial_by_payer: denial analysis by payer
    - v_denial_by_reason: denial analysis by CARC
    - v_procedure_volume: procedure volume and performance
    - v_provider_productivity: provider RVU and productivity
    - v_audit_assignment_status: audit tracking
    - v_reimbursement_analysis: financial analysis

14. **014_create_utility_functions.sql**
    - Validation functions (NPI, ICD-10, CPT/HCPCS)
    - Calculation functions (age, Medicare payment, business days)
    - Data masking functions (MBI)
    - Utility functions (phone standardization, fiscal year)
    - Triggers for data validation and auto-calculation
    - Additional performance indexes (trigram, composite)
    - Materialized views (mv_flag_statistics, mv_denial_statistics)
    - Helper functions for common queries

## Running Migrations

### Prerequisites

- PostgreSQL 14 or higher
- Database created with UTF-8 encoding
- Superuser or sufficient privileges to create schemas and extensions

### Execution Order

Execute migrations in numeric order:

```bash
psql -U username -d database_name -f migrations/001_create_schemas.sql
psql -U username -d database_name -f migrations/002_create_organization_tables.sql
psql -U username -d database_name -f migrations/003_create_provider_tables.sql
# ... continue through 014
```

Or use a migration runner script:

```bash
for file in migrations/*.sql; do
    echo "Running $file..."
    psql -U username -d database_name -f "$file"
done
```

### Verification

After running all migrations, verify:

```sql
-- Check schemas exist
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('staging', 'claims', 'ml');

-- Count tables per schema
SELECT schemaname, COUNT(*)
FROM pg_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
GROUP BY schemaname;

-- Check extensions
SELECT * FROM pg_extension WHERE extname IN ('uuid-ossp', 'citext', 'pg_trgm', 'pgcrypto');

-- Verify views
SELECT schemaname, viewname
FROM pg_views
WHERE schemaname = 'claims'
AND viewname LIKE 'v_%';

-- Verify materialized views
SELECT schemaname, matviewname
FROM pg_matviews
WHERE schemaname = 'claims';
```

## Key Features

### Performance Optimization

- Comprehensive indexing strategy for 10K claims/15 seconds target
- Composite indexes on common query patterns
- Partial indexes for filtered queries
- Trigram indexes for fuzzy text search
- Materialized views for dashboard performance

### Data Integrity

- Foreign key constraints throughout
- Check constraints on critical fields
- Triggers for automatic timestamp updates
- Validation triggers for dates and formats
- Unique constraints to prevent duplicates

### Audit Trail

- created_at, updated_at, created_by, updated_by on all tables
- Soft delete support (is_active, soft_deleted)
- Comprehensive change tracking

### 837p Compliance

- All data elements from ASC X12N Version 005010X222A1
- Loop 1000A/B: Submitter/Receiver
- Loop 2000A: Billing Provider
- Loop 2000B: Subscriber
- Loop 2300: Claim Information
- Loop 2400: Service Lines
- All segments properly mapped

### Healthcare Standards

- ICD-10-CM diagnosis codes (up to 12 per encounter)
- CPT/HCPCS procedure codes
- NPI for provider identification
- Place of Service codes
- Medicare Beneficiary Identifier (MBI)
- CARC/RARC denial reason codes

## Maintenance

### Refreshing Materialized Views

Materialized views should be refreshed regularly:

```sql
-- Refresh flag statistics (daily)
REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_flag_statistics;

-- Refresh denial statistics (daily)
REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_denial_statistics;
```

Consider setting up automated refresh using scheduled jobs in the staging.scheduled_job table.

### Updating Reference Data

Some tables contain reference data that needs periodic updates:

- **claims.rvu_reference**: Update annually with new CMS PFS data
- **claims.conversion_factor**: Update annually (typically January)
- **claims.gpci_reference**: Update annually with new locality data
- **claims.denial_reason_code**: Update as CMS publishes new CARC/RARC codes
- **claims.modifier_adjustment**: Update as payment policies change

### Index Maintenance

Regularly analyze and reindex tables:

```sql
-- Analyze all tables in claims schema
ANALYZE claims.encounter;
ANALYZE claims.service_line;
-- etc.

-- Reindex if needed
REINDEX TABLE CONCURRENTLY claims.encounter;
```

## Rollback

If you need to rollback, drop schemas in reverse order:

```sql
-- WARNING: This will delete all data!
DROP SCHEMA ml CASCADE;
DROP SCHEMA claims CASCADE;
DROP SCHEMA staging CASCADE;
```

## Support

For questions or issues with migrations, refer to:
- Software Requirements Document (docs/srd.md)
- 837p Companion Guide (docs/837p_compguide.pdf)
- Project rules (docs/CLAUDE.md)
