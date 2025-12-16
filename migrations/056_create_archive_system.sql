-- Migration: 056_create_archive_system
-- Description: Create archive schema and procedures for manual data archiving
-- Date: 2025-11-26
-- Purpose: Prevent performance cliff by allowing manual archiving of old data

-- ==============================================================================
-- ARCHIVE SCHEMA
-- ==============================================================================

CREATE SCHEMA IF NOT EXISTS archive;

COMMENT ON SCHEMA archive IS 'Archive schema for storing old/historical data that has been removed from active tables';

-- ==============================================================================
-- ARCHIVE TABLES - Mirror structure of main tables
-- ==============================================================================

-- Archive table for encounters (using DO block for idempotency since LIKE doesn't support IF NOT EXISTS)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'encounter') THEN
        CREATE TABLE archive.encounter (LIKE claims.encounter INCLUDING ALL);
        ALTER TABLE archive.encounter ALTER COLUMN encounter_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.encounter IS 'Archived encounters removed from active claims.encounter table';

-- Archive table for service lines
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'service_line') THEN
        CREATE TABLE archive.service_line (LIKE claims.service_line INCLUDING ALL);
        ALTER TABLE archive.service_line ALTER COLUMN service_line_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.service_line IS 'Archived service lines removed from active claims.service_line table';

-- Archive table for encounter diagnoses
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'encounter_diagnosis') THEN
        CREATE TABLE archive.encounter_diagnosis (LIKE claims.encounter_diagnosis INCLUDING ALL);
        ALTER TABLE archive.encounter_diagnosis ALTER COLUMN diagnosis_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.encounter_diagnosis IS 'Archived diagnoses removed from active claims.encounter_diagnosis table';

-- Archive table for encounter flags
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'encounter_flag') THEN
        CREATE TABLE archive.encounter_flag (LIKE claims.encounter_flag INCLUDING ALL);
        ALTER TABLE archive.encounter_flag ALTER COLUMN flag_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.encounter_flag IS 'Archived encounter flags removed from active claims.encounter_flag table';

-- Archive table for service line flags
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'service_line_flag') THEN
        CREATE TABLE archive.service_line_flag (LIKE claims.service_line_flag INCLUDING ALL);
        ALTER TABLE archive.service_line_flag ALTER COLUMN flag_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.service_line_flag IS 'Archived service line flags removed from active claims.service_line_flag table';

-- Archive table for import batches
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'import_batch') THEN
        CREATE TABLE archive.import_batch (LIKE staging.import_batch INCLUDING ALL);
        ALTER TABLE archive.import_batch ALTER COLUMN batch_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.import_batch IS 'Archived import batches removed from active staging.import_batch table';

-- Archive table for raw claims
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'archive' AND table_name = 'raw_claims') THEN
        CREATE TABLE archive.raw_claims (LIKE staging.raw_claims INCLUDING ALL);
        ALTER TABLE archive.raw_claims ALTER COLUMN raw_claim_id DROP IDENTITY IF EXISTS;
    END IF;
END $$;

COMMENT ON TABLE archive.raw_claims IS 'Archived raw claims removed from active staging.raw_claims table';

-- ==============================================================================
-- ARCHIVE METADATA TABLE
-- ==============================================================================

CREATE TABLE IF NOT EXISTS archive.archive_log (
    archive_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    archive_type VARCHAR(50) NOT NULL, -- ENCOUNTER, IMPORT_BATCH
    archived_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    archived_by VARCHAR(100),
    date_range_start DATE,
    date_range_end DATE,
    organization_id BIGINT,
    facility_id BIGINT,
    records_archived INTEGER NOT NULL,
    child_records_archived INTEGER DEFAULT 0,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_archive_log_type ON archive.archive_log(archive_type);
CREATE INDEX IF NOT EXISTS idx_archive_log_date ON archive.archive_log(archived_at);

COMMENT ON TABLE archive.archive_log IS 'Audit log of all archive operations';

-- ==============================================================================
-- ARCHIVE ENCOUNTERS PROCEDURE
-- ==============================================================================

CREATE OR REPLACE FUNCTION archive.archive_encounters(
    p_cutoff_date DATE,
    p_organization_id BIGINT DEFAULT NULL,
    p_facility_id BIGINT DEFAULT NULL,
    p_archived_by VARCHAR(100) DEFAULT 'SYSTEM'
)
RETURNS TABLE (
    encounters_archived INTEGER,
    service_lines_archived INTEGER,
    diagnoses_archived INTEGER,
    encounter_flags_archived INTEGER,
    service_line_flags_archived INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_encounters_archived INTEGER := 0;
    v_service_lines_archived INTEGER := 0;
    v_diagnoses_archived INTEGER := 0;
    v_encounter_flags_archived INTEGER := 0;
    v_service_line_flags_archived INTEGER := 0;
    v_encounter_ids BIGINT[];
    v_service_line_ids BIGINT[];
BEGIN
    -- Validate cutoff date is not too recent (must be at least 90 days old)
    IF p_cutoff_date > CURRENT_DATE - INTERVAL '90 days' THEN
        RAISE EXCEPTION 'Cutoff date must be at least 90 days in the past. Provided: %, Minimum allowed: %',
            p_cutoff_date, CURRENT_DATE - INTERVAL '90 days';
    END IF;

    -- Get encounter IDs to archive
    SELECT ARRAY_AGG(encounter_id) INTO v_encounter_ids
    FROM claims.encounter
    WHERE date_of_service_from < p_cutoff_date
      AND (p_organization_id IS NULL OR organization_id = p_organization_id)
      AND (p_facility_id IS NULL OR facility_id = p_facility_id)
      AND claim_status NOT IN ('PENDING', 'PROCESSING'); -- Don't archive active claims

    IF v_encounter_ids IS NULL OR array_length(v_encounter_ids, 1) IS NULL THEN
        RAISE NOTICE 'No encounters found to archive for cutoff date %', p_cutoff_date;
        RETURN QUERY SELECT 0, 0, 0, 0, 0;
        RETURN;
    END IF;

    -- Get service line IDs for these encounters
    SELECT ARRAY_AGG(service_line_id) INTO v_service_line_ids
    FROM claims.service_line
    WHERE encounter_id = ANY(v_encounter_ids);

    -- 1. Archive service line flags first (deepest child)
    IF v_service_line_ids IS NOT NULL AND array_length(v_service_line_ids, 1) > 0 THEN
        WITH moved AS (
            DELETE FROM claims.service_line_flag
            WHERE service_line_id = ANY(v_service_line_ids)
            RETURNING *
        )
        INSERT INTO archive.service_line_flag SELECT * FROM moved;

        GET DIAGNOSTICS v_service_line_flags_archived = ROW_COUNT;
    END IF;

    -- 2. Archive encounter flags
    WITH moved AS (
        DELETE FROM claims.encounter_flag
        WHERE encounter_id = ANY(v_encounter_ids)
        RETURNING *
    )
    INSERT INTO archive.encounter_flag SELECT * FROM moved;

    GET DIAGNOSTICS v_encounter_flags_archived = ROW_COUNT;

    -- 3. Archive encounter diagnoses
    WITH moved AS (
        DELETE FROM claims.encounter_diagnosis
        WHERE encounter_id = ANY(v_encounter_ids)
        RETURNING *
    )
    INSERT INTO archive.encounter_diagnosis SELECT * FROM moved;

    GET DIAGNOSTICS v_diagnoses_archived = ROW_COUNT;

    -- 4. Archive service lines
    IF v_service_line_ids IS NOT NULL AND array_length(v_service_line_ids, 1) > 0 THEN
        WITH moved AS (
            DELETE FROM claims.service_line
            WHERE encounter_id = ANY(v_encounter_ids)
            RETURNING *
        )
        INSERT INTO archive.service_line SELECT * FROM moved;

        GET DIAGNOSTICS v_service_lines_archived = ROW_COUNT;
    END IF;

    -- 5. Archive encounters
    WITH moved AS (
        DELETE FROM claims.encounter
        WHERE encounter_id = ANY(v_encounter_ids)
        RETURNING *
    )
    INSERT INTO archive.encounter SELECT * FROM moved;

    GET DIAGNOSTICS v_encounters_archived = ROW_COUNT;

    -- Log the archive operation
    INSERT INTO archive.archive_log (
        archive_type,
        archived_by,
        date_range_start,
        date_range_end,
        organization_id,
        facility_id,
        records_archived,
        child_records_archived,
        notes
    ) VALUES (
        'ENCOUNTER',
        p_archived_by,
        (SELECT MIN(date_of_service_from) FROM archive.encounter WHERE encounter_id = ANY(v_encounter_ids)),
        p_cutoff_date,
        p_organization_id,
        p_facility_id,
        v_encounters_archived,
        v_service_lines_archived + v_diagnoses_archived + v_encounter_flags_archived + v_service_line_flags_archived,
        format('Archived %s encounters with %s service lines, %s diagnoses, %s encounter flags, %s service line flags',
            v_encounters_archived, v_service_lines_archived, v_diagnoses_archived,
            v_encounter_flags_archived, v_service_line_flags_archived)
    );

    RETURN QUERY SELECT
        v_encounters_archived,
        v_service_lines_archived,
        v_diagnoses_archived,
        v_encounter_flags_archived,
        v_service_line_flags_archived;
END;
$$;

COMMENT ON FUNCTION archive.archive_encounters IS 'Archives encounters and all related data (service lines, diagnoses, flags) older than the cutoff date';

-- ==============================================================================
-- ARCHIVE IMPORT BATCHES PROCEDURE
-- ==============================================================================

CREATE OR REPLACE FUNCTION archive.archive_import_batches(
    p_cutoff_date DATE,
    p_organization_id BIGINT DEFAULT NULL,
    p_archived_by VARCHAR(100) DEFAULT 'SYSTEM'
)
RETURNS TABLE (
    batches_archived INTEGER,
    raw_claims_archived INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_batches_archived INTEGER := 0;
    v_raw_claims_archived INTEGER := 0;
    v_batch_ids BIGINT[];
BEGIN
    -- Validate cutoff date
    IF p_cutoff_date > CURRENT_DATE - INTERVAL '90 days' THEN
        RAISE EXCEPTION 'Cutoff date must be at least 90 days in the past';
    END IF;

    -- Get batch IDs to archive (only completed batches)
    SELECT ARRAY_AGG(batch_id) INTO v_batch_ids
    FROM staging.import_batch
    WHERE created_at < p_cutoff_date
      AND (p_organization_id IS NULL OR organization_id = p_organization_id)
      AND import_status IN ('COMPLETED', 'FAILED', 'PARTIAL'); -- Don't archive active batches

    IF v_batch_ids IS NULL OR array_length(v_batch_ids, 1) IS NULL THEN
        RAISE NOTICE 'No import batches found to archive for cutoff date %', p_cutoff_date;
        RETURN QUERY SELECT 0, 0;
        RETURN;
    END IF;

    -- 1. Archive raw claims first
    WITH moved AS (
        DELETE FROM staging.raw_claims
        WHERE batch_id = ANY(v_batch_ids)
        RETURNING *
    )
    INSERT INTO archive.raw_claims SELECT * FROM moved;

    GET DIAGNOSTICS v_raw_claims_archived = ROW_COUNT;

    -- 2. Archive import batches
    WITH moved AS (
        DELETE FROM staging.import_batch
        WHERE batch_id = ANY(v_batch_ids)
        RETURNING *
    )
    INSERT INTO archive.import_batch SELECT * FROM moved;

    GET DIAGNOSTICS v_batches_archived = ROW_COUNT;

    -- Log the archive operation
    INSERT INTO archive.archive_log (
        archive_type,
        archived_by,
        date_range_start,
        date_range_end,
        organization_id,
        records_archived,
        child_records_archived,
        notes
    ) VALUES (
        'IMPORT_BATCH',
        p_archived_by,
        (SELECT MIN(created_at)::DATE FROM archive.import_batch WHERE batch_id = ANY(v_batch_ids)),
        p_cutoff_date,
        p_organization_id,
        v_batches_archived,
        v_raw_claims_archived,
        format('Archived %s import batches with %s raw claims', v_batches_archived, v_raw_claims_archived)
    );

    RETURN QUERY SELECT v_batches_archived, v_raw_claims_archived;
END;
$$;

COMMENT ON FUNCTION archive.archive_import_batches IS 'Archives import batches and raw claims older than the cutoff date';

-- ==============================================================================
-- RESTORE ENCOUNTERS PROCEDURE (for unarchiving if needed)
-- ==============================================================================

CREATE OR REPLACE FUNCTION archive.restore_encounters(
    p_encounter_ids BIGINT[],
    p_restored_by VARCHAR(100) DEFAULT 'SYSTEM'
)
RETURNS TABLE (
    encounters_restored INTEGER,
    service_lines_restored INTEGER,
    diagnoses_restored INTEGER,
    encounter_flags_restored INTEGER,
    service_line_flags_restored INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_encounters_restored INTEGER := 0;
    v_service_lines_restored INTEGER := 0;
    v_diagnoses_restored INTEGER := 0;
    v_encounter_flags_restored INTEGER := 0;
    v_service_line_flags_restored INTEGER := 0;
    v_service_line_ids BIGINT[];
BEGIN
    -- Verify encounters exist in archive
    IF NOT EXISTS (SELECT 1 FROM archive.encounter WHERE encounter_id = ANY(p_encounter_ids)) THEN
        RAISE EXCEPTION 'No archived encounters found with the provided IDs';
    END IF;

    -- Get service line IDs from archive
    SELECT ARRAY_AGG(service_line_id) INTO v_service_line_ids
    FROM archive.service_line
    WHERE encounter_id = ANY(p_encounter_ids);

    -- 1. Restore encounters first (parent)
    WITH moved AS (
        DELETE FROM archive.encounter
        WHERE encounter_id = ANY(p_encounter_ids)
        RETURNING *
    )
    INSERT INTO claims.encounter SELECT * FROM moved;

    GET DIAGNOSTICS v_encounters_restored = ROW_COUNT;

    -- 2. Restore service lines
    IF v_service_line_ids IS NOT NULL AND array_length(v_service_line_ids, 1) > 0 THEN
        WITH moved AS (
            DELETE FROM archive.service_line
            WHERE encounter_id = ANY(p_encounter_ids)
            RETURNING *
        )
        INSERT INTO claims.service_line SELECT * FROM moved;

        GET DIAGNOSTICS v_service_lines_restored = ROW_COUNT;
    END IF;

    -- 3. Restore diagnoses
    WITH moved AS (
        DELETE FROM archive.encounter_diagnosis
        WHERE encounter_id = ANY(p_encounter_ids)
        RETURNING *
    )
    INSERT INTO claims.encounter_diagnosis SELECT * FROM moved;

    GET DIAGNOSTICS v_diagnoses_restored = ROW_COUNT;

    -- 4. Restore encounter flags
    WITH moved AS (
        DELETE FROM archive.encounter_flag
        WHERE encounter_id = ANY(p_encounter_ids)
        RETURNING *
    )
    INSERT INTO claims.encounter_flag SELECT * FROM moved;

    GET DIAGNOSTICS v_encounter_flags_restored = ROW_COUNT;

    -- 5. Restore service line flags
    IF v_service_line_ids IS NOT NULL AND array_length(v_service_line_ids, 1) > 0 THEN
        WITH moved AS (
            DELETE FROM archive.service_line_flag
            WHERE service_line_id = ANY(v_service_line_ids)
            RETURNING *
        )
        INSERT INTO claims.service_line_flag SELECT * FROM moved;

        GET DIAGNOSTICS v_service_line_flags_restored = ROW_COUNT;
    END IF;

    -- Log the restore operation
    INSERT INTO archive.archive_log (
        archive_type,
        archived_by,
        records_archived,
        child_records_archived,
        notes
    ) VALUES (
        'RESTORE_ENCOUNTER',
        p_restored_by,
        v_encounters_restored,
        v_service_lines_restored + v_diagnoses_restored + v_encounter_flags_restored + v_service_line_flags_restored,
        format('Restored %s encounters with %s service lines, %s diagnoses, %s encounter flags, %s service line flags',
            v_encounters_restored, v_service_lines_restored, v_diagnoses_restored,
            v_encounter_flags_restored, v_service_line_flags_restored)
    );

    RETURN QUERY SELECT
        v_encounters_restored,
        v_service_lines_restored,
        v_diagnoses_restored,
        v_encounter_flags_restored,
        v_service_line_flags_restored;
END;
$$;

COMMENT ON FUNCTION archive.restore_encounters IS 'Restores archived encounters and all related data back to active tables';

-- ==============================================================================
-- ARCHIVE STATISTICS VIEW
-- ==============================================================================

CREATE OR REPLACE VIEW archive.v_archive_statistics AS
SELECT
    'encounters' AS table_name,
    COUNT(*) AS archived_records,
    MIN(date_of_service_from) AS oldest_record,
    MAX(date_of_service_from) AS newest_record
FROM archive.encounter
UNION ALL
SELECT
    'service_lines',
    COUNT(*),
    MIN(service_date_from),
    MAX(service_date_from)
FROM archive.service_line
UNION ALL
SELECT
    'encounter_diagnoses',
    COUNT(*),
    NULL,
    NULL
FROM archive.encounter_diagnosis
UNION ALL
SELECT
    'encounter_flags',
    COUNT(*),
    MIN(created_at)::DATE,
    MAX(created_at)::DATE
FROM archive.encounter_flag
UNION ALL
SELECT
    'service_line_flags',
    COUNT(*),
    MIN(created_at)::DATE,
    MAX(created_at)::DATE
FROM archive.service_line_flag
UNION ALL
SELECT
    'import_batches',
    COUNT(*),
    MIN(created_at)::DATE,
    MAX(created_at)::DATE
FROM archive.import_batch
UNION ALL
SELECT
    'raw_claims',
    COUNT(*),
    MIN(ingested_at)::DATE,
    MAX(ingested_at)::DATE
FROM archive.raw_claims;

COMMENT ON VIEW archive.v_archive_statistics IS 'Summary statistics of archived data';

-- ==============================================================================
-- USAGE EXAMPLES
-- ==============================================================================
-- Archive all encounters older than 1 year:
--   SELECT * FROM archive.archive_encounters('2024-11-26'::DATE);
--
-- Archive encounters for a specific organization:
--   SELECT * FROM archive.archive_encounters('2024-11-26'::DATE, 1, NULL, 'admin_user');
--
-- Archive import batches older than 6 months:
--   SELECT * FROM archive.archive_import_batches('2025-05-26'::DATE);
--
-- View archive statistics:
--   SELECT * FROM archive.v_archive_statistics;
--
-- View archive history:
--   SELECT * FROM archive.archive_log ORDER BY archived_at DESC;
--
-- Restore specific encounters:
--   SELECT * FROM archive.restore_encounters(ARRAY[123, 456, 789]);
