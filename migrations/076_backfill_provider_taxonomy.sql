-- Migration: 076_backfill_provider_taxonomy
-- Description: One-shot backfill for claims.provider rows whose taxonomy_code
--              and/or specialty are NULL but for which a valid taxonomy code
--              IS available on a referencing claims.encounter or
--              claims.service_line row.
--
-- Background: prior versions through 2.14.3.0 had three bugs in
-- claims_processor.rs::upsert_providers_in_own_tx that prevented provider
-- master rows from being updated once they existed:
--   A. The cache filter skipped already-cached NPIs before the upsert ran.
--   B. ON CONFLICT only updated updated_at, so even if A had let a row
--      through, the new taxonomy/specialty was discarded.
--   C. Within-batch entry().or_insert() kept the FIRST sample (which might
--      have had no taxonomy) and dropped later, better samples.
--
-- The code in 2.15.0.0 (Bugs A/B/C fix) repairs FORWARD - providers seen
-- from that point on. This migration repairs BACKWARD: existing provider
-- rows with NULL taxonomy that have at least one claim row referencing them
-- with a real taxonomy code.
--
-- Properties:
--   * Idempotent - only updates rows where taxonomy_code IS NULL OR specialty IS NULL.
--   * Safe - joins against claims.provider_taxonomy so only NUCC-valid codes propagate.
--   * Non-destructive - does NOT touch already-set values (preserves NPI-
--     enrichment results from the NPPI registry).
--   * Reports row count via RAISE NOTICE for visibility.

DO $$
DECLARE
    v_updated INTEGER;
BEGIN
    -- Pick a taxonomy code per provider, preferring (in order):
    --   1. Most recent encounter.rendering_provider_taxonomy
    --   2. Most recent encounter.referring_provider_taxonomy
    --   3. Most recent encounter.supervising_provider_taxonomy
    --   4. Most recent service_line.rendering_provider_taxonomy
    -- The DISTINCT ON ... ORDER BY created_at DESC picks the latest sample.
    WITH candidate_per_provider AS (
        SELECT DISTINCT ON (provider_id)
            provider_id,
            taxonomy_code
        FROM (
            SELECT
                rendering_provider_id   AS provider_id,
                rendering_provider_taxonomy AS taxonomy_code,
                created_at
            FROM claims.encounter
            WHERE rendering_provider_id IS NOT NULL
              AND rendering_provider_taxonomy IS NOT NULL
              AND rendering_provider_taxonomy <> ''
            UNION ALL
            SELECT
                referring_provider_id,
                referring_provider_taxonomy,
                created_at
            FROM claims.encounter
            WHERE referring_provider_id IS NOT NULL
              AND referring_provider_taxonomy IS NOT NULL
              AND referring_provider_taxonomy <> ''
            UNION ALL
            SELECT
                supervising_provider_id,
                supervising_provider_taxonomy,
                created_at
            FROM claims.encounter
            WHERE supervising_provider_id IS NOT NULL
              AND supervising_provider_taxonomy IS NOT NULL
              AND supervising_provider_taxonomy <> ''
            UNION ALL
            SELECT
                rendering_provider_id,
                rendering_provider_taxonomy,
                created_at
            FROM claims.service_line
            WHERE rendering_provider_id IS NOT NULL
              AND rendering_provider_taxonomy IS NOT NULL
              AND rendering_provider_taxonomy <> ''
        ) all_refs
        ORDER BY provider_id, created_at DESC
    )
    UPDATE claims.provider p
    SET taxonomy_code = COALESCE(p.taxonomy_code, cpp.taxonomy_code),
        specialty     = COALESCE(p.specialty,     pt.specialty_display),
        updated_at    = CURRENT_TIMESTAMP
    FROM candidate_per_provider cpp
    JOIN claims.provider_taxonomy pt ON pt.taxonomy_code = cpp.taxonomy_code
    WHERE p.provider_id = cpp.provider_id
      AND (p.taxonomy_code IS NULL OR p.specialty IS NULL);

    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RAISE NOTICE 'Provider taxonomy/specialty backfill: filled NULL fields on % provider row(s)', v_updated;

    -- Also queue any rows we just touched (or that still have NULLs because
    -- no claim mentioned a valid taxonomy for them) for NPI enrichment, so
    -- the NPPI registry can fill the gap. ON CONFLICT preserves any
    -- previously-queued entry.
    INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority)
    SELECT p.provider_id, p.npi, 5
    FROM claims.provider p
    WHERE p.taxonomy_code IS NULL OR p.specialty IS NULL
    ON CONFLICT (provider_id) DO NOTHING;

    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RAISE NOTICE 'Provider enrichment queue: enqueued % provider row(s) still missing taxonomy/specialty', v_updated;
END $$;
