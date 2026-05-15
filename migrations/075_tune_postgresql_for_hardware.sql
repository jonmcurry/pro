-- Migration: 075_tune_postgresql_for_hardware
-- Description: Tune PostgreSQL for the target deployment profile -
--              8 vCPU server, 64GB RAM, Postgres co-located with the
--              data processing service.
--
-- Strategy: ALTER SYSTEM SET writes to postgresql.auto.conf, which overrides
--           postgresql.conf without touching the base file. Reload-capable
--           settings take effect after pg_reload_conf(); restart-required
--           settings (shared_buffers, max_connections, max_worker_processes,
--           wal_buffers) take effect on the next PostgreSQL restart.
--
-- Sizing rationale (8 vCPU / 64GB):
--   * shared_buffers = 16GB    -> 25% of RAM, PG community sweet spot
--   * effective_cache_size = 40GB -> planner hint for OS page cache size
--   * work_mem = 64MB          -> already set in migration 066; preserved
--   * maintenance_work_mem = 1GB -> for VACUUM / CREATE INDEX bursts
--   * max_connections = 50     -> app pool (24) + web app + admin + headroom
--   * max_worker_processes = 8 -> match vCPU count
--   * max_parallel_workers = 4 -> cap parallel workers so analytics queries
--                                  don't starve the ingestion pipeline
--   * max_parallel_workers_per_gather = 2 -> reasonable per-query cap
--   * wal_buffers = 16MB       -> matches typical auto-sized value
--   * random_page_cost = 1.1   -> SSD assumption; raise to 4 on spinning disk
--   * checkpoint_completion_target = 0.9 -> smooth checkpoint I/O
--
-- Override per-machine: re-run ALTER SYSTEM SET with site-specific values and
-- pg_reload_conf() (or restart, for the restart-required settings).

-- Memory
ALTER SYSTEM SET shared_buffers = '16GB';
ALTER SYSTEM SET effective_cache_size = '40GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Connections (must align with DB_MAX_CONNECTIONS in the app env)
ALTER SYSTEM SET max_connections = '50';

-- Parallel query
ALTER SYSTEM SET max_worker_processes = '8';
ALTER SYSTEM SET max_parallel_workers = '4';
ALTER SYSTEM SET max_parallel_workers_per_gather = '2';

-- WAL / checkpoint
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';

-- Planner cost (SSD)
ALTER SYSTEM SET random_page_cost = '1.1';

-- Apply reload-capable settings immediately. Settings that require a restart
-- are persisted in postgresql.auto.conf and take effect on next PG restart.
SELECT pg_reload_conf();

DO $$
DECLARE
    v_shared_buffers text;
    v_effective_cache text;
    v_max_conn text;
    v_work_mem text;
    v_pending_restart bool;
BEGIN
    SELECT current_setting('shared_buffers') INTO v_shared_buffers;
    SELECT current_setting('effective_cache_size') INTO v_effective_cache;
    SELECT current_setting('max_connections') INTO v_max_conn;
    SELECT current_setting('work_mem') INTO v_work_mem;

    -- Warn if any setting we just wrote needs a restart to take effect.
    SELECT EXISTS (
        SELECT 1
        FROM pg_settings
        WHERE pending_restart = true
          AND name IN ('shared_buffers', 'max_connections', 'max_worker_processes', 'wal_buffers')
    ) INTO v_pending_restart;

    RAISE NOTICE 'PostgreSQL tuning applied:';
    RAISE NOTICE '  shared_buffers      = % (effective on next restart if pending)', v_shared_buffers;
    RAISE NOTICE '  effective_cache_size = %', v_effective_cache;
    RAISE NOTICE '  max_connections     = % (effective on next restart if pending)', v_max_conn;
    RAISE NOTICE '  work_mem            = %', v_work_mem;

    IF v_pending_restart THEN
        RAISE NOTICE '';
        RAISE NOTICE 'NOTE: Some settings require a PostgreSQL restart to take effect.';
        RAISE NOTICE '      Restart the postgresql-x64-* Windows service when convenient.';
    END IF;
END $$;
