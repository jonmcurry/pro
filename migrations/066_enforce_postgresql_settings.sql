-- Migration: 066_enforce_postgresql_settings
-- Description: Enforce critical PostgreSQL settings to prevent performance issues
--
-- Fixes:
--   1. Ensure autovacuum is enabled (was disabled causing 717k dead tuples, 71x table bloat)
--   2. Set work_mem to 64MB (512MB with 300 connections could exhaust 150GB+ RAM)

-- Enable autovacuum (critical for preventing table bloat)
ALTER SYSTEM SET autovacuum = 'on';

-- Set reasonable work_mem (64MB is safe for concurrent connections)
ALTER SYSTEM SET work_mem = '64MB';

-- Reload configuration to apply changes immediately
SELECT pg_reload_conf();

-- Verify settings were applied
DO $$
DECLARE
    v_autovacuum text;
    v_work_mem text;
BEGIN
    SELECT current_setting('autovacuum') INTO v_autovacuum;
    SELECT current_setting('work_mem') INTO v_work_mem;

    RAISE NOTICE 'PostgreSQL settings verified:';
    RAISE NOTICE '  autovacuum = %', v_autovacuum;
    RAISE NOTICE '  work_mem = %', v_work_mem;
END $$;
