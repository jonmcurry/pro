-- Migration: 001_create_schemas
-- Description: Create the three main schemas for the application
-- Date: 2025-10-14

-- Create schemas
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS claims;
CREATE SCHEMA IF NOT EXISTS ml;

-- Add comments
COMMENT ON SCHEMA staging IS 'Staging area for processing metrics, file tracking, and configuration';
COMMENT ON SCHEMA claims IS 'Main claims processing schema with encounter, provider, and audit data';
COMMENT ON SCHEMA ml IS 'Machine learning schema for predictive analytics';

-- Enable required extensions

CREATE EXTENSION IF NOT EXISTS "citext";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set default search path
-- Note: Database name will be set during installation
-- ALTER DATABASE professional_smart SET search_path TO claims, staging, ml, public;
