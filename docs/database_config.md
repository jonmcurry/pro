Database Configuration Guide
PostgreSQL Setup
1. Installation
bash# Install PostgreSQL (example for Ubuntu)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
2. Database Creation
sql-- Run as postgres superuser
sudo -u postgres psql

-- Create database and user
CREATE DATABASE claims_processing;
CREATE USER edi_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE claims_processing TO edi_user;

-- Connect to the database
\c claims_processing;

-- Run the schema creation script
\i sql/postgresql_create_edi_databases.sql;
3. Performance Tuning
sql-- Recommended settings for claims processing
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();
SQL Server Setup
1. Installation
Install SQL Server Express or higher edition suitable for your environment.
2. Database Creation
sql-- Create database
CREATE DATABASE validation_results;

-- Use the database
USE validation_results;

-- Run the schema creation script
-- Copy and paste contents from sql/sqlserver_create_results_database.sql
3. Performance Optimization
sql-- Set recovery model to SIMPLE for better performance (non-production)
ALTER DATABASE validation_results SET RECOVERY SIMPLE;

-- Configure tempdb for better performance
ALTER DATABASE tempdb MODIFY FILE (NAME = tempdev, SIZE = 1GB, FILEGROWTH = 256MB);
ALTER DATABASE tempdb MODIFY FILE (NAME = templog, SIZE = 256MB, FILEGROWTH = 64MB);
Connection Configuration
PostgreSQL Connection String
yamldatabase:
  postgresql:
    host: localhost
    port: 5432
    database: claims_processing
    user: edi_user
    password_encrypted: "your_encrypted_password"
    min_connections: 2
    max_connections: 10
SQL Server Connection String
yamldatabase:
  sqlserver:
    connection_string: "mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
    pool_size: 10