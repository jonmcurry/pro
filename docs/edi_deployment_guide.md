# EDI Claims Processing System - Complete Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Database Configuration](#database-configuration)
5. [Application Deployment](#application-deployment)
6. [Configuration Management](#configuration-management)
7. [Monitoring Setup](#monitoring-setup)
8. [Security Configuration](#security-configuration)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance and Operations](#maintenance-and-operations)

---

## System Overview

The EDI Claims Processing System is a high-performance healthcare claims validation platform that combines ML-based predictions with rule-based validation. The system processes large volumes of EDI claims using hybrid validation approaches.

### Architecture Components
- **Parser**: Optimized claim parsing with batch processing
- **Validator**: Hybrid ML + pyDatalog rule validation
- **Storage**: High-performance result storage with bulk operations
- **Monitoring**: Prometheus metrics and email notifications
- **ML Training**: XGBoost-based filter prediction

---

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.4GHz
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1Gbps

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0GHz
- **RAM**: 32GB+
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps

### Software Requirements

#### Operating System
- **Windows Server 2019/2022** (Primary)
- **Linux**: Ubuntu 20.04+ or RHEL 8+ (Alternative)

#### Python Environment
```bash
# Python 3.8 or higher
python --version  # Should be 3.8+
pip --version
```

#### Database Systems
- **PostgreSQL 13+** (Source data)
- **SQL Server 2019+** (Results storage)

#### Additional Software
- **Git** (for deployment)
- **Docker** (optional, for containerized deployment)
- **Prometheus** (monitoring)
- **Grafana** (dashboards)

---

## Infrastructure Setup

### 1. Server Provisioning

#### Production Environment
```yaml
# Infrastructure specification
Servers:
  - Application Server:
      CPU: 8 cores
      RAM: 32GB
      Storage: 500GB SSD
      OS: Windows Server 2022
  
  - Database Servers:
      PostgreSQL:
        CPU: 4 cores
        RAM: 16GB
        Storage: 1TB SSD
      SQL Server:
        CPU: 4 cores
        RAM: 16GB
        Storage: 1TB SSD
  
  - Monitoring Server:
      CPU: 2 cores
      RAM: 8GB
      Storage: 100GB SSD
```

### 2. Network Configuration

#### Firewall Rules
```bash
# Application Server
Port 8000: Prometheus metrics (internal)
Port 5432: PostgreSQL access
Port 1433: SQL Server access
Port 587: SMTP (outbound)

# Database Servers
PostgreSQL: Port 5432 (from app server only)
SQL Server: Port 1433 (from app server only)

# Monitoring
Prometheus: Port 9090 (internal)
Grafana: Port 3000 (admin access)
```

### 3. Security Groups (Cloud Deployment)
```yaml
# Example AWS Security Groups
ApplicationSecurityGroup:
  Inbound:
    - Protocol: TCP, Port: 22, Source: AdminIP
    - Protocol: TCP, Port: 8000, Source: MonitoringSecurityGroup
  Outbound:
    - All traffic allowed

DatabaseSecurityGroup:
  Inbound:
    - Protocol: TCP, Port: 5432, Source: ApplicationSecurityGroup
    - Protocol: TCP, Port: 1433, Source: ApplicationSecurityGroup
```

---

## Database Configuration

### 1. PostgreSQL Setup (Source Database)

#### Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-13 postgresql-contrib-13

# Windows
# Download and install PostgreSQL from official website
```

#### Configuration
```sql
-- Create database and user
CREATE DATABASE claims_processing;
CREATE USER edi_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE claims_processing TO edi_user;

-- Connect to claims_processing database
\c claims_processing

-- Create schema
CREATE SCHEMA IF NOT EXISTS edi;

-- Create tables
CREATE TABLE edi.claims (
    claim_id VARCHAR(50) PRIMARY KEY,
    patient_age INTEGER,
    provider_type VARCHAR(100),
    place_of_service VARCHAR(20),
    total_charge_amount DECIMAL(10,2),
    claim_data JSONB,
    diagnoses JSONB DEFAULT '[]',
    procedures JSONB DEFAULT '[]',
    processing_status VARCHAR(20) DEFAULT 'PENDING',
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE edi.processed_chunks (
    chunk_id INTEGER PRIMARY KEY,
    status VARCHAR(20),
    processed_date TIMESTAMP,
    claims_count INTEGER DEFAULT 0,
    processing_duration_seconds DECIMAL(10,3) DEFAULT 0
);

CREATE TABLE edi.retry_queue (
    chunk_id INTEGER PRIMARY KEY,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_retry_date TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE edi.filters (
    filter_id SERIAL PRIMARY KEY,
    filter_name VARCHAR(200) NOT NULL,
    rule_definition TEXT NOT NULL,
    description TEXT,
    rule_type VARCHAR(50) DEFAULT 'DATALOG',
    support DECIMAL(5,4),
    confidence DECIMAL(5,4),
    lift DECIMAL(5,4),
    active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_claims_processing_status ON edi.claims(processing_status);
CREATE INDEX idx_claims_created_date ON edi.claims(created_date);
CREATE INDEX idx_processed_chunks_status ON edi.processed_chunks(status);
CREATE INDEX idx_filters_active ON edi.filters(active);
```

#### Performance Tuning
```sql
-- postgresql.conf optimizations
-- Add these to postgresql.conf and restart PostgreSQL

# Memory settings
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
work_mem = 256MB
maintenance_work_mem = 2GB

# Connection settings
max_connections = 100
max_worker_processes = 8
max_parallel_workers = 8
max_parallel_workers_per_gather = 4

# Checkpoint settings
checkpoint_timeout = 10min
checkpoint_completion_target = 0.9
wal_buffers = 64MB

# Query planner
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage
```

### 2. SQL Server Setup (Results Database)

#### Installation
```sql
-- SQL Server installation and configuration
-- Create database
CREATE DATABASE ClaimsValidationResults;
GO

USE ClaimsValidationResults;
GO

-- Create login and user
CREATE LOGIN edi_results_user WITH PASSWORD = 'secure_password_here';
CREATE USER edi_results_user FOR LOGIN edi_results_user;
ALTER ROLE db_datareader ADD MEMBER edi_results_user;
ALTER ROLE db_datawriter ADD MEMBER edi_results_user;
ALTER ROLE db_ddladmin ADD MEMBER edi_results_user;
GO

-- Create tables
CREATE TABLE dbo.validation_results (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    claim_id VARCHAR(50) NOT NULL,
    filter_id INTEGER,
    filter_name VARCHAR(200),
    validation_status VARCHAR(20),
    passed BIT,
    rule_type VARCHAR(50),
    details NVARCHAR(MAX),
    error_message NVARCHAR(MAX),
    processing_time DECIMAL(8,4),
    created_date DATETIME2 DEFAULT GETUTCDATE()
);

CREATE TABLE dbo.processing_summary (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    processing_date DATE,
    total_claims INTEGER,
    successful_validations INTEGER,
    failed_validations INTEGER,
    error_count INTEGER,
    avg_processing_time DECIMAL(8,4),
    created_date DATETIME2 DEFAULT GETUTCDATE()
);

-- Create indexes
CREATE NONCLUSTERED INDEX IX_validation_results_claim_id 
ON dbo.validation_results(claim_id);

CREATE NONCLUSTERED INDEX IX_validation_results_created_date 
ON dbo.validation_results(created_date);

CREATE NONCLUSTERED INDEX IX_validation_results_filter_id 
ON dbo.validation_results(filter_id);
```

#### Performance Configuration
```sql
-- SQL Server performance settings
-- Execute these as sysadmin

-- Set max degree of parallelism
EXEC sp_configure 'max degree of parallelism', 4;
RECONFIGURE;

-- Set max server memory (leave 4GB for OS)
EXEC sp_configure 'max server memory', 28672; -- 28GB for 32GB system
RECONFIGURE;

-- Enable optimize for ad hoc workloads
EXEC sp_configure 'optimize for ad hoc workloads', 1;
RECONFIGURE;
```

---

## Application Deployment

### 1. Environment Setup

#### Create Application Directory
```bash
# Windows
mkdir C:\EDI-Processing
cd C:\EDI-Processing

# Linux
sudo mkdir -p /opt/edi-processing
cd /opt/edi-processing
```

#### Clone Repository
```bash
git clone https://github.com/yourorg/edi-claims-processing.git .
```

#### Python Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For development environment
pip install -r requirements.txt[dev]
```

### 2. Create Requirements Files

#### requirements.txt
```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0

# Database
psycopg2-binary>=2.9.0
pyodbc>=4.0.30
SQLAlchemy>=1.4.0

# ML and Data Processing
mlxtend>=0.19.0
joblib>=1.1.0

# Logic and Rules
pyDatalog>=0.17.1

# Monitoring and Metrics
prometheus-client>=0.12.0

# System utilities
psutil>=5.8.0
tqdm>=4.62.0

# Configuration
PyYAML>=5.4.0
cryptography>=3.4.0

# Development tools (optional)
pytest>=6.2.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

### 3. Directory Structure
```
C:\EDI-Processing\
├── config/
│   ├── config.yaml
│   └── config.yaml.example
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── parser.py
│   ├── validator.py
│   ├── storage.py
│   ├── train_filter_predictor.py
│   ├── config/
│   ├── database/
│   ├── monitoring/
│   └── utils/
├── models/
├── logs/
├── data/
├── scripts/
├── tests/
├── run_edi.py
├── setup.py
├── requirements.txt
└── README.md
```

### 4. Service Configuration

#### Windows Service Setup
```powershell
# Create service script: edi-service.ps1
$serviceName = "EDI-Processing-Service"
$serviceDisplayName = "EDI Claims Processing Service"
$servicePath = "C:\EDI-Processing\scripts\run_service.bat"

# Install service using NSSM (Non-Sucking Service Manager)
# Download NSSM from https://nssm.cc/download
nssm install $serviceName $servicePath
nssm set $serviceName DisplayName $serviceDisplayName
nssm set $serviceName Description "Healthcare claims validation and processing service"
nssm set $serviceName Start SERVICE_AUTO_START
```

#### Linux Systemd Service
```ini
# /etc/systemd/system/edi-processing.service
[Unit]
Description=EDI Claims Processing Service
After=network.target

[Service]
Type=simple
User=edi
Group=edi
WorkingDirectory=/opt/edi-processing
Environment=PATH=/opt/edi-processing/venv/bin
ExecStart=/opt/edi-processing/venv/bin/python run_edi.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 5. Service Management Scripts

#### Windows: scripts/run_service.bat
```batch
@echo off
cd /d C:\EDI-Processing
call venv\Scripts\activate
python run_edi.py --config config/config.yaml --log-level INFO
```

#### Linux: scripts/start_service.sh
```bash
#!/bin/bash
cd /opt/edi-processing
source venv/bin/activate
python run_edi.py --config config/config.yaml --log-level INFO
```

---

## Configuration Management

### 1. Main Configuration File

#### config/config.yaml
```yaml
# EDI Claims Processing System Configuration

# Database Configuration
database:
  postgresql:
    host: localhost
    port: 5432
    database: claims_processing
    user: edi_user
    password_encrypted: "ENCRYPTED_PASSWORD_HERE"
    min_connections: 2
    max_connections: 10
    
  sqlserver:
    connection_string: "mssql+pyodbc://edi_results_user:PASSWORD@server/ClaimsValidationResults?driver=ODBC+Driver+17+for+SQL+Server"
    pool_size: 10

# Processing Configuration
processing:
  chunk_size: 500
  max_workers: 4
  memory_limit_percent: 70
  cpu_limit_percent: 80
  batch_size: 1000
  enable_batch_processing: true
  adaptive_chunk_sizing: true
  storage_strategy: "bulk"  # bulk, async, parallel, retry

# ML Training Configuration
ml_training:
  training_chunk_size: 10000
  memory_limit_percent: 70
  model_output_path: "models/filter_predictor.pkl"
  use_gpu: false
  n_jobs: -1
  early_stopping_rounds: 20

# Validation Configuration
validation:
  ml_model_path: "models/filter_predictor.pkl"
  prediction_threshold: 0.3
  max_filters_per_claim: 10

# Storage Configuration
storage:
  batch_size: 1000
  max_retries: 3
  storage_workers: 2
  cleanup_days: 90

# Monitoring Configuration
monitoring:
  prometheus_port: 8000
  log_level: INFO
  enable_email_alerts: true
  monitoring_interval: 30

# Email Configuration
email:
  enabled: true
  smtp_server: smtp.company.com
  smtp_port: 587
  username: edi-system@company.com
  password_encrypted: "ENCRYPTED_SMTP_PASSWORD"
  from_email: edi-system@company.com
  recipients:
    - admin@company.com
    - ops@company.com
  severity_levels:
    - ERROR
    - CRITICAL

# Rule Generation Configuration
rule_generation:
  min_support: 0.01
  min_confidence: 0.5
  min_lift: 1.1

# Encryption key for sensitive configuration values
encryption_key: "ENCRYPTION_KEY_HERE"
```

### 2. Environment-Specific Configurations

#### config/production.yaml
```yaml
# Production overrides
database:
  postgresql:
    host: prod-postgres.company.com
    max_connections: 20
  sqlserver:
    connection_string: "mssql+pyodbc://edi_results_user:PASSWORD@prod-sqlserver.company.com/ClaimsValidationResults?driver=ODBC+Driver+17+for+SQL+Server"

processing:
  chunk_size: 1000
  max_workers: 8

monitoring:
  log_level: WARNING
```

#### config/development.yaml
```yaml
# Development overrides
database:
  postgresql:
    host: dev-postgres.company.com
    max_connections: 5
  sqlserver:
    connection_string: "mssql+pyodbc://edi_results_user:PASSWORD@dev-sqlserver.company.com/ClaimsValidationResults_Dev?driver=ODBC+Driver+17+for+SQL+Server"

processing:
  chunk_size: 100
  max_workers: 2

monitoring:
  log_level: DEBUG
  enable_email_alerts: false
```

### 3. Configuration Encryption

#### Generate Encryption Key
```python
# scripts/generate_key.py
from cryptography.fernet import Fernet

key = Fernet.generate_key()
print(f"Encryption Key: {key.decode()}")
```

#### Encrypt Passwords
```python
# scripts/encrypt_password.py
from cryptography.fernet import Fernet

def encrypt_password(password, key):
    f = Fernet(key.encode())
    encrypted = f.encrypt(password.encode())
    return encrypted.decode()

# Usage
key = "YOUR_ENCRYPTION_KEY_HERE"
password = "your_actual_password"
encrypted = encrypt_password(password, key)
print(f"Encrypted password: {encrypted}")
```

---

## Monitoring Setup

### 1. Prometheus Configuration

#### Install Prometheus
```bash
# Download and extract Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.windows-amd64.zip
unzip prometheus-2.40.0.windows-amd64.zip
```

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'edi-processing'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'sqlserver'
    static_configs:
      - targets: ['localhost:4000']
```

### 2. Grafana Dashboard Setup

#### Install Grafana
```bash
# Windows: Download from Grafana website
# Linux:
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
```

#### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "EDI Claims Processing",
    "panels": [
      {
        "title": "Claims Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(claims_processed_total[5m])",
            "legendFormat": "Claims/sec"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_percent",
            "legendFormat": "Memory %"
          },
          {
            "expr": "cpu_usage_percent", 
            "legendFormat": "CPU %"
          }
        ]
      }
    ]
  }
}
```

### 3. Log Management

#### Log Rotation (Windows)
```powershell
# PowerShell script: scripts/rotate_logs.ps1
$logPath = "C:\EDI-Processing\logs"
$maxSizeMB = 50
$maxFiles = 10

Get-ChildItem $logPath -Filter "*.log" | ForEach-Object {
    if ($_.Length -gt ($maxSizeMB * 1MB)) {
        # Rotate log file
        $baseName = $_.BaseName
        $extension = $_.Extension
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $newName = "$baseName.$timestamp$extension"
        Move-Item $_.FullName (Join-Path $logPath $newName)
        
        # Keep only latest files
        Get-ChildItem $logPath -Filter "$baseName.*$extension" | 
            Sort-Object CreationTime -Descending | 
            Select-Object -Skip $maxFiles | 
            Remove-Item
    }
}
```

---

## Security Configuration

### 1. Database Security

#### PostgreSQL Security
```sql
-- Create read-only monitoring user
CREATE USER monitoring_user WITH PASSWORD 'secure_monitoring_password';
GRANT CONNECT ON DATABASE claims_processing TO monitoring_user;
GRANT USAGE ON SCHEMA edi TO monitoring_user;
GRANT SELECT ON ALL TABLES IN SCHEMA edi TO monitoring_user;

-- Configure pg_hba.conf for secure connections
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    claims_processing  edi_user      10.0.0.0/16             md5
host    claims_processing  monitoring_user 10.0.0.0/16          md5
```

#### SQL Server Security
```sql
-- Create monitoring login
CREATE LOGIN monitoring_user WITH PASSWORD = 'secure_monitoring_password';
USE ClaimsValidationResults;
CREATE USER monitoring_user FOR LOGIN monitoring_user;
ALTER ROLE db_datareader ADD MEMBER monitoring_user;

-- Enable encryption
ALTER DATABASE ClaimsValidationResults SET ENCRYPTION ON;
```

### 2. Application Security

#### File Permissions (Linux)
```bash
# Set proper ownership and permissions
sudo chown -R edi:edi /opt/edi-processing
sudo chmod 750 /opt/edi-processing
sudo chmod 640 /opt/edi-processing/config/*.yaml
sudo chmod 750 /opt/edi-processing/scripts/*.sh
```

#### Secure Configuration Storage
```bash
# Store sensitive configs in secure location
sudo mkdir -p /etc/edi-processing
sudo chown root:edi /etc/edi-processing
sudo chmod 750 /etc/edi-processing
sudo mv /opt/edi-processing/config/config.yaml /etc/edi-processing/
sudo ln -s /etc/edi-processing/config.yaml /opt/edi-processing/config/config.yaml
```

### 3. Network Security

#### Firewall Configuration (Ubuntu)
```bash
# Enable UFW
sudo ufw enable

# Allow SSH (adjust IP range as needed)
sudo ufw allow from 10.0.0.0/16 to any port 22

# Allow application ports
sudo ufw allow from 10.0.0.0/16 to any port 8000  # Prometheus metrics
sudo ufw allow from 10.0.0.0/16 to any port 9090  # Prometheus server

# Deny all other traffic
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

---

## Performance Optimization

### 1. System Tuning

#### Windows Performance Settings
```powershell
# PowerShell script: scripts/optimize_windows.ps1

# Set power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable unnecessary services
Set-Service -Name "Windows Search" -StartupType Disabled
Set-Service -Name "Superfetch" -StartupType Disabled

# Set virtual memory
$ram = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB
$pagefile = [math]::Round($ram * 1.5)
wmic computersystem where name="%computername%" set AutomaticManagedPagefile=False
wmic pagefileset where name="C:\\pagefile.sys" set InitialSize=$pagefile,MaximumSize=$pagefile
```

#### Linux Performance Tuning
```bash
# /etc/sysctl.conf additions
# Network tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Apply settings
sudo sysctl -p
```

### 2. Application Performance

#### Python Optimization
```python
# scripts/optimize_python.py
import sys
import os

# Set Python optimization flags
os.environ['PYTHONOPTIMIZE'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Precompile Python modules
import py_compile
import compileall

def precompile_modules():
    """Precompile all Python modules for faster startup."""
    compileall.compile_dir('src', force=True, optimize=2)
    print("Modules precompiled successfully")

if __name__ == "__main__":
    precompile_modules()
```

#### Memory Management
```python
# Add to main.py for memory optimization
import gc
import psutil

def optimize_memory():
    """Memory optimization settings."""
    # Force garbage collection
    gc.collect()
    
    # Adjust garbage collection thresholds
    gc.set_threshold(700, 10, 10)
    
    # Set process priority
    try:
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except:
        pass
```

### 3. Database Optimization

#### Connection Pooling
```python
# Enhanced database connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_optimized_engine(connection_string):
    return create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
```

---

## Troubleshooting

### 1. Common Issues and Solutions

#### Issue: High Memory Usage
```bash
# Check memory usage
# Windows
tasklist /fi "imagename eq python.exe" /fo table

# Linux
ps aux | grep python
```

**Solutions:**
- Reduce chunk_size in configuration
- Enable adaptive_chunk_sizing
- Increase memory_limit_percent threshold
- Add more RAM to system

#### Issue: Database Connection Failures
```python
# Test database connections
def test_connections():
    import psycopg2
    import pyodbc
    
    # Test PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="claims_processing",
            user="edi_user",
            password="password"
        )
        print("PostgreSQL connection: OK")
        conn.close()
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
    
    # Test SQL Server
    try:
        conn = pyodbc.connect("connection_string_here")
        print("SQL Server connection: OK")
        conn.close()
    except Exception as e:
        print(f"SQL Server connection failed: {e}")
```

**Solutions:**
- Verify database services are running
- Check firewall settings
- Validate connection strings
- Confirm user permissions

#### Issue: Slow Processing Performance
```python
# Performance diagnostic script
def diagnose_performance():
    import time
    import psutil
    
    print("System Performance Diagnostics")
    print("-" * 40)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    # Disk I/O
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}%")
    
    # Network I/O
    network = psutil.net_io_counters()
    print(f"Network Bytes Sent: {network.bytes_sent / (1024**2):.2f} MB")
    print(f"Network Bytes Received: {network.bytes_recv / (1024**2):.2f} MB")
```

### 2. Log Analysis

#### Error Pattern Analysis
```bash
# PowerShell script for Windows log analysis
Get-Content "C:\EDI-Processing\logs\edi_processing.log" | 
    Select-String "ERROR|CRITICAL" | 
    Group-Object {($_ -split " - ")[2]} | 
    Sort-Object Count -Descending | 
    Format-Table Name, Count

# Linux equivalent
grep -E "ERROR|CRITICAL" /opt/edi-processing/logs/edi_processing.log | 
    awk -F' - ' '{print $3}' | 
    sort | uniq -c | sort -nr
```

### 3. Health Check Scripts

#### System Health Check
```python
# scripts/health_check.py
#!/usr/bin/env python3
"""
Comprehensive system health check
"""
import sys
import os
import psutil
import psycopg2
import pyodbc
from datetime import datetime, timedelta

def check_system_resources():
    """Check system resource availability."""
    print("System Resource Check:")
    print("-" * 30)
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}% ({memory.available / (1024**3):.2f} GB available)")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Disk
    disk = psutil.disk_usage('/' if os.name != 'nt' else 'C:')
    print(f"Disk Usage: {disk.percent}% ({disk.free / (1024**3):.2f} GB free)")
    
    return memory.percent < 90 and cpu_percent < 95 and disk.percent < 90

def check_database_connections():
    """Check database connectivity."""
    print("\nDatabase Connection Check:")
    print("-" * 30)
    
    # Add your actual connection parameters
    postgres_ok = True
    sqlserver_ok = True
    
    try:
        # Test PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            database="claims_processing", 
            user="edi_user",
            password="password"
        )
        conn.close()
        print("PostgreSQL: OK")
    except Exception as e:
        print(f"PostgreSQL: FAILED - {e}")
        postgres_ok = False
    
    try:
        # Test SQL Server
        conn = pyodbc.connect("your_connection_string_here")
        conn.close()
        print("SQL Server: OK")
    except Exception as e:
        print(f"SQL Server: FAILED - {e}")
        sqlserver_ok = False
    
    return postgres_ok and sqlserver_ok

def check_log_files():
    """Check for recent errors in log files."""
    print("\nLog File Check:")
    print("-" * 30)
    
    log_path = "logs/edi_processing.log"
    if os.name == 'nt':
        log_path = "C:\\EDI-Processing\\logs\\edi_processing.log"
    else:
        log_path = "/opt/edi-processing/logs/edi_processing.log"
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return False
    
    # Check for recent errors (last 24 hours)
    yesterday = datetime.now() - timedelta(days=1)
    error_count = 0
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'ERROR' in line or 'CRITICAL' in line:
                    error_count += 1
        
        print(f"Recent errors found: {error_count}")
        return error_count < 10  # Threshold for acceptable error count
        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return False

def check_required_directories():
    """Check if required directories exist."""
    print("\nDirectory Structure Check:")
    print("-" * 30)
    
    base_path = "C:\\EDI-Processing" if os.name == 'nt' else "/opt/edi-processing"
    required_dirs = ['logs', 'models', 'config', 'data']
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            print(f"{dir_name}: OK")
        else:
            print(f"{dir_name}: MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run comprehensive health check."""
    print("EDI Claims Processing System Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    checks = [
        ("System Resources", check_system_resources),
        ("Database Connections", check_database_connections),
        ("Log Files", check_log_files),
        ("Directory Structure", check_required_directories)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"Error in {check_name}: {e}")
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("✅ All health checks PASSED")
        sys.exit(0)
    else:
        print("❌ Some health checks FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Maintenance and Operations

### 1. Regular Maintenance Tasks

#### Daily Tasks
```bash
# Daily maintenance script: scripts/daily_maintenance.sh
#!/bin/bash

echo "Starting daily maintenance - $(date)"

# 1. Check disk space
df -h

# 2. Check system resources
top -bn1 | head -5

# 3. Check service status
systemctl status edi-processing

# 4. Rotate logs if needed
python scripts/rotate_logs.py

# 5. Database maintenance
psql -h localhost -U edi_user -d claims_processing -c "ANALYZE;"

echo "Daily maintenance completed - $(date)"
```

#### Weekly Tasks
```bash
# Weekly maintenance script: scripts/weekly_maintenance.sh
#!/bin/bash

echo "Starting weekly maintenance - $(date)"

# 1. Full database statistics update
psql -h localhost -U edi_user -d claims_processing -c "VACUUM ANALYZE;"

# 2. Clean up old log files
find /opt/edi-processing/logs -name "*.log.*" -mtime +30 -delete

# 3. Update ML model if needed
python src/train_filter_predictor.py --config config/config.yaml

# 4. Generate performance report
python scripts/generate_performance_report.py

# 5. Backup configuration
cp -r /etc/edi-processing /backup/edi-config-$(date +%Y%m%d)

echo "Weekly maintenance completed - $(date)"
```

#### Monthly Tasks
```bash
# Monthly maintenance script: scripts/monthly_maintenance.sh
#!/bin/bash

echo "Starting monthly maintenance - $(date)"

# 1. Deep database cleanup
python scripts/cleanup_old_data.py --days 90

# 2. Performance optimization analysis
python scripts/analyze_performance_trends.py

# 3. Security updates check
sudo apt update && sudo apt list --upgradable

# 4. Full system backup
python scripts/backup_system.py

echo "Monthly maintenance completed - $(date)"
```

### 2. Backup and Recovery

#### Database Backup Script
```python
# scripts/backup_database.py
#!/usr/bin/env python3
"""
Database backup script
"""
import subprocess
import os
from datetime import datetime
import logging

def backup_postgresql():
    """Backup PostgreSQL database."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"/backup/postgres_claims_{timestamp}.sql"
    
    cmd = [
        'pg_dump',
        '-h', 'localhost',
        '-U', 'edi_user',
        '-d', 'claims_processing',
        '-f', backup_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"PostgreSQL backup created: {backup_file}")
        return backup_file
    except subprocess.CalledProcessError as e:
        logging.error(f"PostgreSQL backup failed: {e}")
        return None

def backup_sqlserver():
    """Backup SQL Server database."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"C:\\Backup\\sqlserver_results_{timestamp}.bak"
    
    sql_command = f"""
    BACKUP DATABASE ClaimsValidationResults 
    TO DISK = '{backup_file}'
    WITH FORMAT, COMPRESSION;
    """
    
    cmd = [
        'sqlcmd',
        '-S', 'localhost',
        '-Q', sql_command
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"SQL Server backup created: {backup_file}")
        return backup_file
    except subprocess.CalledProcessError as e:
        logging.error(f"SQL Server backup failed: {e}")
        return None

def cleanup_old_backups(backup_dir, days_to_keep=30):
    """Remove old backup files."""
    import time
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    for file in os.listdir(backup_dir):
        file_path = os.path.join(backup_dir, file)
        if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
            os.remove(file_path)
            logging.info(f"Removed old backup: {file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create backup directories
    os.makedirs("/backup", exist_ok=True)
    os.makedirs("C:\\Backup", exist_ok=True)
    
    # Perform backups
    backup_postgresql()
    backup_sqlserver()
    
    # Cleanup old backups
    cleanup_old_backups("/backup")
    cleanup_old_backups("C:\\Backup")
```

#### Recovery Procedures
```bash
# PostgreSQL Recovery
# 1. Stop the application service
sudo systemctl stop edi-processing

# 2. Restore database
psql -h localhost -U edi_user -d claims_processing < /backup/postgres_claims_20240101_120000.sql

# 3. Restart application
sudo systemctl start edi-processing

# SQL Server Recovery
# 1. Stop application
# 2. Use SQL Server Management Studio or sqlcmd:
RESTORE DATABASE ClaimsValidationResults 
FROM DISK = 'C:\Backup\sqlserver_results_20240101_120000.bak'
WITH REPLACE;
```

### 3. Performance Monitoring

#### Performance Report Generator
```python
# scripts/generate_performance_report.py
#!/usr/bin/env python3
"""
Generate comprehensive performance report
"""
import sys
import os
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.postgresql_handler import PostgreSQLHandler
from database.sqlserver_handler import SQLServerHandler
from config.config_manager import ConfigurationManager

def generate_report():
    """Generate performance report."""
    
    # Load configuration
    config_manager = ConfigurationManager("config/config.yaml")
    config = config_manager.get_config()
    
    # Initialize database handlers
    postgres_handler = PostgreSQLHandler(config['database']['postgresql'])
    sqlserver_handler = SQLServerHandler(config['database']['sqlserver'])
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'period': '30 days',
        'metrics': {}
    }
    
    try:
        # Processing statistics
        processing_stats = get_processing_statistics(postgres_handler)
        report['metrics']['processing'] = processing_stats
        
        # Validation statistics
        validation_stats = get_validation_statistics(sqlserver_handler)
        report['metrics']['validation'] = validation_stats
        
        # System performance
        system_stats = get_system_performance()
        report['metrics']['system'] = system_stats
        
        # Database performance
        db_stats = get_database_performance(postgres_handler, sqlserver_handler)
        report['metrics']['database'] = db_stats
        
    except Exception as e:
        report['error'] = str(e)
    
    # Save report
    report_file = f"reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report generated: {report_file}")
    return report

def get_processing_statistics(postgres_handler):
    """Get processing statistics from PostgreSQL."""
    query = """
    SELECT 
        COUNT(*) as total_claims,
        SUM(CASE WHEN processing_status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_claims,
        AVG(CASE WHEN processing_status = 'COMPLETED' THEN 
            EXTRACT(EPOCH FROM (updated_date - created_date)) ELSE NULL END) as avg_processing_time,
        MAX(created_date) as latest_claim_date
    FROM edi.claims 
    WHERE created_date >= NOW() - INTERVAL '30 days'
    """
    
    result = postgres_handler.execute_query(query)
    return result[0] if result else {}

def get_validation_statistics(sqlserver_handler):
    """Get validation statistics from SQL Server."""
    query = """
    SELECT 
        COUNT(*) as total_validations,
        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_validations,
        AVG(processing_time) as avg_validation_time,
        COUNT(DISTINCT claim_id) as unique_claims_validated
    FROM dbo.validation_results 
    WHERE created_date >= DATEADD(day, -30, GETUTCDATE())
    """
    
    result = sqlserver_handler.execute_query(query)
    return result[0] if result else {}

def get_system_performance():
    """Get system performance metrics."""
    import psutil
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3)
    }

def get_database_performance(postgres_handler, sqlserver_handler):
    """Get database performance metrics."""
    postgres_stats = postgres_handler.get_connection_stats()
    sqlserver_stats = sqlserver_handler.get_connection_stats()
    
    return {
        'postgresql': postgres_stats,
        'sqlserver': sqlserver_stats
    }

if __name__ == "__main__":
    generate_report()
```

### 4. Alerting and Notifications

#### Alert Rules Configuration
```yaml
# monitoring/alert_rules.yml
groups:
  - name: edi_processing_alerts
    rules:
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}%"
      
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/second"
      
      - alert: DatabaseConnectionFailure
        expr: database_connections_active == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "No active database connections"
      
      - alert: ProcessingStopped
        expr: rate(claims_processed_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Claims processing stopped"
          description: "No claims processed in the last 10 minutes"
```

#### Custom Alert Script
```python
# scripts/custom_alerts.py
#!/usr/bin/env python3
"""
Custom alerting script
"""
import smtplib
import psutil
import logging
from email.mime.text import MIMEText
from datetime import datetime

class AlertManager:
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)
    
    def check_system_health(self):
        """Check system health and send alerts if needed."""
        alerts = []
        
        # Memory check
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            alerts.append({
                'severity': 'CRITICAL',
                'title': 'High Memory Usage',
                'message': f'Memory usage is {memory_percent:.1f}%'
            })
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            alerts.append({
                'severity': 'CRITICAL',
                'title': 'High CPU Usage',
                'message': f'CPU usage is {cpu_percent:.1f}%'
            })
        
        # Disk check
        disk_percent = psutil.disk_usage('/').percent
        if disk_percent > 95:
            alerts.append({
                'severity': 'CRITICAL',
                'title': 'High Disk Usage',
                'message': f'Disk usage is {disk_percent:.1f}%'
            })
        
        # Send alerts
        for alert in alerts:
            self.send_alert(alert)
    
    def send_alert(self, alert):
        """Send alert email."""
        try:
            subject = f"[EDI Processing {alert['severity']}] {alert['title']}"
            body = f"""
            Alert: {alert['title']}
            Severity: {alert['severity']}
            Time: {datetime.now()}
            Message: {alert['message']}
            
            Please check the system immediately.
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.smtp_config['recipients'])
            
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Alert sent: {alert['title']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

if __name__ == "__main__":
    # Configure SMTP settings
    smtp_config = {
        'server': 'smtp.company.com',
        'port': 587,
        'username': 'alerts@company.com',
        'password': 'password',
        'from_email': 'alerts@company.com',
        'recipients': ['admin@company.com', 'ops@company.com']
    }
    
    alert_manager = AlertManager(smtp_config)
    alert_manager.check_system_health()
```

### 5. Deployment Automation

#### Deployment Script
```bash
#!/bin/bash
# scripts/deploy.sh - Automated deployment script

set -e

ENVIRONMENT=${1:-development}
VERSION=${2:-latest}

echo "Starting deployment to $ENVIRONMENT environment"
echo "Version: $VERSION"

# 1. Stop the service
echo "Stopping EDI processing service..."
sudo systemctl stop edi-processing

# 2. Backup current version
echo "Creating backup..."
sudo cp -r /opt/edi-processing /opt/edi-processing.backup.$(date +%Y%m%d_%H%M%S)

# 3. Pull latest code
echo "Updating code..."
cd /opt/edi-processing
git fetch origin
git checkout $VERSION
git pull origin $VERSION

# 4. Update dependencies
echo "Updating dependencies..."
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 5. Run database migrations if any
echo "Running database updates..."
python scripts/migrate_database.py

# 6. Update configuration for environment
echo "Updating configuration..."
cp config/$ENVIRONMENT.yaml config/config.yaml

# 7. Precompile Python modules
echo "Precompiling modules..."
python scripts/optimize_python.py

# 8. Run health checks
echo "Running health checks..."
python scripts/health_check.py

# 9. Start the service
echo "Starting EDI processing service..."
sudo systemctl start edi-processing

# 10. Verify deployment
sleep 30
if sudo systemctl is-active --quiet edi-processing; then
    echo "✅ Deployment successful"
    echo "Service is running and healthy"
else
    echo "❌ Deployment failed"
    echo "Service is not running"
    exit 1
fi

echo "Deployment completed successfully"
```

#### Rollback Script
```bash
#!/bin/bash
# scripts/rollback.sh - Rollback to previous version

set -e

BACKUP_DIR=${1:-$(ls -1d /opt/edi-processing.backup.* | tail -1)}

if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ No backup directory found or specified"
    exit 1
fi

echo "Rolling back to: $BACKUP_DIR"

# 1. Stop service
echo "Stopping service..."
sudo systemctl stop edi-processing

# 2. Move current version
echo "Backing up current version..."
sudo mv /opt/edi-processing /opt/edi-processing.failed.$(date +%Y%m%d_%H%M%S)

# 3. Restore backup
echo "Restoring backup..."
sudo mv $BACKUP_DIR /opt/edi-processing

# 4. Start service
echo "Starting service..."
sudo systemctl start edi-processing

# 5. Verify
sleep 30
if sudo systemctl is-active --quiet edi-processing; then
    echo "✅ Rollback successful"
else
    echo "❌ Rollback failed"
    exit 1
fi
```

---