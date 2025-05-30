Prometheus Setup for Windows
1. Download and Install Prometheus
Download Prometheus

Go to https://prometheus.io/download/
Download the Windows version (prometheus-x.x.x.windows-amd64.zip)
Extract to C:\prometheus

Download Node Exporter (Optional)

Download windows_exporter from https://github.com/prometheus-community/windows_exporter
Extract to C:\node_exporter

2. Configuration
Copy Prometheus Config
cmdcopy monitoring\prometheus\prometheus.yml C:\prometheus\prometheus.yml
Modify prometheus.yml for Windows paths
yaml# Update the EDI processing target to match your setup
scrape_configs:
  - job_name: 'edi-processing'
    static_configs:
      - targets: ['localhost:8000']
3. Install as Windows Service
Create Prometheus Service
cmd# Install NSSM (Non-Sucking Service Manager)
# Download from https://nssm.cc/download

# Install Prometheus service
nssm install Prometheus "C:\prometheus\prometheus.exe"
nssm set Prometheus Arguments "--config.file=C:\prometheus\prometheus.yml --storage.tsdb.path=C:\prometheus\data"
nssm set Prometheus DisplayName "Prometheus Monitoring"
nssm set Prometheus Description "Prometheus time-series database for EDI monitoring"
Create Node Exporter Service (Optional)
cmdnssm install NodeExporter "C:\node_exporter\windows_exporter.exe"
nssm set NodeExporter DisplayName "Node Exporter"
nssm set NodeExporter Description "System metrics exporter"
4. Start Services
cmd# Start Prometheus
net start Prometheus

# Start Node Exporter (if installed)
net start NodeExporter
5. Verify Installation

Open browser to http://localhost:9090
Verify Prometheus UI loads
Check Status > Targets to see if EDI processing endpoint is being scraped
Query edi_claims_processed_total to verify metrics are being collected

6. Windows Firewall Configuration
cmd# Allow Prometheus port
netsh advfirewall firewall add rule name="Prometheus" dir=in action=allow protocol=TCP localport=9090

# Allow Node Exporter port (if used)
netsh advfirewall firewall add rule name="NodeExporter" dir=in action=allow protocol=TCP localport=9100

# Allow EDI metrics port
netsh advfirewall firewall add rule name="EDI-Metrics" dir=in action=allow protocol=TCP localport=8000
7. Basic Queries for EDI Monitoring
promql# Claims processing rate
rate(claims_processed_total[5m])

# System memory usage
memory_usage_percent

# Active database connections
database_connections_active

# Error rate
rate(errors_total[5m])

# Processing duration
histogram_quantile(0.95, rate(processing_duration_seconds_bucket[5m]))