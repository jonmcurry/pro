## Project Directory Structure
```
edi_claims_processing/
├── config/
│   ├── config.example.yaml
│   ├── config.yaml
│   ├── app_config.yaml
│   └── database_config.yaml
├── docs/
│   ├── architecture.md
│   ├── database_config.md
│   └── prometheus_setup_windows.md
├── sql/
│   ├── postgresql_create_edi_databases.sql
│   └── sqlserver_create_results_database.sql
├── monitoring/
│   └── prometheus/
│       └── prometheus.yml
├── logs/
├── models/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── validator.py
│   ├── parser.py
│   ├── storage.py
│   ├── rule_generator.py
│   ├── train_filter_predictor.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_manager.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── notifications.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgresql_handler.py
│   │   └── sqlserver_handler.py
│   └── utils/
│       ├── __init__.py
│       ├── encryption.py
│       ├── logging_config.py
│       └── resource_monitor.py
├── tests/
├── requirements.txt
├── run_edi.py
├── README.md
└── setup.py