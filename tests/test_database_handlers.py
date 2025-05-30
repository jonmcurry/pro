# tests/test_database_handlers.py
"""
Tests for database handlers
"""
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.postgresql_handler import PostgreSQLHandler
from database.sqlserver_handler import SQLServerHandler


class TestPostgreSQLHandler:
    """Tests for PostgreSQL handler."""
    
    def test_init_with_config(self):
        """Test handler initialization with config."""
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        with patch('psycopg2.pool.ThreadedConnectionPool'):
            handler = PostgreSQLHandler(config)
            assert handler.host == 'localhost'
            assert handler.port == 5432
            assert handler.database == 'test_db'
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_success(self, mock_pool):
        """Test successful query execution."""
        # Setup mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [('id',), ('name',)]
        mock_cursor.fetchall.return_value = [(1, 'test')]
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.return_value.getconn.return_value = mock_connection
        
        config = {
            'host': 'localhost', 'port': 5432, 'database': 'test_db',
            'user': 'test_user', 'password': 'test_pass'
        }
        
        handler = PostgreSQLHandler(config)
        result = handler.execute_query("SELECT * FROM test")
        
        assert len(result) == 1
        assert result[0]['id'] == 1
        assert result[0]['name'] == 'test'


class TestSQLServerHandler:
    """Tests for SQL Server handler."""
    
    @patch('pyodbc.connect')
    def test_init_with_config(self, mock_connect):
        """Test handler initialization."""
        config = {
            'connection_string': 'test_connection_string',
            'pool_size': 5
        }
        
        mock_connect.return_value = MagicMock()
        handler = SQLServerHandler(config)
        assert handler.connection_string == 'test_connection_string'
        assert handler.pool_size == 5


# tests/test_config_manager.py
"""
Tests for configuration manager
"""
import pytest
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_manager import ConfigurationManager


class TestConfigurationManager:
    """Tests for configuration manager."""
    
    def test_load_valid_config(self):
        """Test loading valid configuration."""
        config_data = {
            'database': {
                'postgresql': {'host': 'localhost'},
                'sqlserver': {'connection_string': 'test'}
            },
            'processing': {'chunk_size': 500}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigurationManager(config_path)
            config = manager.get_config()
            
            assert 'database' in config
            assert 'processing' in config
            assert config['processing']['chunk_size'] == 500
        finally:
            Path(config_path).unlink()
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            manager = ConfigurationManager('nonexistent.yaml')
            manager.get_config()


# tests/test_validators.py
"""
Tests for claim validator
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validator import ClaimValidator


class TestClaimValidator:
    """Tests for claim validator."""
    
    def test_init_validator(self):
        """Test validator initialization."""
        mock_db = MagicMock()
        config = {'prediction_threshold': 0.3}
        
        with patch.object(ClaimValidator, '_load_ml_components'):
            with patch.object(ClaimValidator, '_load_datalog_rules'):
                validator = ClaimValidator(mock_db, config)
                assert validator.config == config
    
    def test_validate_claim_structure(self):
        """Test claim validation returns proper structure."""
        mock_db = MagicMock()
        config = {'prediction_threshold': 0.3}
        
        with patch.object(ClaimValidator, '_load_ml_components'):
            with patch.object(ClaimValidator, '_load_datalog_rules'):
                with patch.object(ClaimValidator, '_predict_applicable_filters', return_value=[]):
                    with patch.object(ClaimValidator, '_validate_with_rules', return_value=[]):
                        validator = ClaimValidator(mock_db, config)
                        
                        claim = {'claim_id': 'TEST123', 'patient_age': 30}
                        result = validator.validate_claim(claim)
                        
                        assert 'claim_id' in result
                        assert 'validation_status' in result
                        assert 'processing_time' in result
                        assert result['claim_id'] == 'TEST123'


# tests/conftest.py
"""
Pytest configuration and fixtures
"""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_content = """
database:
  postgresql:
    host: localhost
    port: 5432
    database: test_db
    user: test_user
    password: test_pass
  sqlserver:
    connection_string: "test_connection"

processing:
  chunk_size: 100
  max_workers: 2

validation:
  prediction_threshold: 0.3
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_claim():
    """Sample claim data for testing."""
    return {
        'claim_id': 'CLM_TEST_001',
        'patient_id': 'PAT_123456',
        'provider_id': 'PRV_789',
        'total_charge_amount': 150.00,
        'patient_age': 35,
        'provider_type': 'PRIMARY_CARE',
        'place_of_service': '11',
        'diagnoses': [
            {'code': 'Z00.00', 'type': 'ICD10'}
        ],
        'procedures': [
            {'code': '99213', 'type': 'CPT'}
        ]
    }


# tests/__init__.py
"""
Test package for EDI Claims Processing System
"""

# tests/run_tests.py
"""
Test runner script
"""
import pytest
import sys
from pathlib import Path

def run_tests():
    """Run all tests."""
    test_dir = Path(__file__).parent
    
    # Run pytest with coverage
    exit_code = pytest.main([
        str(test_dir),
        '-v',
        '--tb=short',
        '--cov=src',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests())