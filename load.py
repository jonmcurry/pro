### load.py
"""
Data Loading and Ingestion Module
Handles loading sample data and claim files into the EDI processing system
"""
import logging
import pandas as pd
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import random

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.postgresql_handler import PostgreSQLHandler
from src.config.config_manager import ConfigurationManager
from src.utils.logging_config import setup_logging


class DataLoader:
    """Handles loading and ingestion of claims data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.db_handler = PostgreSQLHandler(config['database']['postgresql'])
        
        # Data pools for realistic sample generation
        self.first_names = ["JOHN", "MARY", "DAVID", "SARAH", "MICHAEL", "EMMA", "JAMES", "LISA",
                           "ROBERT", "JENNIFER", "WILLIAM", "LINDA", "RICHARD", "PATRICIA", "CHARLES", "BARBARA",
                           "THOMAS", "ELIZABETH", "CHRISTOPHER", "MARIA", "DANIEL", "SUSAN", "MATTHEW", "MARGARET"]
        
        self.last_names = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS",
                          "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ", "WILSON", "ANDERSON", "THOMAS",
                          "TAYLOR", "MOORE", "JACKSON", "MARTIN", "LEE", "PEREZ", "THOMPSON", "WHITE"]
        
        # Standard healthcare codes
        self.place_of_service = [str(i) for i in range(1, 82)]  # CMS codes up to 81
        self.sex_values = ["F", "M", "U"]
        self.payers = ["Medicare", "Medicaid", "Blue Cross", "Others", "Self Pay", "HMO", 
                      "Tricare", "Commercial", "Workers Comp", "MC Advantage", "Aetna", 
                      "Cigna", "UnitedHealth", "Humana"]
        
        # Load codes from CSV files in same directory as script
        self.icd10_codes = self._load_icd10_codes()
        self.cpt_codes = self._load_cpt_codes()
    
    def _load_icd10_codes(self) -> List[str]:
        """Load ICD-10 codes from icd10.csv in same directory as script."""
        script_dir = Path(__file__).parent
        icd10_file = script_dir / "icd10.csv"
        
        if icd10_file.exists():
            try:
                codes = self._load_codes_from_csv(str(icd10_file))
                if codes:
                    self.logger.info(f"✓ Loaded {len(codes)} ICD-10 codes from {icd10_file}")
                    return codes
            except Exception as e:
                self.logger.warning(f"Failed to load ICD-10 codes from {icd10_file}: {e}")
        
        # Fallback to default codes
        default_codes = [
            "Z00.00", "I10", "E11.9", "M79.89", "R06.02", "K21.9", "F32.9", "M25.50",
            "E78.5", "N39.0", "J44.1", "M54.5", "R50.9", "K59.00", "F41.9", "M19.90",
            "I25.10", "E66.9", "K76.0", "R51", "J06.9", "M62.81", "F43.10", "N18.6",
            "J02.9", "R06.00", "M79.3", "K30", "R05", "G44.1", "M25.511", "J45.9"
        ]
        self.logger.info(f"ℹ No icd10.csv file found in {script_dir}. Using default ICD-10 codes")
        return default_codes
    
    def _load_cpt_codes(self) -> List[str]:
        """Load CPT codes from cpts.csv in same directory as script."""
        script_dir = Path(__file__).parent
        cpt_file = script_dir / "cpts.csv"
        
        if cpt_file.exists():
            try:
                codes = self._load_codes_from_csv(str(cpt_file))
                if codes:
                    self.logger.info(f"✓ Loaded {len(codes)} CPT codes from {cpt_file}")
                    return codes
            except Exception as e:
                self.logger.warning(f"Failed to load CPT codes from {cpt_file}: {e}")
        
        # Fallback to default codes
        default_codes = [
            "99213", "99214", "99215", "99202", "99203", "99204", "99212", "99291",
            "99232", "99233", "99238", "99239", "36415", "85025", "80053", "93000",
            "71020", "73060", "76700", "99395", "99396", "99385", "99386", "90471",
            "99211", "99205", "99381", "99391", "90686", "90715", "87804", "80048"
        ]
        self.logger.info(f"ℹ No cpts.csv file found in {script_dir}. Using default CPT codes")
        return default_codes
    
    def _load_codes_from_csv(self, file_path: str) -> List[str]:
        """Load codes from a CSV file (handles both comma-separated single row and multi-row formats)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                
                # Check if it's a single-row comma-separated format (like generate_837p_data.py)
                if len(first_row) > 1:
                    # Single row with multiple codes separated by commas
                    codes = [code.strip() for code in first_row if code.strip()]
                else:
                    # Multi-row format or single row with one code
                    codes = [first_row[0].strip()] if first_row[0].strip() else []
                    
                    # Read remaining rows
                    for row in reader:
                        if row and row[0].strip():
                            codes.append(row[0].strip())
                
                return codes
                
        except Exception as e:
            raise Exception(f"Error reading codes from {file_path}: {str(e)}")
    
    def random_npi(self, used_npis: set) -> str:
        """Generate a unique 10-digit NPI."""
        npi = str(random.randint(1000000000, 9999999999))
        while npi in used_npis:
            npi = str(random.randint(1000000000, 9999999999))
        return npi

    def load_sample_data(self, num_claims: int = 1000) -> bool:
        """Generate and load sample claims data for testing."""
        try:
            self.logger.info(f"Generating {num_claims} sample claims")
            
            # Generate sample claims with realistic data
            sample_claims = self._generate_realistic_sample_claims(num_claims)
            
            # Load claims into database
            self._load_claims_to_db(sample_claims)
            
            # Generate sample diagnoses and procedures using loaded codes
            self._load_diagnoses_and_procedures_realistic(sample_claims)
            
            # Load sample filters
            self._load_sample_filters()
            
            self.logger.info(f"Successfully loaded {num_claims} sample claims")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading sample data: {str(e)}")
            return False
    
    def load_csv_file(self, file_path: str) -> bool:
        """Load claims data from CSV file."""
        try:
            self.logger.info(f"Loading claims from CSV: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['claim_id', 'patient_id', 'provider_id', 'total_charge_amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert DataFrame to claims list
            claims = df.to_dict('records')
            
            # Load to database
            self._load_claims_to_db(claims)
            
            self.logger.info(f"Successfully loaded {len(claims)} claims from CSV")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            return False
    
    def load_json_file(self, file_path: str) -> bool:
        """Load claims data from JSON file."""
        try:
            self.logger.info(f"Loading claims from JSON: {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                claims = data
            elif isinstance(data, dict) and 'claims' in data:
                claims = data['claims']
            else:
                raise ValueError("Invalid JSON structure. Expected list or object with 'claims' key")
            
            # Load to database
            self._load_claims_to_db(claims)
            
            self.logger.info(f"Successfully loaded {len(claims)} claims from JSON")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {str(e)}")
            return False
    
    def _generate_realistic_sample_claims(self, num_claims: int) -> List[Dict[str, Any]]:
        """Generate realistic sample claims data using loaded codes and realistic patterns."""
        sample_claims = []
        used_npis = set()
        
        # Date range for realistic claims (last 2 years)
        date_range_start = datetime.now() - timedelta(days=730)
        date_range_end = datetime.now()
        
        for i in range(num_claims):
            claim_id = f"CLM_{str(uuid.uuid4())[:8].upper()}"
            patient_id = f"PAT_{random.randint(100000, 999999)}"
            
            # Generate realistic NPI
            provider_npi = self.random_npi(used_npis)
            used_npis.add(provider_npi)
            
            # Random service dates
            service_date_from = self._random_date(date_range_start, date_range_end)
            service_date_to = self._random_date(service_date_from, 
                                               service_date_from + timedelta(days=random.randint(0, 30)))
            
            # Realistic charge amount
            charge_amount = round(random.uniform(25.00, 5000.0), 2)
            
            # Patient demographics
            patient_age = random.randint(1, 95)
            sex = random.choice(self.sex_values)
            
            # Provider and service info
            place_of_service = random.choice(self.place_of_service[:20])  # Use common POS codes
            payer = random.choice(self.payers)
            
            # Provider type based on place of service
            provider_type = self._get_provider_type_from_pos(place_of_service)
            
            claim = {
                'claim_id': claim_id,
                'patient_id': patient_id,
                'provider_id': provider_npi,
                'claim_data': json.dumps({
                    'submission_date': datetime.now().isoformat(),
                    'member_id': f"MBR_{random.randint(1000000, 9999999)}",
                    'patient_name': f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
                    'provider_name': f"Dr. {random.choice(self.first_names)} {random.choice(self.last_names)}",
                    'sex': sex,
                    'payer': payer,
                    'service_date_from': service_date_from.isoformat(),
                    'service_date_to': service_date_to.isoformat()
                }),
                'total_charge_amount': charge_amount,
                'service_date': service_date_from.date(),
                'patient_age': patient_age,
                'provider_type': provider_type,
                'place_of_service': place_of_service,
                'processing_status': random.choice(['PENDING', 'PROCESSING', 'APPROVED'])
            }
            
            sample_claims.append(claim)
        
        return sample_claims
    
    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate a random date between start and end."""
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    def _get_provider_type_from_pos(self, pos_code: str) -> str:
        """Determine provider type based on place of service code."""
        pos_to_type = {
            '11': 'PRIMARY_CARE',  # Office
            '12': 'PRIMARY_CARE',  # Home
            '21': 'HOSPITAL',      # Inpatient Hospital
            '22': 'HOSPITAL',      # Outpatient Hospital
            '23': 'EMERGENCY',     # Emergency Room
            '24': 'URGENT_CARE',   # Ambulatory Surgical Center
            '31': 'SPECIALIST',    # Skilled Nursing Facility
            '41': 'URGENT_CARE',   # Ambulance
            '49': 'SPECIALIST',    # Independent Clinic
            '50': 'SPECIALIST',    # Federally Qualified Health Center
            '71': 'SPECIALIST',    # Public Health Clinic
            '81': 'SPECIALIST'     # Independent Laboratory
        }
        return pos_to_type.get(pos_code, 'SPECIALIST')
    
    def _load_claims_to_db(self, claims: List[Dict[str, Any]]):
        """Load claims data to database."""
        query = """
        INSERT INTO edi.claims (
            claim_id, patient_id, provider_id, claim_data, 
            total_charge_amount, service_date, patient_age, 
            provider_type, place_of_service, processing_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (claim_id) DO NOTHING
        """
        
        params_list = []
        for claim in claims:
            params = (
                claim['claim_id'],
                claim['patient_id'],
                claim['provider_id'],
                claim.get('claim_data'),
                claim['total_charge_amount'],
                claim.get('service_date'),
                claim.get('patient_age'),
                claim.get('provider_type'),
                claim.get('place_of_service'),
                claim.get('processing_status', 'PENDING')
            )
            params_list.append(params)
        
        self.db_handler.execute_many(query, params_list)
    
    def _load_diagnoses_and_procedures_realistic(self, claims: List[Dict[str, Any]]):
        """Generate and load realistic diagnoses and procedures using loaded codes."""
        diagnosis_params = []
        procedure_params = []
        
        for claim in claims:
            claim_id = claim['claim_id']
            
            # Add 1-5 diagnoses per claim (more realistic range)
            num_diagnoses = random.randint(1, min(5, len(self.icd10_codes)))
            if num_diagnoses > 0:
                selected_diagnoses = random.sample(self.icd10_codes, num_diagnoses)
                for i, diagnosis_code in enumerate(selected_diagnoses):
                    is_principal = i == 0  # First diagnosis is principal
                    diagnosis_params.append((
                        claim_id,
                        diagnosis_code,
                        'ICD10',
                        f'Diagnosis description for {diagnosis_code}',
                        i + 1,
                        is_principal
                    ))
            
            # Add 1-3 procedures per claim
            num_procedures = random.randint(1, min(3, len(self.cpt_codes)))
            if num_procedures > 0:
                selected_procedures = random.sample(self.cpt_codes, num_procedures)
                for i, procedure_code in enumerate(selected_procedures):
                    # Generate service line data
                    line_charge = round(claim['total_charge_amount'] / num_procedures, 2)
                    if i == num_procedures - 1:  # Last procedure gets remaining amount
                        line_charge = round(claim['total_charge_amount'] - 
                                          (line_charge * (num_procedures - 1)), 2)
                    
                    # Create diagnosis pointers (reference to diagnoses)
                    max_pointers = min(4, num_diagnoses) if num_diagnoses > 0 else 0
                    num_pointers = random.randint(1, max_pointers) if max_pointers > 0 else 0
                    pointers = sorted(random.sample(range(1, num_diagnoses + 1), 
                                                  num_pointers)) if num_pointers > 0 else []
                    
                    procedure_params.append((
                        claim_id,
                        procedure_code,
                        'CPT',
                        f'Procedure description for {procedure_code}',
                        claim['service_date'],
                        i + 1,
                        line_charge,
                        json.dumps(pointers)  # Store as JSON array
                    ))
        
        # Insert diagnoses with principal flag
        diagnosis_query = """
        INSERT INTO edi.diagnoses (
            claim_id, diagnosis_code, diagnosis_type, description, 
            diagnosis_sequence, is_principal
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.db_handler.execute_many(diagnosis_query, diagnosis_params)
        
        # Insert procedures with enhanced data
        procedure_query = """
        INSERT INTO edi.procedures (
            claim_id, procedure_code, procedure_type, description, 
            service_date, procedure_sequence, charge_amount, diagnosis_pointers
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.db_handler.execute_many(procedure_query, procedure_params)
    
    def _load_sample_filters(self):
        """Load sample validation filters."""
        sample_filters = [
            {
                'name': 'Age_Under_18_Preventive_Care',
                'definition': 'has_diagnosis(X, Code) & patient_age(X, Age) & (Age < 18) & preventive_code(Code)',
                'description': 'Validate preventive care codes for patients under 18',
                'rule_type': 'MANUAL'
            },
            {
                'name': 'High_Cost_Specialist_Review',
                'definition': 'charge_amount(X, Amount) & provider_type(X, "SPECIALIST") & (Amount > 1000)',
                'description': 'Flag high-cost specialist claims for review',
                'rule_type': 'MANUAL'
            },
            {
                'name': 'Emergency_Room_Diagnosis_Check',
                'definition': 'place_of_service(X, "23") & has_diagnosis(X, Code) & emergency_appropriate(Code)',
                'description': 'Validate diagnosis codes are appropriate for emergency room visits',
                'rule_type': 'MANUAL'
            },
            {
                'name': 'Diabetes_Management_Protocol',
                'definition': 'has_diagnosis(X, "E11.9") & has_procedure(X, ProcCode) & diabetes_management(ProcCode)',
                'description': 'Ensure proper procedures for diabetes management',
                'rule_type': 'MANUAL'
            }
        ]
        
        filter_params = []
        for filter_rule in sample_filters:
            filter_params.append((
                filter_rule['name'],
                filter_rule['definition'],
                filter_rule['description'],
                filter_rule['rule_type']
            ))
        
        query = """
        INSERT INTO edi.filters (
            filter_name, rule_definition, description, rule_type, active
        ) VALUES (%s, %s, %s, %s, true)
        ON CONFLICT (filter_name) DO NOTHING
        """
        
        self.db_handler.execute_many(query, filter_params)
    
    def clear_all_data(self) -> bool:
        """Clear all data from the database (use with caution)."""
        try:
            self.logger.warning("Clearing all data from database")
            
            # Clear in reverse dependency order
            tables = ['edi.procedures', 'edi.diagnoses', 'edi.processed_chunks', 
                     'edi.retry_queue', 'edi.claims', 'edi.filters']
            
            for table in tables:
                self.db_handler.execute_query(f"DELETE FROM {table}")
                self.logger.info(f"Cleared table: {table}")
            
            self.logger.info("All data cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            return False


def main():
    """Main function for data loading operations - simplified to load sample data automatically."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting EDI data loading - generating sample claims data")
        
        # Load configuration from default location
        config_file = "config/config.yaml"
        config_manager = ConfigurationManager(config_file)
        config = config_manager.get_config()
        
        # Initialize data loader
        loader = DataLoader(config)
        
        # Load sample data with 100,000 claims
        num_claims = 100000
        success = loader.load_sample_data(num_claims)
        
        if success:
            logger.info(f"✓ Successfully loaded {num_claims} sample claims into database")
            print(f"✓ Data loading completed successfully! Generated {num_claims} claims.")
            return 0
        else:
            logger.error("✗ Data loading operation failed")
            print("✗ Data loading failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())