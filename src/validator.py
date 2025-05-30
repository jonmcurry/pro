### src/validator.py
"""
OPTIMIZED Claim Validator - Fixes ML model loading and pyDatalog performance issues
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from pyDatalog import pyDatalog
import functools
from dataclasses import dataclass

from database.postgresql_handler import PostgreSQLHandler


@dataclass
class ValidationResult:
    """Structured validation result."""
    filter_id: int
    filter_name: str
    passed: bool
    rule_type: str
    details: str = ""
    error: str = ""


class ModelCache:
    """OPTIMIZED: Singleton ML model cache with lazy loading."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.ml_model = None
            self.label_encoders = {}
            self.feature_columns = []
            self.model_path = None
            self.model_timestamp = None
            self._model_lock = threading.Lock()
            self._initialized = True
    
    def load_model(self, model_path: str, force_reload: bool = False):
        """Load ML model with caching and hot reloading support."""
        import os
        
        if not os.path.exists(model_path):
            return False
            
        # Check if model needs reloading
        current_timestamp = os.path.getmtime(model_path)
        
        with self._model_lock:
            if (not force_reload and 
                self.ml_model is not None and 
                self.model_path == model_path and
                self.model_timestamp == current_timestamp):
                return True
            
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ml_model = model_data['model']
                self.label_encoders = model_data['encoders']
                self.feature_columns = model_data.get('features', [])
                self.model_path = model_path
                self.model_timestamp = current_timestamp
                
                return True
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error loading ML model: {str(e)}")
                return False
    
    def predict_batch(self, features: np.ndarray, threshold: float = 0.3, max_filters: int = 10) -> List[List[int]]:
        """OPTIMIZED: Batch prediction for multiple claims."""
        if self.ml_model is None:
            return [[] for _ in range(len(features))]
        
        with self._model_lock:
            try:
                # Batch prediction - much faster than individual predictions
                probabilities = self.ml_model.predict_proba(features)
                
                results = []
                for probs in probabilities:
                    applicable_filters = []
                    for i, prob in enumerate(probs):
                        if prob > threshold:
                            applicable_filters.append(i)
                    
                    # Limit number of filters per claim
                    results.append(applicable_filters[:max_filters])
                
                return results
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Batch prediction error: {str(e)}")
                return [[] for _ in range(len(features))]


class RuleCache:
    """OPTIMIZED: pyDatalog rule cache with compilation."""
    
    def __init__(self):
        self.compiled_rules = {}
        self.rule_definitions = {}
        self._rules_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize pyDatalog once
        pyDatalog.clear()
        self._setup_predicates()
    
    def _setup_predicates(self):
        """Setup pyDatalog predicates once."""
        try:
            pyDatalog.create_terms('has_diagnosis, has_procedure, patient_age, charge_amount')
            pyDatalog.create_terms('provider_type, place_of_service, rule_applies')
            pyDatalog.create_terms('X, Age, Code, Amount, Provider, POS')
        except Exception as e:
            self.logger.error(f"Error setting up predicates: {str(e)}")
    
    def load_rules(self, rules: List[Dict[str, Any]]):
        """Load and compile rules for faster execution."""
        with self._rules_lock:
            self.rule_definitions.clear()
            self.compiled_rules.clear()
            
            for rule in rules:
                filter_id = rule['filter_id']
                self.rule_definitions[filter_id] = rule
                
                # Pre-compile rule if possible
                try:
                    rule_query = rule['rule_definition']
                    # Store compiled rule representation
                    self.compiled_rules[filter_id] = {
                        'query': rule_query,
                        'name': rule['filter_name'],
                        'description': rule.get('description', '')
                    }
                except Exception as e:
                    self.logger.error(f"Error compiling rule {filter_id}: {str(e)}")
    
    def evaluate_rules_batch(self, claims: List[Dict[str, Any]], filter_ids_list: List[List[int]]) -> List[List[ValidationResult]]:
        """OPTIMIZED: Batch rule evaluation for multiple claims."""
        results = []
        
        with self._rules_lock:
            # Setup facts for all claims at once
            self._setup_batch_facts(claims)
            
            for i, (claim, filter_ids) in enumerate(zip(claims, filter_ids_list)):
                claim_results = []
                claim_id = claim.get('claim_id', f'claim_{i}')
                
                for filter_id in filter_ids:
                    if filter_id in self.compiled_rules:
                        result = self._evaluate_single_rule(claim_id, filter_id)
                        claim_results.append(result)
                
                results.append(claim_results)
        
        return results
    
    def _setup_batch_facts(self, claims: List[Dict[str, Any]]):
        """OPTIMIZED: Setup facts for all claims in batch."""
        try:
            # Clear existing facts
            pyDatalog.clear()
            self._setup_predicates()
            
            # Add facts for all claims
            for claim in claims:
                claim_id = claim.get('claim_id', 'unknown')
                
                # Diagnosis facts
                for diagnosis in claim.get('diagnoses', []):
                    if diagnosis.get('code'):
                        pyDatalog.assert_fact('has_diagnosis', claim_id, diagnosis['code'])
                
                # Procedure facts
                for procedure in claim.get('procedures', []):
                    if procedure.get('code'):
                        pyDatalog.assert_fact('has_procedure', claim_id, procedure['code'])
                
                # Patient facts
                pyDatalog.assert_fact('patient_age', claim_id, claim.get('patient_age', 0))
                pyDatalog.assert_fact('charge_amount', claim_id, claim.get('total_charge_amount', 0))
                
                # Provider facts
                pyDatalog.assert_fact('provider_type', claim_id, claim.get('provider_type', ''))
                pyDatalog.assert_fact('place_of_service', claim_id, claim.get('place_of_service', ''))
                
        except Exception as e:
            self.logger.error(f"Error setting up batch facts: {str(e)}")
    
    def _evaluate_single_rule(self, claim_id: str, filter_id: int) -> ValidationResult:
        """Evaluate a single rule against a claim."""
        try:
            rule_info = self.compiled_rules[filter_id]
            
            # Create a specific query for this claim
            rule_query = f"rule_applies('{claim_id}')"
            
            # Execute rule
            results = pyDatalog.ask(rule_query)
            
            return ValidationResult(
                filter_id=filter_id,
                filter_name=rule_info['name'],
                passed=bool(results),
                rule_type='DATALOG',
                details=rule_info['description']
            )
            
        except Exception as e:
            return ValidationResult(
                filter_id=filter_id,
                filter_name=f'Filter_{filter_id}',
                passed=False,
                rule_type='DATALOG',
                error=str(e)
            )


class ClaimValidator:
    """OPTIMIZED: Hybrid validation engine with batch processing and caching."""
    
    def __init__(self, db_handler: PostgreSQLHandler, config: Dict[str, Any]):
        self.db_handler = db_handler
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self.model_cache = ModelCache()
        self.rule_cache = RuleCache()
        
        # Validation settings
        self.prediction_threshold = config.get('prediction_threshold', 0.3)
        self.max_filters_per_claim = config.get('max_filters_per_claim', 10)
        
        # Initialize components
        self._load_ml_model()
        self._load_rules()
        
        # Feature extraction optimization
        self._feature_extractors = self._build_feature_extractors()
    
    def validate_claims_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Validate multiple claims in batch for better performance.
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract features for all claims in batch
            features_matrix = self._extract_features_batch(claims)
            
            # Step 2: Batch ML prediction
            predicted_filters_list = self.model_cache.predict_batch(
                features_matrix, 
                self.prediction_threshold,
                self.max_filters_per_claim
            )
            
            # Step 3: Batch rule validation
            validation_results_list = self.rule_cache.evaluate_rules_batch(
                claims, predicted_filters_list
            )
            
            # Step 4: Compile results
            results = []
            processing_time = time.time() - start_time
            avg_time_per_claim = processing_time / len(claims) if claims else 0
            
            for i, claim in enumerate(claims):
                result = {
                    'claim_id': claim.get('claim_id'),
                    'predicted_filters': predicted_filters_list[i],
                    'validation_results': [
                        {
                            'filter_id': vr.filter_id,
                            'filter_name': vr.filter_name,
                            'passed': vr.passed,
                            'rule_type': vr.rule_type,
                            'details': vr.details,
                            'error': vr.error
                        }
                        for vr in validation_results_list[i]
                    ],
                    'processing_time': avg_time_per_claim,
                    'validation_status': 'COMPLETED'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch validation error: {str(e)}")
            # Return error results for all claims
            return [
                {
                    'claim_id': claim.get('claim_id'),
                    'validation_status': 'ERROR',
                    'error_message': str(e),
                    'processing_time': 0
                }
                for claim in claims
            ]
    
    def validate_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single claim (wrapper for batch processing)."""
        results = self.validate_claims_batch([claim])
        return results[0] if results else {
            'claim_id': claim.get('claim_id'),
            'validation_status': 'ERROR',
            'error_message': 'Validation failed',
            'processing_time': 0
        }
    
    def _extract_features_batch(self, claims: List[Dict[str, Any]]) -> np.ndarray:
        """
        OPTIMIZED: Extract features for multiple claims using vectorized operations.
        """
        try:
            features_list = []
            
            for claim in claims:
                features = []
                
                # Numerical features
                features.extend([
                    float(claim.get('patient_age', 0)),
                    float(claim.get('total_charge_amount', 0)),
                    float(claim.get('diagnosis_count', len(claim.get('diagnoses', [])))),
                    float(claim.get('procedure_count', len(claim.get('procedures', []))))
                ])
                
                # Categorical features (pre-encoded)
                categorical_fields = ['provider_type', 'place_of_service', 'primary_diagnosis']
                for field in categorical_fields:
                    value = claim.get(field, 'UNKNOWN')
                    encoded_value = self._encode_categorical_value(field, value)
                    features.append(float(encoded_value))
                
                features_list.append(features)
            
            return np.array(features_list, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            # Return zero matrix as fallback
            num_features = len(self._get_expected_features())
            return np.zeros((len(claims), num_features), dtype=np.float32)
    
    @functools.lru_cache(maxsize=1000)
    def _encode_categorical_value(self, field: str, value: str) -> int:
        """Cached categorical value encoding."""
        if field in self.model_cache.label_encoders:
            try:
                return self.model_cache.label_encoders[field].transform([value])[0]
            except ValueError:
                return -1  # Unknown category
        return 0
    
    def _get_expected_features(self) -> List[str]:
        """Get expected feature names."""
        return [
            'patient_age', 'total_charge_amount', 'diagnosis_count', 'procedure_count',
            'provider_type', 'place_of_service', 'primary_diagnosis'
        ]
    
    def _build_feature_extractors(self) -> Dict[str, callable]:
        """Build optimized feature extractors."""
        return {
            'numerical': lambda claim: [
                float(claim.get('patient_age', 0)),
                float(claim.get('total_charge_amount', 0)),
                float(len(claim.get('diagnoses', []))),
                float(len(claim.get('procedures', [])))
            ],
            'categorical': lambda claim: [
                self._encode_categorical_value('provider_type', claim.get('provider_type', 'UNKNOWN')),
                self._encode_categorical_value('place_of_service', claim.get('place_of_service', 'UNKNOWN')),
                self._encode_categorical_value('primary_diagnosis', claim.get('primary_diagnosis', 'UNKNOWN'))
            ]
        }
    
    def _load_ml_model(self):
        """Load ML model with error handling."""
        try:
            model_path = self.config.get('ml_model_path', 'models/filter_predictor.pkl')
            success = self.model_cache.load_model(model_path)
            
            if success:
                self.logger.info("ML model loaded successfully")
            else:
                self.logger.warning("ML model not available, running without ML predictions")
                
        except Exception as e:
            self.logger.error(f"Error loading ML model: {str(e)}")
    
    def _load_rules(self):
        """Load and compile pyDatalog rules."""
        try:
            rules = self.db_handler.get_active_filters()
            self.rule_cache.load_rules(rules)
            
            self.logger.info(f"Loaded {len(rules)} validation rules")
            
        except Exception as e:
            self.logger.error(f"Error loading rules: {str(e)}")
    
    def reload_components(self):
        """Reload ML model and rules (for hot reloading)."""
        self._load_ml_model()
        self._load_rules()
        self._encode_categorical_value.cache_clear()  # Clear encoding cache