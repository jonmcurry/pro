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
            self.ml_model_classes_ = None # To store the actual class labels (filter_ids)
            self.model_timestamp = None
            self._model_lock = threading.Lock()
            self._initialized = True
    
    def load_model(self, model_path: str, force_reload: bool = False):
        """Load ML model with caching and hot reloading support."""
        import os
        
        if not os.path.exists(model_path):
            logging.getLogger(__name__).error(f"ML model file not found: {model_path}")
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
                # Load model's class labels, falling back to model.classes_ if available from the model object itself
                # This assumes 'classes_' is saved during training (e.g., model.classes_)
                self.ml_model_classes_ = model_data.get('classes_', getattr(self.ml_model, 'classes_', None))
                self.model_path = model_path
                self.model_timestamp = current_timestamp
                logging.getLogger(__name__).info(f"ML model loaded successfully from: {model_path}. Features: {self.feature_columns}, Classes: {self.ml_model_classes_ is not None}")
                return True
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error loading ML model from {model_path}: {str(e)}", exc_info=True)
                # Reset to ensure a partial load doesn't persist
                self.ml_model = None 
                return False
    
    def predict_batch(self, features: np.ndarray, threshold: float = 0.3, max_filters: int = 10) -> List[List[int]]:
        """OPTIMIZED: Batch prediction for multiple claims."""
        if self.ml_model is None or features.shape[0] == 0: # Handle empty features array
            return [[] for _ in range(features.shape[0])]
        
        with self._model_lock:
            try:
                # Batch prediction - much faster than individual predictions
                all_probabilities = self.ml_model.predict_proba(features)
                
                # Get the actual class labels (filter_ids) the model predicts
                model_classes = self.ml_model_classes_
                if model_classes is None:
                    logging.getLogger(__name__).warning("ml_model_classes_ not available in ModelCache. Falling back to index-based filter IDs. This may be incorrect.")
                    # Fallback: assume index is class_id if classes_ were not loaded (less robust)
                    model_classes = list(range(all_probabilities.shape[1]))
                
                results = []
                for probs_for_one_claim in all_probabilities:
                    scored_filters = []
                    for i, prob in enumerate(probs_for_one_claim):
                        if prob > threshold:
                            # Map model output index to actual filter_id
                            filter_id = model_classes[i] 
                            scored_filters.append({'id': filter_id, 'score': prob})
                    
                    # Sort by probability (score) in descending order
                    scored_filters.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Limit number of filters per claim
                    results.append([sf['id'] for sf in scored_filters[:max_filters]])
                
                return results
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Batch prediction error: {str(e)}", exc_info=True)
                return [[] for _ in range(len(features))]


class RuleCache:
    """OPTIMIZED: pyDatalog rule cache with compilation."""
    
    def __init__(self):
        self.compiled_rules = {}
        self.rule_definitions = {}
        self._rules_lock = threading.Lock() # Ensures thread-safe access to pyDatalog global state
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
            self.logger.error(f"Error setting up predicates: {str(e)}", exc_info=True)
    
    def load_rules(self, rules: List[Dict[str, Any]]):
        """Load and compile rules for faster execution."""
        with self._rules_lock: # Protect access to shared rule structures
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
                    self.logger.error(f"Error compiling rule {filter_id}: {str(e)}", exc_info=True)
    
    def evaluate_rules_batch(self, claims: List[Dict[str, Any]], filter_ids_list: List[List[int]]) -> List[List[ValidationResult]]:
        """OPTIMIZED: Batch rule evaluation for multiple claims."""
        results = []
        
        # The _rules_lock is crucial here because pyDatalog.clear() and fact assertion
        # modify global state within pyDatalog. If multiple threads from
        # OptimizedProcessingOrchestrator call this method on the same ClaimValidator instance,
        # they would interfere with each other's pyDatalog state without this lock.
        with self._rules_lock:
            # Setup facts for all claims at once
            self._setup_batch_facts(claims) # This clears and re-asserts facts
            
            for i, (claim, filter_ids) in enumerate(zip(claims, filter_ids_list)):
                claim_results = []
                claim_id = claim.get('claim_id', f'claim_{i}') # Use a unique ID for the claim in Datalog
                
                for filter_id in filter_ids:
                    if filter_id in self.compiled_rules:
                        # Pass the unique claim_id for this specific claim to _evaluate_single_rule
                        result = self._evaluate_single_rule(claim_id, filter_id)
                        claim_results.append(result)
                
                results.append(claim_results)
        
        return results
    
    def _setup_batch_facts(self, claims: List[Dict[str, Any]]):
        """OPTIMIZED: Setup facts for all claims in batch."""
        try:
            # Clear existing facts
            pyDatalog.clear()
            self._setup_predicates() # Re-create terms after clearing
            
            # Add facts for all claims
            for i, claim in enumerate(claims):
                # Use a unique ID for each claim within this batch's Datalog context
                # This is important if rules refer to a generic 'X' that needs to be specific per claim.
                # The claim_id from the data is used here.
                datalog_claim_id = claim.get('claim_id', f'claim_{i}') 
                
                # Diagnosis facts
                for diagnosis in claim.get('diagnoses', []):
                    if diagnosis.get('code'):
                        pyDatalog.assert_fact('has_diagnosis', datalog_claim_id, diagnosis['code'])
                
                # Procedure facts
                for procedure in claim.get('procedures', []):
                    if procedure.get('code'):
                        pyDatalog.assert_fact('has_procedure', datalog_claim_id, procedure['code'])
                
                # Patient facts
                pyDatalog.assert_fact('patient_age', datalog_claim_id, claim.get('patient_age', 0))
                pyDatalog.assert_fact('charge_amount', datalog_claim_id, claim.get('total_charge_amount', 0))
                
                # Provider facts
                pyDatalog.assert_fact('provider_type', datalog_claim_id, claim.get('provider_type', ''))
                pyDatalog.assert_fact('place_of_service', datalog_claim_id, claim.get('place_of_service', ''))
                
        except Exception as e:
            self.logger.error(f"Error setting up batch facts: {str(e)}", exc_info=True)
    
    def _evaluate_single_rule(self, datalog_claim_id: str, filter_id: int) -> ValidationResult:
        """Evaluate a single rule against a claim, using the specific datalog_claim_id."""
        try:
            rule_info = self.compiled_rules[filter_id]
            
            # The rule definition (e.g., "rule_applies(X) <= has_diagnosis(X, 'A01')")
            # needs to be asserted for the Datalog engine to use it.
            # We assert it here, specific to the current evaluation context.
            # This assumes rule_info['query'] is a complete Datalog rule string.
            # pyDatalog.load allows loading rule strings.
            pyDatalog.load(rule_info['query'])
            
            # Create a specific query for this claim using its datalog_claim_id
            # This asks if the 'rule_applies' predicate is true for this specific claim.
            query_for_claim = f"rule_applies('{datalog_claim_id}')"
            
            # Execute rule
            results = pyDatalog.ask(query_for_claim)
            
            return ValidationResult(
                filter_id=filter_id,
                filter_name=rule_info['name'],
                passed=bool(results), # True if the query returns any results
                rule_type='DATALOG',
                details=rule_info['description']
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {filter_id} for claim {datalog_claim_id}: {str(e)}", exc_info=True)
            return ValidationResult(
                filter_id=filter_id,
                filter_name=self.compiled_rules.get(filter_id, {}).get('name', f'Filter_{filter_id}'),
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
    
    def validate_claims_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Validate multiple claims in batch for better performance.
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract features for all claims in batch
            features_matrix, extraction_success = self._extract_features_batch(claims)
            
            if not extraction_success:
                self.logger.error("Feature extraction failed for the batch. ML prediction will be skipped or use empty features.")
                # Handle case where feature extraction itself fails for the whole batch
                # For now, predict_batch will handle empty/zero features_matrix if it occurs.

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
                    'predicted_filters': predicted_filters_list[i] if i < len(predicted_filters_list) else [],
                    'validation_results': [
                        {
                            'filter_id': vr.filter_id,
                            'filter_name': vr.filter_name,
                            'passed': vr.passed,
                            'rule_type': vr.rule_type,
                            'details': vr.details,
                            'error': vr.error
                        }
                        for vr in (validation_results_list[i] if i < len(validation_results_list) else [])
                    ],
                    'processing_time': avg_time_per_claim,
                    'validation_status': 'COMPLETED'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch validation error: {str(e)}", exc_info=True)
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
    
    def _extract_features_batch(self, claims: List[Dict[str, Any]]) -> Tuple[np.ndarray, bool]:
        """
        OPTIMIZED: Extract features for multiple claims using vectorized operations,
        aligning with the features defined in the loaded ML model.
        Returns a tuple: (features_matrix, success_flag).
        """
        try:
            expected_model_features = self.model_cache.feature_columns
            if not expected_model_features:
                self.logger.error("Model feature_columns not loaded. Cannot extract features for ML prediction.")
                # Return an empty array of shape (num_claims, 0 features) and False for success
                return np.array([[] for _ in claims], dtype=np.float32), False

            all_claims_feature_vectors = []
            for claim in claims:
                feature_vector = []
                for feature_name in expected_model_features:
                    value_to_append = 0.0  # Default for missing or unhandled features

                    # Numerical features directly from claim or calculated
                    if feature_name == 'patient_age':
                        value_to_append = float(claim.get('patient_age', 0))
                    elif feature_name == 'total_charge_amount':
                        value_to_append = float(claim.get('total_charge_amount', 0))
                    elif feature_name == 'diagnosis_count':
                        # Use pre-calculated 'diagnosis_count' if available from parser, else calculate
                        value_to_append = float(claim.get('diagnosis_count', len(claim.get('diagnoses', []))))
                    elif feature_name == 'procedure_count':
                        # Use pre-calculated 'procedure_count' if available from parser, else calculate
                        value_to_append = float(claim.get('procedure_count', len(claim.get('procedures', []))))
                    
                    # Categorical features that need encoding
                    # Assumes feature_name in expected_model_features is like 'provider_type_encoded'
                    elif feature_name.endswith('_encoded'):
                        raw_field_name = feature_name.replace('_encoded', '')
                        # Check if an encoder exists for this raw field name
                        if raw_field_name in self.model_cache.label_encoders:
                            # 'primary_diagnosis' is prepared by ClaimParser.
                            # If 'primary_diagnosis_encoded' is an expected feature, it will be handled here.
                            claim_value = claim.get(raw_field_name, 'UNKNOWN')
                            value_to_append = float(self._encode_categorical_value(raw_field_name, claim_value))
                        else:
                            self.logger.warning(
                                f"Encoder not found for base field '{raw_field_name}' (derived from '{feature_name}') "
                                f"for claim {claim.get('claim_id', 'N/A')}. Using 0.0."
                            )
                    else:
                        # This case might occur if a raw feature name is in expected_model_features
                        # but isn't one of the explicitly handled numerical ones or doesn't end with _encoded.
                        # This could be an oversight or a feature that's numerical but not handled above.
                        self.logger.warning(
                            f"Unrecognized or unhandled feature '{feature_name}' in model's expected features "
                            f"for claim {claim.get('claim_id', 'N/A')}. Using 0.0."
                        )
                    feature_vector.append(value_to_append)
                all_claims_feature_vectors.append(feature_vector)
            
            return np.array(all_claims_feature_vectors, dtype=np.float32), True
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}", exc_info=True)
            num_expected_features = len(self.model_cache.feature_columns) if self.model_cache.feature_columns else 0
            return np.zeros((len(claims), num_expected_features), dtype=np.float32), False
    
    @functools.lru_cache(maxsize=1000)
    def _encode_categorical_value(self, field: str, value: str) -> int:
        """Cached categorical value encoding."""
        if field in self.model_cache.label_encoders:
            try:
                return self.model_cache.label_encoders[field].transform([value])[0]
            except ValueError: # Value not seen during fit
                # Try to fit the new value if dynamic updates are allowed, or return unknown
                # For simplicity, returning -1 for unknown, consistent with training.
                return -1  # Unknown category
        self.logger.warning(f"No label encoder found for field '{field}'. Returning 0.")
        return 0 # Should ideally not happen if model_cache.label_encoders is comprehensive
    
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
            self.logger.error(f"Error loading ML model: {str(e)}", exc_info=True)
    
    def _load_rules(self):
        """Load and compile pyDatalog rules."""
        try:
            rules = self.db_handler.get_active_filters() # Fetches from edi.filters
            self.rule_cache.load_rules(rules)
            
            self.logger.info(f"Loaded {len(rules)} validation rules")
            
        except Exception as e:
            self.logger.error(f"Error loading rules: {str(e)}", exc_info=True)
    
    def reload_components(self):
        """Reload ML model and rules (for hot reloading)."""
        self.logger.info("Reloading validator components...")
        self._load_ml_model() # This will use ModelCache.load_model which checks timestamp
        self._load_rules()
        self._encode_categorical_value.cache_clear()  # Clear encoding cache as encoders might change
        self.logger.info("Validator components reloaded.")

