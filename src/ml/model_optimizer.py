# src/ml/model_optimizer.py
"""
Optimized ML Model Management with Caching and Hot Reloading
"""
import logging
import pickle
import numpy as np
import threading
import time
import os
from typing import Dict, List, Any, Optional
import joblib
from concurrent.futures import ThreadPoolExecutor
import psutil


class OptimizedModelManager:
    """Thread-safe ML model manager with advanced caching."""
    
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
            self.logger = logging.getLogger(__name__)
            self.models = {}
            self.model_metadata = {}
            self._model_locks = {}
            self._prediction_cache = {}
            self._cache_lock = threading.Lock()
            self._initialized = True
    
    def load_model(self, model_path: str, model_name: str = 'default') -> bool:
        """Load model with optimizations."""
        try:
            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.Lock()
            
            with self._model_locks[model_name]:
                # Check if model needs reloading
                if self._should_reload_model(model_path, model_name):
                    self.logger.info(f"Loading/reloading model: {model_name}")
                    
                    # Try joblib first (faster), then pickle
                    model_data = self._load_model_file(model_path)
                    
                    if model_data:
                        self.models[model_name] = model_data
                        self.model_metadata[model_name] = {
                            'path': model_path,
                            'loaded_time': time.time(),
                            'file_mtime': os.path.getmtime(model_path),
                            'memory_usage': self._estimate_model_memory(model_data)
                        }
                        
                        # Clear prediction cache for this model
                        self._clear_prediction_cache(model_name)
                        
                        self.logger.info(f"Model {model_name} loaded successfully")
                        return True
                    
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def predict_batch(self, features: np.ndarray, model_name: str = 'default', 
                     use_cache: bool = True) -> Optional[np.ndarray]:
        """Optimized batch prediction with caching."""
        if model_name not in self.models:
            return None
        
        try:
            # Generate cache key
            cache_key = None
            if use_cache:
                cache_key = self._generate_cache_key(features, model_name)
                cached_result = self._get_cached_prediction(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Perform prediction
            with self._model_locks[model_name]:
                model_data = self.models[model_name]
                model = model_data['model']
                
                # Optimize prediction based on model type
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(features)
                else:
                    predictions = model.predict(features)
            
            # Cache result
            if use_cache and cache_key:
                self._cache_prediction(cache_key, predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error for model {model_name}: {str(e)}")
            return None
    
    def get_feature_extractors(self, model_name: str = 'default') -> Optional[Dict]:
        """Get feature extractors for the model."""
        if model_name in self.models:
            return self.models[model_name].get('feature_extractor')
        return None
    
    def get_label_encoders(self, model_name: str = 'default') -> Optional[Dict]:
        """Get label encoders for the model."""
        if model_name in self.models:
            return self.models[model_name].get('encoders', {})
        return {}
    
    def _should_reload_model(self, model_path: str, model_name: str) -> bool:
        """Check if model should be reloaded."""
        if model_name not in self.models:
            return True
        
        if not os.path.exists(model_path):
            return False
        
        # Check file modification time
        current_mtime = os.path.getmtime(model_path)
        stored_mtime = self.model_metadata[model_name].get('file_mtime', 0)
        
        return current_mtime > stored_mtime
    
    def _load_model_file(self, model_path: str) -> Optional[Dict]:
        """Load model file with multiple methods."""
        try:
            # Try joblib first (usually faster)
            joblib_path = model_path.replace('.pkl', '_joblib.pkl')
            if os.path.exists(joblib_path):
                return joblib.load(joblib_path)
            
            # Fall back to pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading model file: {str(e)}")
            return None
    
    def _estimate_model_memory(self, model_data: Dict) -> float:
        """Estimate memory usage of loaded model."""
        try:
            import sys
            return sys.getsizeof(model_data) / (1024 * 1024)  # MB
        except:
            return 0.0
    
    def _generate_cache_key(self, features: np.ndarray, model_name: str) -> str:
        """Generate cache key for predictions."""
        import hashlib
        feature_hash = hashlib.md5(features.tobytes()).hexdigest()[:16]
        return f"{model_name}_{feature_hash}_{features.shape}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[np.ndarray]:
        """Get cached prediction."""
        with self._cache_lock:
            return self._prediction_cache.get(cache_key)
    
    def _cache_prediction(self, cache_key: str, prediction: np.ndarray):
        """Cache prediction with memory management."""
        with self._cache_lock:
            # Simple cache size management
            if len(self._prediction_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._prediction_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._prediction_cache[key]
            
            self._prediction_cache[cache_key] = prediction.copy()
    
    def _clear_prediction_cache(self, model_name: str):
        """Clear prediction cache for specific model."""
        with self._cache_lock:
            keys_to_remove = [k for k in self._prediction_cache.keys() if k.startswith(model_name)]
            for key in keys_to_remove:
                del self._prediction_cache[key]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = {
            'loaded_models': list(self.models.keys()),
            'total_memory_usage_mb': sum(
                meta.get('memory_usage', 0) for meta in self.model_metadata.values()
            ),
            'cache_size': len(self._prediction_cache),
            'models_detail': {}
        }
        
        for model_name, metadata in self.model_metadata.items():
            stats['models_detail'][model_name] = {
                'loaded_time': metadata.get('loaded_time'),
                'memory_usage_mb': metadata.get('memory_usage', 0),
                'file_path': metadata.get('path')
            }
        
        return stats
    
    def cleanup_cache(self):
        """Clean up prediction cache and unused models."""
        with self._cache_lock:
            self._prediction_cache.clear()
        
        # Clean up models if memory usage is high
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            self.logger.warning(f"High memory usage ({memory_percent}%), clearing model caches")
            for model_name in list(self.models.keys()):
                self._clear_prediction_cache(model_name)


class VectorizedFeatureExtractor:
    """Optimized feature extractor using vectorized operations."""
    
    def __init__(self, model_manager: OptimizedModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
    
    def extract_features_batch(self, claims: List[Dict[str, Any]], 
                             model_name: str = 'default') -> np.ndarray:
        """Extract features for multiple claims using vectorized operations."""
        try:
            # Get encoders from model
            encoders = self.model_manager.get_label_encoders(model_name)
            
            # Pre-allocate feature matrix
            n_claims = len(claims)
            n_features = self._get_feature_count()
            features = np.zeros((n_claims, n_features), dtype=np.float32)
            
            # Vectorized extraction
            self._extract_numerical_features(claims, features)
            self._extract_categorical_features(claims, features, encoders)
            self._extract_derived_features(claims, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            # Return zero matrix as fallback
            return np.zeros((len(claims), self._get_feature_count()), dtype=np.float32)
    
    def _extract_numerical_features(self, claims: List[Dict], features: np.ndarray):
        """Extract numerical features using vectorized operations."""
        for i, claim in enumerate(claims):
            features[i, 0] = float(claim.get('patient_age', 0))
            features[i, 1] = float(claim.get('total_charge_amount', 0))
            features[i, 2] = float(len(claim.get('diagnoses', [])))
            features[i, 3] = float(len(claim.get('procedures', [])))
    
    def _extract_categorical_features(self, claims: List[Dict], 
                                    features: np.ndarray, encoders: Dict):
        """Extract categorical features with efficient encoding."""
        categorical_fields = ['provider_type', 'place_of_service']
        
        for field_idx, field in enumerate(categorical_fields):
            feature_col = 4 + field_idx  # After numerical features
            
            if field in encoders:
                encoder = encoders[field]
                
                # Extract all values for this field
                values = [claim.get(field, 'UNKNOWN') for claim in claims]
                
                # Batch encode
                try:
                    encoded_values = encoder.transform(values)
                    features[:, feature_col] = encoded_values.astype(np.float32)
                except ValueError:
                    # Handle unknown categories
                    for i, value in enumerate(values):
                        try:
                            features[i, feature_col] = encoder.transform([value])[0]
                        except ValueError:
                            features[i, feature_col] = -1  # Unknown category
    
    def _extract_derived_features(self, claims: List[Dict], features: np.ndarray):
        """Extract derived features."""
        for i, claim in enumerate(claims):
            # Primary diagnosis encoding
            diagnoses = claim.get('diagnoses', [])
            primary_diag = self._get_primary_diagnosis(diagnoses)
            features[i, 6] = hash(primary_diag) % 10000 if primary_diag else 0
            
            # Charge per procedure
            procedure_count = len(claim.get('procedures', []))
            total_charge = float(claim.get('total_charge_amount', 0))
            features[i, 7] = total_charge / max(procedure_count, 1)
    
    def _get_primary_diagnosis(self, diagnoses: List[Dict]) -> str:
        """Get primary diagnosis code."""
        try:
            # Look for principal diagnosis
            for diag in diagnoses:
                if diag.get('is_principal', False):
                    return diag.get('code', '')
            
            # Get first by sequence
            if diagnoses:
                sorted_diags = sorted(diagnoses, key=lambda x: x.get('sequence', 999))
                return sorted_diags[0].get('code', '')
            
            return ''
        except:
            return ''
    
    def _get_feature_count(self) -> int:
        """Get total number of features."""
        return 8  # 4 numerical + 2 categorical + 2 derived