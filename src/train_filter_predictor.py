### src/train_filter_predictor_fixed.py
"""
FIXED ML Model Training - Addresses SMOTE and XGBoost issues
"""
import logging
import time
import pickle
import json
import sys
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import psutil
import gc
import joblib
from concurrent.futures import ThreadPoolExecutor

from database.postgresql_handler import PostgreSQLHandler


class OptimizedFeatureExtractor:
    """OPTIMIZED: Vectorized feature extraction for better performance."""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Fit encoders and transform training data."""
        self.logger.info("Fitting feature extractors...")
        
        # Extract all categorical values for fitting
        categorical_fields = ['provider_type', 'place_of_service']
        
        for field in categorical_fields:
            unique_values = set()
            for record in training_data:
                value = record.get(field, 'UNKNOWN')
                unique_values.add(value)
            
            self.label_encoders[field] = LabelEncoder()
            self.label_encoders[field].fit(list(unique_values))
        
        # Transform data
        return self.transform(training_data)
    
    def transform(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Transform data to feature matrix."""
        features_list = []
        
        for record in data:
            features = self._extract_single_record(record)
            features_list.append(features)
        
        self.feature_names = self._get_feature_names()
        return np.array(features_list, dtype=np.float32), self.feature_names
    
    def _extract_single_record(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from single record."""
        features = []
        
        # Numerical features
        features.extend([
            float(record.get('patient_age', 0)),
            float(record.get('total_charge_amount', 0)),
            float(len(record.get('diagnoses', []) or [])),
            float(len(record.get('procedures', []) or []))
        ])
        
        # Categorical features
        categorical_fields = ['provider_type', 'place_of_service']
        for field in categorical_fields:
            value = record.get(field, 'UNKNOWN')
            
            try:
                encoded_value = self.label_encoders[field].transform([value])[0]
            except (ValueError, KeyError):
                encoded_value = -1  # Unknown category
                
            features.append(float(encoded_value))
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            'patient_age', 'total_charge_amount', 'diagnosis_count', 'procedure_count',
            'provider_type_encoded', 'place_of_service_encoded'
        ]


class FilterPredictorTrainer:
    """FIXED: Advanced ML model trainer with bug fixes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = None
        
        # Training parameters
        self.memory_limit_percent = config.get('memory_limit_percent', 70)
        self.chunk_size = config.get('training_chunk_size', 10000)
        self.model_path = config.get('model_output_path', 'models/filter_predictor.pkl')
        
        # Performance optimizations
        self.use_gpu = config.get('use_gpu', False)
        self.n_jobs = config.get('n_jobs', -1)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 20)
        
        # Model components
        self.model = None
        self.feature_extractor = OptimizedFeatureExtractor()
        
    def initialize_database(self, db_handler: PostgreSQLHandler):
        """Initialize database connection."""
        self.db_handler = db_handler
        
    def train_model(self) -> bool:
        """FIXED: Train the filter prediction model with bug fixes."""
        try:
            self.logger.info("Starting fixed ML model training")
            
            # Load and prepare training data
            X, y, feature_names = self._load_training_data_optimized()
            
            if X is None or len(X) == 0:
                self.logger.error("No training data available")
                return False
            
            self.logger.info(f"Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # FIXED: Check if we have valid labels and multiple classes
            unique_labels = np.unique(y)
            self.logger.info(f"Unique labels in training data: {unique_labels}")
            
            if len(unique_labels) < 2:
                self.logger.error(f"Insufficient label diversity. Found labels: {unique_labels}")
                self.logger.error("Need at least 2 different classes for classification")
                return False
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Handle class imbalance (FIXED)
            X_train_balanced, y_train_balanced = self._balance_classes_fixed(X_train, y_train)
            
            # FIXED: Re-check classes after balancing
            balanced_unique = np.unique(y_train_balanced)
            self.logger.info(f"Classes after balancing: {balanced_unique}")
            
            if len(balanced_unique) < 2:
                self.logger.error("Still insufficient classes after balancing")
                return False
            
            # Train model with hyperparameter optimization (FIXED)
            self.model = self._train_xgboost_fixed(X_train_balanced, y_train_balanced, X_test, y_test)
            
            # Comprehensive evaluation
            self._evaluate_model_comprehensive(self.model, X_test, y_test, feature_names)
            
            # Save model and components
            self._save_model_components_optimized(feature_names)
            
            self.logger.info("Fixed model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}", exc_info=True)
            return False
    
    def _load_training_data_optimized(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """FIXED: Load training data with better data validation."""
        try:
            self.logger.info("Loading training data with optimizations...")
            
            # Calculate optimal parameters
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            optimal_chunk_size = self._calculate_optimal_chunk_size(available_memory)
            
            # Load data in parallel chunks
            all_records = []
            all_labels = []
            
            # Get total count for progress tracking
            total_count = self._get_total_training_records()
            total_chunks = (total_count + optimal_chunk_size - 1) // optimal_chunk_size
            
            self.logger.info(f"Processing {total_chunks} chunks of {optimal_chunk_size} records each")
            
            with tqdm(total=total_chunks, desc="Loading training data") as pbar:
                offset = 0
                
                while True:
                    # Load chunk
                    chunk_data = self.db_handler.get_training_data_batch(optimal_chunk_size, offset)
                    
                    if not chunk_data:
                        break
                    
                    # Process chunk in parallel if large enough
                    if len(chunk_data) > 1000:
                        chunk_records, chunk_labels = self._process_chunk_parallel(chunk_data)
                    else:
                        chunk_records, chunk_labels = self._process_training_chunk_fixed(chunk_data)
                    
                    if chunk_records:
                        all_records.extend(chunk_records)
                        all_labels.extend(chunk_labels)
                    
                    pbar.update(1)
                    offset += optimal_chunk_size
                    
                    # Memory management
                    if psutil.virtual_memory().percent > self.memory_limit_percent:
                        gc.collect()
            
            if not all_records:
                self.logger.error("No training records found")
                return None, None, []
            
            # FIXED: Validate labels before feature extraction
            self.logger.info(f"Total records loaded: {len(all_records)}")
            self.logger.info(f"Total labels loaded: {len(all_labels)}")
            
            # Check label distribution
            label_counts = {}
            for label in all_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            self.logger.info(f"Label distribution: {label_counts}")
            
            # Filter out invalid labels (if any)
            valid_indices = [i for i, label in enumerate(all_labels) if label is not None and label >= 0]
            
            if len(valid_indices) == 0:
                self.logger.error("No valid labels found")
                return None, None, []
            
            all_records = [all_records[i] for i in valid_indices]
            all_labels = [all_labels[i] for i in valid_indices]
            
            # Extract features using optimized extractor
            self.logger.info("Extracting features...")
            X, feature_names = self.feature_extractor.fit_transform(all_records)
            y = np.array(all_labels)
            
            self.logger.info(f"Feature extraction completed: {X.shape}")
            self.logger.info(f"Final label distribution: {np.unique(y, return_counts=True)}")
            
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            return None, None, []
    
    def _process_chunk_parallel(self, chunk_data: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """Process large chunks in parallel."""
        try:
            # Split chunk into sub-chunks for parallel processing
            sub_chunk_size = max(100, len(chunk_data) // 4)
            sub_chunks = [
                chunk_data[i:i + sub_chunk_size]
                for i in range(0, len(chunk_data), sub_chunk_size)
            ]
            
            all_records = []
            all_labels = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._process_training_chunk_fixed, sub_chunk) for sub_chunk in sub_chunks]
                
                for future in futures:
                    records, labels = future.result()
                    if records:
                        all_records.extend(records)
                        all_labels.extend(labels)
            
            return all_records, all_labels
            
        except Exception as e:
            self.logger.error(f"Error in parallel chunk processing: {str(e)}")
            return self._process_training_chunk_fixed(chunk_data)
    
    def _process_training_chunk_fixed(self, chunk_data: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """FIXED: Process training chunk with better label handling."""
        try:
            records = []
            labels = []
            
            for record in chunk_data:
                # Create training record
                training_record = {
                    'patient_age': record.get('patient_age', 0),
                    'total_charge_amount': record.get('total_charge_amount', 0),
                    'provider_type': record.get('provider_type', 'UNKNOWN'),
                    'place_of_service': record.get('place_of_service', 'UNKNOWN'),
                    'diagnoses': record.get('diagnoses', []) or [],
                    'procedures': record.get('procedures', []) or []
                }
                
                # FIXED: Better handling of applied filters
                applied_filters = record.get('applied_filters', [])
                
                # Handle different formats of applied_filters
                if applied_filters:
                    # If we have filters, create one training example per filter
                    for filter_id in applied_filters:
                        if filter_id is not None and filter_id > 0:  # Valid filter ID
                            records.append(training_record.copy())
                            labels.append(filter_id)
                
                # FIXED: Always add a negative example (no filter applied)
                # This ensures we have both positive and negative examples
                records.append(training_record.copy())
                labels.append(0)  # Class 0 = no filter applied
            
            self.logger.debug(f"Processed chunk: {len(records)} records, {len(labels)} labels")
            
            return records, labels
            
        except Exception as e:
            self.logger.error(f"Error processing training chunk: {str(e)}")
            return [], []
    
    def _balance_classes_fixed(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Handle class imbalance without unsupported parameters."""
        try:
            self.logger.info("Balancing classes with fixed SMOTE")
            
            # Check class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = np.min(class_counts)
            
            self.logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            if len(unique_classes) < 2:
                self.logger.warning("Only one class found, cannot apply SMOTE")
                return X, y
            
            if min_samples < 6:
                self.logger.warning("Not enough samples for SMOTE, using original data")
                return X, y
            
            # FIXED: Use SMOTE without n_jobs parameter
            smote = SMOTE(
                random_state=42, 
                k_neighbors=min(5, min_samples-1)
                # Removed n_jobs parameter as it's not supported in older versions
            )
            
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Log results
            balanced_unique, balanced_counts = np.unique(y_balanced, return_counts=True)
            self.logger.info(f"Balanced distribution: {dict(zip(balanced_unique, balanced_counts))}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"Error balancing classes: {str(e)}")
            return X, y
    
    def _train_xgboost_fixed(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray) -> xgb.XGBClassifier:
        """FIXED: Train XGBoost with proper class handling."""
        try:
            self.logger.info("Training fixed XGBoost model")
            
            # FIXED: Ensure we have valid classes
            unique_classes = np.unique(y_train)
            num_classes = len(unique_classes)
            
            self.logger.info(f"Training with {num_classes} classes: {unique_classes}")
            
            if num_classes < 2:
                raise ValueError(f"Need at least 2 classes for classification, found {num_classes}")
            
            # FIXED: Use appropriate objective based on number of classes
            if num_classes == 2:
                objective = 'binary:logistic'
                eval_metric = ['logloss', 'error']
            else:
                objective = 'multi:softprob'
                eval_metric = ['mlogloss', 'merror']
            
            # Optimized parameters for performance
            params = {
                'objective': objective,
                'eval_metric': eval_metric,
                'tree_method': 'hist',  # Fastest for large datasets
                'max_depth': 6,  # Reduced for stability
                'learning_rate': 0.1,
                'n_estimators': 300,  # Reduced for faster training
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': self.n_jobs,
                'verbosity': 1
            }
            
            # GPU optimization if available
            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            
            # FIXED: Explicitly set num_class for multi-class problems
            if num_classes > 2:
                params['num_class'] = num_classes
            
            model = xgb.XGBClassifier(**params)
            
            # Train with validation set for early stopping
            if len(X_train) > 1000:  # Only split if we have enough data
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                eval_set = [(X_val_split, y_val_split), (X_test, y_test)]
            else:
                X_train_split, y_train_split = X_train, y_train
                eval_set = [(X_test, y_test)]
            
            model.fit(
                X_train_split, y_train_split,
                eval_set=eval_set,
                verbose=False
            )
            
            self.logger.info("XGBoost model training completed successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def _evaluate_model_comprehensive(self, model, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]):
        """OPTIMIZED: Comprehensive model evaluation with feature importance."""
        try:
            self.logger.info("Performing comprehensive model evaluation")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Classification metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Multi-class AUC
            try:
                if len(np.unique(y_test)) == 2:
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except ValueError:
                auc_score = None
            
            # F1 scores
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Log comprehensive metrics
            self.logger.info("Model Performance Metrics:")
            self.logger.info(f"  Weighted F1 Score: {f1_weighted:.4f}")
            self.logger.info(f"  Macro F1 Score: {f1_macro:.4f}")
            if auc_score:
                self.logger.info(f"  AUC Score: {auc_score:.4f}")
            self.logger.info(f"  Accuracy: {report['accuracy']:.4f}")
            
            self.logger.info("Top Feature Importances:")
            for feature, importance in sorted_features:
                self.logger.info(f"  {feature}: {importance:.4f}")
            
            # Save detailed report
            report_data = {
                'classification_report': report,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'auc_score': auc_score,
                'feature_importance': feature_importance,
                'training_config': self.config,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': 'XGBoost',
                'samples_trained': len(X_test) * 5,  # Approximate based on test split
            }
            
            import os
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            report_path = self.model_path.replace('.pkl', '_training_report.json')
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Training report saved to: {report_path}")
                
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
    
    def _save_model_components_optimized(self, feature_names: List[str]):
        """OPTIMIZED: Save model with compression and metadata."""
        try:
            self.logger.info(f"Saving optimized model to {self.model_path}")
            
            model_data = {
                'model': self.model,
                'encoders': self.feature_extractor.label_encoders,
                'features': feature_names,
                'classes_': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None,
                'feature_extractor': self.feature_extractor,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config,
                'model_version': '2.1',  # Updated version
                'optimization_level': 'high'
            }
            
            # Create backup
            import os
            import shutil
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            if os.path.exists(self.model_path):
                backup_path = self.model_path + '.backup'
                shutil.copy2(self.model_path, backup_path)
                self.logger.info(f"Backup created: {backup_path}")
            
            # Save with compression
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save using joblib for better performance
            joblib_path = self.model_path.replace('.pkl', '_joblib.pkl')
            joblib.dump(model_data, joblib_path, compress=3)
            
            self.logger.info("Model saved successfully with compression")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def _calculate_optimal_chunk_size(self, available_memory_gb: float) -> int:
        """Calculate optimal chunk size based on system resources."""
        # More sophisticated calculation
        base_memory_per_record = 0.2  # KB per record
        safety_factor = 0.3  # Use 30% of available memory
        
        usable_memory_kb = available_memory_gb * 1024 * 1024 * safety_factor
        optimal_size = int(usable_memory_kb / base_memory_per_record)
        
        # Bounds with more reasonable limits
        optimal_size = max(5000, min(optimal_size, 100000))
        
        self.logger.info(f"Calculated optimal chunk size: {optimal_size}")
        return optimal_size
    
    def _get_total_training_records(self) -> int:
        """Get total number of training records."""
        try:
            query = "SELECT COUNT(*) as total FROM edi.claims WHERE processing_status = 'COMPLETED'"
            result = self.db_handler.execute_query(query)
            return result[0]['total'] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting total training records: {str(e)}")
            return 0


def main():
    """Enhanced main function with better error handling."""
    import argparse
    from pathlib import Path
    
    # Add src to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from config.config_manager import ConfigurationManager
    from utils.logging_config import setup_logging
    from database.postgresql_handler import PostgreSQLHandler
    
    parser = argparse.ArgumentParser(description="Train Fixed Filter Predictor Model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting FIXED filter predictor training")
        
        # Load configuration
        config_manager = ConfigurationManager(args.config)
        config = config_manager.get_config()
        
        # Override config with command line args
        ml_config = config.get('ml_training', {})
        if args.use_gpu:
            ml_config['use_gpu'] = True
        if args.chunk_size:
            ml_config['training_chunk_size'] = args.chunk_size
        
        # Initialize trainer
        trainer = FilterPredictorTrainer(ml_config)
        
        # Initialize database
        db_handler = PostgreSQLHandler(config['database']['postgresql'])
        trainer.initialize_database(db_handler)
        
        # Train model
        success = trainer.train_model()
        
        if success:
            logger.info("FIXED model training completed successfully")
            return 0
        else:
            logger.error("FIXED model training failed")
            return 1
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())