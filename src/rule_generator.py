### src/rule_generator.py
"""
Rule Generator - Association rule mining for filter generation
"""
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

from database.postgresql_handler import PostgreSQLHandler


class RuleGenerator:
    """Generates association rules for claim validation filters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = None
        
        # Rule generation parameters
        self.min_support = config.get('min_support', 0.01)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.min_lift = config.get('min_lift', 1.1)
        self.data_load_limit = config.get('rule_generation_data_limit', 10000) # Make limit configurable
        
    def initialize_database(self, db_handler: PostgreSQLHandler):
        """Initialize database connection."""
        self.db_handler = db_handler
        
    def generate_rules(self) -> List[Dict[str, Any]]:
        """Generate association rules from historical claims data."""
        try:
            self.logger.info("Starting association rule generation")
            
            # Load transaction data
            transactions = self._load_transaction_data(limit=self.data_load_limit)
            
            if self.data_load_limit < 50000: # Arbitrary threshold for warning
                self.logger.warning(f"Rule generation is using a limited dataset of {self.data_load_limit} records. "
                                    "Generated rules might not be representative of the entire dataset.")
            
            if not transactions:
                self.logger.warning("No transaction data available for rule generation")
                return []
            
            # Convert to binary matrix
            binary_matrix = self._create_binary_matrix(transactions)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(
                binary_matrix, 
                min_support=self.min_support, 
                use_colnames=True
            )
            
            if frequent_itemsets.empty:
                self.logger.warning("No frequent itemsets found")
                return []
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
            
            # Filter rules by lift
            rules = rules[rules['lift'] >= self.min_lift]
            
            # Convert to rule definitions
            rule_definitions = self._convert_to_rule_definitions(rules)
            
            self.logger.info(f"Generated {len(rule_definitions)} association rules")
            return rule_definitions
            
        except Exception as e:
            self.logger.error(f"Rule generation error: {str(e)}", exc_info=True)
            return []
    
    def _load_transaction_data(self, limit: int) -> List[List[str]]:
        """Load transaction data from database."""
        try:
            # Use edi.schema prefix for tables
            query = """
            SELECT 
                c.claim_id,
                COALESCE(array_agg(DISTINCT 'DIAG_' || d.diagnosis_code) FILTER (WHERE d.diagnosis_code IS NOT NULL), ARRAY[]::text[]) as diagnoses,
                COALESCE(array_agg(DISTINCT 'PROC_' || p.procedure_code) FILTER (WHERE p.procedure_code IS NOT NULL), ARRAY[]::text[]) as procedures,
                COALESCE(ARRAY['PROVIDER_' || c.provider_type] FILTER (WHERE c.provider_type IS NOT NULL), ARRAY[]::text[]) as provider_info,
                COALESCE(ARRAY['POS_' || c.place_of_service] FILTER (WHERE c.place_of_service IS NOT NULL), ARRAY[]::text[]) as service_location,
                CASE 
                    WHEN c.patient_age < 18 THEN ARRAY['AGE_CHILD']
                    WHEN c.patient_age >= 65 THEN ARRAY['AGE_SENIOR']
                    ELSE ARRAY['AGE_ADULT']
                END as age_group
            FROM edi.claims c
            LEFT JOIN edi.diagnoses d ON c.claim_id = d.claim_id
            LEFT JOIN edi.procedures p ON c.claim_id = p.claim_id
            WHERE c.processing_status = 'COMPLETED'
            GROUP BY c.claim_id, c.provider_type, c.place_of_service, c.patient_age
            LIMIT %s 
            """
            
            results = self.db_handler.execute_query(query, (limit,))
            
            transactions = []
            for row in results:
                transaction = []
                
                # Add diagnoses
                # COALESCE in SQL ensures row['diagnoses'] is an empty array, not [None] or None
                if row.get('diagnoses'):
                    transaction.extend([d for d in row['diagnoses'] if d])
                
                # Add procedures
                if row.get('procedures'):
                    transaction.extend([p for p in row['procedures'] if p])
                
                # Add provider info
                if row.get('provider_info'):
                    transaction.extend(row['provider_info'])
                
                # Add service location
                if row.get('service_location'):
                    transaction.extend(row['service_location'])
                
                # Add age group
                if row.get('age_group'):
                    transaction.extend(row['age_group'])
                
                if transaction:
                    transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error loading transaction data: {str(e)}", exc_info=True)
            return []
    
    def _create_binary_matrix(self, transactions: List[List[str]]) -> pd.DataFrame:
        """Create binary matrix from transactions."""
        try:
            # Use transaction encoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            
            # Create DataFrame
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            self.logger.info(f"Created binary matrix: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating binary matrix: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _convert_to_rule_definitions(self, rules: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert association rules to rule definitions."""
        rule_definitions = []
        
        try:
            for idx, rule in rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # Create rule definition
                rule_def = {
                    'name': f"AssocRule_{idx}",
                    'description': f"If {', '.join(antecedents)} then {', '.join(consequents)}",
                    'antecedents': antecedents,
                    'consequents': consequents,
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'rule_type': 'ASSOCIATION',
                    'query': self._generate_datalog_query(antecedents, consequents)
                }
                
                rule_definitions.append(rule_def)
            
            return rule_definitions
            
        except Exception as e:
            self.logger.error(f"Error converting rules: {str(e)}", exc_info=True)
            return []
    
    def _generate_datalog_query(self, antecedents: List[str], consequents: List[str]) -> str:
        """Generate pyDatalog query from rule components."""
        try:
            conditions = []
            
            for item in antecedents:
                if item.startswith('DIAG_'):
                    code = item.replace('DIAG_', '')
                    conditions.append(f"has_diagnosis(X, '{code}')")
                elif item.startswith('PROC_'):
                    code = item.replace('PROC_', '')
                    conditions.append(f"has_procedure(X, '{code}')")
                elif item.startswith('PROVIDER_'):
                    provider = item.replace('PROVIDER_', '')
                    conditions.append(f"provider_type(X, '{provider}')")
                elif item.startswith('POS_'):
                    pos = item.replace('POS_', '')
                    conditions.append(f"place_of_service(X, '{pos}')")
                elif item.startswith('AGE_'):
                    age_group = item.replace('AGE_', '')
                    if age_group == 'CHILD':
                        conditions.append("patient_age(X, Age) & (Age < 18)")
                    elif age_group == 'SENIOR':
                        conditions.append("patient_age(X, Age) & (Age >= 65)")
                    else:
                        conditions.append("patient_age(X, Age) & (Age >= 18) & (Age < 65)")
            
            # Create query
            if conditions:
                query = f"rule_applies(X) <= {' & '.join(conditions)}"
                return query
            else:
                return "rule_applies(X) <= True"
                
        except Exception as e:
            self.logger.error(f"Error generating datalog query: {str(e)}", exc_info=True)
            return "rule_applies(X) <= True"
    
    def save_rules_to_database(self, rules: List[Dict[str, Any]]) -> bool:
        """Save generated rules to database."""
        try:
            if not rules:
                return True
            
            # Clear existing generated rules
            self.db_handler.execute_query(
                "DELETE FROM edi.filters WHERE rule_type = 'ASSOCIATION'" # Use schema prefix
            )
            
            # Insert new rules
            for rule in rules:
                query = """
                INSERT INTO edi.filters (
                    filter_name, rule_definition, description, rule_type,
                    support, confidence, lift, active, created_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """
                
                params = (
                    rule['name'],
                    rule['query'],
                    rule['description'],
                    rule['rule_type'],
                    rule['support'],
                    rule['confidence'],
                    rule['lift'],
                    True
                )
                
                self.db_handler.execute_query(query, params)
            
            self.logger.info(f"Saved {len(rules)} rules to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving rules to database: {str(e)}", exc_info=True)
            return False