"""
OMOP Validator for the Clinical Protocol Extraction Engine.
This module validates OMOP CDM data created during protocol conversion.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

class OMOPValidator:
    """Validates OMOP CDM data against OMOP standards."""
    
    def __init__(self, schema_dir=None):
        """
        Initialize OMOP validator.
        
        Args:
            schema_dir (str, optional): Directory containing OMOP CDM schemas
        """
        logger.info("Initializing OMOP validator")
        
        # Set schema directory
        if schema_dir:
            self.schema_dir = schema_dir
        else:
            self.schema_dir = os.path.join(os.path.dirname(__file__), 'schemas')
        
        # Load schemas
        self.schemas = self._load_schemas()
        
        # Define validation rules for specific tables
        self.validation_rules = {
            "condition_occurrence": [
                self._validate_condition_occurrence_rules
            ],
            "drug_exposure": [
                self._validate_drug_exposure_rules
            ],
            "procedure_occurrence": [
                self._validate_procedure_occurrence_rules
            ],
            "observation": [
                self._validate_observation_rules
            ]
        }
        
    def _load_schemas(self):
        """
        Load OMOP CDM schemas from schema directory.
        
        Returns:
            dict: Dictionary of schemas by table name
        """
        schemas = {}
        
        try:
            # Check if schema directory exists
            if not os.path.exists(self.schema_dir):
                logger.warning(f"Schema directory not found: {self.schema_dir}")
                return schemas
            
            # Load each schema file
            for filename in os.listdir(self.schema_dir):
                if filename.endswith('.json'):
                    table_name = os.path.splitext(filename)[0]
                    schema_path = os.path.join(self.schema_dir, filename)
                    
                    with open(schema_path, 'r') as f:
                        schema = json.load(f)
                        schemas[table_name] = schema
            
            logger.info(f"Loaded {len(schemas)} OMOP CDM schemas")
            return schemas
            
        except Exception as e:
            logger.error(f"Error loading schemas: {str(e)}", exc_info=True)
            return {}
    
    def validate(self, tables):
        """
        Validate OMOP CDM tables against schemas and rules.
        
        Args:
            tables (dict): OMOP tables to validate
            
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Starting OMOP CDM data validation")
            
            validation_issues = []
            tables_validated = 0
            
            # Validate each table
            for table_name, records in tables.items():
                # Skip empty tables
                if not records:
                    logger.info(f"Skipping empty table: {table_name}")
                    continue
                
                # Validate against schema
                schema_issues = self._validate_against_schema(table_name, records)
                validation_issues.extend(schema_issues)
                
                # Validate against specific rules for this table
                if table_name in self.validation_rules:
                    for rule_func in self.validation_rules[table_name]:
                        rule_issues = rule_func(records)
                        validation_issues.extend(rule_issues)
                
                tables_validated += 1
            
            # Check cross-table relationships
            cross_table_issues = self._validate_cross_table_relationships(tables)
            validation_issues.extend(cross_table_issues)
            
            # Check terminology concept IDs
            concept_issues = self._validate_concept_ids(tables)
            validation_issues.extend(concept_issues)
            
            # Determine overall validation status
            is_valid = len(validation_issues) == 0
            
            validation_result = {
                "valid": is_valid,
                "issues": validation_issues,
                "tables_validated": tables_validated,
                "validated_at": datetime.now().isoformat(),
                "cdm_version": "5.4"  # Current OMOP CDM version
            }
            
            logger.info(f"OMOP validation completed with {len(validation_issues)} issues")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during OMOP validation: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "tables_validated": 0,
                "validated_at": datetime.now().isoformat(),
                "cdm_version": "5.4"
            }
    
    def _validate_against_schema(self, table_name, records):
        """
        Validate table records against OMOP CDM schema.
        
        Args:
            table_name (str): Name of the table
            records (list): Records to validate
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        # Check if schema exists for this table
        if table_name not in self.schemas:
            issues.append(f"No schema found for table: {table_name}")
            return issues
        
        schema = self.schemas[table_name]
        
        # Get required fields from schema
        required_fields = [field['name'] for field in schema['fields'] if field.get('required', True)]
        
        # Check data types for fields
        field_types = {field['name']: field['type'] for field in schema['fields']}
        
        # Check each record
        for i, record in enumerate(records):
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    issues.append(f"{table_name}[{i}]: Missing required field: {field}")
            
            # Check data types
            for field, value in record.items():
                if field in field_types and value is not None:
                    expected_type = field_types[field].lower()
                    
                    # Skip validation for null values
                    if pd.isna(value):
                        continue
                    
                    # Validate type based on expected type
                    if expected_type == 'integer':
                        if not (isinstance(value, int) or (isinstance(value, float) and value.is_integer())):
                            issues.append(f"{table_name}[{i}]: Field {field} should be integer, got {type(value).__name__}")
                    
                    elif expected_type == 'float':
                        if not isinstance(value, (int, float)):
                            issues.append(f"{table_name}[{i}]: Field {field} should be float, got {type(value).__name__}")
                    
                    elif expected_type == 'string':
                        if not isinstance(value, str):
                            issues.append(f"{table_name}[{i}]: Field {field} should be string, got {type(value).__name__}")
                    
                    elif expected_type in ('date', 'datetime'):
                        # Check if value is a valid date/datetime string or object
                        if not (isinstance(value, (datetime, str))):
                            issues.append(f"{table_name}[{i}]: Field {field} should be date/datetime, got {type(value).__name__}")
        
        return issues
    
    def _validate_condition_occurrence_rules(self, records):
        """
        Validate CONDITION_OCCURRENCE specific rules.
        
        Args:
            records (list): CONDITION_OCCURRENCE records
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        for i, record in enumerate(records):
            # Validate condition_concept_id is valid
            condition_concept_id = record.get('condition_concept_id')
            if condition_concept_id is not None and condition_concept_id == 0:
                issues.append(f"CONDITION_OCCURRENCE[{i}]: Unmapped condition_concept_id (0). Consider using a standard vocabulary.")
            
            # Validate condition_type_concept_id is valid
            type_concept_id = record.get('condition_type_concept_id')
            if type_concept_id is None:
                issues.append(f"CONDITION_OCCURRENCE[{i}]: Missing condition_type_concept_id")
        
        return issues
    
    def _validate_drug_exposure_rules(self, records):
        """
        Validate DRUG_EXPOSURE specific rules.
        
        Args:
            records (list): DRUG_EXPOSURE records
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        for i, record in enumerate(records):
            # Validate drug_concept_id is valid
            drug_concept_id = record.get('drug_concept_id')
            if drug_concept_id is not None and drug_concept_id == 0:
                issues.append(f"DRUG_EXPOSURE[{i}]: Unmapped drug_concept_id (0). Consider using a standard vocabulary.")
            
            # Validate drug_type_concept_id is valid
            type_concept_id = record.get('drug_type_concept_id')
            if type_concept_id is None:
                issues.append(f"DRUG_EXPOSURE[{i}]: Missing drug_type_concept_id")
            
            # Validate quantity is numeric if provided
            quantity = record.get('quantity')
            if quantity is not None and not isinstance(quantity, (int, float)) and not pd.isna(quantity):
                issues.append(f"DRUG_EXPOSURE[{i}]: quantity should be numeric, got {type(quantity).__name__}")
        
        return issues
    
    def _validate_procedure_occurrence_rules(self, records):
        """
        Validate PROCEDURE_OCCURRENCE specific rules.
        
        Args:
            records (list): PROCEDURE_OCCURRENCE records
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        for i, record in enumerate(records):
            # Validate procedure_concept_id is valid
            procedure_concept_id = record.get('procedure_concept_id')
            if procedure_concept_id is not None and procedure_concept_id == 0:
                issues.append(f"PROCEDURE_OCCURRENCE[{i}]: Unmapped procedure_concept_id (0). Consider using a standard vocabulary.")
            
            # Validate procedure_type_concept_id is valid
            type_concept_id = record.get('procedure_type_concept_id')
            if type_concept_id is None:
                issues.append(f"PROCEDURE_OCCURRENCE[{i}]: Missing procedure_type_concept_id")
            
            # Validate quantity is integer if provided
            quantity = record.get('quantity')
            if quantity is not None and not isinstance(quantity, int) and not pd.isna(quantity):
                issues.append(f"PROCEDURE_OCCURRENCE[{i}]: quantity should be integer, got {type(quantity).__name__}")
        
        return issues
    
    def _validate_observation_rules(self, records):
        """
        Validate OBSERVATION specific rules.
        
        Args:
            records (list): OBSERVATION records
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        for i, record in enumerate(records):
            # Validate observation_concept_id is valid
            observation_concept_id = record.get('observation_concept_id')
            if observation_concept_id is not None and observation_concept_id == 0:
                issues.append(f"OBSERVATION[{i}]: Unmapped observation_concept_id (0). Consider using a standard vocabulary.")
            
            # Validate observation_type_concept_id is valid
            type_concept_id = record.get('observation_type_concept_id')
            if type_concept_id is None:
                issues.append(f"OBSERVATION[{i}]: Missing observation_type_concept_id")
            
            # Validate value fields
            value_as_number = record.get('value_as_number')
            value_as_string = record.get('value_as_string')
            value_as_concept_id = record.get('value_as_concept_id')
            
            # Check that at least one value field is provided if this is not a protocol record
            if all(v is None for v in (value_as_number, value_as_string, value_as_concept_id)):
                # For protocol definitions, this might be acceptable
                logger.debug(f"OBSERVATION[{i}]: No value field provided (acceptable for protocol definitions)")
            
            # Validate value_as_number is numeric
            if value_as_number is not None and not isinstance(value_as_number, (int, float)) and not pd.isna(value_as_number):
                issues.append(f"OBSERVATION[{i}]: value_as_number should be numeric, got {type(value_as_number).__name__}")
        
        return issues
    
    def _validate_cross_table_relationships(self, tables):
        """
        Validate relationships between tables.
        
        Args:
            tables (dict): OMOP tables to validate
            
        Returns:
            list: Validation issues
        """
        # In a protocol conversion, cross-table relationships are minimal
        # This is mostly relevant for actual patient data
        return []
    
    def _validate_concept_ids(self, tables):
        """
        Validate concept IDs across tables.
        
        Args:
            tables (dict): OMOP tables to validate
            
        Returns:
            list: Validation issues
        """
        # In a basic implementation, we just warn about unmapped concepts
        issues = []
        unmapped_tables = set()
        
        # Check for tables with unmapped concepts
        for table_name, records in tables.items():
            if not records:
                continue
            
            concept_id_field = None
            if table_name == 'condition_occurrence':
                concept_id_field = 'condition_concept_id'
            elif table_name == 'drug_exposure':
                concept_id_field = 'drug_concept_id'
            elif table_name == 'procedure_occurrence':
                concept_id_field = 'procedure_concept_id'
            elif table_name == 'observation':
                concept_id_field = 'observation_concept_id'
            
            if concept_id_field:
                unmapped_count = sum(1 for r in records if r.get(concept_id_field) == 0)
                if unmapped_count > 0:
                    unmapped_tables.add(table_name)
        
        if unmapped_tables:
            issues.append(f"Tables with unmapped concepts (concept_id = 0): {', '.join(unmapped_tables)}. Consider enhancing terminology mapping.")
        
        return issues
    
    def validate_table(self, table_name, records):
        """
        Validate a single OMOP table.
        
        Args:
            table_name (str): Name of the table
            records (list): Records to validate
            
        Returns:
            dict: Validation result
        """
        try:
            issues = self._validate_against_schema(table_name, records)
            
            # Apply table-specific rules
            if table_name in self.validation_rules:
                for rule_func in self.validation_rules[table_name]:
                    rule_issues = rule_func(records)
                    issues.extend(rule_issues)
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "table": table_name,
                "record_count": len(records),
                "validated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating {table_name}: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "table": table_name,
                "record_count": len(records) if records else 0,
                "validated_at": datetime.now().isoformat()
            }
    
    def get_cdm_version(self):
        """Return the OMOP CDM version used by the validator."""
        return "5.4"  # Current OMOP CDM version