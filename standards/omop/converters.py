"""
OMOP Converter for the Clinical Protocol Extraction Engine.
This module handles the conversion of extracted protocol data to OMOP CDM format.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sqlalchemy as sa

# Set up logging
logger = logging.getLogger(__name__)

class OMOPConverter:
    """Converts extracted protocol data to OMOP CDM format."""
    
    def __init__(self, terminology_mapper=None, config=None):
        """
        Initialize OMOP converter with optional terminology mapper.
        
        Args:
            terminology_mapper: Optional mapper for clinical terminology
            config (dict, optional): Configuration options
        """
        self.terminology_mapper = terminology_mapper
        self.config = config or {}
        
        # Load schema directory path
        self.schema_dir = os.path.join(os.path.dirname(__file__), 'schemas')
        if not os.path.exists(self.schema_dir):
            os.makedirs(self.schema_dir, exist_ok=True)
            self._create_default_schemas()
        
        # Set default concept mappings
        self.default_concept_mappings = {
            # Concept type concept_ids
            "type_concept": {
                "EHR": 32817,
                "Claim": 32818,
                "Protocol": 32879,
                "Protocol Inclusion": 32880,
                "Protocol Exclusion": 32881
            },
            # Standard concept placeholder IDs (in real implementation, these would be real OMOP concept_ids)
            "standard_concepts": {
                "Unknown": 0,
                "No matching concept": 0
            }
        }
        
        logger.info(f"Initialized OMOP converter with schema directory: {self.schema_dir}")
    
    def _create_default_schemas(self):
        """Create default OMOP CDM schema files if they don't exist."""
        # Create CONDITION_OCCURRENCE schema
        condition_occurrence_schema = {
            "name": "CONDITION_OCCURRENCE",
            "description": "Records of conditions or diagnoses",
            "fields": [
                {"name": "condition_occurrence_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "condition_concept_id", "type": "integer", "required": True},
                {"name": "condition_start_date", "type": "date", "required": True},
                {"name": "condition_start_datetime", "type": "datetime", "required": False},
                {"name": "condition_end_date", "type": "date", "required": False},
                {"name": "condition_end_datetime", "type": "datetime", "required": False},
                {"name": "condition_type_concept_id", "type": "integer", "required": True},
                {"name": "condition_status_concept_id", "type": "integer", "required": False},
                {"name": "stop_reason", "type": "string", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "condition_source_value", "type": "string", "required": False},
                {"name": "condition_source_concept_id", "type": "integer", "required": False},
                {"name": "condition_status_source_value", "type": "string", "required": False}
            ]
        }
        
        # Create DRUG_EXPOSURE schema
        drug_exposure_schema = {
            "name": "DRUG_EXPOSURE",
            "description": "Records of drugs or medications",
            "fields": [
                {"name": "drug_exposure_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "drug_concept_id", "type": "integer", "required": True},
                {"name": "drug_exposure_start_date", "type": "date", "required": True},
                {"name": "drug_exposure_start_datetime", "type": "datetime", "required": False},
                {"name": "drug_exposure_end_date", "type": "date", "required": False},
                {"name": "drug_exposure_end_datetime", "type": "datetime", "required": False},
                {"name": "verbatim_end_date", "type": "date", "required": False},
                {"name": "drug_type_concept_id", "type": "integer", "required": True},
                {"name": "stop_reason", "type": "string", "required": False},
                {"name": "refills", "type": "integer", "required": False},
                {"name": "quantity", "type": "float", "required": False},
                {"name": "days_supply", "type": "integer", "required": False},
                {"name": "sig", "type": "string", "required": False},
                {"name": "route_concept_id", "type": "integer", "required": False},
                {"name": "lot_number", "type": "string", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "drug_source_value", "type": "string", "required": False},
                {"name": "drug_source_concept_id", "type": "integer", "required": False},
                {"name": "route_source_value", "type": "string", "required": False},
                {"name": "dose_unit_source_value", "type": "string", "required": False}
            ]
        }
        
        # Create PROCEDURE_OCCURRENCE schema
        procedure_occurrence_schema = {
            "name": "PROCEDURE_OCCURRENCE",
            "description": "Records of procedures",
            "fields": [
                {"name": "procedure_occurrence_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "procedure_concept_id", "type": "integer", "required": True},
                {"name": "procedure_date", "type": "date", "required": True},
                {"name": "procedure_datetime", "type": "datetime", "required": False},
                {"name": "procedure_type_concept_id", "type": "integer", "required": True},
                {"name": "modifier_concept_id", "type": "integer", "required": False},
                {"name": "quantity", "type": "integer", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "procedure_source_value", "type": "string", "required": False},
                {"name": "procedure_source_concept_id", "type": "integer", "required": False},
                {"name": "modifier_source_value", "type": "string", "required": False}
            ]
        }
        
        # Create OBSERVATION schema
        observation_schema = {
            "name": "OBSERVATION",
            "description": "Records of observations or measurements",
            "fields": [
                {"name": "observation_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "observation_concept_id", "type": "integer", "required": True},
                {"name": "observation_date", "type": "date", "required": True},
                {"name": "observation_datetime", "type": "datetime", "required": False},
                {"name": "observation_type_concept_id", "type": "integer", "required": True},
                {"name": "value_as_number", "type": "float", "required": False},
                {"name": "value_as_string", "type": "string", "required": False},
                {"name": "value_as_concept_id", "type": "integer", "required": False},
                {"name": "qualifier_concept_id", "type": "integer", "required": False},
                {"name": "unit_concept_id", "type": "integer", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "observation_source_value", "type": "string", "required": False},
                {"name": "observation_source_concept_id", "type": "integer", "required": False},
                {"name": "unit_source_value", "type": "string", "required": False},
                {"name": "qualifier_source_value", "type": "string", "required": False}
            ]
        }
        
        # Save schemas
        with open(os.path.join(self.schema_dir, 'condition_occurrence.json'), 'w') as f:
            json.dump(condition_occurrence_schema, f, indent=2)
        
        with open(os.path.join(self.schema_dir, 'drug_exposure.json'), 'w') as f:
            json.dump(drug_exposure_schema, f, indent=2)
        
        with open(os.path.join(self.schema_dir, 'procedure_occurrence.json'), 'w') as f:
            json.dump(procedure_occurrence_schema, f, indent=2)
        
        with open(os.path.join(self.schema_dir, 'observation.json'), 'w') as f:
            json.dump(observation_schema, f, indent=2)
        
        logger.info("Created default OMOP CDM schema files")
    
    def convert(self, protocol_data):
        """
        Convert protocol data to OMOP CDM format.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            dict: OMOP CDM data with validation results
        """
        try:
            logger.info("Converting protocol data to OMOP CDM format")
            
            # Create OMOP tables
            condition_occurrence = self.create_condition_occurrence(protocol_data)
            drug_exposure = self.create_drug_exposure(protocol_data)
            procedure_occurrence = self.create_procedure_occurrence(protocol_data)
            observation = self.create_observation(protocol_data)
            
            # Package tables
            tables = {
                "condition_occurrence": condition_occurrence,
                "drug_exposure": drug_exposure,
                "procedure_occurrence": procedure_occurrence,
                "observation": observation
            }
            
            # Create DataFrames from tables
            dataframes = {}
            for table_name, records in tables.items():
                if records:
                    dataframes[table_name] = pd.DataFrame(records)
                else:
                    dataframes[table_name] = pd.DataFrame()
            
            # Validate data
            validation_results = self.validate_omop_data(tables)
            
            logger.info(f"OMOP conversion completed with validation status: {validation_results['valid']}")
            
            # Convert DataFrames back to records for return
            omop_data = {}
            for table_name, df in dataframes.items():
                if not df.empty:
                    omop_data[table_name] = df.to_dict('records')
                else:
                    omop_data[table_name] = []
            
            return {
                "tables": omop_data,
                "validation": validation_results
            }
        
        except Exception as e:
            logger.error(f"Error converting to OMOP: {str(e)}", exc_info=True)
            raise
    
    def create_condition_occurrence(self, protocol_data):
        """
        Map conditions to CONDITION_OCCURRENCE table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: CONDITION_OCCURRENCE records
        """
        try:
            logger.info("Creating CONDITION_OCCURRENCE records")
            condition_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            inclusion_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol Inclusion"]
            exclusion_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol Exclusion"]
            
            # Process eligibility criteria for conditions
            if 'eligibility_criteria' in protocol_data and protocol_data['eligibility_criteria']:
                for i, criterion in enumerate(protocol_data['eligibility_criteria']):
                    # Skip if not a condition
                    if 'text' not in criterion:
                        continue
                    
                    # Determine if this is an inclusion or exclusion criterion
                    criterion_type = criterion.get('type', '').lower()
                    type_concept_id = inclusion_type_concept_id if criterion_type == 'inclusion' else exclusion_type_concept_id
                    
                    # Get concept_id if terminology mapper is available
                    condition_concept_id = 0  # Default to 0 (Unknown)
                    if self.terminology_mapper:
                        condition_text = criterion.get('text', '')
                        condition_concept_id = self.terminology_mapper.map_to_snomed(condition_text) or 0
                    
                    # Create record
                    record = {
                        "condition_occurrence_id": len(condition_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "condition_concept_id": condition_concept_id,
                        "condition_start_date": None,  # Not applicable for protocols
                        "condition_type_concept_id": type_concept_id,
                        "condition_source_value": criterion.get('text', '')
                    }
                    
                    condition_records.append(record)
            
            # Process conditions from other sections (if available)
            if 'conditions' in protocol_data and protocol_data['conditions']:
                for i, condition in enumerate(protocol_data['conditions']):
                    # Get concept_id if terminology mapper is available
                    condition_concept_id = 0  # Default to 0 (Unknown)
                    if self.terminology_mapper:
                        condition_text = condition.get('text', '')
                        condition_concept_id = self.terminology_mapper.map_to_snomed(condition_text) or 0
                    
                    # Create record
                    record = {
                        "condition_occurrence_id": len(condition_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "condition_concept_id": condition_concept_id,
                        "condition_start_date": None,  # Not applicable for protocols
                        "condition_type_concept_id": protocol_type_concept_id,
                        "condition_source_value": condition.get('text', '')
                    }
                    
                    condition_records.append(record)
            
            logger.info(f"Created {len(condition_records)} CONDITION_OCCURRENCE records")
            return condition_records
        
        except Exception as e:
            logger.error(f"Error creating CONDITION_OCCURRENCE: {str(e)}", exc_info=True)
            raise
    
    def create_drug_exposure(self, protocol_data):
        """
        Map medications to DRUG_EXPOSURE table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: DRUG_EXPOSURE records
        """
        try:
            logger.info("Creating DRUG_EXPOSURE records")
            drug_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Process medications
            if 'medications' in protocol_data and protocol_data['medications']:
                for i, medication in enumerate(protocol_data['medications']):
                    # Get concept_id if terminology mapper is available
                    drug_concept_id = 0  # Default to 0 (Unknown)
                    if self.terminology_mapper:
                        drug_text = medication.get('text', '')
                        drug_concept_id = self.terminology_mapper.map_to_rxnorm(drug_text) or 0
                    
                    # Process dosage information if available
                    quantity = None
                    days_supply = None
                    sig = None
                    route_source_value = None
                    
                    if 'dosage' in medication:
                        dosage = medication['dosage']
                        sig = dosage if isinstance(dosage, str) else None
                    
                    if 'route' in medication:
                        route_source_value = medication['route']
                    
                    # Create record
                    record = {
                        "drug_exposure_id": len(drug_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "drug_concept_id": drug_concept_id,
                        "drug_exposure_start_date": None,  # Not applicable for protocols
                        "drug_type_concept_id": protocol_type_concept_id,
                        "quantity": quantity,
                        "days_supply": days_supply,
                        "sig": sig,
                        "route_source_value": route_source_value,
                        "drug_source_value": medication.get('text', '')
                    }
                    
                    drug_records.append(record)
            
            logger.info(f"Created {len(drug_records)} DRUG_EXPOSURE records")
            return drug_records
        
        except Exception as e:
            logger.error(f"Error creating DRUG_EXPOSURE: {str(e)}", exc_info=True)
            raise
    
    def create_procedure_occurrence(self, protocol_data):
        """
        Map procedures to PROCEDURE_OCCURRENCE table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: PROCEDURE_OCCURRENCE records
        """
        try:
            logger.info("Creating PROCEDURE_OCCURRENCE records")
            procedure_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Process procedures
            if 'procedures' in protocol_data and protocol_data['procedures']:
                for i, procedure in enumerate(protocol_data['procedures']):
                    # Get concept_id if terminology mapper is available
                    procedure_concept_id = 0  # Default to 0 (Unknown)
                    if self.terminology_mapper:
                        procedure_text = procedure.get('text', '')
                        procedure_concept_id = self.terminology_mapper.map_to_snomed(procedure_text) or 0
                    
                    # Process additional information if available
                    quantity = None
                    if 'quantity' in procedure:
                        try:
                            quantity = int(procedure['quantity'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Create record
                    record = {
                        "procedure_occurrence_id": len(procedure_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "procedure_concept_id": procedure_concept_id,
                        "procedure_date": None,  # Not applicable for protocols
                        "procedure_type_concept_id": protocol_type_concept_id,
                        "quantity": quantity,
                        "procedure_source_value": procedure.get('text', '')
                    }
                    
                    procedure_records.append(record)
            
            logger.info(f"Created {len(procedure_records)} PROCEDURE_OCCURRENCE records")
            return procedure_records
        
        except Exception as e:
            logger.error(f"Error creating PROCEDURE_OCCURRENCE: {str(e)}", exc_info=True)
            raise
    
    def create_observation(self, protocol_data):
        """
        Map measurements and endpoints to OBSERVATION table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: OBSERVATION records
        """
        try:
            logger.info("Creating OBSERVATION records")
            observation_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Process measurements
            if 'measurements' in protocol_data and protocol_data['measurements']:
                for i, measurement in enumerate(protocol_data['measurements']):
                    # Get concept_id if terminology mapper is available
                    observation_concept_id = 0  # Default to 0 (Unknown)
                    unit_concept_id = 0
                    
                    if self.terminology_mapper:
                        measurement_text = measurement.get('text', '')
                        observation_concept_id = self.terminology_mapper.map_to_loinc(measurement_text) or 0
                        
                        # Map units if available
                        if 'units' in measurement and self.terminology_mapper:
                            unit_source_value = measurement['units']
                            unit_concept_id = self.terminology_mapper.map_unit(unit_source_value) or 0
                    
                    # Create record
                    record = {
                        "observation_id": len(observation_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "observation_concept_id": observation_concept_id,
                        "observation_date": None,  # Not applicable for protocols
                        "observation_type_concept_id": protocol_type_concept_id,
                        "value_as_number": None,  # No value for protocol definition
                        "unit_concept_id": unit_concept_id,
                        "observation_source_value": measurement.get('text', ''),
                        "unit_source_value": measurement.get('units')
                    }
                    
                    observation_records.append(record)
            
            # Process endpoints
            if 'endpoints' in protocol_data and protocol_data['endpoints']:
                for i, endpoint in enumerate(protocol_data['endpoints']):
                    # Get concept_id if terminology mapper is available
                    observation_concept_id = 0  # Default to 0 (Unknown)
                    
                    if self.terminology_mapper:
                        endpoint_text = endpoint.get('text', '')
                        observation_concept_id = self.terminology_mapper.map_to_loinc(endpoint_text) or 0
                    
                    # Create record
                    record = {
                        "observation_id": len(observation_records) + 1,
                        "person_id": None,  # Not applicable for protocols
                        "observation_concept_id": observation_concept_id,
                        "observation_date": None,  # Not applicable for protocols
                        "observation_type_concept_id": protocol_type_concept_id,
                        "value_as_string": "Protocol Endpoint",  # Flag as endpoint
                        "observation_source_value": endpoint.get('text', '')
                    }
                    
                    observation_records.append(record)
            
            logger.info(f"Created {len(observation_records)} OBSERVATION records")
            return observation_records
        
        except Exception as e:
            logger.error(f"Error creating OBSERVATION: {str(e)}", exc_info=True)
            raise
    
    def validate_omop_data(self, tables):
        """
        Validate OMOP CDM data against schemas.
        
        Args:
            tables (dict): OMOP tables to validate
            
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Validating OMOP CDM data")
            validation_issues = []
            
            # Load schemas for validation
            schemas = {}
            for table_name in ['condition_occurrence', 'drug_exposure', 'procedure_occurrence', 'observation']:
                schema_path = os.path.join(self.schema_dir, f"{table_name}.json")
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schemas[table_name] = json.load(f)
            
            # Validate each table against its schema
            for table_name, records in tables.items():
                if not records:
                    continue
                
                if table_name not in schemas:
                    validation_issues.append(f"No schema found for table {table_name}")
                    continue
                
                schema = schemas[table_name]
                required_fields = [field['name'] for field in schema['fields'] if field.get('required', False)]
                
                # Check records for required fields
                for i, record in enumerate(records):
                    for field in required_fields:
                        if field not in record or record[field] is None:
                            validation_issues.append(f"{table_name} record {i+1} missing required field: {field}")
            
            # Determine overall validation status
            is_valid = len(validation_issues) == 0
            
            validation_results = {
                "valid": is_valid,
                "issues": validation_issues
            }
            
            logger.info(f"Validation completed with {len(validation_issues)} issues")
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating OMOP data: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"]
            }
    
    def generate_sql(self, tables, dialect='sqlite'):
        """
        Generate SQL statements for OMOP tables.
        
        Args:
            tables (dict): OMOP tables data
            dialect (str): SQL dialect (sqlite, postgresql, etc.)
            
        Returns:
            dict: SQL statements for each table
        """
        try:
            logger.info(f"Generating {dialect} SQL statements")
            
            # Create SQLAlchemy engine for the specified dialect
            engine = sa.create_engine(f'{dialect}://')
            sql_statements = {}
            
            # Load schemas
            schemas = {}
            for table_name in tables.keys():
                schema_path = os.path.join(self.schema_dir, f"{table_name}.json")
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schemas[table_name] = json.load(f)
            
            # Generate SQL for each table
            for table_name, records in tables.items():
                if not records:
                    continue
                
                # Convert records to DataFrame
                df = pd.DataFrame(records)
                
                # Create SQLAlchemy table
                metadata = sa.MetaData()
                
                # Define columns based on schema
                columns = []
                if table_name in schemas:
                    schema = schemas[table_name]
                    for field in schema['fields']:
                        # Map field type to SQLAlchemy type
                        field_type = field['type'].lower()
                        
                        if field_type == 'integer':
                            sa_type = sa.Integer()
                        elif field_type == 'float':
                            sa_type = sa.Float()
                        elif field_type == 'string':
                            sa_type = sa.String(255)
                        elif field_type == 'date':
                            sa_type = sa.Date()
                        elif field_type == 'datetime':
                            sa_type = sa.DateTime()
                        else:
                            sa_type = sa.String(255)  # Default to string
                        
                        # Define column
                        column = sa.Column(
                            field['name'],
                            sa_type,
                            primary_key=field.get('primary_key', False),
                            nullable=not field.get('required', False)
                        )
                        columns.append(column)
                
                # Create table
                table = sa.Table(table_name, metadata, *columns)
                
                # Generate CREATE TABLE statement
                create_stmt = str(sa.schema.CreateTable(table).compile(engine))
                
                # Generate INSERT statements
                insert_statements = []
                for _, row in df.iterrows():
                    # Clean row data (replace NaN with None)
                    row_dict = {k: None if pd.isna(v) else v for k, v in row.to_dict().items()}
                    
                    # Generate insert statement
                    insert = table.insert().values(**row_dict)
                    insert_stmt = str(insert.compile(engine, compile_kwargs={"literal_binds": True}))
                    insert_statements.append(insert_stmt)
                
                # Combine statements
                sql_statements[table_name] = {
                    "create": create_stmt,
                    "inserts": insert_statements
                }
            
            logger.info(f"Generated SQL statements for {len(sql_statements)} tables")
            return sql_statements
        
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}", exc_info=True)
            raise
    
    def get_cdm_version(self):
        """Return the OMOP CDM version used by the converter."""
        return "5.4"  # Current OMOP CDM version


# For easy testing
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test OMOP conversion with sample data')
    parser.add_argument('--output', help='Output directory for OMOP data')
    parser.add_argument('--sql', action='store_true', help='Generate SQL statements')
    parser.add_argument('--dialect', default='sqlite', help='SQL dialect (sqlite, postgresql, etc.)')
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create sample protocol data
    sample_protocol = {
        "title": "Sample Clinical Trial Protocol",
        "description": "This is a sample protocol for testing OMOP conversion",
        "protocol_id": "CT-2023-001",
        "eligibility_criteria": [
            {
                "type": "inclusion",
                "text": "Age >= 18 years"
            },
            {
                "type": "inclusion",
                "text": "Diagnosed with condition X"
            },
            {
                "type": "exclusion",
                "text": "History of condition Y"
            }
        ],
        "procedures": [
            {
                "text": "Blood sample collection",
                "description": "Collect 10ml blood sample",
                "quantity": 1
            },
            {
                "text": "MRI scan",
                "description": "Full-body MRI scan"
            }
        ],
        "medications": [
            {
                "text": "Drug A",
                "description": "Experimental treatment",
                "dosage": "10mg daily",
                "route": "oral"
            }
        ],
        "endpoints": [
            {
                "text": "Change in biomarker X level"
            },
            {
                "text": "Frequency of adverse events"
            }
        ],
        "measurements": [
            {
                "text": "Blood pressure",
                "units": "mmHg"
            },
            {
                "text": "Body weight",
                "units": "kg"
            }
        ]
    }
    
    try:
        # Create converter and convert sample data
        converter = OMOPConverter()
        result = converter.convert(sample_protocol)

        # Save output if directory specified
        if output_dir:
            # Save tables to CSV
            for table_name, records in result['tables'].items():
                if records:
                    df = pd.DataFrame(records)
                    csv_path = os.path.join(output_dir, f"{table_name}.csv")
                    df.to_csv(csv_path, index=False)
            
            # Generate and save SQL if requested
            if args.sql:
                sql_results = converter.generate_sql(result['tables'], dialect=args.dialect)
                for table_name, statements in sql_results.items():
                    sql_path = os.path.join(output_dir, f"{table_name}.sql")
                    with open(sql_path, 'w') as f:
                        f.write(f"-- {table_name} table CREATE statement\n")
                        f.write(statements['create'])
                        f.write("\n\n-- INSERT statements\n")
                        for insert in statements['inserts']:
                            f.write(insert + ";\n")
            
            print(f"\nOMOP data saved to {output_dir}")
        else:
            # Print summary of records
            print("\nConverted Tables:")
            for table_name, records in result['tables'].items():
                print(f"{table_name}: {len(records)} records")
        
        # Print validation results
        print("\nValidation Results:")
        print(f"Valid: {result['validation']['valid']}")
        if not result['validation']['valid']:
            print("Issues:")
            for issue in result['validation']['issues']:
                print(f"  - {issue}")
    
    except Exception as e:
        logger.error(f"Error in sample conversion: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")