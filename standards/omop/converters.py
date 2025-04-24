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
        
        # Create MEASUREMENT schema
        measurement_schema = {
            "name": "MEASUREMENT",
            "description": "Records of measurements and lab tests",
            "fields": [
                {"name": "measurement_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "measurement_concept_id", "type": "integer", "required": True},
                {"name": "measurement_date", "type": "date", "required": True},
                {"name": "measurement_datetime", "type": "datetime", "required": False},
                {"name": "measurement_type_concept_id", "type": "integer", "required": True},
                {"name": "operator_concept_id", "type": "integer", "required": False},
                {"name": "value_as_number", "type": "float", "required": False},
                {"name": "value_as_concept_id", "type": "integer", "required": False},
                {"name": "unit_concept_id", "type": "integer", "required": False},
                {"name": "range_low", "type": "float", "required": False},
                {"name": "range_high", "type": "float", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "measurement_source_value", "type": "string", "required": False},
                {"name": "measurement_source_concept_id", "type": "integer", "required": False},
                {"name": "unit_source_value", "type": "string", "required": False},
                {"name": "value_source_value", "type": "string", "required": False}
            ]
        }
        
        # Create DEVICE_EXPOSURE schema
        device_exposure_schema = {
            "name": "DEVICE_EXPOSURE",
            "description": "Records of device usage",
            "fields": [
                {"name": "device_exposure_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "device_concept_id", "type": "integer", "required": True},
                {"name": "device_exposure_start_date", "type": "date", "required": True},
                {"name": "device_exposure_start_datetime", "type": "datetime", "required": False},
                {"name": "device_exposure_end_date", "type": "date", "required": False},
                {"name": "device_exposure_end_datetime", "type": "datetime", "required": False},
                {"name": "device_type_concept_id", "type": "integer", "required": True},
                {"name": "unique_device_id", "type": "string", "required": False},
                {"name": "quantity", "type": "integer", "required": False},
                {"name": "provider_id", "type": "integer", "required": False},
                {"name": "visit_occurrence_id", "type": "integer", "required": False},
                {"name": "visit_detail_id", "type": "integer", "required": False},
                {"name": "device_source_value", "type": "string", "required": False},
                {"name": "device_source_concept_id", "type": "integer", "required": False}
            ]
        }
        
        # Create SPECIMEN schema
        specimen_schema = {
            "name": "SPECIMEN",
            "description": "Records of specimens",
            "fields": [
                {"name": "specimen_id", "type": "integer", "required": True, "primary_key": True},
                {"name": "person_id", "type": "integer", "required": True},
                {"name": "specimen_concept_id", "type": "integer", "required": True},
                {"name": "specimen_type_concept_id", "type": "integer", "required": True},
                {"name": "specimen_date", "type": "date", "required": True},
                {"name": "specimen_datetime", "type": "datetime", "required": False},
                {"name": "quantity", "type": "float", "required": False},
                {"name": "unit_concept_id", "type": "integer", "required": False},
                {"name": "anatomic_site_concept_id", "type": "integer", "required": False},
                {"name": "disease_status_concept_id", "type": "integer", "required": False},
                {"name": "specimen_source_id", "type": "string", "required": False},
                {"name": "specimen_source_value", "type": "string", "required": False},
                {"name": "unit_source_value", "type": "string", "required": False},
                {"name": "anatomic_site_source_value", "type": "string", "required": False},
                {"name": "disease_status_source_value", "type": "string", "required": False}
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
            
        with open(os.path.join(self.schema_dir, 'measurement.json'), 'w') as f:
            json.dump(measurement_schema, f, indent=2)
            
        with open(os.path.join(self.schema_dir, 'device_exposure.json'), 'w') as f:
            json.dump(device_exposure_schema, f, indent=2)
            
        with open(os.path.join(self.schema_dir, 'specimen.json'), 'w') as f:
            json.dump(specimen_schema, f, indent=2)
        
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
            
            # First map all clinical terms to standard terminologies
            mapped_data = self.map_extracted_data(protocol_data)
            
            # Create OMOP tables
            condition_occurrence = self.create_condition_occurrence(mapped_data)
            drug_exposure = self.create_drug_exposure(mapped_data)
            procedure_occurrence = self.create_procedure_occurrence(mapped_data)
            observation = self.create_observation(mapped_data)
            measurement = self.create_measurement(mapped_data)
            device_exposure = self.create_device_exposure(mapped_data)
            specimen = self.create_specimen(mapped_data)
            
            # Package tables
            tables = {
                "condition_occurrence": condition_occurrence,
                "drug_exposure": drug_exposure,
                "procedure_occurrence": procedure_occurrence,
                "observation": observation,
                "measurement": measurement,
                "device_exposure": device_exposure,
                "specimen": specimen
            }
            
            # Create DataFrames from tables for easier manipulation
            dataframes = {}
            for table_name, records in tables.items():
                if records:
                    dataframes[table_name] = pd.DataFrame(records)
                else:
                    dataframes[table_name] = pd.DataFrame()
            
            # Validate data against OMOP CDM schemas
            validation_results = self.validate_omop_data(tables)
            
            # Generate SQL statements if configured
            sql_statements = None
            if self.config.get("generate_sql", False):
                dialect = self.config.get("sql_dialect", "sqlite")
                sql_statements = self.generate_sql(tables, dialect)
            
            logger.info(f"OMOP conversion completed with validation status: {validation_results['valid']}")
            
            # Convert DataFrames back to records for return
            omop_data = {}
            for table_name, df in dataframes.items():
                if not df.empty:
                    omop_data[table_name] = df.to_dict('records')
                else:
                    omop_data[table_name] = []
            
            result = {
                "tables": omop_data,
                "validation": validation_results,
                "version": self.get_cdm_version()
            }
            
            # Add SQL statements if generated
            if sql_statements:
                result["sql"] = sql_statements
                
            # Add terminology mapping statistics
            if hasattr(self.terminology_mapper, 'get_statistics'):
                result["terminology_mapping"] = {
                    "statistics": self.terminology_mapper.get_statistics()
                }
            
            return result
        
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
    
    def map_extracted_data(self, extracted_data):
        """
        Map all extractable entities in the data to standard terminologies.
        
        This method processes the entire extracted dataset, mapping all clinical
        terms to their standardized codes as appropriate.
        
        Args:
            extracted_data (dict): The extracted protocol data
            
        Returns:
            dict: Processed data with terminology mappings
        """
        try:
            logger.info("Mapping extracted clinical terms to standard OMOP terminologies")
            
            # Create a copy of the data to avoid modifying the original
            mapped_data = extracted_data.copy()
            
            # Helper function for mapping
            def map_term(term, text, context=None, term_type=None):
                if not self.terminology_mapper:
                    return {
                        "concept_id": 0,
                        "source_value": text,
                        "source_concept_id": 0
                    }
                
                if term_type == "condition":
                    mapping = self.terminology_mapper.map_to_snomed(text, context)
                elif term_type == "medication":
                    mapping = self.terminology_mapper.map_to_rxnorm(text, context)
                elif term_type == "observation" or term_type == "measurement":
                    mapping = self.terminology_mapper.map_to_loinc(text, context)
                elif term_type == "procedure":
                    mapping = self.terminology_mapper.map_to_snomed(text, context)
                else:
                    # Default to SNOMED
                    mapping = self.terminology_mapper.map_to_snomed(text, context)
                
                if not mapping or not isinstance(mapping, dict):
                    return {
                        "concept_id": 0,
                        "source_value": text,
                        "source_concept_id": 0
                    }
                
                return {
                    "concept_id": mapping.get("code", 0),
                    "source_value": text,
                    "source_concept_id": 0,
                    "system": mapping.get("system", ""),
                    "display": mapping.get("display", text),
                    "found": mapping.get("found", False)
                }
            
            # Map eligibility criteria
            if 'eligibility_criteria' in mapped_data and mapped_data['eligibility_criteria']:
                for i, criterion in enumerate(mapped_data['eligibility_criteria']):
                    if 'text' in criterion:
                        criterion_text = criterion.get('text', '')
                        context = "eligibility_criteria"
                        criterion_type = criterion.get('type', '').lower()
                        if criterion_type:
                            context += f"_{criterion_type}"
                        
                        mapped_data['eligibility_criteria'][i]['mapping'] = map_term(
                            'condition', criterion_text, context, "condition"
                        )
            
            # Map procedures
            if 'procedures' in mapped_data and mapped_data['procedures']:
                for i, procedure in enumerate(mapped_data['procedures']):
                    if 'text' in procedure:
                        procedure_text = procedure.get('text', '')
                        context = procedure.get('description', 'procedure')
                        
                        mapped_data['procedures'][i]['mapping'] = map_term(
                            'procedure', procedure_text, context, "procedure"
                        )
                        
                        # Check if this is a specimen collection procedure
                        if any(kw in procedure_text.lower() for kw in ["collect", "sample", "specimen", "biopsy"]):
                            mapped_data['procedures'][i]['specimen_type'] = "specimen collection"
            
            # Map medications
            if 'medications' in mapped_data and mapped_data['medications']:
                for i, medication in enumerate(mapped_data['medications']):
                    if 'text' in medication:
                        medication_text = medication.get('text', '')
                        context = medication.get('description', '')
                        if 'dosage' in medication:
                            context += " " + str(medication['dosage'])
                        if 'route' in medication:
                            context += " " + str(medication['route'])
                        
                        mapped_data['medications'][i]['mapping'] = map_term(
                            'medication', medication_text, context, "medication"
                        )
            
            # Map measurements
            if 'measurements' in mapped_data and mapped_data['measurements']:
                for i, measurement in enumerate(mapped_data['measurements']):
                    if 'text' in measurement:
                        measurement_text = measurement.get('text', '')
                        context = "measurement"
                        if 'units' in measurement:
                            context += f" ({measurement['units']})"
                        
                        mapped_data['measurements'][i]['mapping'] = map_term(
                            'measurement', measurement_text, context, "measurement"
                        )
                        
                        # Map units if available
                        if 'units' in measurement:
                            unit_mapping = self._map_unit(measurement['units'])
                            mapped_data['measurements'][i]['unit_mapping'] = unit_mapping
            
            # Map endpoints
            if 'endpoints' in mapped_data and mapped_data['endpoints']:
                for i, endpoint in enumerate(mapped_data['endpoints']):
                    if 'text' in endpoint:
                        endpoint_text = endpoint.get('text', '')
                        
                        mapped_data['endpoints'][i]['mapping'] = map_term(
                            'observation', endpoint_text, "endpoint", "observation"
                        )
            
            # Map devices
            if 'devices' in mapped_data and mapped_data['devices']:
                for i, device in enumerate(mapped_data['devices']):
                    if 'text' in device:
                        device_text = device.get('text', '')
                        
                        mapped_data['devices'][i]['mapping'] = map_term(
                            'device', device_text, "device", "device"
                        )
            
            logger.info("Terminology mapping completed for OMOP conversion")
            return mapped_data
        
        except Exception as e:
            logger.error(f"Error mapping terminology for OMOP: {str(e)}", exc_info=True)
            # Return original data if mapping fails
            return extracted_data
    
    def _map_unit(self, unit_source_value):
        """Map a unit string to OMOP concept ID."""
        if not unit_source_value:
            return {"concept_id": 0, "source_value": None}
        
        # Common unit mappings in OMOP
        unit_mappings = {
            "mg": 8576,
            "ml": 8587,
            "mmHg": 8876,
            "kg": 9529,
            "cm": 8582,
            "g": 8505,
            "mmol/L": 8840,
            "U/L": 8645,
            "cells/μL": 8848,
            "%": 8554,
            "years": 8532,
            "months": 8531,
            "days": 8507,
            "hours": 8520,
            "minutes": 8517,
            "seconds": 8555
        }
        
        unit_source_value = str(unit_source_value).strip().lower()
        concept_id = unit_mappings.get(unit_source_value, 0)
        
        # Try alternative formats
        if concept_id == 0:
            # Try common variations
            variations = {
                "milligrams": "mg",
                "milliliters": "ml",
                "kilograms": "kg",
                "grams": "g",
                "centimeters": "cm",
                "millimeters": "mm",
                "percent": "%",
                "mmol/l": "mmol/L",
                "u/l": "U/L"
            }
            
            if unit_source_value in variations:
                std_unit = variations[unit_source_value]
                concept_id = unit_mappings.get(std_unit, 0)
        
        return {
            "concept_id": concept_id,
            "source_value": unit_source_value
        }
    
    def create_observation(self, protocol_data):
        """
        Map protocol-specific observations to OBSERVATION table.
        
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
            
            # Process endpoints (primary focus for Observation table)
            if 'endpoints' in protocol_data and protocol_data['endpoints']:
                for i, endpoint in enumerate(protocol_data['endpoints']):
                    # Get concept_id from mapping if available
                    observation_concept_id = 0
                    if 'mapping' in endpoint and endpoint['mapping']:
                        observation_concept_id = endpoint['mapping'].get('concept_id', 0)
                    
                    # Create record
                    record = {
                        "observation_id": len(observation_records) + 1,
                        "person_id": 0,  # Placeholder for protocol
                        "observation_concept_id": observation_concept_id,
                        "observation_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                        "observation_type_concept_id": protocol_type_concept_id,
                        "value_as_string": "Protocol Endpoint",  # Flag as endpoint
                        "observation_source_value": endpoint.get('text', ''),
                        "observation_source_concept_id": 0
                    }
                    
                    observation_records.append(record)
            
            # Process other observations not suitable for Measurement table
            if 'observations' in protocol_data and protocol_data['observations']:
                for i, obs in enumerate(protocol_data['observations']):
                    # Get concept_id from mapping if available
                    observation_concept_id = 0
                    if 'mapping' in obs and obs['mapping']:
                        observation_concept_id = obs['mapping'].get('concept_id', 0)
                    
                    # Create record
                    record = {
                        "observation_id": len(observation_records) + 1,
                        "person_id": 0,  # Placeholder for protocol
                        "observation_concept_id": observation_concept_id,
                        "observation_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                        "observation_type_concept_id": protocol_type_concept_id,
                        "value_as_string": obs.get('value', None),
                        "observation_source_value": obs.get('text', ''),
                        "observation_source_concept_id": 0
                    }
                    
                    observation_records.append(record)
            
            # Add eligibility criteria that don't map well to conditions
            if 'eligibility_criteria' in protocol_data and protocol_data['eligibility_criteria']:
                for i, criterion in enumerate(protocol_data['eligibility_criteria']):
                    criterion_text = criterion.get('text', '')
                    
                    # Skip if this is clearly a condition
                    if any(x in criterion_text.lower() for x in ['disease', 'disorder', 'syndrome', 'condition']):
                        continue
                        
                    # Check if this is demographic or other criteria better suited for Observation
                    if any(x in criterion_text.lower() for x in ['age', 'gender', 'sex', 'consent', 'language', 'able to']):
                        # Get concept_id from mapping if available
                        observation_concept_id = 0
                        if 'mapping' in criterion and criterion['mapping']:
                            observation_concept_id = criterion['mapping'].get('concept_id', 0)
                        
                        criterion_type = criterion.get('type', '').lower()
                        
                        # Create record
                        record = {
                            "observation_id": len(observation_records) + 1,
                            "person_id": 0,  # Placeholder for protocol
                            "observation_concept_id": observation_concept_id,
                            "observation_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                            "observation_type_concept_id": protocol_type_concept_id,
                            "value_as_string": f"{criterion_type} criterion",
                            "observation_source_value": criterion_text,
                            "observation_source_concept_id": 0
                        }
                        
                        observation_records.append(record)
            
            logger.info(f"Created {len(observation_records)} OBSERVATION records")
            return observation_records
        
        except Exception as e:
            logger.error(f"Error creating OBSERVATION: {str(e)}", exc_info=True)
            raise
    
    def create_measurement(self, protocol_data):
        """
        Map measurements to MEASUREMENT table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: MEASUREMENT records
        """
        try:
            logger.info("Creating MEASUREMENT records")
            measurement_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Process measurements
            if 'measurements' in protocol_data and protocol_data['measurements']:
                for i, measurement in enumerate(protocol_data['measurements']):
                    # Get concept_id from mapping if available
                    measurement_concept_id = 0
                    if 'mapping' in measurement and measurement['mapping']:
                        measurement_concept_id = measurement['mapping'].get('concept_id', 0)
                    
                    # Get unit concept ID from mapping if available
                    unit_concept_id = 0
                    if 'unit_mapping' in measurement and measurement['unit_mapping']:
                        unit_concept_id = measurement['unit_mapping'].get('concept_id', 0)
                    
                    # Set range values if available
                    range_low = None
                    range_high = None
                    if 'range' in measurement:
                        range_data = measurement['range']
                        if isinstance(range_data, dict):
                            range_low = range_data.get('low')
                            range_high = range_data.get('high')
                    
                    # Create record
                    record = {
                        "measurement_id": len(measurement_records) + 1,
                        "person_id": 0,  # Placeholder for protocol
                        "measurement_concept_id": measurement_concept_id,
                        "measurement_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                        "measurement_type_concept_id": protocol_type_concept_id,
                        "value_as_number": None,  # No exact value for protocols
                        "unit_concept_id": unit_concept_id,
                        "range_low": range_low,
                        "range_high": range_high,
                        "measurement_source_value": measurement.get('text', ''),
                        "unit_source_value": measurement.get('units')
                    }
                    
                    measurement_records.append(record)
            
            # Look for lab tests in endpoints
            if 'endpoints' in protocol_data and protocol_data['endpoints']:
                for i, endpoint in enumerate(protocol_data['endpoints']):
                    endpoint_text = endpoint.get('text', '').lower()
                    
                    # Check if this is a lab test endpoint
                    is_lab_test = any(kw in endpoint_text for kw in [
                        'lab', 'test', 'level', 'count', 'concentration', 'value'
                    ])
                    
                    if is_lab_test:
                        # Get concept_id from mapping if available
                        measurement_concept_id = 0
                        if 'mapping' in endpoint and endpoint['mapping']:
                            measurement_concept_id = endpoint['mapping'].get('concept_id', 0)
                        
                        # Create record
                        record = {
                            "measurement_id": len(measurement_records) + 1,
                            "person_id": 0,  # Placeholder for protocol
                            "measurement_concept_id": measurement_concept_id,
                            "measurement_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                            "measurement_type_concept_id": protocol_type_concept_id,
                            "value_as_number": None,  # No exact value for protocols
                            "measurement_source_value": endpoint.get('text', ''),
                            "value_source_value": "Protocol Endpoint"
                        }
                        
                        measurement_records.append(record)
            
            logger.info(f"Created {len(measurement_records)} MEASUREMENT records")
            return measurement_records
        
        except Exception as e:
            logger.error(f"Error creating MEASUREMENT: {str(e)}", exc_info=True)
            raise
    
    def create_device_exposure(self, protocol_data):
        """
        Map devices to DEVICE_EXPOSURE table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: DEVICE_EXPOSURE records
        """
        try:
            logger.info("Creating DEVICE_EXPOSURE records")
            device_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Process explicit devices
            if 'devices' in protocol_data and protocol_data['devices']:
                for i, device in enumerate(protocol_data['devices']):
                    # Get concept_id from mapping if available
                    device_concept_id = 0
                    if 'mapping' in device and device['mapping']:
                        device_concept_id = device['mapping'].get('concept_id', 0)
                    
                    # Create record
                    record = {
                        "device_exposure_id": len(device_records) + 1,
                        "person_id": 0,  # Placeholder for protocol
                        "device_concept_id": device_concept_id,
                        "device_exposure_start_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                        "device_type_concept_id": protocol_type_concept_id,
                        "device_source_value": device.get('text', ''),
                        "quantity": device.get('quantity', None)
                    }
                    
                    device_records.append(record)
            
            # Look for devices mentioned in procedures
            if 'procedures' in protocol_data and protocol_data['procedures']:
                for i, procedure in enumerate(protocol_data['procedures']):
                    procedure_text = procedure.get('text', '').lower()
                    
                    # Check if this procedure involves a device
                    device_keywords = [
                        'device', 'implant', 'catheter', 'pacemaker', 'stent',
                        'monitor', 'machine', 'ventilator', 'pump'
                    ]
                    
                    is_device_procedure = any(kw in procedure_text for kw in device_keywords)
                    
                    if is_device_procedure:
                        # Extract device information from procedure
                        device_text = procedure_text
                        
                        # Try to map the device
                        device_concept_id = 0
                        # Use the procedure mapping as a fallback
                        if 'mapping' in procedure and procedure['mapping']:
                            device_concept_id = procedure['mapping'].get('concept_id', 0)
                        
                        # Create record
                        record = {
                            "device_exposure_id": len(device_records) + 1,
                            "person_id": 0,  # Placeholder for protocol
                            "device_concept_id": device_concept_id,
                            "device_exposure_start_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                            "device_type_concept_id": protocol_type_concept_id,
                            "device_source_value": device_text,
                            "quantity": procedure.get('quantity', None)
                        }
                        
                        device_records.append(record)
            
            logger.info(f"Created {len(device_records)} DEVICE_EXPOSURE records")
            return device_records
        
        except Exception as e:
            logger.error(f"Error creating DEVICE_EXPOSURE: {str(e)}", exc_info=True)
            raise
    
    def create_specimen(self, protocol_data):
        """
        Map specimens to SPECIMEN table.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: SPECIMEN records
        """
        try:
            logger.info("Creating SPECIMEN records")
            specimen_records = []
            
            # Set default values for a protocol conversion
            protocol_type_concept_id = self.default_concept_mappings["type_concept"]["Protocol"]
            
            # Look for specimen collection procedures
            if 'procedures' in protocol_data and protocol_data['procedures']:
                for i, procedure in enumerate(protocol_data['procedures']):
                    procedure_text = procedure.get('text', '').lower()
                    is_specimen_procedure = False
                    
                    # Check if this is a specimen collection procedure
                    if 'specimen_type' in procedure and procedure['specimen_type'] == "specimen collection":
                        is_specimen_procedure = True
                    elif any(kw in procedure_text for kw in [
                        'sample', 'specimen', 'collection', 'biopsy', 'aspirate',
                        'blood draw', 'venipuncture', 'tissue'
                    ]):
                        is_specimen_procedure = True
                    
                    if is_specimen_procedure:
                        # Extract specimen information
                        anatomic_site = None
                        disease_status = None
                        quantity = None
                        
                        # Determine specimen type from procedure text
                        specimen_type = "Unknown"
                        if 'blood' in procedure_text:
                            specimen_type = "Blood"
                        elif 'urine' in procedure_text:
                            specimen_type = "Urine"
                        elif 'tissue' in procedure_text or 'biopsy' in procedure_text:
                            specimen_type = "Tissue"
                        elif 'fluid' in procedure_text:
                            specimen_type = "Fluid"
                        elif 'swab' in procedure_text:
                            specimen_type = "Swab"
                        
                        # Get anatomic site if available
                        body_sites = ['arm', 'leg', 'chest', 'abdomen', 'head', 'neck', 'back', 'liver', 'kidney', 'heart', 'lung']
                        for site in body_sites:
                            if site in procedure_text:
                                anatomic_site = site
                                break
                        
                        # Get quantity if available
                        if 'quantity' in procedure:
                            quantity = procedure['quantity']
                        
                        # Get concept IDs
                        specimen_concept_id = 0
                        # Use the procedure mapping as a fallback
                        if 'mapping' in procedure and procedure['mapping']:
                            specimen_concept_id = procedure['mapping'].get('concept_id', 0)
                        
                        # Create record
                        record = {
                            "specimen_id": len(specimen_records) + 1,
                            "person_id": 0,  # Placeholder for protocol
                            "specimen_concept_id": specimen_concept_id,
                            "specimen_type_concept_id": protocol_type_concept_id,
                            "specimen_date": datetime.now().strftime("%Y-%m-%d"),  # Protocol creation date
                            "quantity": quantity,
                            "specimen_source_value": specimen_type,
                            "anatomic_site_source_value": anatomic_site,
                            "disease_status_source_value": disease_status
                        }
                        
                        specimen_records.append(record)
            
            logger.info(f"Created {len(specimen_records)} SPECIMEN records")
            return specimen_records
        
        except Exception as e:
            logger.error(f"Error creating SPECIMEN: {str(e)}", exc_info=True)
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
            validation_warnings = []
            
            # Load schemas for validation
            schemas = {}
            schema_files = [
                'condition_occurrence',
                'drug_exposure',
                'procedure_occurrence',
                'observation',
                'measurement',
                'device_exposure',
                'specimen'
            ]
            
            for table_name in schema_files:
                schema_path = os.path.join(self.schema_dir, f"{table_name}.json")
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schemas[table_name] = json.load(f)
                else:
                    validation_warnings.append(f"Schema file not found for table: {table_name}")
            
            # Validate each table against its schema
            for table_name, records in tables.items():
                if not records:
                    # Empty tables are allowed but generate a warning
                    validation_warnings.append(f"Table {table_name} has no records")
                    continue
                
                if table_name not in schemas:
                    validation_issues.append(f"No schema found for table {table_name}")
                    continue
                
                schema = schemas[table_name]
                required_fields = [field['name'] for field in schema['fields'] if field.get('required', False)]
                field_types = {field['name']: field['type'] for field in schema['fields']}
                
                # Check records for required fields and data types
                for i, record in enumerate(records):
                    # Check required fields
                    for field in required_fields:
                        if field not in record or record[field] is None:
                            # For protocol data, person_id is often not applicable, so make this a warning
                            if field == 'person_id':
                                validation_warnings.append(f"{table_name} record {i+1} missing person_id (using placeholder)")
                                # Automatically fix by setting a placeholder person_id (0 is standard for placeholder)
                                record['person_id'] = 0
                            # For protocol data, dates are often not applicable either
                            elif field.endswith('_date'):
                                validation_warnings.append(f"{table_name} record {i+1} missing {field} (using protocol date)")
                                # Automatically fix by setting today's date as protocol date
                                record[field] = datetime.now().strftime("%Y-%m-%d")
                            else:
                                validation_issues.append(f"{table_name} record {i+1} missing required field: {field}")
                    
                    # Check data types
                    for field, value in record.items():
                        if field in field_types and value is not None:
                            expected_type = field_types[field]
                            
                            # Type checking based on OMOP CDM data types
                            if expected_type == 'integer':
                                try:
                                    if not isinstance(value, int) and not (isinstance(value, str) and value.isdigit()):
                                        validation_issues.append(f"{table_name} record {i+1} field {field} should be integer, got {type(value).__name__}")
                                except:
                                    validation_issues.append(f"{table_name} record {i+1} field {field} has invalid value: {value}")
                            
                            elif expected_type == 'string':
                                if not isinstance(value, str):
                                    validation_warnings.append(f"{table_name} record {i+1} field {field} should be string, got {type(value).__name__}")
                            
                            elif expected_type == 'float':
                                try:
                                    float(value)  # Try to convert to float
                                except (ValueError, TypeError):
                                    validation_warnings.append(f"{table_name} record {i+1} field {field} should be numeric, got {type(value).__name__}")
                            
                            elif expected_type == 'date':
                                if not isinstance(value, str) or not self._is_valid_date(value):
                                    validation_warnings.append(f"{table_name} record {i+1} field {field} has invalid date format: {value}")
            
            # Check for table relationships (referential integrity)
            # This would be more comprehensive in a full implementation
            
            # Sort and deduplicate issues
            unique_issues = list(set(validation_issues))
            unique_warnings = list(set(validation_warnings))
            
            # Count issues by table
            table_issues = {}
            for table_name in tables.keys():
                table_issues[table_name] = {
                    "errors": len([i for i in unique_issues if table_name in i]),
                    "warnings": len([w for w in unique_warnings if table_name in w]),
                    "records": len(tables[table_name]) if table_name in tables else 0
                }
            
            # Determine overall validation status - only errors make it invalid
            is_valid = len(unique_issues) == 0
            
            validation_results = {
                "valid": is_valid,
                "issues": unique_issues,
                "warnings": unique_warnings,
                "tables": table_issues,
                "summary": {
                    "total_records": sum(len(records) for records in tables.values()),
                    "total_tables": len(tables),
                    "tables_with_data": len([t for t, r in tables.items() if r]),
                    "error_count": len(unique_issues),
                    "warning_count": len(unique_warnings)
                }
            }
            
            logger.info(f"Validation completed with {len(unique_issues)} issues and {len(unique_warnings)} warnings")
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating OMOP data: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "summary": {
                    "error_count": 1,
                    "warning_count": 0
                }
            }
    
    def _is_valid_date(self, date_string):
        """Check if a string is a valid date in YYYY-MM-DD format."""
        try:
            if not date_string:
                return False
            # Check format
            if not isinstance(date_string, str) or len(date_string) != 10:
                return False
            if date_string[4] != '-' or date_string[7] != '-':
                return False
            # Parse date components
            year = int(date_string[0:4])
            month = int(date_string[5:7])
            day = int(date_string[8:10])
            # Basic validation
            if not (1900 <= year <= 2100):
                return False
            if not (1 <= month <= 12):
                return False
            if not (1 <= day <= 31):
                return False
            # More specific validation could be added
            return True
        except:
            return False
    
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
            schema_files = [
                'condition_occurrence',
                'drug_exposure',
                'procedure_occurrence',
                'observation',
                'measurement',
                'device_exposure',
                'specimen'
            ]
            
            for table_name in schema_files:
                schema_path = os.path.join(self.schema_dir, f"{table_name}.json")
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schemas[table_name] = json.load(f)
            
            # Define SQL Data Types based on dialect
            sql_type_map = {
                'sqlite': {
                    'integer': sa.Integer(),
                    'float': sa.Float(),
                    'string': sa.String(255),
                    'date': sa.Date(),
                    'datetime': sa.DateTime()
                },
                'postgresql': {
                    'integer': sa.Integer(),
                    'float': sa.Float(),
                    'string': sa.String(255),
                    'date': sa.Date(),
                    'datetime': sa.DateTime(timezone=True)
                },
                'mysql': {
                    'integer': sa.Integer(),
                    'float': sa.Float(),
                    'string': sa.String(255),
                    'date': sa.Date(),
                    'datetime': sa.DateTime()
                },
                'mssql': {
                    'integer': sa.Integer(),
                    'float': sa.Float(),
                    'string': sa.String(255),
                    'date': sa.Date(),
                    'datetime': sa.DateTime()
                }
            }
            
            # Use default type map if dialect not supported
            type_map = sql_type_map.get(dialect, sql_type_map['sqlite'])
            
            # Create table creation statements for all tables (even empty ones)
            # This ensures all schema tables are included
            all_tables_sql = {}
            for table_name, schema in schemas.items():
                # Define columns based on schema
                columns = []
                for field in schema['fields']:
                    # Map field type to SQLAlchemy type
                    field_type = field['type'].lower()
                    sa_type = type_map.get(field_type, sa.String(255))
                    
                    # Define column
                    column = sa.Column(
                        field['name'],
                        sa_type,
                        primary_key=field.get('primary_key', False),
                        nullable=not field.get('required', False)
                    )
                    columns.append(column)
                
                # Create metadata and table
                metadata = sa.MetaData()
                table = sa.Table(table_name, metadata, *columns)
                
                # Generate CREATE TABLE statement
                create_stmt = str(sa.schema.CreateTable(table).compile(engine))
                
                # Initialize table SQL with empty inserts
                all_tables_sql[table_name] = {
                    "create": create_stmt,
                    "inserts": []
                }
            
            # Generate INSERT statements for tables with data
            for table_name, records in tables.items():
                if not records or table_name not in all_tables_sql:
                    continue
                
                # Convert records to DataFrame
                df = pd.DataFrame(records)
                
                # Get the SQLAlchemy table definition
                metadata = sa.MetaData()
                columns = []
                
                if table_name in schemas:
                    schema = schemas[table_name]
                    for field in schema['fields']:
                        field_type = field['type'].lower()
                        sa_type = type_map.get(field_type, sa.String(255))
                        
                        column = sa.Column(
                            field['name'],
                            sa_type,
                            primary_key=field.get('primary_key', False),
                            nullable=not field.get('required', False)
                        )
                        columns.append(column)
                
                table = sa.Table(table_name, metadata, *columns)
                
                # Generate INSERT statements
                insert_statements = []
                for _, row in df.iterrows():
                    # Clean row data (replace NaN with None)
                    row_dict = {k: None if pd.isna(v) else v for k, v in row.to_dict().items()}
                    
                    # Filter out fields not in the schema
                    if table_name in schemas:
                        schema_fields = [f['name'] for f in schemas[table_name]['fields']]
                        row_dict = {k: v for k, v in row_dict.items() if k in schema_fields}
                    
                    # Generate insert statement
                    try:
                        insert = table.insert().values(**row_dict)
                        insert_stmt = str(insert.compile(engine, compile_kwargs={"literal_binds": True}))
                        insert_statements.append(insert_stmt)
                    except Exception as stmt_error:
                        logger.warning(f"Error generating insert statement for {table_name}: {str(stmt_error)}")
                        # Attempt a simpler INSERT format as fallback
                        fields = ', '.join(f'"{k}"' for k in row_dict.keys())
                        values = ', '.join(f"'{v}'" if isinstance(v, str) else str(v) if v is not None else "NULL" for v in row_dict.values())
                        simple_insert = f"INSERT INTO {table_name} ({fields}) VALUES ({values});"
                        insert_statements.append(simple_insert)
                
                # Update the inserts in the all_tables_sql
                all_tables_sql[table_name]['inserts'] = insert_statements
            
            # Generate index creation statements (optional enhancement)
            for table_name in all_tables_sql.keys():
                if table_name in schemas:
                    schema = schemas[table_name]
                    
                    # Add foreign key constraints (commented out for reference)
                    fk_fields = []
                    if table_name == 'condition_occurrence':
                        fk_fields = ['person_id', 'condition_concept_id', 'condition_type_concept_id']
                    elif table_name == 'drug_exposure':
                        fk_fields = ['person_id', 'drug_concept_id', 'drug_type_concept_id']
                    elif table_name == 'procedure_occurrence':
                        fk_fields = ['person_id', 'procedure_concept_id', 'procedure_type_concept_id']
                    elif table_name == 'observation':
                        fk_fields = ['person_id', 'observation_concept_id', 'observation_type_concept_id']
                    
                    # Add commented index statements for reference
                    index_statements = []
                    for field in fk_fields:
                        index_stmt = f"-- CREATE INDEX idx_{table_name}_{field} ON {table_name}({field});"
                        index_statements.append(index_stmt)
                    
                    if index_statements:
                        all_tables_sql[table_name]['indexes'] = index_statements
            
            # Add statements to enable foreign key constraints (SQLite specific)
            if dialect == 'sqlite':
                all_tables_sql['_preamble'] = {
                    "setup": ["PRAGMA foreign_keys = ON;"]
                }
            
            logger.info(f"Generated SQL statements for {len(all_tables_sql)} tables")
            return all_tables_sql
        
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}", exc_info=True)
            # Return a minimal SQL structure even if there's an error
            return {
                "_error": {
                    "message": f"Error generating SQL: {str(e)}",
                    "dialect": dialect
                }
            }
    
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