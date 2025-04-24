"""
FHIR Converter for the Clinical Protocol Extraction Engine.
This module handles the conversion of extracted protocol data to FHIR resources.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Import FHIR resources library
from fhir.resources.plandefinition import PlanDefinition
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.library import Library
from fhir.resources.questionnaire import Questionnaire
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.identifier import Identifier
from fhir.resources.period import Period
from fhir.resources.fhirtypes import Code, Uri, String, Boolean, DateTime

# Import our terminology mapper
from standards.terminology.mapper import TerminologyMapper

# Set up logging
logger = logging.getLogger(__name__)

class FHIRConverter:
    """Converts extracted protocol data to FHIR resources."""
    
    def __init__(self, terminology_mapper=None, config=None):
        """
        Initialize FHIR converter with optional terminology mapper.
        
        Args:
            terminology_mapper: Optional mapper for clinical terminology
            config: Optional configuration dictionary
        """
        # Create a new terminology mapper if one wasn't provided
        if terminology_mapper is None:
            logger.info("No terminology mapper provided, creating a new one")
            self.terminology_mapper = TerminologyMapper(config)
        else:
            self.terminology_mapper = terminology_mapper
        
        # Load template directory path
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            self._create_default_templates()
        
        logger.info(f"Initialized FHIR converter with template directory: {self.template_dir}")
    
    def _create_default_templates(self):
        """Create default FHIR resource templates if they don't exist."""
        # Create PlanDefinition template
        plan_definition_template = {
            "resourceType": "PlanDefinition",
            "status": "draft",
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/plan-definition-type",
                    "code": "protocol",
                    "display": "Protocol"
                }]
            },
            "action": []
        }
        
        # Create ActivityDefinition template
        activity_definition_template = {
            "resourceType": "ActivityDefinition",
            "status": "draft",
            "kind": "ServiceRequest",
            "participant": [{
                "type": "practitioner",
                "role": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/participant-role",
                        "code": "performer",
                        "display": "Performer"
                    }]
                }
            }]
        }
        
        # Create Library template
        library_template = {
            "resourceType": "Library",
            "status": "draft",
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/library-type",
                    "code": "logic-library",
                    "display": "Logic Library"
                }]
            },
            "content": []
        }
        
        # Create Questionnaire template
        questionnaire_template = {
            "resourceType": "Questionnaire",
            "status": "draft",
            "item": []
        }
        
        # Save templates
        with open(os.path.join(self.template_dir, 'plan_definition.json'), 'w') as f:
            json.dump(plan_definition_template, f, indent=2)
        
        with open(os.path.join(self.template_dir, 'activity_definition.json'), 'w') as f:
            json.dump(activity_definition_template, f, indent=2)
        
        with open(os.path.join(self.template_dir, 'library.json'), 'w') as f:
            json.dump(library_template, f, indent=2)
        
        with open(os.path.join(self.template_dir, 'questionnaire.json'), 'w') as f:
            json.dump(questionnaire_template, f, indent=2)
        
        logger.info("Created default FHIR resource templates")
    
    def convert(self, protocol_data):
        """
        Convert protocol data to FHIR resources.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            dict: FHIR resources with validation results
        """
        try:
            logger.info("Converting protocol data to FHIR resources")
            
            # First map all clinical terms to standard terminologies
            mapped_data = self.map_extracted_data(protocol_data)
            
            # Generate FHIR resources using the mapped data
            plan_definition = self.create_plan_definition(mapped_data)
            activity_definitions = self.create_activity_definitions(mapped_data)
            library = self.create_library(mapped_data)
            questionnaire = self.create_questionnaire(mapped_data)
            
            # Package resources
            resources = {
                "planDefinition": plan_definition.dict(),
                "activityDefinitions": [ad.dict() for ad in activity_definitions],
                "library": library.dict(),
                "questionnaire": questionnaire.dict()
            }
            
            # Validate resources
            validation_results = self.validate_resources(resources)
            
            logger.info(f"FHIR conversion completed with validation status: {validation_results['valid']}")
            
            # Return both resources and mapping statistics
            mapping_stats = self.terminology_mapper.get_statistics() if hasattr(self.terminology_mapper, 'get_statistics') else {}
            
            return {
                "resources": resources,
                "validation": validation_results,
                "terminology_mapping": {
                    "statistics": mapping_stats
                }
            }
        
        except Exception as e:
            logger.error(f"Error converting to FHIR: {str(e)}", exc_info=True)
            raise
    
    def create_plan_definition(self, protocol_data):
        """
        Create PlanDefinition resource from protocol data.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            PlanDefinition: FHIR PlanDefinition resource
        """
        try:
            logger.info("Creating PlanDefinition resource")
            
            # Create unique ID for resource
            resource_id = f"protocol-{str(uuid.uuid4())[:8]}"
            
            # Create PlanDefinition
            plan_definition = PlanDefinition.construct()
            
            # Set required fields FIRST
            plan_definition.status = "draft"
            
            # Then set other fields
            plan_definition.id = resource_id
            plan_definition.date = datetime.now().strftime("%Y-%m-%d")
            
            # Add title and description if available
            if 'title' in protocol_data:
                plan_definition.title = protocol_data['title']
            
            if 'description' in protocol_data:
                plan_definition.description = protocol_data['description']
            
            # Add protocol type
            plan_definition.type = CodeableConcept(
                coding=[
                    Coding(
                        system="http://terminology.hl7.org/CodeSystem/plan-definition-type",
                        code="protocol",
                        display="Protocol"
                    )
                ]
            )
            
            # Add identifiers
            if 'protocol_id' in protocol_data:
                plan_definition.identifier = [
                    Identifier(
                        system="http://example.org/fhir/identifier/protocols",
                        value=protocol_data['protocol_id']
                    )
                ]
            
            # Add usage context if available
            if 'usage' in protocol_data:
                pass  # Implement usage context here
            
            # Add actions for eligibility criteria
            actions = []
            
            if 'eligibility_criteria' in protocol_data:
                for criterion in protocol_data['eligibility_criteria']:
                    criterion_type = criterion.get('type', '').lower()
                    criterion_text = criterion.get('text', '')
                    
                    if criterion_type == 'inclusion':
                        # Create inclusion criterion action
                        action = {
                            "title": "Inclusion Criterion",
                            "description": criterion_text,
                            "condition": [{
                                "kind": "applicability",
                                "expression": {
                                    "language": "text/plain",
                                    "expression": criterion_text
                                }
                            }]
                        }
                        actions.append(action)
                    
                    elif criterion_type == 'exclusion':
                        # Create exclusion criterion action
                        action = {
                            "title": "Exclusion Criterion",
                            "description": criterion_text,
                            "condition": [{
                                "kind": "applicability",
                                "expression": {
                                    "language": "text/plain",
                                    "expression": criterion_text,
                                    "extension": [{
                                        "url": "http://example.org/fhir/StructureDefinition/exclusion",
                                        "valueBoolean": True
                                    }]
                                }
                            }]
                        }
                        actions.append(action)
            
            # Add actions for procedures
            if 'procedures' in protocol_data:
                for procedure in protocol_data['procedures']:
                    procedure_text = procedure.get('text', '')
                    
                    # Map to terminology using our mapper
                    procedure_code = None
                    if procedure_text:
                        mapping_result = self.terminology_mapper.map_to_snomed(procedure_text)
                        if mapping_result and mapping_result.get("found", False):
                            procedure_code = mapping_result.get("code")
                            # Create a coded concept for the procedure
                            code_concept = {
                                "coding": [{
                                    "system": mapping_result.get("system", "http://snomed.info/sct"),
                                    "code": mapping_result.get("code"),
                                    "display": mapping_result.get("display")
                                }],
                                "text": procedure_text
                            }
                    
                    # Create procedure action
                    action = {
                        "title": procedure_text,
                        "description": procedure.get('description', ''),
                        "type": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/action-type",
                                "code": "create",
                                "display": "Create"
                            }]
                        }
                    }
                    
                    # Add reference to ActivityDefinition if we have a code
                    if procedure_code:
                        action["definitionCanonical"] = f"ActivityDefinition/activity-{procedure_code}"
                    
                    actions.append(action)
            
            # Add timing information if available
            if 'timing' in protocol_data:
                for timing_info in protocol_data['timing']:
                    timing_text = timing_info.get('text', '')
                    action = {
                        "title": "Timing Information",
                        "description": timing_text,
                        "timingPeriod": {
                            "start": timing_info.get('start'),
                            "end": timing_info.get('end')
                        }
                    }
                    actions.append(action)
            
            # Set actions in PlanDefinition
            if actions:
                plan_definition.action = actions
            
            logger.info(f"Created PlanDefinition with {len(actions)} actions")
            return plan_definition
        
        except Exception as e:
            logger.error(f"Error creating PlanDefinition: {str(e)}", exc_info=True)
            raise
    
    def create_activity_definitions(self, protocol_data):
        """
        Create ActivityDefinition resources from protocol data.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            list: FHIR ActivityDefinition resources
        """
        try:
            logger.info("Creating ActivityDefinition resources")
            activity_definitions = []
            
            # Process procedures
            if 'procedures' in protocol_data and protocol_data['procedures']:
                for procedure in protocol_data['procedures']:
                    # Create unique ID for resource
                    resource_id = f"activity-{str(uuid.uuid4())[:8]}"
                    
                    # Create ActivityDefinition
                    activity_definition = ActivityDefinition.construct()
                    
                    # Set required fields FIRST
                    activity_definition.status = "draft"
                    
                    # Then set other fields
                    activity_definition.id = resource_id
                    activity_definition.date = datetime.now().strftime("%Y-%m-%d")
                    
                    # Add procedure details
                    activity_definition.title = procedure.get('text', '')
                    activity_definition.description = procedure.get('description', '')
                    
                    # Set kind
                    activity_definition.kind = "ServiceRequest"
                    
                    # Add participant information
                    activity_definition.participant = [{
                        "type": "practitioner",
                        "role": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/participant-role",
                                "code": "performer",
                                "display": "Performer"
                            }]
                        }
                    }]
                    
                    # Add timing information if available
                    if 'timing' in procedure:
                        pass  # Implement timing here
                    
                    # Add to collection
                    activity_definitions.append(activity_definition)
            
            # Process medications
            if 'medications' in protocol_data and protocol_data['medications']:
                for medication in protocol_data['medications']:
                    # Create unique ID for resource
                    resource_id = f"activity-{str(uuid.uuid4())[:8]}"
                    
                    # Create ActivityDefinition
                    activity_definition = ActivityDefinition.construct()
                    
                    # Set required fields FIRST
                    activity_definition.status = "draft"
                    
                    # Then set other fields
                    activity_definition.id = resource_id
                    activity_definition.date = datetime.now().strftime("%Y-%m-%d")
                    
                    # Add medication details
                    activity_definition.title = medication.get('text', '')
                    activity_definition.description = medication.get('description', '')
                    
                    # Set kind
                    activity_definition.kind = "MedicationRequest"
                    
                    # Add dosage information if available
                    if 'dosage' in medication:
                        pass  # Implement dosage here
                    
                    # Add to collection
                    activity_definitions.append(activity_definition)
            
            logger.info(f"Created {len(activity_definitions)} ActivityDefinition resources")
            return activity_definitions
        
        except Exception as e:
            logger.error(f"Error creating ActivityDefinitions: {str(e)}", exc_info=True)
            raise
    
    def create_library(self, protocol_data):
        """
        Create Library resource from protocol data.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            Library: FHIR Library resource
        """
        try:
            logger.info("Creating Library resource")
            
            # Create unique ID for resource
            resource_id = f"library-{str(uuid.uuid4())[:8]}"
            
            # Create Library
            library = Library.construct()
            
            # Set required fields FIRST
            library.status = "draft"
            
            # Then set other fields
            library.id = resource_id
            library.date = datetime.now().strftime("%Y-%m-%d")
            
            # Add title and description if available
            if 'title' in protocol_data:
                library.title = f"Logic for {protocol_data['title']}"
            
            if 'description' in protocol_data:
                library.description = protocol_data['description']
            
            # Set type
            library.type = CodeableConcept(
                coding=[
                    Coding(
                        system="http://terminology.hl7.org/CodeSystem/library-type",
                        code="logic-library",
                        display="Logic Library"
                    )
                ]
            )
            
            # Add content (in a real implementation, this would include logic expressions)
            # For now, just create a placeholder content entry
            library.content = []
            
            logger.info(f"Created Library resource: {resource_id}")
            return library
        
        except Exception as e:
            logger.error(f"Error creating Library: {str(e)}", exc_info=True)
            raise
    
    def create_questionnaire(self, protocol_data):
        """
        Create Questionnaire resource from protocol data.
        
        Args:
            protocol_data (dict): Extracted protocol data
            
        Returns:
            Questionnaire: FHIR Questionnaire resource
        """
        try:
            logger.info("Creating Questionnaire resource")
            
            # Create unique ID for resource
            resource_id = f"questionnaire-{str(uuid.uuid4())[:8]}"
            
            # Create Questionnaire
            questionnaire = Questionnaire.construct()
            
            # Set required fields FIRST
            questionnaire.status = "draft"
            
            # Then set other fields
            questionnaire.id = resource_id
            questionnaire.date = datetime.now().strftime("%Y-%m-%d")
            
            # Add title
            if 'title' in protocol_data:
                questionnaire.title = f"Data Collection for {protocol_data['title']}"
            
            # Create questionnaire items based on endpoints/measurements
            items = []
            
            # Add items for endpoints
            if 'endpoints' in protocol_data and protocol_data['endpoints']:
                for endpoint in protocol_data['endpoints']:
                    endpoint_text = endpoint.get('text', '')
                    
                    # Create item
                    item = {
                        "linkId": f"endpoint-{len(items)+1}",
                        "text": endpoint_text,
                        "type": "string"  # Default to string, adjust based on data type
                    }
                    
                    items.append(item)
            
            # Add items for measurements
            if 'measurements' in protocol_data and protocol_data['measurements']:
                for measurement in protocol_data['measurements']:
                    measurement_text = measurement.get('text', '')
                    
                    # Determine appropriate type based on measurement
                    item_type = "decimal"  # Default to decimal
                    
                    # Create item
                    item = {
                        "linkId": f"measurement-{len(items)+1}",
                        "text": measurement_text,
                        "type": item_type
                    }
                    
                    # Add units if available
                    if 'units' in measurement:
                        units = measurement['units']
                        # Add units extension
                        item["extension"] = [{
                            "url": "http://hl7.org/fhir/StructureDefinition/questionnaire-unit",
                            "valueCoding": {
                                "display": units
                            }
                        }]
                    
                    items.append(item)
            
            # Set items in Questionnaire
            if items:
                questionnaire.item = items
            
            logger.info(f"Created Questionnaire with {len(items)} items")
            return questionnaire
        
        except Exception as e:
            logger.error(f"Error creating Questionnaire: {str(e)}", exc_info=True)
            raise
    
    def validate_resources(self, resources):
        """
        Validate FHIR resources.
        
        Args:
            resources (dict): FHIR resources to validate
            
        Returns:
            dict: Validation results
        """
        # In a real implementation, this would use a FHIR validator
        # For now, we'll do basic validation checks
        
        try:
            logger.info("Validating FHIR resources")
            validation_issues = []
            
            # Check PlanDefinition
            if 'planDefinition' in resources:
                plan_def = resources['planDefinition']
                if not plan_def.get('status'):
                    validation_issues.append("PlanDefinition missing required 'status' element")
                if not plan_def.get('action') and not isinstance(plan_def.get('action'), list):
                    validation_issues.append("PlanDefinition should have at least one action")
            
            # Check ActivityDefinitions
            if 'activityDefinitions' in resources:
                activity_defs = resources['activityDefinitions']
                for i, ad in enumerate(activity_defs):
                    if not ad.get('status'):
                        validation_issues.append(f"ActivityDefinition[{i}] missing required 'status' element")
                    if not ad.get('kind'):
                        validation_issues.append(f"ActivityDefinition[{i}] missing required 'kind' element")
            
            # Check Library
            if 'library' in resources:
                library = resources['library']
                if not library.get('status'):
                    validation_issues.append("Library missing required 'status' element")
                if not library.get('type'):
                    validation_issues.append("Library missing required 'type' element")
            
            # Check Questionnaire
            if 'questionnaire' in resources:
                questionnaire = resources['questionnaire']
                if not questionnaire.get('status'):
                    validation_issues.append("Questionnaire missing required 'status' element")
            
            # Determine overall validation status
            is_valid = len(validation_issues) == 0
            
            validation_results = {
                "valid": is_valid,
                "issues": validation_issues
            }
            
            logger.info(f"Validation completed with {len(validation_issues)} issues")
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating FHIR resources: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"]
            }
    
    def get_fhir_version(self):
        """Return the FHIR version used by the converter."""
        return "4.0.1"  # FHIR R4
        
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
            logger.info("Mapping extracted clinical terms to standard terminologies")
            
            # Create a copy of the data to avoid modifying the original
            mapped_data = extracted_data.copy()
            
            # Map procedures to SNOMED CT
            if 'procedures' in mapped_data and mapped_data['procedures']:
                for i, procedure in enumerate(mapped_data['procedures']):
                    if 'text' in procedure:
                        mapping_result = self.terminology_mapper.map_to_snomed(procedure['text'], 'procedure')
                        if mapping_result:
                            mapped_data['procedures'][i]['terminology'] = {
                                'system': mapping_result.get('system'),
                                'code': mapping_result.get('code'),
                                'display': mapping_result.get('display'),
                                'found': mapping_result.get('found', False)
                            }
            
            # Map medications to RxNorm
            if 'medications' in mapped_data and mapped_data['medications']:
                for i, medication in enumerate(mapped_data['medications']):
                    if 'text' in medication:
                        mapping_result = self.terminology_mapper.map_to_rxnorm(medication['text'], 'medication')
                        if mapping_result:
                            mapped_data['medications'][i]['terminology'] = {
                                'system': mapping_result.get('system'),
                                'code': mapping_result.get('code'),
                                'display': mapping_result.get('display'),
                                'found': mapping_result.get('found', False)
                            }
            
            # Map measurements to LOINC
            if 'measurements' in mapped_data and mapped_data['measurements']:
                for i, measurement in enumerate(mapped_data['measurements']):
                    if 'text' in measurement:
                        mapping_result = self.terminology_mapper.map_to_loinc(measurement['text'], 'measurement')
                        if mapping_result:
                            mapped_data['measurements'][i]['terminology'] = {
                                'system': mapping_result.get('system'),
                                'code': mapping_result.get('code'),
                                'display': mapping_result.get('display'),
                                'found': mapping_result.get('found', False)
                            }
            
            # Map conditions to SNOMED CT
            if 'conditions' in mapped_data and mapped_data['conditions']:
                for i, condition in enumerate(mapped_data['conditions']):
                    if 'text' in condition:
                        mapping_result = self.terminology_mapper.map_to_snomed(condition['text'], 'condition')
                        if mapping_result:
                            mapped_data['conditions'][i]['terminology'] = {
                                'system': mapping_result.get('system'),
                                'code': mapping_result.get('code'),
                                'display': mapping_result.get('display'),
                                'found': mapping_result.get('found', False)
                            }
            
            logger.info("Terminology mapping completed")
            return mapped_data
        
        except Exception as e:
            logger.error(f"Error mapping terminology: {str(e)}", exc_info=True)
            # Return original data if mapping fails
            return extracted_data