"""
FHIR Validator for the Clinical Protocol Extraction Engine.
This module validates FHIR resources created during protocol conversion.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Import FHIR resources library
from fhir.resources.plandefinition import PlanDefinition
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.library import Library
from fhir.resources.questionnaire import Questionnaire

# Set up logging
logger = logging.getLogger(__name__)

class FHIRValidator:
    """Validates FHIR resources against FHIR standards."""
    
    def __init__(self):
        """Initialize FHIR validator."""
        logger.info("Initializing FHIR validator")
        
        # Define required elements for each resource type
        self.required_elements = {
            "PlanDefinition": ["resourceType", "status"],
            "ActivityDefinition": ["resourceType", "status", "kind"],
            "Library": ["resourceType", "status", "type"],
            "Questionnaire": ["resourceType", "status"]
        }
        
        # Define allowed values for certain elements
        self.allowed_values = {
            "status": ["draft", "active", "retired", "unknown"],
            "kind": ["ServiceRequest", "MedicationRequest", "Task", "Appointment"]
        }
    
    def validate(self, resources):
        """
        Validate FHIR resources against FHIR standards.
        
        Args:
            resources (dict): FHIR resources to validate
            
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Starting FHIR resource validation")
            
            validation_issues = []
            
            # Validate PlanDefinition
            if "planDefinition" in resources:
                plan_def_issues = self.validate_plan_definition(resources["planDefinition"])
                validation_issues.extend(plan_def_issues)
            
            # Validate ActivityDefinitions
            if "activityDefinitions" in resources:
                activity_def_issues = self.validate_activity_definitions(resources["activityDefinitions"])
                validation_issues.extend(activity_def_issues)
            
            # Validate Library
            if "library" in resources:
                library_issues = self.validate_library(resources["library"])
                validation_issues.extend(library_issues)
            
            # Validate Questionnaire
            if "questionnaire" in resources:
                questionnaire_issues = self.validate_questionnaire(resources["questionnaire"])
                validation_issues.extend(questionnaire_issues)
            
            # Check for resource references integrity
            reference_issues = self.validate_references(resources)
            validation_issues.extend(reference_issues)
            
            # Determine overall validation status
            is_valid = len(validation_issues) == 0
            
            validation_result = {
                "valid": is_valid,
                "issues": validation_issues,
                "validated_at": datetime.now().isoformat(),
                "fhir_version": "4.0.1"  # FHIR R4
            }
            
            logger.info(f"FHIR validation completed with {len(validation_issues)} issues")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during FHIR validation: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "validated_at": datetime.now().isoformat(),
                "fhir_version": "4.0.1"
            }
    
    def validate_plan_definition(self, plan_definition):
        """
        Validate PlanDefinition resource.
        
        Args:
            plan_definition (dict): PlanDefinition resource
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        # Check resource type
        if plan_definition.get("resourceType") != "PlanDefinition":
            issues.append("Invalid resourceType for PlanDefinition")
        
        # Check required elements
        for element in self.required_elements["PlanDefinition"]:
            if element not in plan_definition or not plan_definition[element]:
                issues.append(f"PlanDefinition missing required element: {element}")
        
        # Check status value
        if "status" in plan_definition and plan_definition["status"] not in self.allowed_values["status"]:
            issues.append(f"PlanDefinition has invalid status: {plan_definition['status']}")
        
        # Check actions structure
        if "action" in plan_definition:
            actions = plan_definition["action"]
            if not isinstance(actions, list):
                issues.append("PlanDefinition action must be an array")
            else:
                for i, action in enumerate(actions):
                    # Check action has title or description
                    if not action.get("title") and not action.get("description"):
                        issues.append(f"PlanDefinition action[{i}] missing both title and description")
                    
                    # Check condition structure if present
                    if "condition" in action:
                        if not isinstance(action["condition"], list):
                            issues.append(f"PlanDefinition action[{i}] condition must be an array")
                        else:
                            for j, condition in enumerate(action["condition"]):
                                if "kind" not in condition:
                                    issues.append(f"PlanDefinition action[{i}] condition[{j}] missing kind")
                                if "expression" not in condition:
                                    issues.append(f"PlanDefinition action[{i}] condition[{j}] missing expression")
        
        return issues
    
    def validate_activity_definitions(self, activity_definitions):
        """
        Validate ActivityDefinition resources.
        
        Args:
            activity_definitions (list): ActivityDefinition resources
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        if not isinstance(activity_definitions, list):
            issues.append("ActivityDefinitions must be an array")
            return issues
        
        for i, activity_def in enumerate(activity_definitions):
            # Check resource type
            if activity_def.get("resourceType") != "ActivityDefinition":
                issues.append(f"ActivityDefinition[{i}] has invalid resourceType")
            
            # Check required elements
            for element in self.required_elements["ActivityDefinition"]:
                if element not in activity_def or not activity_def[element]:
                    issues.append(f"ActivityDefinition[{i}] missing required element: {element}")
            
            # Check status value
            if "status" in activity_def and activity_def["status"] not in self.allowed_values["status"]:
                issues.append(f"ActivityDefinition[{i}] has invalid status: {activity_def['status']}")
            
            # Check kind value
            if "kind" in activity_def and activity_def["kind"] not in self.allowed_values["kind"]:
                issues.append(f"ActivityDefinition[{i}] has invalid kind: {activity_def['kind']}")
            
            # Check participant structure if present
            if "participant" in activity_def:
                if not isinstance(activity_def["participant"], list):
                    issues.append(f"ActivityDefinition[{i}] participant must be an array")
                else:
                    for j, participant in enumerate(activity_def["participant"]):
                        if "type" not in participant:
                            issues.append(f"ActivityDefinition[{i}] participant[{j}] missing type")
        
        return issues
    
    def validate_library(self, library):
        """
        Validate Library resource.
        
        Args:
            library (dict): Library resource
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        # Check resource type
        if library.get("resourceType") != "Library":
            issues.append("Invalid resourceType for Library")
        
        # Check required elements
        for element in self.required_elements["Library"]:
            if element not in library or not library[element]:
                issues.append(f"Library missing required element: {element}")
        
        # Check status value
        if "status" in library and library["status"] not in self.allowed_values["status"]:
            issues.append(f"Library has invalid status: {library['status']}")
        
        # Check type structure
        if "type" in library:
            if not isinstance(library["type"], dict):
                issues.append("Library type must be a CodeableConcept")
            else:
                if "coding" not in library["type"] or not isinstance(library["type"]["coding"], list):
                    issues.append("Library type.coding must be an array")
        
        return issues
    
    def validate_questionnaire(self, questionnaire):
        """
        Validate Questionnaire resource.
        
        Args:
            questionnaire (dict): Questionnaire resource
            
        Returns:
            list: Validation issues
        """
        issues = []
        
        # Check resource type
        if questionnaire.get("resourceType") != "Questionnaire":
            issues.append("Invalid resourceType for Questionnaire")
        
        # Check required elements
        for element in self.required_elements["Questionnaire"]:
            if element not in questionnaire or not questionnaire[element]:
                issues.append(f"Questionnaire missing required element: {element}")
        
        # Check status value
        if "status" in questionnaire and questionnaire["status"] not in self.allowed_values["status"]:
            issues.append(f"Questionnaire has invalid status: {questionnaire['status']}")
        
        # Check items structure if present
        if "item" in questionnaire:
            if not isinstance(questionnaire["item"], list):
                issues.append("Questionnaire item must be an array")
            else:
                for i, item in enumerate(questionnaire["item"]):
                    # Check required item elements
                    if "linkId" not in item:
                        issues.append(f"Questionnaire item[{i}] missing linkId")
                    if "type" not in item:
                        issues.append(f"Questionnaire item[{i}] missing type")
                    
                    # Check for valid item types
                    valid_types = ["group", "display", "boolean", "decimal", "integer", "date", "dateTime", 
                                   "time", "string", "text", "url", "choice", "open-choice", "attachment", 
                                   "reference", "quantity"]
                    if "type" in item and item["type"] not in valid_types:
                        issues.append(f"Questionnaire item[{i}] has invalid type: {item['type']}")
        
        return issues
    
    def validate_references(self, resources):
        """
        Validate references between resources.
        
        Args:
            resources (dict): All FHIR resources
            
        Returns:
            list: Validation issues related to references
        """
        issues = []
        
        # Check references from PlanDefinition to ActivityDefinition
        if "planDefinition" in resources and "action" in resources["planDefinition"]:
            actions = resources["planDefinition"]["action"]
            if isinstance(actions, list):
                activity_def_ids = []
                
                # Collect all ActivityDefinition IDs
                if "activityDefinitions" in resources:
                    for activity_def in resources["activityDefinitions"]:
                        if "id" in activity_def:
                            activity_def_ids.append(f"ActivityDefinition/{activity_def['id']}")
                
                # Check references in actions
                for i, action in enumerate(actions):
                    if "definitionCanonical" in action:
                        ref = action["definitionCanonical"]
                        if ref.startswith("ActivityDefinition/") and ref not in activity_def_ids:
                            issues.append(f"PlanDefinition action[{i}] references non-existent ActivityDefinition: {ref}")
        
        return issues
    
    def validate_fhir_resource(self, resource_type, resource):
        """
        Validate a single FHIR resource against its schema.
        
        Args:
            resource_type (str): Type of FHIR resource
            resource (dict): FHIR resource to validate
            
        Returns:
            dict: Validation result
        """
        try:
            # Map resource type to validation function
            validation_functions = {
                "PlanDefinition": self.validate_plan_definition,
                "ActivityDefinition": lambda r: self.validate_activity_definitions([r]),
                "Library": self.validate_library,
                "Questionnaire": self.validate_questionnaire
            }
            
            if resource_type not in validation_functions:
                return {
                    "valid": False,
                    "issues": [f"Unsupported resource type: {resource_type}"]
                }
            
            # Validate resource
            issues = validation_functions[resource_type](resource)
            
            return {
                "valid": len(issues) == 0,
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Error validating {resource_type}: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"]
            }
    
    def get_fhir_version(self):
        """Return the FHIR version used by the validator."""
        return "4.0.1"  # FHIR R4