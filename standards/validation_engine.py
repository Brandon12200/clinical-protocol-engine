"""
Validation Engine for the Clinical Protocol Extraction Engine.
This module provides a unified interface for validating healthcare data standards.
"""

import os
import json
import logging
from datetime import datetime

from standards.fhir.validators import FHIRValidator
from standards.omop.validators import OMOPValidator

# Set up logging
logger = logging.getLogger(__name__)

class ValidationEngine:
    """Validates converted data against healthcare standards."""
    
    def __init__(self):
        """Initialize validation engine with standard-specific validators."""
        logger.info("Initializing ValidationEngine")
        
        # Initialize validators
        self.fhir_validator = FHIRValidator()
        self.omop_validator = OMOPValidator()
    
    def validate_fhir(self, resources):
        """
        Validate FHIR resources against FHIR standards.
        
        Args:
            resources (dict): FHIR resources to validate
            
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Validating FHIR resources")
            
            # Use the FHIR validator
            validation_results = self.fhir_validator.validate(resources)
            
            logger.info(f"FHIR validation completed with status: {validation_results['valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in FHIR validation: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "validated_at": datetime.now().isoformat(),
                "standard": "FHIR",
                "version": self.fhir_validator.get_fhir_version()
            }
    
    def validate_omop(self, omop_data):
        """
        Validate OMOP CDM data against OMOP standards.
        
        Args:
            omop_data (dict): OMOP tables to validate
            
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Validating OMOP CDM data")
            
            # Use the OMOP validator
            validation_results = self.omop_validator.validate(omop_data)
            
            logger.info(f"OMOP validation completed with status: {validation_results['valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in OMOP validation: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "validated_at": datetime.now().isoformat(),
                "standard": "OMOP CDM",
                "version": self.omop_validator.get_cdm_version()
            }
    
    def check_cross_standard_consistency(self, fhir_resources, omop_data):
        """
        Verify consistency between FHIR and OMOP representations.
        
        Args:
            fhir_resources (dict): FHIR resources
            omop_data (dict): OMOP tables
            
        Returns:
            dict: Consistency check results
        """
        try:
            logger.info("Checking cross-standard consistency")
            
            consistency_issues = []
            
            # Check that all significant entities are represented in both standards
            # This requires knowledge of the mapping between standards
            
            # 1. Check conditions/diagnoses
            fhir_conditions = self._extract_fhir_conditions(fhir_resources)
            omop_conditions = self._extract_omop_conditions(omop_data)
            
            condition_diff = len(fhir_conditions) - len(omop_conditions)
            if abs(condition_diff) > 0:
                consistency_issues.append(
                    f"Condition count mismatch: {len(fhir_conditions)} in FHIR, {len(omop_conditions)} in OMOP"
                )
            
            # 2. Check procedures
            fhir_procedures = self._extract_fhir_procedures(fhir_resources)
            omop_procedures = self._extract_omop_procedures(omop_data)
            
            procedure_diff = len(fhir_procedures) - len(omop_procedures)
            if abs(procedure_diff) > 0:
                consistency_issues.append(
                    f"Procedure count mismatch: {len(fhir_procedures)} in FHIR, {len(omop_procedures)} in OMOP"
                )
            
            # 3. Check medications
            fhir_medications = self._extract_fhir_medications(fhir_resources)
            omop_medications = self._extract_omop_medications(omop_data)
            
            medication_diff = len(fhir_medications) - len(omop_medications)
            if abs(medication_diff) > 0:
                consistency_issues.append(
                    f"Medication count mismatch: {len(fhir_medications)} in FHIR, {len(omop_medications)} in OMOP"
                )
            
            # 4. Check observations/measurements
            fhir_observations = self._extract_fhir_observations(fhir_resources)
            omop_observations = self._extract_omop_observations(omop_data)
            
            observation_diff = len(fhir_observations) - len(omop_observations)
            if abs(observation_diff) > 0:
                consistency_issues.append(
                    f"Observation count mismatch: {len(fhir_observations)} in FHIR, {len(omop_observations)} in OMOP"
                )
            
            # Determine overall consistency status
            is_consistent = len(consistency_issues) == 0
            
            consistency_results = {
                "consistent": is_consistent,
                "issues": consistency_issues,
                "checked_at": datetime.now().isoformat(),
                "fhir_version": self.fhir_validator.get_fhir_version(),
                "omop_version": self.omop_validator.get_cdm_version()
            }
            
            logger.info(f"Cross-standard consistency check completed with status: {is_consistent}")
            return consistency_results
            
        except Exception as e:
            logger.error(f"Error in cross-standard consistency check: {str(e)}", exc_info=True)
            return {
                "consistent": False,
                "issues": [f"Consistency check error: {str(e)}"],
                "checked_at": datetime.now().isoformat(),
                "fhir_version": self.fhir_validator.get_fhir_version(),
                "omop_version": self.omop_validator.get_cdm_version()
            }
    
    def generate_validation_report(self, validation_results):
        """
        Create human-readable validation report.
        
        Args:
            validation_results (dict): Validation results to report
            
        Returns:
            dict: Formatted validation report
        """
        try:
            logger.info("Generating validation report")
            
            # Determine if this is FHIR or OMOP validation
            standard = validation_results.get('standard', 'Unknown')
            if 'fhir_version' in validation_results:
                standard = 'FHIR'
            elif 'cdm_version' in validation_results:
                standard = 'OMOP CDM'
            
            # Format issues for better readability
            formatted_issues = []
            for issue in validation_results.get('issues', []):
                formatted_issues.append({
                    "description": issue,
                    "severity": "Error" if "missing required" in issue else "Warning"
                })
            
            # Generate summary statistics
            issue_count = len(formatted_issues)
            error_count = sum(1 for i in formatted_issues if i['severity'] == 'Error')
            warning_count = issue_count - error_count
            
            # Create report
            report = {
                "standard": standard,
                "valid": validation_results.get('valid', False),
                "summary": {
                    "total_issues": issue_count,
                    "errors": error_count,
                    "warnings": warning_count
                },
                "issues": formatted_issues,
                "generated_at": datetime.now().isoformat(),
                "recommendations": self._generate_recommendations(formatted_issues, standard)
            }
            
            logger.info(f"Validation report generated with {issue_count} issues")
            return report
            
        except Exception as e:
            logger.error(f"Error generating validation report: {str(e)}", exc_info=True)
            return {
                "standard": "Unknown",
                "valid": False,
                "summary": {
                    "total_issues": 1,
                    "errors": 1,
                    "warnings": 0
                },
                "issues": [{
                    "description": f"Report generation error: {str(e)}",
                    "severity": "Error"
                }],
                "generated_at": datetime.now().isoformat(),
                "recommendations": ["Review the validation process for errors."]
            }
    
    def _generate_recommendations(self, issues, standard):
        """
        Generate recommendations based on validation issues.
        
        Args:
            issues (list): Formatted validation issues
            standard (str): Standard being validated
            
        Returns:
            list: Recommendations for resolving issues
        """
        recommendations = []
        
        # Count issue types
        missing_required = sum(1 for i in issues if "missing required" in i['description'])
        invalid_type = sum(1 for i in issues if "should be" in i['description'])
        unmapped_concepts = sum(1 for i in issues if "unmapped" in i['description'])
        
        # Generate standard-specific recommendations
        if standard == 'FHIR':
            if missing_required > 0:
                recommendations.append(f"Add required elements that are missing in {missing_required} instances.")
            
            if "PlanDefinition should have at least one action" in str(issues):
                recommendations.append("Add actions to the PlanDefinition resource.")
            
            if "reference" in str(issues):
                recommendations.append("Fix invalid references between resources.")
        
        elif standard == 'OMOP CDM':
            if missing_required > 0:
                recommendations.append(f"Add required fields that are missing in {missing_required} instances.")
            
            if invalid_type > 0:
                recommendations.append(f"Fix data type issues in {invalid_type} fields.")
            
            if unmapped_concepts > 0:
                recommendations.append("Improve terminology mapping to reduce unmapped concepts.")
        
        # Add general recommendations
        if not recommendations:
            if len(issues) > 0:
                recommendations.append("Review each issue individually and address according to standard requirements.")
            else:
                recommendations.append("No issues found. The data meets validation requirements.")
        
        return recommendations
    
    # Helper methods for cross-standard consistency check
    
    def _extract_fhir_conditions(self, fhir_resources):
        """Extract conditions from FHIR resources."""
        conditions = []
        
        # Check PlanDefinition for eligibility criteria conditions
        if 'planDefinition' in fhir_resources and 'action' in fhir_resources['planDefinition']:
            actions = fhir_resources['planDefinition']['action']
            for action in actions:
                if 'condition' in action:
                    conditions.append(action)
        
        return conditions
    
    def _extract_omop_conditions(self, omop_data):
        """Extract conditions from OMOP tables."""
        if 'condition_occurrence' in omop_data:
            return omop_data['condition_occurrence']
        return []
    
    def _extract_fhir_procedures(self, fhir_resources):
        """Extract procedures from FHIR resources."""
        procedures = []
        
        # Check ActivityDefinitions for procedures
        if 'activityDefinitions' in fhir_resources:
            for ad in fhir_resources['activityDefinitions']:
                if ad.get('kind') == 'ServiceRequest':
                    procedures.append(ad)
        
        return procedures
    
    def _extract_omop_procedures(self, omop_data):
        """Extract procedures from OMOP tables."""
        if 'procedure_occurrence' in omop_data:
            return omop_data['procedure_occurrence']
        return []
    
    def _extract_fhir_medications(self, fhir_resources):
        """Extract medications from FHIR resources."""
        medications = []
        
        # Check ActivityDefinitions for medications
        if 'activityDefinitions' in fhir_resources:
            for ad in fhir_resources['activityDefinitions']:
                if ad.get('kind') == 'MedicationRequest':
                    medications.append(ad)
        
        return medications
    
    def _extract_omop_medications(self, omop_data):
        """Extract medications from OMOP tables."""
        if 'drug_exposure' in omop_data:
            return omop_data['drug_exposure']
        return []
    
    def _extract_fhir_observations(self, fhir_resources):
        """Extract observations from FHIR resources."""
        observations = []
        
        # Check Questionnaire for observation items
        if 'questionnaire' in fhir_resources and 'item' in fhir_resources['questionnaire']:
            observations = fhir_resources['questionnaire']['item']
        
        return observations
    
    def _extract_omop_observations(self, omop_data):
        """Extract observations from OMOP tables."""
        if 'observation' in omop_data:
            return omop_data['observation']
        return []