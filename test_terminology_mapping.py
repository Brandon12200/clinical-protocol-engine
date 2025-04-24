#!/usr/bin/env python
"""
Test the terminology mapping with real-world examples.

This script demonstrates the use of the terminology mapping system
with real-world clinical text, showing exact matching, fuzzy matching,
and context-enhanced mapping.
"""

import os
import sys
import json
import logging
from standards.terminology.mapper import TerminologyMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_with_clinical_examples():
    """Test the terminology mapper with clinical examples."""
    logger.info("Testing terminology mapping with clinical examples")
    
    # Create a mapper instance
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'terminology'
    )
    
    config = {
        "data_dir": data_dir,
        "use_fuzzy_matching": True,
        "use_external_services": True  # Will fall back to embedded DB if not available
    }
    
    mapper = TerminologyMapper(config)
    mapper.initialize()
    
    # Test cases with clinical snippets
    test_cases = [
        {
            "description": "Exact match with SNOMED",
            "term": "hypertension",
            "system": "snomed",
            "context": "Patient has a history of hypertension and is on lisinopril."
        },
        {
            "description": "Abbreviation with SNOMED",
            "term": "htn",
            "system": "snomed",
            "context": "Medical history: HTN, controlled on meds."
        },
        {
            "description": "Misspelling with SNOMED",
            "term": "diabetus",
            "system": "snomed",
            "context": "Patient has poorly controlled diabetus with glucose of 210."
        },
        {
            "description": "Context-enhanced matching",
            "term": "type 2",  # Ambiguous without context
            "system": "snomed",
            "context": "Patient has type 2 diabetes controlled with metformin."
        },
        {
            "description": "Exact match with LOINC",
            "term": "hemoglobin a1c",
            "system": "loinc",
            "context": "Lab results show hemoglobin A1c of 7.2%."
        },
        {
            "description": "Abbreviation with LOINC",
            "term": "hba1c",
            "system": "loinc",
            "context": "The patient's HbA1c was elevated at 8.5%."
        },
        {
            "description": "Exact match with RxNorm",
            "term": "metformin",
            "system": "rxnorm",
            "context": "Current medications: Metformin 500mg bid."
        },
        {
            "description": "Misspelling with RxNorm",
            "term": "atorvastatyn",
            "system": "rxnorm",
            "context": "The patient is taking atorvastatyn 40mg daily for hyperlipidemia."
        },
        {
            "description": "Brand name to generic mapping",
            "term": "lipitor", 
            "system": "rxnorm",
            "context": "Current medications include Lipitor 20mg daily."
        }
    ]
    
    # Run the tests
    logger.info("Running terminology mapping tests with clinical examples:\n")
    
    for i, test in enumerate(test_cases, 1):
        description = test["description"]
        term = test["term"]
        system = test["system"]
        context = test["context"]
        
        # Perform the mapping
        result = mapper.map_term(term, system, context)
        
        # Display results
        found = result.get("found", False)
        code = result.get("code", "none")
        display = result.get("display", "")
        
        logger.info(f"{i}. {description}:")
        logger.info(f"   Term: '{term}'")
        logger.info(f"   Context: '{context}'")
        logger.info(f"   Mapped to: {'✓' if found else '✗'} {display} ({code})")
        
        # Display match details if available
        if "match_type" in result:
            logger.info(f"   Match type: {result['match_type']}")
        if "score" in result:
            logger.info(f"   Confidence: {result['score']}%")
        if "context_enhanced" in result and result["context_enhanced"]:
            logger.info(f"   Context enhanced: Yes (term: '{result['context_term']}')")
            
        logger.info("")
    
    # Get overall statistics
    stats = mapper.get_statistics()
    logger.info(f"Terminology mapping statistics: {json.dumps(stats, indent=2)}")
    
    return True

def main():
    """Main entry point for the test script."""
    try:
        return 0 if test_with_clinical_examples() else 1
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())