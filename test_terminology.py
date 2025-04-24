#!/usr/bin/env python
"""
Simple test script to verify the terminology mapping implementation.
"""

import os
import json
from standards.terminology.mapper import TerminologyMapper
from standards.terminology.fuzzy_matcher import FuzzyMatcher
from standards.terminology.embedded_db import EmbeddedDatabaseManager

def main():
    """Test the terminology mapping functionality."""
    print("Testing terminology mapping...")
    
    # Create a mapper
    mapper = TerminologyMapper()
    
    # Test exact mapping
    print("\n1. Testing exact mapping...")
    result = mapper.map_to_snomed("hypertension")
    print(f"  - Mapped 'hypertension' to: {result}")
    
    result = mapper.map_to_loinc("hemoglobin a1c")
    print(f"  - Mapped 'hemoglobin a1c' to: {result}")
    
    result = mapper.map_to_rxnorm("metformin")
    print(f"  - Mapped 'metformin' to: {result}")
    
    # Test term normalization
    print("\n2. Testing term normalization...")
    result = mapper.map_to_snomed("HYPERTENSION")
    print(f"  - Mapped 'HYPERTENSION' to: {result}")
    
    result = mapper.map_to_snomed("history of hypertension")
    print(f"  - Mapped 'history of hypertension' to: {result}")
    
    # Test fuzzy matching
    print("\n3. Testing fuzzy matching...")
    if mapper.fuzzy_matcher:
        print("  - Fuzzy matching is available")
        
        # Try a synonym
        result = mapper.map_to_snomed("htn")
        print(f"  - Mapped 'htn' to: {result}")
        
        # Try a misspelling
        result = mapper.map_to_snomed("hypertention")
        print(f"  - Mapped 'hypertention' to: {result}")
    else:
        print("  - Fuzzy matching is not available")
    
    # Test external services
    print("\n4. Testing external services...")
    if mapper.external_service and mapper.external_service.is_available():
        print("  - External services are available")
        
        # Try mapping a term
        result = mapper.map_to_rxnorm("aspirin")
        print(f"  - Mapped 'aspirin' using external service to: {result}")
    else:
        print("  - External services are not available")
    
    # Test adding a mapping
    print("\n5. Testing custom mapping addition...")
    mapper.add_custom_mapping(
        "snomed", 
        "pneumonia", 
        "233604007", 
        "Pneumonia"
    )
    
    result = mapper.map_to_snomed("pneumonia")
    print(f"  - Mapped 'pneumonia' to: {result}")
    
    # Get statistics
    print("\n6. Getting statistics...")
    stats = mapper.get_statistics()
    print(f"  - Statistics: {json.dumps(stats, indent=2)}")
    
    print("\nTerminology mapping tests completed.")

if __name__ == "__main__":
    main()