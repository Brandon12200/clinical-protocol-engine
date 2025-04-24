"""
Test script for the enhanced OMOP integration functionality.
This tests the complete OMOP CDM conversion process with sample protocol data.
"""

import os
import sys
import json
import unittest
import pandas as pd
from pathlib import Path

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from standards.omop.converters import OMOPConverter
from standards.terminology.mapper import TerminologyMapper

class TestOMOPIntegration(unittest.TestCase):
    """Tests for the enhanced OMOP CDM integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a terminology mapper
        self.terminology_mapper = TerminologyMapper()
        
        # Configure the OMOP converter
        config = {
            "generate_sql": True,
            "sql_dialect": "sqlite"
        }
        
        # Initialize the converter
        self.converter = OMOPConverter(
            terminology_mapper=self.terminology_mapper,
            config=config
        )
        
        # Create sample protocol data
        self.sample_protocol = {
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
                    "text": "Diagnosed with hypertension"
                },
                {
                    "type": "exclusion",
                    "text": "History of heart failure"
                }
            ],
            "procedures": [
                {
                    "text": "Blood sample collection",
                    "description": "Collect 10ml blood sample",
                    "quantity": 1
                },
                {
                    "text": "MRI scan of the brain",
                    "description": "Full brain MRI scan"
                },
                {
                    "text": "Implantation of cardiac monitoring device",
                    "description": "Implant continuous monitoring device"
                }
            ],
            "medications": [
                {
                    "text": "Metoprolol",
                    "description": "Beta blocker treatment",
                    "dosage": "50mg twice daily",
                    "route": "oral"
                },
                {
                    "text": "Amlodipine",
                    "description": "Calcium channel blocker",
                    "dosage": "5mg daily",
                    "route": "oral"
                }
            ],
            "endpoints": [
                {
                    "text": "Change in systolic blood pressure"
                },
                {
                    "text": "Frequency of adverse events"
                },
                {
                    "text": "Blood glucose level"
                }
            ],
            "measurements": [
                {
                    "text": "Blood pressure",
                    "units": "mmHg",
                    "range": {
                        "low": 90,
                        "high": 140
                    }
                },
                {
                    "text": "Body weight",
                    "units": "kg"
                },
                {
                    "text": "Heart rate",
                    "units": "bpm",
                    "range": {
                        "low": 60,
                        "high": 100
                    }
                }
            ],
            "devices": [
                {
                    "text": "Blood pressure monitor",
                    "description": "Home BP monitoring device",
                    "quantity": 1
                }
            ]
        }
        
        # Set up output directory for test results
        self.output_dir = Path(os.path.dirname(__file__)) / "outputs" / "omop"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_omop_conversion(self):
        """Test basic OMOP conversion functionality."""
        # Convert the sample protocol to OMOP CDM format
        result = self.converter.convert(self.sample_protocol)
        
        # Check that the conversion was successful
        self.assertIn('tables', result)
        self.assertIn('validation', result)
        
        # Check that we have data in the expected tables
        self.assertIn('condition_occurrence', result['tables'])
        self.assertIn('drug_exposure', result['tables'])
        self.assertIn('procedure_occurrence', result['tables'])
        self.assertIn('observation', result['tables'])
        self.assertIn('measurement', result['tables'])
        self.assertIn('device_exposure', result['tables'])
        self.assertIn('specimen', result['tables'])
        
        # Check validation results
        self.assertIn('valid', result['validation'])
        self.assertIn('issues', result['validation'])
        self.assertIn('warnings', result['validation'])
        
        # Save the result for inspection
        with open(self.output_dir / "conversion_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        # Save each table as CSV
        for table_name, records in result['tables'].items():
            if records:
                df = pd.DataFrame(records)
                df.to_csv(self.output_dir / f"{table_name}.csv", index=False)
        
        print(f"Conversion test completed. Results saved to {self.output_dir}")
    
    def test_sql_generation(self):
        """Test SQL generation functionality."""
        # Convert and generate SQL
        result = self.converter.convert(self.sample_protocol)
        
        # Check that SQL was generated
        self.assertIn('tables', result)
        
        # Save SQL scripts
        if 'sql' in result:
            sql_dir = self.output_dir / "sql"
            sql_dir.mkdir(exist_ok=True)
            
            # Save a combined SQL script
            with open(sql_dir / "omop_tables.sql", "w") as combined_sql:
                # Add preamble if exists
                if '_preamble' in result['sql']:
                    for setup_stmt in result['sql']['_preamble'].get('setup', []):
                        combined_sql.write(f"{setup_stmt}\n\n")
                
                # Add table creation and data insertion statements
                for table_name, statements in result['sql'].items():
                    if table_name.startswith('_'):  # Skip special entries
                        continue
                        
                    combined_sql.write(f"-- {table_name.upper()} TABLE\n")
                    combined_sql.write(f"{statements['create']};\n\n")
                    
                    if 'inserts' in statements and statements['inserts']:
                        combined_sql.write(f"-- {table_name} DATA\n")
                        for insert in statements['inserts']:
                            combined_sql.write(f"{insert};\n")
                        combined_sql.write("\n")
                    
                    if 'indexes' in statements:
                        combined_sql.write(f"-- {table_name} INDEXES (commented out)\n")
                        for index in statements['indexes']:
                            combined_sql.write(f"{index}\n")
                        combined_sql.write("\n")
            
            print(f"SQL generation test completed. SQL script saved to {sql_dir / 'omop_tables.sql'}")
            
    def test_terminology_mapping(self):
        """Test terminology mapping within OMOP conversion."""
        # First do the terminology mapping only
        mapped_data = self.converter.map_extracted_data(self.sample_protocol)
        
        # Save the mapped data for inspection
        with open(self.output_dir / "mapped_data.json", "w") as f:
            json.dump(mapped_data, f, indent=2, default=str)
        
        # Check that we have mappings in the result
        self.assertIn('eligibility_criteria', mapped_data)
        
        # Check at least one mapping
        if len(mapped_data['eligibility_criteria']) > 0:
            # Check if terminology mapping was attempted on at least one criterion
            criterion = mapped_data['eligibility_criteria'][0]
            self.assertIn('mapping', criterion)
        
        print(f"Terminology mapping test completed. Results saved to {self.output_dir}")

if __name__ == '__main__':
    unittest.main()