import os
import pytest
import json
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from standards.terminology.mapper import TerminologyMapper
from standards.terminology.embedded_db import EmbeddedDatabaseManager
from standards.terminology.fuzzy_matcher import FuzzyMatcher
from standards.terminology.external_service import ExternalTerminologyService
from standards.fhir.converters import FHIRConverter

class TestTerminologyMapper:
    """Tests for the terminology mapping functionality."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        
        # Create a custom mappings file
        test_mappings = {
            "snomed": {
                "hypertension": {
                    "code": "38341003",
                    "display": "Hypertension",
                    "system": "http://snomed.info/sct",
                    "found": True
                },
                "asthma": {
                    "code": "195967001",
                    "display": "Asthma",
                    "system": "http://snomed.info/sct",
                    "found": True
                }
            },
            "loinc": {
                "hemoglobin a1c": {
                    "code": "4548-4",
                    "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
                    "system": "http://loinc.org",
                    "found": True
                }
            },
            "rxnorm": {
                "metformin": {
                    "code": "6809",
                    "display": "metformin",
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "found": True
                }
            }
        }
        
        # Create the terminology directory
        os.makedirs(os.path.join(temp_dir, "terminology"), exist_ok=True)
        
        # Save the test mappings
        with open(os.path.join(temp_dir, "terminology", "custom_mappings.json"), "w") as f:
            json.dump(test_mappings, f, indent=2)
        
        yield temp_dir
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mapper(self, test_data_dir):
        """Create a terminology mapper with test data."""
        config = {"data_dir": os.path.join(test_data_dir, "terminology")}
        mapper = TerminologyMapper(config)
        mapper.initialize()
        return mapper
    
    def test_initialization(self, mapper):
        """Test that the mapper initializes correctly."""
        assert mapper is not None
        assert mapper.db_manager is not None
    
    def test_map_to_snomed(self, mapper):
        """Test mapping to SNOMED CT."""
        # Test with exact match from custom mappings
        result = mapper.map_to_snomed("hypertension")
        assert result is not None
        assert result["code"] == "38341003"
        assert result["found"] is True
        
        # Test with non-existent term
        result = mapper.map_to_snomed("nonexistent term")
        assert result is not None
        assert result["code"] is None
        assert result["found"] is False
        
        # Test with normalized term (uppercase)
        result = mapper.map_to_snomed("HYPERTENSION")
        assert result is not None
        assert result["code"] == "38341003"
        assert result["found"] is True
    
    def test_map_to_loinc(self, mapper):
        """Test mapping to LOINC."""
        # Test with exact match from custom mappings
        result = mapper.map_to_loinc("hemoglobin a1c")
        assert result is not None
        assert result["code"] == "4548-4"
        assert result["found"] is True
        
        # Test with non-existent term
        result = mapper.map_to_loinc("nonexistent term")
        assert result is not None
        assert result["code"] is None
        assert result["found"] is False
    
    def test_map_to_rxnorm(self, mapper):
        """Test mapping to RxNorm."""
        # Test with exact match from custom mappings
        result = mapper.map_to_rxnorm("metformin")
        assert result is not None
        assert result["code"] == "6809"
        assert result["found"] is True
        
        # Test with non-existent term
        result = mapper.map_to_rxnorm("nonexistent term")
        assert result is not None
        assert result["code"] is None
        assert result["found"] is False
    
    def test_term_normalization(self, mapper):
        """Test term normalization for better matching."""
        # Test with prefixes that should be removed
        result = mapper.map_to_snomed("history of asthma")
        assert result is not None
        assert result["code"] == "195967001"
        assert result["found"] is True
        
        # Test with capitalization
        result = mapper.map_to_snomed("ASTHMA")
        assert result is not None
        assert result["code"] == "195967001"
        assert result["found"] is True
        
        # Test with punctuation
        result = mapper.map_to_snomed("asthma,")
        assert result is not None
        assert result["code"] == "195967001"
        assert result["found"] is True
    
    def test_add_custom_mapping(self, mapper):
        """Test adding a custom mapping."""
        # Add a new mapping
        success = mapper.add_custom_mapping(
            "snomed", 
            "pneumonia", 
            "233604007", 
            "Pneumonia"
        )
        assert success is True
        
        # Test that the new mapping works
        result = mapper.map_to_snomed("pneumonia")
        assert result is not None
        assert result["code"] == "233604007"
        assert result["found"] is True
        
    def test_map_term_generic(self, mapper):
        """Test the generic map_term method."""
        # Test mapping to SNOMED
        result = mapper.map_term("hypertension", "snomed")
        assert result is not None
        assert result["code"] == "38341003"
        assert result["found"] is True
        
        # Test mapping to LOINC
        result = mapper.map_term("hemoglobin a1c", "loinc")
        assert result is not None
        assert result["code"] == "4548-4"
        assert result["found"] is True
        
        # Test with invalid system
        result = mapper.map_term("hypertension", "invalid_system")
        assert result is not None
        assert result["found"] is False
        assert "error" in result
        
    def test_get_statistics(self, mapper):
        """Test getting statistics."""
        stats = mapper.get_statistics()
        assert stats is not None
        assert "snomed" in stats
        assert "loinc" in stats
        assert "rxnorm" in stats
        assert "custom" in stats
        assert "fuzzy_matching" in stats
        assert "external_services" in stats


class TestFuzzyMatcher:
    """Tests for the fuzzy matching functionality."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        
        # Create terminology directory
        terminology_dir = os.path.join(temp_dir, "terminology")
        os.makedirs(terminology_dir, exist_ok=True)
        
        # Create synonyms directory
        os.makedirs(os.path.join(terminology_dir, "synonyms"), exist_ok=True)
        
        # Create a test SNOMED database
        conn = sqlite3.connect(os.path.join(terminology_dir, "snomed_core.sqlite"))
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS snomed_concepts (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            term TEXT NOT NULL,
            display TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # Insert some test data
        test_concepts = [
            (1, "38341003", "hypertension", "Hypertension", 1),
            (2, "195967001", "asthma", "Asthma", 1),
            (3, "73211009", "diabetes mellitus", "Diabetes mellitus", 1),
            (4, "44054006", "type 2 diabetes mellitus", "Type 2 diabetes mellitus", 1)
        ]
        cursor.executemany(
            "INSERT INTO snomed_concepts (id, code, term, display, is_active) VALUES (?, ?, ?, ?, ?)",
            test_concepts
        )
        conn.commit()
        conn.close()
        
        # Create a synonyms file
        synonyms = {
            "diabetes_syn": [
                "diabetes mellitus",
                "dm",
                "diabetes"
            ],
            "hypertension_syn": [
                "hypertension",
                "htn", 
                "high blood pressure"
            ]
        }
        
        with open(os.path.join(terminology_dir, "synonyms", "medical_synonyms.json"), "w") as f:
            json.dump(synonyms, f, indent=2)
        
        yield temp_dir
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def db_manager(self, test_data_dir):
        """Create a database manager with test data."""
        data_dir = os.path.join(test_data_dir, "terminology")
        db_manager = EmbeddedDatabaseManager(data_dir=data_dir)
        db_manager.connect()
        return db_manager
    
    @pytest.fixture
    def fuzzy_matcher(self, db_manager):
        """Create a fuzzy matcher with test data."""
        config = {
            "synonyms_path": os.path.join(db_manager.data_dir, "synonyms", "medical_synonyms.json"),
            "ratio_threshold": 85,
            "partial_ratio_threshold": 90,
            "token_sort_ratio_threshold": 80
        }
        matcher = FuzzyMatcher(db_manager, config)
        matcher.initialize()
        return matcher
    
    def test_matcher_initialization(self, fuzzy_matcher):
        """Test that the fuzzy matcher initializes correctly."""
        assert fuzzy_matcher is not None
        assert fuzzy_matcher.db_manager is not None
        assert "snomed" in fuzzy_matcher.term_index
        assert len(fuzzy_matcher.term_index["snomed"]) > 0
    
    def test_fuzzy_matching(self, fuzzy_matcher):
        """Test fuzzy matching capabilities."""
        # Test with a synonym
        result = fuzzy_matcher.find_fuzzy_match("htn", "snomed")
        assert result is not None
        assert result["code"] == "38341003"  # hypertension
        assert result["found"] is True
        
        # Test with a misspelling (if string similarity is high enough)
        result = fuzzy_matcher.find_fuzzy_match("hypertention", "snomed")
        assert result is not None
        assert result["code"] == "38341003"  # hypertension
        assert result["found"] is True
        
        # Test with a non-existent term
        result = fuzzy_matcher.find_fuzzy_match("something completely different", "snomed")
        assert result is None
    
    def test_add_synonym(self, fuzzy_matcher):
        """Test adding synonyms."""
        # Add new synonyms
        success = fuzzy_matcher.add_synonym("asthma", ["asthmatic disease", "bronchial asthma"])
        assert success is True
        
        # Test with the new synonym
        result = fuzzy_matcher.find_fuzzy_match("asthmatic disease", "snomed")
        assert result is not None
        assert result["code"] == "195967001"  # asthma
        assert result["found"] is True


class TestExternalService:
    """Tests for the external terminology service functionality."""
    
    @pytest.fixture
    def mock_external_service(self):
        """Create a mocked external service."""
        with patch("standards.terminology.external_service.requests.get") as mock_get:
            # Mock the RxNav response
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "approximateGroup": {
                    "candidate": [
                        {
                            "rxcui": "6809",
                            "name": "metformin",
                            "score": 100
                        }
                    ]
                }
            }
            mock_get.return_value = mock_response
            
            # Create the service
            service = ExternalTerminologyService({
                "use_rxnav_api": True
            })
            
            # Mock initialize to return success
            service.initialize = MagicMock(return_value=True)
            service.is_available = MagicMock(return_value=True)
            
            # Add the mocked service
            service.services = {
                "rxnav": {
                    "base_url": "https://rxnav.nlm.nih.gov/REST/",
                    "active": True
                }
            }
            
            yield service
    
    def test_external_mapping(self, mock_external_service):
        """Test mapping terms using external services."""
        # Test RxNav mapping
        result = mock_external_service.search_rxnav("metformin")
        assert result is not None
        assert result["code"] == "6809"
        assert result["found"] is True
        
        # Test generic mapping
        result = mock_external_service.map_term("metformin", "rxnorm")
        assert result is not None
        assert result["code"] == "6809"
        assert result["found"] is True


class TestIntegratedMapping:
    """Tests for the full terminology mapping pipeline with all components."""
    
    @pytest.fixture
    def integrated_mapper(self, test_data_dir):
        """Create a mapper with fuzzy matching and external services."""
        config = {
            "data_dir": os.path.join(test_data_dir, "terminology"),
            "use_fuzzy_matching": True,
            "use_external_services": True
        }
        
        # Create the mapper
        mapper = TerminologyMapper(config)
        
        # Mock the external service
        mock_external = MagicMock()
        mock_external.is_available.return_value = True
        mock_external.map_term.return_value = {
            "code": "12345",
            "display": "External Term",
            "system": "http://example.org",
            "found": True
        }
        mapper.external_service = mock_external
        
        # Mock the fuzzy matcher
        mock_fuzzy = MagicMock()
        mock_fuzzy.find_fuzzy_match.return_value = {
            "code": "67890",
            "display": "Fuzzy Term",
            "system": "http://example.org",
            "found": True,
            "match_type": "fuzzy",
            "score": 95
        }
        mapper.fuzzy_matcher = mock_fuzzy
        
        return mapper
    
    def test_integrated_pipeline(self, integrated_mapper):
        """Test the full mapping pipeline with all components."""
        # Mock the database lookup to return None (forcing fuzzy matcher use)
        integrated_mapper.db_manager.lookup_snomed = MagicMock(return_value=None)
        
        # Test with a term that would use fuzzy matching
        result = integrated_mapper.map_to_snomed("test term")
        assert result is not None
        assert result["code"] == "67890"  # from fuzzy matcher
        assert result["found"] is True
        assert result["match_type"] == "fuzzy"
        
        # Mock the fuzzy matcher to return None (forcing external service use)
        integrated_mapper.fuzzy_matcher.find_fuzzy_match.return_value = None
        
        # Test with a term that would use external service
        result = integrated_mapper.map_to_snomed("another term")
        assert result is not None
        assert result["code"] == "12345"  # from external service
        assert result["found"] is True


class TestFHIRTerminologyIntegration:
    """Tests for integrating terminology mapping with FHIR conversion."""
    
    @pytest.fixture
    def mapper(self, test_data_dir):
        """Create a terminology mapper with test data."""
        config = {"data_dir": os.path.join(test_data_dir, "terminology")}
        mapper = TerminologyMapper(config)
        mapper.initialize()
        return mapper
    
    @pytest.fixture
    def converter(self, mapper):
        """Create a FHIR converter with the test mapper."""
        return FHIRConverter(mapper)
    
    def test_fhir_with_terminology(self, converter):
        """Test FHIR conversion with terminology mapping."""
        # Create a simple protocol with conditions and procedures
        protocol_data = {
            "title": "Test Protocol",
            "description": "Protocol for testing terminology mapping",
            "procedures": [
                {"text": "blood pressure measurement"},
                {"text": "hypertension screening"}
            ],
            "conditions": [
                {"text": "hypertension"},
                {"text": "asthma"}
            ],
            "measurements": [
                {"text": "hemoglobin a1c", "units": "%"}
            ],
            "medications": [
                {"text": "metformin", "dosage": "500mg daily"}
            ]
        }
        
        # Convert to FHIR
        result = converter.convert(protocol_data)
        
        # Check that we get back valid resources
        assert "resources" in result
        assert "validation" in result
        assert result["validation"]["valid"] is True
        
        # Check that terminology mapping statistics are included
        assert "terminology_mapping" in result
        
        # Check that mapped_data includes terminology mappings
        mapped_data = converter.map_extracted_data(protocol_data)
        
        # Verify condition mappings
        assert "conditions" in mapped_data
        assert len(mapped_data["conditions"]) == 2
        assert "terminology" in mapped_data["conditions"][0]
        assert mapped_data["conditions"][0]["terminology"]["code"] == "38341003"  # hypertension
        
        # Verify medication mappings
        assert "medications" in mapped_data
        assert len(mapped_data["medications"]) == 1
        assert "terminology" in mapped_data["medications"][0]
        assert mapped_data["medications"][0]["terminology"]["code"] == "6809"  # metformin