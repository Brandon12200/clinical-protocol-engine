import os
import pytest
import tempfile
import uuid
import logging
import json
import datetime
from unittest.mock import MagicMock, patch
from pathlib import Path

from utils.file_handler import FileHandler
from utils.logger import setup_logger, get_performance_logger, get_error_logger
from utils.visualization import ResultsVisualization


class TestFileHandler:
    """Tests for the FileHandler class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after test
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def file_handler(self, temp_dir):
        """Create a FileHandler with a temporary base directory."""
        handler = FileHandler(base_directory=temp_dir)
        return handler
    
    def test_initialization(self, temp_dir):
        """Test FileHandler initialization."""
        handler = FileHandler(base_directory=temp_dir)
        
        assert handler.base_directory == temp_dir
        assert handler.uploads_dir == os.path.join(temp_dir, 'uploads')
        assert handler.processed_dir == os.path.join(temp_dir, 'processed')
        assert os.path.exists(handler.uploads_dir)
        assert os.path.exists(handler.processed_dir)
    
    def test_save_file(self, file_handler):
        """Test saving a file."""
        # Create a mock file
        mock_file = MagicMock()
        mock_file.filename = "test_document.pdf"
        mock_file.save = MagicMock()
        
        # Save the file
        result = file_handler.save_file(mock_file)
        
        # Verify results
        assert mock_file.save.called
        assert "test_document" in result["saved_filename"]
        assert result["original_filename"] == "test_document.pdf"
        assert result["file_path"].startswith(file_handler.uploads_dir)
        assert result["file_path"].endswith(".pdf")
    
    def test_sanitize_filename(self, file_handler):
        """Test filename sanitization."""
        # Test with spaces and special characters
        sanitized = file_handler._sanitize_filename("Test File Name !@#$%.txt")
        assert sanitized == "Test_File_Name_txt"
        
        # Test with very long filename
        long_name = "A" * 100
        sanitized = file_handler._sanitize_filename(long_name)
        assert len(sanitized) == 50
    
    def test_get_file_path(self, file_handler, temp_dir):
        """Test retrieving file paths."""
        # Create test files
        upload_file = os.path.join(file_handler.uploads_dir, "upload_test.txt")
        processed_file = os.path.join(file_handler.processed_dir, "processed_test.txt")
        
        with open(upload_file, 'w') as f:
            f.write("test")
        with open(processed_file, 'w') as f:
            f.write("test")
        
        # Test finding files in uploads
        path = file_handler.get_file_path("upload_test.txt")
        assert path == upload_file
        
        # Test finding files in processed
        path = file_handler.get_file_path("processed_test.txt")
        assert path == processed_file
        
        # Test with non-existent file
        path = file_handler.get_file_path("nonexistent.txt")
        assert path is None
        
        # Test with specific directory
        path = file_handler.get_file_path("upload_test.txt", file_handler.uploads_dir)
        assert path == upload_file
    
    def test_remove_file(self, file_handler):
        """Test file removal."""
        # Create a test file
        test_file = os.path.join(file_handler.uploads_dir, "remove_test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Remove the file
        result = file_handler.remove_file("remove_test.txt")
        assert result is True
        assert not os.path.exists(test_file)
        
        # Try to remove non-existent file
        result = file_handler.remove_file("nonexistent.txt")
        assert result is False
    
    def test_list_files(self, file_handler):
        """Test listing files."""
        # Create some test files
        test_files = [
            os.path.join(file_handler.uploads_dir, "test1.pdf"),
            os.path.join(file_handler.uploads_dir, "test2.docx"),
            os.path.join(file_handler.uploads_dir, "test3.txt")
        ]
        
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("test")
        
        # List all files
        files = file_handler.list_files()
        assert len(files) == 3
        
        # Check file info structure
        for file_info in files:
            assert "filename" in file_info
            assert "path" in file_info
            assert "size" in file_info
            assert "modified" in file_info
            assert "extension" in file_info
        
        # Filter by extension
        pdf_files = file_handler.list_files(extensions=['.pdf'])
        assert len(pdf_files) == 1
        assert pdf_files[0]["filename"] == "test1.pdf"
    
    def test_save_and_load_result(self, file_handler):
        """Test saving and loading JSON results."""
        # Test data
        test_data = {
            "entities": [{"text": "test", "label": "TEST"}],
            "sections": [{"type": "SECTION", "text": "test section"}]
        }
        
        # Save the result
        result_path = file_handler.save_result(test_data)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Extract filename from path
        filename = os.path.basename(result_path)
        
        # Load the result
        loaded_data = file_handler.load_result(filename)
        
        # Verify data
        assert loaded_data == test_data


class TestLogger:
    """Tests for the logger utility functions."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_setup_logger(self, temp_log_dir):
        """Test setting up a basic logger."""
        log_file = os.path.join(temp_log_dir, "test.log")
        logger = setup_logger("test_logger", log_file)
        
        # Check logger configuration
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # Console and file handler
        
        # Test logging
        logger.info("Test message")
        
        # Verify file was created
        assert os.path.exists(log_file)
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_get_performance_logger(self, temp_log_dir):
        """Test creating a performance logger."""
        logger = get_performance_logger("perf_test", temp_log_dir)
        
        # Check logger configuration
        assert logger.name == "perf_test"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # File handler only
        assert logger.propagate is False  # No propagation to root
        
        # Test logging
        logger.info("Performance data: 100ms")
        
        # Verify file was created
        log_file = os.path.join(temp_log_dir, "perf_test.log")
        assert os.path.exists(log_file)
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Performance data: 100ms" in content
    
    def test_get_error_logger(self, temp_log_dir):
        """Test creating an error logger."""
        logger = get_error_logger("error_test", temp_log_dir)
        
        # Check logger configuration
        assert logger.name == "error_test"
        assert logger.level == logging.ERROR
        assert len(logger.handlers) == 2  # Console and file handler
        assert logger.propagate is False
        
        # Test logging
        logger.error("Test error message")
        
        # Verify file was created
        log_file = os.path.join(temp_log_dir, "error_test.log")
        assert os.path.exists(log_file)
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test error message" in content


class TestVisualization:
    """Tests for the ResultsVisualization class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a ResultsVisualization instance."""
        return ResultsVisualization()
    
    def test_highlight_entities(self, visualizer):
        """Test highlighting entities in text."""
        text = "This is a test document with inclusion and exclusion criteria."
        entities = [
            {
                "text": "inclusion",
                "label": "ELIGIBILITY",
                "start": 28,
                "end": 37,
                "confidence": 0.95
            },
            {
                "text": "exclusion criteria",
                "label": "ELIGIBILITY",
                "start": 42,
                "end": 60,
                "confidence": 0.85
            }
        ]
        
        highlighted = visualizer.highlight_entities(text, entities)
        
        # Check for span elements and entity information
        assert "<span class=\"entity\"" in highlighted
        assert "background-color:" in highlighted
        assert "data-type=\"ELIGIBILITY\"" in highlighted
        assert "Confidence: 0.95" in highlighted
        assert "Confidence: 0.85" in highlighted
    
    def test_get_entity_color(self, visualizer):
        """Test entity color mapping."""
        # Test known entity types
        assert visualizer._get_entity_color("ELIGIBILITY") == "#ffcccb"
        assert visualizer._get_entity_color("PROCEDURE") == "#c2f0c2"
        
        # Test with B- or I- prefix
        assert visualizer._get_entity_color("B-ELIGIBILITY") == "#ffcccb"
        
        # Test unknown entity type (should get default color)
        assert visualizer._get_entity_color("UNKNOWN_TYPE") == "#e0e0e0"
    
    def test_create_relation_graph(self, visualizer):
        """Test creating relationship graph data."""
        entities = [
            {"text": "Aspirin", "label": "MEDICATION", "start": 0, "end": 7},
            {"text": "headache", "label": "CONDITION", "start": 12, "end": 20}
        ]
        
        relations = [
            {"source": 0, "target": 1, "type": "TREATS", "confidence": 0.8}
        ]
        
        graph = visualizer.create_relation_graph(entities, relations)
        
        # Check graph structure
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        
        # Check node properties
        assert graph["nodes"][0]["label"] == "Aspirin"
        assert graph["nodes"][0]["group"] == "MEDICATION"
        
        # Check edge properties
        assert graph["edges"][0]["from"] == 0
        assert graph["edges"][0]["to"] == 1
        assert graph["edges"][0]["label"] == "TREATS"
    
    def test_format_fhir_json(self, visualizer):
        """Test formatting FHIR resources."""
        fhir_resources = {
            "planDefinition": {"resourceType": "PlanDefinition", "id": "test", "status": "draft"},
            "activityDefinitions": [{"resourceType": "ActivityDefinition", "id": "activity1"}]
        }
        
        formatted = visualizer.format_fhir_json(fhir_resources)
        
        # Check for formatted JSON
        assert "planDefinition" in formatted
        assert "activityDefinitions" in formatted
        assert "\"resourceType\": \"PlanDefinition\"" in formatted["planDefinition"]
        assert "\"resourceType\": \"ActivityDefinition\"" in formatted["activityDefinitions"]
    
    def test_generate_confidence_visualization(self, visualizer):
        """Test generating confidence visualization data."""
        entities = [
            {"text": "inclusion", "label": "ELIGIBILITY", "confidence": 0.9},
            {"text": "exclusion", "label": "ELIGIBILITY", "confidence": 0.8},
            {"text": "procedure", "label": "PROCEDURE", "confidence": 0.95}
        ]
        
        visualization = visualizer.generate_confidence_visualization(entities)
        
        # Check structure and calculations
        assert "chart_data" in visualization
        assert "average_confidence" in visualization
        assert abs(visualization["average_confidence"] - (0.9 + 0.8 + 0.95) / 3) < 0.01
        
        # Check chart data
        chart_data = visualization["chart_data"]
        assert len(chart_data) == 2  # ELIGIBILITY and PROCEDURE
        
        # Check ELIGIBILITY data
        eligibility_data = next((item for item in chart_data if item["label"] == "ELIGIBILITY"), None)
        assert eligibility_data is not None
        assert eligibility_data["count"] == 2
        assert abs(eligibility_data["value"] - (0.9 + 0.8) / 2) < 0.01