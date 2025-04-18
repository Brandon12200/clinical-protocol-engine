import os
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    # Look for the test_data directory
    root_dir = Path(__file__).parent.parent  # Go up from tests folder to root
    test_data_path = root_dir / 'tests' / 'test_data'
    
    # If test_data doesn't exist, create a temporary structure
    if not test_data_path.exists():
        temp_dir = tempfile.mkdtemp(prefix="test_data_")
        
        # Create the expected directory structure
        os.makedirs(os.path.join(temp_dir, 'clinical_trials'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'edge_cases'), exist_ok=True)
        
        # Create minimal test files to prevent test failures
        # Minimal PDF file
        with open(os.path.join(temp_dir, 'clinical_trials', 'phase3_trial_protocol.pdf'), 'wb') as f:
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Resources<<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000015 00000 n \n0000000060 00000 n \n0000000111 00000 n \ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n183\n%%EOF\n')
        
        # Minimal DOCX file (actually just a ZIP file with basic structure)
        with open(os.path.join(temp_dir, 'clinical_trials', 'fnih_phase1_2_template.docx'), 'wb') as f:
            f.write(b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        
        # Minimal "corrupted" PDF
        with open(os.path.join(temp_dir, 'edge_cases', 'truncated_document.pdf'), 'wb') as f:
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</')
        
        # Minimal "large" PDF (just a larger minimal PDF)
        with open(os.path.join(temp_dir, 'edge_cases', 'large_protocol_document.pdf'), 'wb') as f:
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n' + b'A' * 10000 + b'\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<</Size 1>>\nstartxref\n10050\n%%EOF\n')
        
        return temp_dir
    
    return str(test_data_path)

@pytest.fixture
def mock_model_manager():
    """Create a mock model manager for testing."""
    import torch
    from unittest.mock import MagicMock
    
    mock_manager = MagicMock()
    mock_manager.is_initialized = True
    mock_manager.device = 'cpu'
    
    # Mock tokenizer and model outputs
    mock_manager.tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]]),
        'offset_mapping': torch.tensor([[[0, 2], [2, 5], [5, 8]]])
    }
    
    # Mock model predictions (logits)
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    mock_manager.model.return_value = mock_outputs
    
    # Mock entity labels
    mock_manager.get_entity_labels.return_value = ['O', 'ELIGIBILITY']
    
    return mock_manager