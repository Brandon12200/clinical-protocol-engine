#!/usr/bin/env python
"""
Terminology mapping setup script.

This script sets up the terminology mapping system by initializing
databases, configuring services, and testing the mapping functionality.
"""

import os
import sys
import logging
from standards.terminology.configure_mappings import main as configure_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the setup script."""
    logger.info("Setting up terminology mapping system...")
    
    # Create sample data directory if it doesn't exist
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'terminology'
    )
    sample_dir = os.path.join(data_dir, 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Run the configuration script
    # Set sys.argv to include our arguments
    sys.argv = [
        sys.argv[0],
        '--data-dir', data_dir,
        '--force-update'
    ]
    
    return configure_main()

if __name__ == "__main__":
    sys.exit(main())