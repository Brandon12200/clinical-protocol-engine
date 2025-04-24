#!/usr/bin/env python
"""
Configure and update the terminology mapping system.

This script configures the necessary resources for the terminology
mapping system to function properly, including database setup, 
external service configuration, and synonym expansion.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from standards.terminology.db_updater import TerminologyDatabaseUpdater
from standards.terminology.mapper import TerminologyMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_terminology_databases(data_dir: str, force_update: bool = False) -> bool:
    """
    Set up and populate the terminology databases.
    
    Args:
        data_dir: Path to the data directory
        force_update: Whether to force an update even if databases exist
        
    Returns:
        bool: True if setup was successful
    """
    logger.info(f"Setting up terminology databases in {data_dir}")
    
    # Check if databases already exist
    snomed_db = os.path.join(data_dir, "snomed_core.sqlite")
    loinc_db = os.path.join(data_dir, "loinc_core.sqlite")
    rxnorm_db = os.path.join(data_dir, "rxnorm_core.sqlite")
    
    # Check if all databases exist and have data
    databases_exist = all([
        os.path.exists(snomed_db),
        os.path.exists(loinc_db),
        os.path.exists(rxnorm_db)
    ])
    
    if databases_exist and not force_update:
        # Try to open each database and check if it has data
        import sqlite3
        try:
            non_empty = True
            for db_path in [snomed_db, loinc_db, rxnorm_db]:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get the table name based on the database
                table_name = os.path.basename(db_path).replace("_core.sqlite", "_concepts")
                
                # Check if the table has any rows
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                conn.close()
                
                if count == 0:
                    non_empty = False
                    break
                    
            if non_empty:
                logger.info("Terminology databases already exist and contain data, skipping update")
                return True
        except Exception as e:
            logger.warning(f"Error checking databases, will update: {e}")
    
    # Create and run the database updater
    updater = TerminologyDatabaseUpdater(data_dir=data_dir)
    return updater.update_all()
    
def check_and_configure_external_services(config_path: str = None) -> dict:
    """
    Check and configure external terminology services.
    
    Args:
        config_path: Path to configuration file with API keys
        
    Returns:
        dict: Configuration for external services
    """
    logger.info("Checking external terminology services")
    
    # Default configuration
    config = {
        "use_external_services": True,
        "use_fuzzy_matching": True,
        "use_rxnav_api": True,
        "use_redis_cache": False,
    }
    
    # Load configuration from file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Loaded external service configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Check if we have API keys for each service
    if "umls_api_key" not in config or not config["umls_api_key"]:
        logger.warning("No UMLS API key found. UMLS services will be unavailable.")
        config["use_umls_api"] = False
    else:
        config["use_umls_api"] = True
        logger.info("UMLS API key found.")
    
    if "bioportal_api_key" not in config or not config["bioportal_api_key"]:
        logger.warning("No BioPortal API key found. BioPortal services will be unavailable.")
        config["use_bioportal_api"] = False
    else:
        config["use_bioportal_api"] = True
        logger.info("BioPortal API key found.")
    
    # Test RxNav API since it doesn't require authentication
    import requests
    try:
        response = requests.get("https://rxnav.nlm.nih.gov/REST/version")
        if response.status_code == 200:
            logger.info("RxNav API is available.")
            config["use_rxnav_api"] = True
        else:
            logger.warning("RxNav API is not responding. RxNav services will be unavailable.")
            config["use_rxnav_api"] = False
    except Exception:
        logger.warning("Could not connect to RxNav API. RxNav services will be unavailable.")
        config["use_rxnav_api"] = False
    
    return config

def test_terminology_mapper(data_dir: str, config: dict) -> bool:
    """
    Test the terminology mapper with the current configuration.
    
    Args:
        data_dir: Path to the data directory
        config: Configuration dictionary
        
    Returns:
        bool: True if tests pass
    """
    logger.info("Testing terminology mapper")
    
    # Set the data directory in the config
    config["data_dir"] = data_dir
    
    # Create a mapper instance
    mapper = TerminologyMapper(config)
    
    # Test initialization
    if not mapper.initialize():
        logger.error("Terminology mapper initialization failed")
        return False
        
    # Test some basic mappings
    logger.info("Testing SNOMED mappings:")
    test_terms = ["hypertension", "diabetes", "asthma", "pneumonia"]
    for term in test_terms:
        result = mapper.map_to_snomed(term)
        found = result.get("found", False)
        code = result.get("code", "none")
        logger.info(f"  - {term}: {'✓' if found else '✗'} (code: {code})")
        
    logger.info("Testing LOINC mappings:")
    test_terms = ["hemoglobin a1c", "blood pressure", "glucose"]
    for term in test_terms:
        result = mapper.map_to_loinc(term)
        found = result.get("found", False)
        code = result.get("code", "none")
        logger.info(f"  - {term}: {'✓' if found else '✗'} (code: {code})")
        
    logger.info("Testing RxNorm mappings:")
    test_terms = ["metformin", "lisinopril", "aspirin"]
    for term in test_terms:
        result = mapper.map_to_rxnorm(term)
        found = result.get("found", False)
        code = result.get("code", "none")
        logger.info(f"  - {term}: {'✓' if found else '✗'} (code: {code})")
        
    # Test fuzzy matching
    logger.info("Testing fuzzy matching:")
    fuzzy_tests = [
        ("htn", "snomed"),  # Abbreviation
        ("hypertention", "snomed"),  # Misspelling
        ("a1c", "loinc"),  # Abbreviation
        ("metphormin", "rxnorm")  # Misspelling
    ]
    
    for term, system in fuzzy_tests:
        result = mapper.map_term(term, system)
        found = result.get("found", False)
        code = result.get("code", "none")
        display = result.get("display", "")
        logger.info(f"  - {term} → {display}: {'✓' if found else '✗'} (code: {code})")
        
    # Get statistics
    stats = mapper.get_statistics()
    logger.info(f"Database statistics: {json.dumps(stats, indent=2)}")
    
    return True

def main():
    """Main entry point for the configuration script."""
    parser = argparse.ArgumentParser(description='Configure terminology mapping system')
    parser.add_argument('--config', dest='config_path', help='Path to configuration file')
    parser.add_argument('--data-dir', dest='data_dir', help='Path to data directory')
    parser.add_argument('--force-update', action='store_true', help='Force database update')
    parser.add_argument('--skip-db-update', action='store_true', help='Skip database update')
    args = parser.parse_args()
    
    # Set up data directory
    data_dir = args.data_dir
    if not data_dir:
        # Default to the standard data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'terminology'
        )
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Update databases if requested
    if not args.skip_db_update:
        if not setup_terminology_databases(data_dir, args.force_update):
            logger.error("Failed to set up terminology databases")
            return 1
    
    # Configure external services
    config = check_and_configure_external_services(args.config_path)
    
    # Test the mapper
    if not test_terminology_mapper(data_dir, config):
        logger.error("Terminology mapper tests failed")
        return 1
    
    logger.info("Terminology mapping system configured successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())