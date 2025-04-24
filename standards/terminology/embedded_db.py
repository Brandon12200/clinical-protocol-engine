"""
Embedded database manager for terminology mapping.

This module handles the storage, retrieval, and management of embedded
terminology databases for offline mapping of clinical terms.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddedDatabaseManager:
    """
    Manages embedded terminology databases for offline mapping.
    
    This class provides a lightweight, file-based database system for
    mapping clinical terms to standardized terminologies without requiring
    external services.
    
    Attributes:
        data_dir: Directory containing the terminology databases
        connections: Dictionary of database connections
        custom_mappings: User-defined custom mappings
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            data_dir: Optional path to data directory. If not provided,
                     defaults to the standard data/terminology directory.
        """
        # Default to standard data directory
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), 
            'data', 'terminology'
        )
        
        # Initialize database connections
        self.connections = {}
        self.custom_mappings = {}
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """
        Connect to the embedded databases.
        
        Returns:
            bool: True if connections were successful
        """
        try:
            # Define the database files to connect to
            databases = {
                "snomed": os.path.join(self.data_dir, "snomed_core.sqlite"),
                "loinc": os.path.join(self.data_dir, "loinc_core.sqlite"),
                "rxnorm": os.path.join(self.data_dir, "rxnorm_core.sqlite")
            }
            
            # Connect to each database if it exists
            for db_name, db_path in databases.items():
                if os.path.exists(db_path):
                    logger.info(f"Connecting to {db_name} database at {db_path}")
                    self.connections[db_name] = sqlite3.connect(db_path)
                    # Enable foreign keys
                    self.connections[db_name].execute("PRAGMA foreign_keys = ON")
                else:
                    # If database doesn't exist, create a minimal schema
                    logger.warning(f"{db_name} database not found at {db_path}, creating empty database")
                    self._create_empty_database(db_name, db_path)
                    
            # Load custom mappings if available
            custom_path = os.path.join(self.data_dir, "custom_mappings.json")
            if os.path.exists(custom_path):
                with open(custom_path, 'r') as f:
                    self.custom_mappings = json.load(f)
                logger.info(f"Loaded {sum(len(mappings) for mappings in self.custom_mappings.values())} custom mappings")
            else:
                # Create empty custom mappings file
                self.custom_mappings = {"snomed": {}, "loinc": {}, "rxnorm": {}}
                with open(custom_path, 'w') as f:
                    json.dump(self.custom_mappings, f, indent=2)
                logger.info(f"Created empty custom mappings file at {custom_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to databases: {e}")
            return False
    
    def _create_empty_database(self, db_name: str, db_path: str) -> None:
        """
        Create an empty database with the required schema.
        
        Args:
            db_name: Name of the database (snomed, loinc, rxnorm)
            db_path: Path to the database file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Connect to the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables based on database type
            if db_name == "snomed":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS snomed_concepts (
                    id INTEGER PRIMARY KEY,
                    code TEXT NOT NULL,
                    term TEXT NOT NULL,
                    display TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
                ''')
                # Create index for faster lookups
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_snomed_term ON snomed_concepts(term)')
                
            elif db_name == "loinc":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS loinc_concepts (
                    id INTEGER PRIMARY KEY,
                    code TEXT NOT NULL,
                    term TEXT NOT NULL,
                    display TEXT NOT NULL,
                    component TEXT,
                    property TEXT,
                    time TEXT,
                    system TEXT,
                    scale TEXT,
                    method TEXT
                )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loinc_term ON loinc_concepts(term)')
                
            elif db_name == "rxnorm":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS rxnorm_concepts (
                    id INTEGER PRIMARY KEY,
                    code TEXT NOT NULL,
                    term TEXT NOT NULL,
                    display TEXT NOT NULL,
                    tty TEXT, /* Term Type */
                    is_active INTEGER DEFAULT 1
                )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_rxnorm_term ON rxnorm_concepts(term)')
            
            # Commit changes and add to connections
            conn.commit()
            self.connections[db_name] = conn
            logger.info(f"Created empty {db_name} database at {db_path}")
        except Exception as e:
            logger.error(f"Error creating {db_name} database: {e}")
    
    def lookup_snomed(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Look up a term in the SNOMED CT database.
        
        Args:
            term: The term to look up
            
        Returns:
            Dictionary with mapping information or None if not found
        """
        # Check custom mappings first
        if term in self.custom_mappings.get("snomed", {}):
            return self.custom_mappings["snomed"][term]
            
        # Then check the database
        if "snomed" in self.connections:
            try:
                conn = self.connections["snomed"]
                cursor = conn.cursor()
                
                # Look for exact match first
                cursor.execute(
                    "SELECT code, display FROM snomed_concepts WHERE LOWER(term) = ? AND is_active = 1", 
                    (term.lower(),)
                )
                result = cursor.fetchone()
                
                if result:
                    return {
                        "code": result[0],
                        "display": result[1],
                        "system": "http://snomed.info/sct",
                        "found": True
                    }
                
                # If no exact match, try alternative query approaches (e.g., LIKE)
                # This will be enhanced in Phase 2 with fuzzy matching
                
            except Exception as e:
                logger.error(f"Error looking up SNOMED term '{term}': {e}")
        
        return None
    
    def lookup_loinc(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Look up a term in the LOINC database.
        
        Args:
            term: The term to look up
            
        Returns:
            Dictionary with mapping information or None if not found
        """
        # Check custom mappings first
        if term in self.custom_mappings.get("loinc", {}):
            return self.custom_mappings["loinc"][term]
            
        # Then check the database
        if "loinc" in self.connections:
            try:
                conn = self.connections["loinc"]
                cursor = conn.cursor()
                
                # Look for exact match
                cursor.execute(
                    "SELECT code, display FROM loinc_concepts WHERE LOWER(term) = ?", 
                    (term.lower(),)
                )
                result = cursor.fetchone()
                
                if result:
                    return {
                        "code": result[0],
                        "display": result[1],
                        "system": "http://loinc.org",
                        "found": True
                    }
                
                # If no exact match, try LIKE query
                # This will be enhanced in Phase 2
                
            except Exception as e:
                logger.error(f"Error looking up LOINC term '{term}': {e}")
        
        return None
    
    def lookup_rxnorm(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Look up a term in the RxNorm database.
        
        Args:
            term: The term to look up
            
        Returns:
            Dictionary with mapping information or None if not found
        """
        # Check custom mappings first
        if term in self.custom_mappings.get("rxnorm", {}):
            return self.custom_mappings["rxnorm"][term]
            
        # Then check the database
        if "rxnorm" in self.connections:
            try:
                conn = self.connections["rxnorm"]
                cursor = conn.cursor()
                
                # Look for exact match
                cursor.execute(
                    "SELECT code, display FROM rxnorm_concepts WHERE LOWER(term) = ? AND is_active = 1", 
                    (term.lower(),)
                )
                result = cursor.fetchone()
                
                if result:
                    return {
                        "code": result[0],
                        "display": result[1],
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "found": True
                    }
                
                # If no exact match, try LIKE query
                # This will be enhanced in Phase 2
                
            except Exception as e:
                logger.error(f"Error looking up RxNorm term '{term}': {e}")
        
        return None
    
    def add_mapping(self, system: str, term: str, mapping: Dict[str, Any]) -> bool:
        """
        Add a mapping to the custom mappings.
        
        Args:
            system: The terminology system (snomed, loinc, rxnorm)
            term: The term to map
            mapping: The mapping information
            
        Returns:
            bool: True if the mapping was added successfully
        """
        try:
            # Ensure the system exists in custom mappings
            if system not in self.custom_mappings:
                self.custom_mappings[system] = {}
            
            # Add the mapping
            self.custom_mappings[system][term] = mapping
            
            # Save to file
            custom_path = os.path.join(self.data_dir, "custom_mappings.json")
            with open(custom_path, 'w') as f:
                json.dump(self.custom_mappings, f, indent=2)
            
            logger.info(f"Added custom {system} mapping for '{term}': {mapping['code']}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom mapping for '{term}': {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the databases.
        
        Returns:
            Dictionary with statistics about the databases
        """
        stats = {
            "snomed": {"count": 0, "database_size": 0},
            "loinc": {"count": 0, "database_size": 0},
            "rxnorm": {"count": 0, "database_size": 0},
            "custom": {
                "snomed": len(self.custom_mappings.get("snomed", {})),
                "loinc": len(self.custom_mappings.get("loinc", {})),
                "rxnorm": len(self.custom_mappings.get("rxnorm", {}))
            }
        }
        
        # Get statistics for each database
        for system in ["snomed", "loinc", "rxnorm"]:
            if system in self.connections:
                try:
                    conn = self.connections[system]
                    cursor = conn.cursor()
                    
                    # Get row count
                    table_name = f"{system}_concepts"
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    stats[system]["count"] = cursor.fetchone()[0]
                    
                    # Get database file size
                    db_path = os.path.join(self.data_dir, f"{system}_core.sqlite")
                    if os.path.exists(db_path):
                        stats[system]["database_size"] = os.path.getsize(db_path)
                except Exception as e:
                    logger.error(f"Error getting statistics for {system}: {e}")
        
        return stats
    
    def close(self):
        """Close all database connections."""
        for conn in self.connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
        
        self.connections = {}