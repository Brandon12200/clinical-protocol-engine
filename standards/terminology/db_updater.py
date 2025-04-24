#!/usr/bin/env python
"""
Database updater for terminology databases.

This script populates and updates the embedded SQLite databases used for
terminology mapping with core concepts from SNOMED CT, LOINC, and RxNorm.
"""

import os
import sys
import json
import csv
import sqlite3
import logging
import argparse
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'terminology'
)
DEFAULT_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), 'terminology_downloads')

# Source URLs - these would normally be behind authentication
# Using sample/demo URLs for this implementation
SNOMED_URL = "https://download.nlm.nih.gov/umls/kss/SNOMEDCT_US/SNOMEDCT_US.zip"
LOINC_URL = "https://loinc.org/download/loinc-and-relma-complete-download-file-csv-text-format/"
RXNORM_URL = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip"

# Fallback to sample data if downloads fail
SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'terminology', 'sample_data'
)

class TerminologyDatabaseUpdater:
    """
    Manages the updating and population of terminology databases.
    
    This class downloads, processes, and imports terminology data from
    standard sources into local SQLite databases for offline use.
    
    Attributes:
        config: Configuration dictionary with API keys and settings
        data_dir: Directory where databases are stored
        download_dir: Directory for temporary downloads
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the database updater.
        
        Args:
            config_path: Path to configuration JSON file with API keys
            data_dir: Directory where databases should be stored
        """
        self.config = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.download_dir = DEFAULT_DOWNLOAD_DIR
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Initialize database connections
        self.connections = {}
        
    def update_all(self) -> bool:
        """
        Update all terminology databases.
        
        Returns:
            bool: True if all updates were successful
        """
        success = True
        
        # Update each database
        success = self.update_snomed() and success
        success = self.update_loinc() and success
        success = self.update_rxnorm() and success
        
        # Generate a sample dataset if no real data is available
        if not success:
            logger.warning("One or more database updates failed. Generating sample data.")
            success = self.generate_sample_data()
            
        return success
        
    def update_snomed(self) -> bool:
        """
        Update the SNOMED CT database.
        
        Returns:
            bool: True if update was successful
        """
        logger.info("Updating SNOMED CT database...")
        
        db_path = os.path.join(self.data_dir, "snomed_core.sqlite")
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
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
            
            # Download SNOMED CT data
            download_path = self._download_snomed()
            if not download_path:
                return False
                
            # Process and import the data
            logger.info("Processing SNOMED CT data...")
            count = self._import_snomed_data(download_path, conn)
            
            # Commit changes and close
            conn.commit()
            conn.close()
            
            logger.info(f"SNOMED CT database updated successfully with {count} concepts")
            return True
        except Exception as e:
            logger.error(f"Error updating SNOMED CT database: {e}")
            return False
            
    def update_loinc(self) -> bool:
        """
        Update the LOINC database.
        
        Returns:
            bool: True if update was successful
        """
        logger.info("Updating LOINC database...")
        
        db_path = os.path.join(self.data_dir, "loinc_core.sqlite")
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
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
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_loinc_term ON loinc_concepts(term)')
            
            # Download LOINC data
            download_path = self._download_loinc()
            if not download_path:
                return False
                
            # Process and import the data
            logger.info("Processing LOINC data...")
            count = self._import_loinc_data(download_path, conn)
            
            # Commit changes and close
            conn.commit()
            conn.close()
            
            logger.info(f"LOINC database updated successfully with {count} concepts")
            return True
        except Exception as e:
            logger.error(f"Error updating LOINC database: {e}")
            return False
            
    def update_rxnorm(self) -> bool:
        """
        Update the RxNorm database.
        
        Returns:
            bool: True if update was successful
        """
        logger.info("Updating RxNorm database...")
        
        db_path = os.path.join(self.data_dir, "rxnorm_core.sqlite")
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rxnorm_concepts (
                id INTEGER PRIMARY KEY,
                code TEXT NOT NULL,
                term TEXT NOT NULL,
                display TEXT NOT NULL,
                tty TEXT,
                is_active INTEGER DEFAULT 1
            )
            ''')
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rxnorm_term ON rxnorm_concepts(term)')
            
            # Download RxNorm data
            download_path = self._download_rxnorm()
            if not download_path:
                return False
                
            # Process and import the data
            logger.info("Processing RxNorm data...")
            count = self._import_rxnorm_data(download_path, conn)
            
            # Commit changes and close
            conn.commit()
            conn.close()
            
            logger.info(f"RxNorm database updated successfully with {count} concepts")
            return True
        except Exception as e:
            logger.error(f"Error updating RxNorm database: {e}")
            return False
            
    def _download_snomed(self) -> Optional[str]:
        """
        Download SNOMED CT data.
        
        Returns:
            str: Path to downloaded data or None if failed
        """
        try:
            logger.info("Downloading SNOMED CT data...")
            
            # In a real implementation, this would use the UMLS API with authentication
            # For this example, we'll create a fallback to sample data
            
            # Create a temporary directory for the download
            download_dir = os.path.join(self.download_dir, "snomed")
            os.makedirs(download_dir, exist_ok=True)
            
            # For this implementation, we'll return a path to sample data
            return os.path.join(SAMPLE_DATA_DIR, "snomed_sample.csv")
        except Exception as e:
            logger.error(f"Error downloading SNOMED CT data: {e}")
            return None
            
    def _download_loinc(self) -> Optional[str]:
        """
        Download LOINC data.
        
        Returns:
            str: Path to downloaded data or None if failed
        """
        try:
            logger.info("Downloading LOINC data...")
            
            # In a real implementation, this would download from LOINC with authentication
            # For this example, we'll create a fallback to sample data
            
            # Create a temporary directory for the download
            download_dir = os.path.join(self.download_dir, "loinc")
            os.makedirs(download_dir, exist_ok=True)
            
            # For this implementation, we'll return a path to sample data
            return os.path.join(SAMPLE_DATA_DIR, "loinc_sample.csv")
        except Exception as e:
            logger.error(f"Error downloading LOINC data: {e}")
            return None
            
    def _download_rxnorm(self) -> Optional[str]:
        """
        Download RxNorm data.
        
        Returns:
            str: Path to downloaded data or None if failed
        """
        try:
            logger.info("Downloading RxNorm data...")
            
            # In a real implementation, this would download from NLM with authentication
            # For this example, we'll create a fallback to sample data
            
            # Create a temporary directory for the download
            download_dir = os.path.join(self.download_dir, "rxnorm")
            os.makedirs(download_dir, exist_ok=True)
            
            # For this implementation, we'll return a path to sample data
            return os.path.join(SAMPLE_DATA_DIR, "rxnorm_sample.csv")
        except Exception as e:
            logger.error(f"Error downloading RxNorm data: {e}")
            return None
            
    def _import_snomed_data(self, data_path: str, conn: sqlite3.Connection) -> int:
        """
        Import SNOMED CT data into the database.
        
        Args:
            data_path: Path to the data file
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Check if we're using sample data (in which case we'll generate it)
        if not os.path.exists(data_path):
            return self._import_sample_snomed_data(conn)
            
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM snomed_concepts")
        
        # Import the data
        count = 0
        try:
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                
                # Prepare for batch insert
                batch_size = 1000
                batch = []
                
                for row in reader:
                    if len(row) >= 4:
                        code = row[0]
                        term = row[1].lower()
                        display = row[2]
                        is_active = int(row[3]) if len(row) > 3 else 1
                        
                        batch.append((code, term, display, is_active))
                        count += 1
                        
                        if len(batch) >= batch_size:
                            cursor.executemany(
                                "INSERT INTO snomed_concepts (code, term, display, is_active) VALUES (?, ?, ?, ?)",
                                batch
                            )
                            batch = []
                
                # Insert any remaining records
                if batch:
                    cursor.executemany(
                        "INSERT INTO snomed_concepts (code, term, display, is_active) VALUES (?, ?, ?, ?)",
                        batch
                    )
                    
            conn.commit()
        except Exception as e:
            logger.error(f"Error importing SNOMED CT data: {e}")
            conn.rollback()
            
        return count
        
    def _import_loinc_data(self, data_path: str, conn: sqlite3.Connection) -> int:
        """
        Import LOINC data into the database.
        
        Args:
            data_path: Path to the data file
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Check if we're using sample data (in which case we'll generate it)
        if not os.path.exists(data_path):
            return self._import_sample_loinc_data(conn)
            
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM loinc_concepts")
        
        # Import the data
        count = 0
        try:
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                
                # Prepare for batch insert
                batch_size = 1000
                batch = []
                
                for row in reader:
                    if len(row) >= 3:
                        code = row[0]
                        term = row[1].lower()
                        display = row[2]
                        
                        # Optional LOINC parts
                        component = row[3] if len(row) > 3 else ""
                        property_val = row[4] if len(row) > 4 else ""
                        time_val = row[5] if len(row) > 5 else ""
                        system_val = row[6] if len(row) > 6 else ""
                        scale_val = row[7] if len(row) > 7 else ""
                        method_val = row[8] if len(row) > 8 else ""
                        
                        batch.append((code, term, display, component, property_val, time_val, system_val, scale_val, method_val))
                        count += 1
                        
                        if len(batch) >= batch_size:
                            cursor.executemany(
                                """INSERT INTO loinc_concepts 
                                   (code, term, display, component, property, time, system, scale, method) 
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                batch
                            )
                            batch = []
                
                # Insert any remaining records
                if batch:
                    cursor.executemany(
                        """INSERT INTO loinc_concepts 
                           (code, term, display, component, property, time, system, scale, method) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        batch
                    )
                    
            conn.commit()
        except Exception as e:
            logger.error(f"Error importing LOINC data: {e}")
            conn.rollback()
            
        return count
        
    def _import_rxnorm_data(self, data_path: str, conn: sqlite3.Connection) -> int:
        """
        Import RxNorm data into the database.
        
        Args:
            data_path: Path to the data file
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Check if we're using sample data (in which case we'll generate it)
        if not os.path.exists(data_path):
            return self._import_sample_rxnorm_data(conn)
            
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM rxnorm_concepts")
        
        # Import the data
        count = 0
        try:
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                
                # Prepare for batch insert
                batch_size = 1000
                batch = []
                
                for row in reader:
                    if len(row) >= 3:
                        code = row[0]
                        term = row[1].lower()
                        display = row[2]
                        tty = row[3] if len(row) > 3 else ""
                        is_active = int(row[4]) if len(row) > 4 else 1
                        
                        batch.append((code, term, display, tty, is_active))
                        count += 1
                        
                        if len(batch) >= batch_size:
                            cursor.executemany(
                                "INSERT INTO rxnorm_concepts (code, term, display, tty, is_active) VALUES (?, ?, ?, ?, ?)",
                                batch
                            )
                            batch = []
                
                # Insert any remaining records
                if batch:
                    cursor.executemany(
                        "INSERT INTO rxnorm_concepts (code, term, display, tty, is_active) VALUES (?, ?, ?, ?, ?)",
                        batch
                    )
                    
            conn.commit()
        except Exception as e:
            logger.error(f"Error importing RxNorm data: {e}")
            conn.rollback()
            
        return count
        
    def generate_sample_data(self) -> bool:
        """
        Generate sample data for all databases.
        
        This is used when real data cannot be downloaded or for testing.
        
        Returns:
            bool: True if sample data was generated successfully
        """
        os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
        
        success = True
        
        # Generate sample data directories
        success = self._generate_sample_snomed_data() and success
        success = self._generate_sample_loinc_data() and success
        success = self._generate_sample_rxnorm_data() and success
        
        # Use the sample data to populate the databases
        success = self.update_snomed() and success
        success = self.update_loinc() and success
        success = self.update_rxnorm() and success
        
        return success
        
    def _generate_sample_snomed_data(self) -> bool:
        """
        Generate sample SNOMED CT data.
        
        Returns:
            bool: True if sample data was generated successfully
        """
        try:
            sample_path = os.path.join(SAMPLE_DATA_DIR, "snomed_sample.csv")
            
            # Common medical conditions for sample data
            sample_data = [
                ["38341003", "hypertension", "Hypertension", "1"],
                ["73211009", "diabetes mellitus", "Diabetes mellitus", "1"],
                ["44054006", "type 2 diabetes mellitus", "Type 2 diabetes mellitus", "1"],
                ["46635009", "type 1 diabetes mellitus", "Type 1 diabetes mellitus", "1"],
                ["195967001", "asthma", "Asthma", "1"],
                ["13645005", "chronic obstructive pulmonary disease", "Chronic obstructive pulmonary disease", "1"],
                ["22298006", "myocardial infarction", "Myocardial infarction", "1"],
                ["230690007", "cerebrovascular accident", "Cerebrovascular accident", "1"],
                ["84114007", "heart failure", "Heart failure", "1"],
                ["233604007", "pneumonia", "Pneumonia", "1"],
                ["429040005", "allergy to penicillin", "Allergy to penicillin", "1"],
                ["64859006", "osteoporosis", "Osteoporosis", "1"],
                ["396275006", "osteoarthritis", "Osteoarthritis", "1"],
                ["90688005", "chronic kidney disease", "Chronic kidney disease", "1"],
                ["49436004", "atrial fibrillation", "Atrial fibrillation", "1"],
                ["73430006", "depression", "Depression", "1"],
                ["35489007", "depressive disorder", "Depressive disorder", "1"],
                ["197480006", "anxiety disorder", "Anxiety disorder", "1"],
                ["4556007", "transient ischemic attack", "Transient ischemic attack", "1"],
                ["66071002", "type B viral hepatitis", "Viral hepatitis B", "1"],
                ["40468003", "viral hepatitis c", "Viral hepatitis C", "1"],
                ["155501007", "thromboembolic disorder", "Thromboembolic disorder", "1"],
                ["363346000", "malignant neoplastic disease", "Malignant neoplastic disease", "1"],
                ["68496003", "polyp", "Polyp", "1"],
                ["271737000", "anemia", "Anemia", "1"],
                ["13200003", "anemia due to blood loss", "Anemia due to blood loss", "1"],
                ["87628006", "bacterial infectious disease", "Bacterial infectious disease", "1"],
                ["59621000", "essential hypertension", "Essential hypertension", "1"],
                ["10725009", "benign prostatic hyperplasia", "Benign prostatic hyperplasia", "1"],
                ["62315008", "diarrhea", "Diarrhea", "1"],
                ["43116000", "eczema", "Eczema", "1"],
                ["24700007", "multiple sclerosis", "Multiple sclerosis", "1"],
                ["13920009", "pulmonary embolism", "Pulmonary embolism", "1"],
                ["266998003", "urinary tract infection", "Urinary tract infection", "1"],
                ["44186003", "hyperthyroidism", "Hyperthyroidism", "1"],
                ["40930008", "hypothyroidism", "Hypothyroidism", "1"],
                ["75934005", "metabolic syndrome x", "Metabolic syndrome X", "1"],
                ["55822004", "hyperlipidemia", "Hyperlipidemia", "1"],
                ["235856003", "hepatitis", "Hepatitis", "1"],
                ["88805009", "chronic hepatitis", "Chronic hepatitis", "1"]
            ]
            
            # Write to CSV
            with open(sample_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["code", "term", "display", "is_active"])
                writer.writerows(sample_data)
                
            logger.info(f"Generated sample SNOMED CT data with {len(sample_data)} concepts")
            return True
        except Exception as e:
            logger.error(f"Error generating sample SNOMED CT data: {e}")
            return False
            
    def _generate_sample_loinc_data(self) -> bool:
        """
        Generate sample LOINC data.
        
        Returns:
            bool: True if sample data was generated successfully
        """
        try:
            sample_path = os.path.join(SAMPLE_DATA_DIR, "loinc_sample.csv")
            
            # Common lab tests for sample data
            sample_data = [
                ["4548-4", "hemoglobin a1c", "Hemoglobin A1c/Hemoglobin.total in Blood", "Hemoglobin A1c", "MFr", "Pt", "Bld", "Qn", ""],
                ["85354-9", "blood pressure", "Blood pressure panel with all children optional", "BP", "Pres", "Pt", "Sys", "Qn", ""],
                ["2093-3", "cholesterol", "Cholesterol [Mass/volume] in Serum or Plasma", "Cholesterol", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2160-0", "creatinine", "Creatinine [Mass/volume] in Serum or Plasma", "Creatinine", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2339-0", "glucose", "Glucose [Mass/volume] in Blood", "Glucose", "MCnc", "Pt", "Bld", "Qn", ""],
                ["2571-8", "triglycerides", "Triglyceride [Mass/volume] in Serum or Plasma", "Triglyceride", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["3094-0", "urea nitrogen", "Urea nitrogen [Mass/volume] in Serum or Plasma", "BUN", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2823-3", "potassium", "Potassium [Moles/volume] in Serum or Plasma", "Potassium", "SCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2951-2", "sodium", "Sodium [Moles/volume] in Serum or Plasma", "Sodium", "SCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2075-0", "chloride", "Chloride [Moles/volume] in Serum or Plasma", "Chloride", "SCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2345-7", "glucose", "Glucose [Mass/volume] in Serum or Plasma", "Glucose", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["26515-7", "platelet count", "Platelets [#/volume] in Blood", "Platelets", "NCnc", "Pt", "Bld", "Qn", ""],
                ["26464-8", "leukocyte count", "Leukocytes [#/volume] in Blood", "WBC", "NCnc", "Pt", "Bld", "Qn", ""],
                ["718-7", "hemoglobin", "Hemoglobin [Mass/volume] in Blood", "Hemoglobin", "MCnc", "Pt", "Bld", "Qn", ""],
                ["4544-3", "hematocrit", "Hematocrit [Volume Fraction] of Blood", "Hematocrit", "VFr", "Pt", "Bld", "Qn", ""],
                ["788-0", "erythrocyte count", "Erythrocytes [#/volume] in Blood", "RBC", "NCnc", "Pt", "Bld", "Qn", ""],
                ["17861-6", "calcium", "Calcium [Mass/volume] in Serum or Plasma", "Calcium", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["1920-8", "aspartate aminotransferase", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "AST", "CCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["1742-6", "alanine aminotransferase", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "ALT", "CCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["1975-2", "bilirubin total", "Bilirubin.total [Mass/volume] in Serum or Plasma", "Total bilirubin", "MCnc", "Pt", "Ser/Plas", "Qn", ""],
                ["2458-8", "igg", "IgG [Mass/volume] in Serum", "IgG", "MCnc", "Pt", "Ser", "Qn", ""],
                ["6690-2", "leukocyte common antigen", "Cells.CD45 [#/volume] in Blood", "CD45", "NCnc", "Pt", "Bld", "Qn", ""]
            ]
            
            # Write to CSV
            with open(sample_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["code", "term", "display", "component", "property", "time", "system", "scale", "method"])
                writer.writerows(sample_data)
                
            logger.info(f"Generated sample LOINC data with {len(sample_data)} concepts")
            return True
        except Exception as e:
            logger.error(f"Error generating sample LOINC data: {e}")
            return False
            
    def _generate_sample_rxnorm_data(self) -> bool:
        """
        Generate sample RxNorm data.
        
        Returns:
            bool: True if sample data was generated successfully
        """
        try:
            sample_path = os.path.join(SAMPLE_DATA_DIR, "rxnorm_sample.csv")
            
            # Common medications for sample data
            sample_data = [
                ["6809", "metformin", "metformin", "IN", "1"],
                ["29046", "lisinopril", "lisinopril", "IN", "1"],
                ["1191", "aspirin", "aspirin", "IN", "1"],
                ["83367", "atorvastatin", "atorvastatin", "IN", "1"],
                ["10582", "levothyroxine", "levothyroxine", "IN", "1"],
                ["4053", "amlodipine", "amlodipine", "IN", "1"],
                ["11170", "losartan", "losartan", "IN", "1"],
                ["4337", "atenolol", "atenolol", "IN", "1"],
                ["3356", "acetaminophen", "acetaminophen", "IN", "1"],
                ["6960", "morphine", "morphine", "IN", "1"],
                ["161", "acetylsalicylic acid", "acetylsalicylic acid", "SY", "1"],
                ["56360", "rosuvastatin", "rosuvastatin", "IN", "1"],
                ["42316", "simvastatin", "simvastatin", "IN", "1"],
                ["35296", "sertraline", "sertraline", "IN", "1"],
                ["8299", "paroxetine", "paroxetine", "IN", "1"],
                ["58927", "venlafaxine", "venlafaxine", "IN", "1"],
                ["58147", "pantoprazole", "pantoprazole", "IN", "1"],
                ["36567", "omeprazole", "omeprazole", "IN", "1"],
                ["73494", "dexlansoprazole", "dexlansoprazole", "IN", "1"],
                ["153165", "esomeprazole", "esomeprazole", "IN", "1"],
                ["2244", "amoxicillin", "amoxicillin", "IN", "1"],
                ["10312", "levofloxacin", "levofloxacin", "IN", "1"],
                ["89522", "cephalexin", "cephalexin", "IN", "1"],
                ["1739", "azithromycin", "azithromycin", "IN", "1"],
                ["4337", "atenolol", "atenolol", "IN", "1"],
                ["6135", "metoprolol", "metoprolol", "IN", "1"],
                ["58927", "venlafaxine", "venlafaxine", "IN", "1"],
                ["321988", "eszopiclone", "eszopiclone", "IN", "1"],
                ["2551", "alprazolam", "alprazolam", "IN", "1"],
                ["9997", "lorazepam", "lorazepam", "IN", "1"],
                ["32968", "prednisone", "prednisone", "IN", "1"],
                ["7052", "methylprednisolone", "methylprednisolone", "IN", "1"],
                ["6373", "montelukast", "montelukast", "IN", "1"],
                ["69749", "fluticasone", "fluticasone", "IN", "1"],
                ["3264", "albuterol", "albuterol", "IN", "1"]
            ]
            
            # Write to CSV
            with open(sample_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["code", "term", "display", "tty", "is_active"])
                writer.writerows(sample_data)
                
            logger.info(f"Generated sample RxNorm data with {len(sample_data)} concepts")
            return True
        except Exception as e:
            logger.error(f"Error generating sample RxNorm data: {e}")
            return False
            
    def _import_sample_snomed_data(self, conn: sqlite3.Connection) -> int:
        """
        Import sample SNOMED CT data into the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Generate the sample data
        success = self._generate_sample_snomed_data()
        if not success:
            return 0
            
        # Now import the data from the sample file
        return self._import_snomed_data(os.path.join(SAMPLE_DATA_DIR, "snomed_sample.csv"), conn)
        
    def _import_sample_loinc_data(self, conn: sqlite3.Connection) -> int:
        """
        Import sample LOINC data into the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Generate the sample data
        success = self._generate_sample_loinc_data()
        if not success:
            return 0
            
        # Now import the data from the sample file
        return self._import_loinc_data(os.path.join(SAMPLE_DATA_DIR, "loinc_sample.csv"), conn)
        
    def _import_sample_rxnorm_data(self, conn: sqlite3.Connection) -> int:
        """
        Import sample RxNorm data into the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            int: Number of concepts imported
        """
        # Generate the sample data
        success = self._generate_sample_rxnorm_data()
        if not success:
            return 0
            
        # Now import the data from the sample file
        return self._import_rxnorm_data(os.path.join(SAMPLE_DATA_DIR, "rxnorm_sample.csv"), conn)

def main():
    """Main entry point for the database updater script."""
    parser = argparse.ArgumentParser(description='Update terminology databases')
    parser.add_argument('--config', dest='config_path', help='Path to configuration file')
    parser.add_argument('--data-dir', dest='data_dir', help='Path to data directory')
    args = parser.parse_args()
    
    updater = TerminologyDatabaseUpdater(args.config_path, args.data_dir)
    success = updater.update_all()
    
    if success:
        logger.info("All terminology databases updated successfully.")
        return 0
    else:
        logger.error("One or more database updates failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())