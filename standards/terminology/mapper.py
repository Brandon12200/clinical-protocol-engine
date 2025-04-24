"""
Terminology mapper for clinical terms.

This module provides the core functionality for mapping clinical terms to
standardized terminologies like SNOMED CT, LOINC, and RxNorm.
"""

import os
import logging
import importlib.util
from typing import Dict, List, Optional, Any, Union
from standards.terminology.embedded_db import EmbeddedDatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

class TerminologyMapper:
    """
    Maps clinical terms to standard terminologies (SNOMED CT, LOINC, RxNorm).
    
    This class provides methods to map extracted clinical terms to standardized
    terminology codes, enabling interoperability with healthcare systems.
    
    Attributes:
        db_manager: Manager for embedded terminology databases
        config: Configuration options
        fuzzy_matcher: Fuzzy matching module for inexact matching
        external_service: External terminology service client
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the terminology mapper.
        
        Args:
            config: Optional configuration dictionary with settings
        """
        self.db_manager = EmbeddedDatabaseManager(
            data_dir=config.get("data_dir") if config else None
        )
        self.config = config or {}
        self.fuzzy_matcher = None
        self.external_service = None
        
        self._setup_fuzzy_matching()
        self._setup_external_services()
        self.initialize()
    
    def _setup_fuzzy_matching(self):
        """Initialize fuzzy matching if enabled in config."""
        if self.config.get("use_fuzzy_matching", True):
            try:
                from standards.terminology.fuzzy_matcher import FuzzyMatcher
                self.fuzzy_matcher = FuzzyMatcher(self.db_manager, self.config)
                logger.info("Fuzzy matching initialized")
            except ImportError as e:
                logger.warning(f"Fuzzy matching dependencies not available: {e}")
    
    def _setup_external_services(self):
        """Initialize external terminology services if enabled in config."""
        if self.config.get("use_external_services", False):
            try:
                from standards.terminology.external_service import ExternalTerminologyService
                self.external_service = ExternalTerminologyService(self.config)
                logger.info("External terminology services initialized")
            except ImportError as e:
                logger.warning(f"External terminology service dependencies not available: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize the mapping database and services.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Connect to embedded databases
            db_success = self.db_manager.connect()
            
            # Initialize fuzzy matcher if available
            fuzzy_success = True
            if self.fuzzy_matcher:
                fuzzy_success = self.fuzzy_matcher.initialize()
            
            # Initialize external services if available
            external_success = True
            if self.external_service:
                external_success = self.external_service.initialize()
            
            # Log overall initialization status
            if db_success:
                logger.info("Terminology mapper initialized successfully")
                
                # Log component status
                if self.fuzzy_matcher and not fuzzy_success:
                    logger.warning("Fuzzy matching initialization incomplete")
                if self.external_service and not external_success:
                    logger.warning("External services initialization incomplete")
            else:
                logger.warning("Terminology mapper initialization incomplete")
                
            return db_success
        except Exception as e:
            logger.error(f"Error initializing terminology mapper: {e}")
            return False
    
    def map_to_snomed(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Map a clinical term to SNOMED CT code.
        
        Args:
            term: The clinical term to map
            context: Optional context information to improve mapping accuracy
            
        Returns:
            Dictionary with mapping results including code, display name,
            terminology system, and confidence score
        """
        if not term:
            return {
                "code": None, 
                "display": "", 
                "system": "http://snomed.info/sct", 
                "found": False
            }
            
        # Clean and normalize the term
        clean_term = self._normalize_term(term)
        
        # 1. Try exact match in embedded database
        result = self.db_manager.lookup_snomed(clean_term)
        if result:
            logger.debug(f"Found exact SNOMED match for '{term}': {result['code']}")
            return result
            
        # 2. Try fuzzy matching if available
        if self.fuzzy_matcher and self.config.get("use_fuzzy_matching", True):
            fuzzy_result = self.fuzzy_matcher.find_fuzzy_match(clean_term, "snomed", context)
            if fuzzy_result:
                logger.debug(f"Found fuzzy SNOMED match for '{term}': {fuzzy_result['code']} (match type: {fuzzy_result.get('match_type', 'unknown')}, score: {fuzzy_result.get('score', 0)})")
                return fuzzy_result
                
        # 3. Try external service if available
        if self.external_service and self.config.get("use_external_services", False) and self.external_service.is_available():
            external_result = self.external_service.map_term(clean_term, "snomed", context)
            if external_result:
                logger.debug(f"Found external SNOMED match for '{term}': {external_result['code']}")
                
                # Add the mapping to the custom mappings for future use
                self.add_custom_mapping(
                    "snomed", 
                    clean_term, 
                    external_result["code"], 
                    external_result["display"]
                )
                
                return external_result
                
        # 4. Return not found with the original term
        logger.debug(f"No SNOMED mapping found for '{term}'")
        return {
            "code": None, 
            "display": term, 
            "system": "http://snomed.info/sct", 
            "found": False,
            "attempted_methods": ["exact", 
                                 "fuzzy" if self.fuzzy_matcher else "", 
                                 "external" if (self.external_service and self.external_service.is_available()) else ""]
        }
    
    def map_to_loinc(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Map a clinical term to LOINC code.
        
        Args:
            term: The clinical term to map
            context: Optional context information to improve mapping accuracy
            
        Returns:
            Dictionary with mapping results including code, display name,
            terminology system, and confidence score
        """
        if not term:
            return {
                "code": None, 
                "display": "", 
                "system": "http://loinc.org", 
                "found": False
            }
            
        # Clean and normalize the term
        clean_term = self._normalize_term(term)
        
        # 1. Try exact match in embedded database
        result = self.db_manager.lookup_loinc(clean_term)
        if result:
            logger.debug(f"Found exact LOINC match for '{term}': {result['code']}")
            return result
            
        # 2. Try fuzzy matching if available
        if self.fuzzy_matcher and self.config.get("use_fuzzy_matching", True):
            fuzzy_result = self.fuzzy_matcher.find_fuzzy_match(clean_term, "loinc", context)
            if fuzzy_result:
                logger.debug(f"Found fuzzy LOINC match for '{term}': {fuzzy_result['code']} (match type: {fuzzy_result.get('match_type', 'unknown')}, score: {fuzzy_result.get('score', 0)})")
                return fuzzy_result
                
        # 3. Try external service if available
        if self.external_service and self.config.get("use_external_services", False) and self.external_service.is_available():
            external_result = self.external_service.map_term(clean_term, "loinc", context)
            if external_result:
                logger.debug(f"Found external LOINC match for '{term}': {external_result['code']}")
                
                # Add the mapping to the custom mappings for future use
                self.add_custom_mapping(
                    "loinc", 
                    clean_term, 
                    external_result["code"], 
                    external_result["display"]
                )
                
                return external_result
                
        # 4. Return not found with the original term
        logger.debug(f"No LOINC mapping found for '{term}'")
        return {
            "code": None, 
            "display": term, 
            "system": "http://loinc.org", 
            "found": False,
            "attempted_methods": ["exact", 
                                 "fuzzy" if self.fuzzy_matcher else "", 
                                 "external" if (self.external_service and self.external_service.is_available()) else ""]
        }
    
    def map_to_rxnorm(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Map a medication term to RxNorm code.
        
        Args:
            term: The medication term to map
            context: Optional context information to improve mapping accuracy
            
        Returns:
            Dictionary with mapping results including code, display name,
            terminology system, and confidence score
        """
        if not term:
            return {
                "code": None, 
                "display": "", 
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm", 
                "found": False
            }
            
        # Clean and normalize the term
        clean_term = self._normalize_term(term)
        
        # 1. Try exact match in embedded database
        result = self.db_manager.lookup_rxnorm(clean_term)
        if result:
            logger.debug(f"Found exact RxNorm match for '{term}': {result['code']}")
            return result
            
        # 2. Try fuzzy matching if available
        if self.fuzzy_matcher and self.config.get("use_fuzzy_matching", True):
            fuzzy_result = self.fuzzy_matcher.find_fuzzy_match(clean_term, "rxnorm", context)
            if fuzzy_result:
                logger.debug(f"Found fuzzy RxNorm match for '{term}': {fuzzy_result['code']} (match type: {fuzzy_result.get('match_type', 'unknown')}, score: {fuzzy_result.get('score', 0)})")
                return fuzzy_result
                
        # 3. Try external service if available
        if self.external_service and self.config.get("use_external_services", False) and self.external_service.is_available():
            # For RxNorm, the external service has direct RxNav integration 
            # which is optimized for medication terms
            external_result = self.external_service.map_term(clean_term, "rxnorm", context)
            if external_result:
                logger.debug(f"Found external RxNorm match for '{term}': {external_result['code']}")
                
                # Add the mapping to the custom mappings for future use
                self.add_custom_mapping(
                    "rxnorm", 
                    clean_term, 
                    external_result["code"], 
                    external_result["display"]
                )
                
                return external_result
                
        # 4. Return not found with the original term
        logger.debug(f"No RxNorm mapping found for '{term}'")
        return {
            "code": None, 
            "display": term, 
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm", 
            "found": False,
            "attempted_methods": ["exact", 
                                 "fuzzy" if self.fuzzy_matcher else "", 
                                 "external" if (self.external_service and self.external_service.is_available()) else ""]
        }
    
    def _normalize_term(self, term: str) -> str:
        """
        Normalize a clinical term for better mapping.
        
        Args:
            term: The term to normalize
            
        Returns:
            Normalized term
        """
        if not term:
            return ""
        
        # Convert to lowercase
        term = term.lower()
        
        # Remove common prefix/suffix terms that might affect matching
        prefixes_to_remove = ["history of ", "chronic ", "acute ", "suspected "]
        for prefix in prefixes_to_remove:
            if term.startswith(prefix):
                term = term[len(prefix):]
        
        # Remove punctuation that doesn't affect meaning
        import re
        term = re.sub(r'[,.;:!?()]', ' ', term)
        
        # Normalize whitespace
        term = re.sub(r'\s+', ' ', term).strip()
        
        return term

    def add_custom_mapping(self, system: str, term: str, code: str, display: str) -> bool:
        """
        Add a custom mapping to the database.
        
        Args:
            system: The terminology system (snomed, loinc, rxnorm)
            term: The term to map
            code: The code to map to
            display: The display name for the code
            
        Returns:
            bool: True if the mapping was added successfully
        """
        mapping = {
            "code": code,
            "display": display,
            "system": self._get_system_uri(system),
            "found": True
        }
        
        return self.db_manager.add_mapping(system, self._normalize_term(term), mapping)
    
    def _get_system_uri(self, system: str) -> str:
        """Get the URI for a terminology system."""
        systems = {
            "snomed": "http://snomed.info/sct",
            "loinc": "http://loinc.org",
            "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm"
        }
        return systems.get(system.lower(), "unknown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the terminology mapper.
        
        Returns:
            Dictionary with statistics about available mappings
        """
        stats = self.db_manager.get_statistics()
        
        # Add fuzzy matching statistics if available
        if self.fuzzy_matcher:
            stats["fuzzy_matching"] = {
                "available": True,
                "thresholds": getattr(self.fuzzy_matcher, "thresholds", {})
            }
        else:
            stats["fuzzy_matching"] = {"available": False}
            
        # Add external service statistics if available
        if self.external_service:
            services_available = []
            if hasattr(self.external_service, "services"):
                for service_name, service_info in self.external_service.services.items():
                    if service_info.get("active", False):
                        services_available.append(service_name)
                        
            stats["external_services"] = {
                "available": self.external_service.is_available() if hasattr(self.external_service, "is_available") else False,
                "services": services_available
            }
        else:
            stats["external_services"] = {"available": False}
            
        return stats
        
    def map_term(self, term: str, system: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Map a clinical term to the specified terminology system.
        
        Args:
            term: The clinical term to map
            system: The terminology system to map to (snomed, loinc, rxnorm)
            context: Optional context information to improve mapping accuracy
            
        Returns:
            Dictionary with mapping results including code, display name,
            terminology system, and confidence score
        """
        if not term or not system:
            return {
                "code": None,
                "display": term or "",
                "system": self._get_system_uri(system),
                "found": False
            }
            
        # Route to the appropriate mapping method
        system = system.lower()
        if system == "snomed":
            return self.map_to_snomed(term, context)
        elif system == "loinc":
            return self.map_to_loinc(term, context)
        elif system == "rxnorm":
            return self.map_to_rxnorm(term, context)
        else:
            logger.warning(f"Unsupported terminology system: {system}")
            return {
                "code": None,
                "display": term,
                "system": "unknown",
                "found": False,
                "error": f"Unsupported terminology system: {system}"
            }
            
    def add_synonyms(self, term: str, synonyms: List[str]) -> bool:
        """
        Add synonym mappings for a term.
        
        Args:
            term: The primary term
            synonyms: List of synonyms for the term
            
        Returns:
            bool: True if synonyms were added successfully
        """
        if not self.fuzzy_matcher:
            logger.warning("Fuzzy matcher not available, cannot add synonyms")
            return False
            
        return self.fuzzy_matcher.add_synonym(term, synonyms)
        
    def close(self):
        """Close all database connections and resources."""
        # Close database connections
        if self.db_manager:
            self.db_manager.close()
            
        # Close external services if available
        if self.external_service and hasattr(self.external_service, "close"):
            self.external_service.close()