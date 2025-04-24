"""
External terminology service manager.

This module manages connections to external terminology services like
UMLS API, BioPortal, and other clinical terminology APIs.
"""

import os
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import urljoin
import redis

# Configure logging
logger = logging.getLogger(__name__)

class ExternalTerminologyService:
    """
    Interface with external terminology services.
    
    This class provides methods for connecting to and retrieving data
    from external terminology services when available.
    
    Attributes:
        config: Configuration dictionary with API keys and service endpoints
        services: Dictionary of active service connections
        cache: Redis cache client for caching responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the external terminology service manager.
        
        Args:
            config: Optional configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.services = {}
        self.cache = None
        
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Set up the Redis cache if enabled."""
        if self.config.get("use_redis_cache", False):
            try:
                redis_host = self.config.get("redis_host", "localhost")
                redis_port = self.config.get("redis_port", 6379)
                redis_db = self.config.get("redis_db", 0)
                
                self.cache = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                logger.info(f"Connected to Redis cache at {redis_host}:{redis_port}")
            except Exception as e:
                logger.error(f"Error connecting to Redis cache: {e}")
                self.cache = None
    
    def initialize(self) -> bool:
        """
        Initialize connections to external terminology services.
        
        Returns:
            bool: True if at least one service was initialized successfully
        """
        success = False
        
        # Check for UMLS API access
        if self.config.get("umls_api_key"):
            success = self._initialize_umls() or success
            
        # Check for BioPortal access
        if self.config.get("bioportal_api_key"):
            success = self._initialize_bioportal() or success
        
        # Check for RxNav API access
        if self.config.get("use_rxnav_api", True):
            success = self._initialize_rxnav() or success
        
        if success:
            logger.info("Successfully initialized external terminology services")
        else:
            logger.warning("No external terminology services were successfully initialized")
            
        return success
    
    def _initialize_umls(self) -> bool:
        """
        Initialize the UMLS API service.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            api_key = self.config.get("umls_api_key")
            if not api_key:
                logger.warning("No UMLS API key provided")
                return False
                
            # Store the API information
            self.services["umls"] = {
                "api_key": api_key,
                "base_url": "https://uts-ws.nlm.nih.gov/rest/",
                "active": True,
                "tgt": None,  # Will store the ticket-granting ticket
                "tgt_timestamp": 0  # When the TGT was obtained
            }
            
            # Test the connection by getting a ticket-granting ticket
            self._get_umls_tgt()
            
            logger.info("UMLS API service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing UMLS API service: {e}")
            return False
    
    def _get_umls_tgt(self) -> Optional[str]:
        """
        Get a ticket-granting ticket from the UMLS API.
        
        Returns:
            str: The ticket-granting ticket or None if error
        """
        if "umls" not in self.services:
            logger.error("UMLS service not initialized")
            return None
            
        service = self.services["umls"]
        
        # Check if we already have a valid TGT (less than 8 hours old)
        if service.get("tgt") and (time.time() - service.get("tgt_timestamp", 0)) < 28800:
            return service["tgt"]
            
        try:
            auth_endpoint = urljoin(service["base_url"], "auth/authenticateUser")
            payload = {"apikey": service["api_key"]}
            
            response = requests.post(auth_endpoint, data=payload)
            response.raise_for_status()
            
            # Extract TGT from response
            tgt = response.text
            
            # Store the TGT and timestamp
            service["tgt"] = tgt
            service["tgt_timestamp"] = time.time()
            
            logger.debug("Retrieved new UMLS ticket-granting ticket")
            return tgt
        except Exception as e:
            logger.error(f"Error getting UMLS ticket-granting ticket: {e}")
            service["active"] = False
            return None
    
    def _initialize_bioportal(self) -> bool:
        """
        Initialize the BioPortal API service.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            api_key = self.config.get("bioportal_api_key")
            if not api_key:
                logger.warning("No BioPortal API key provided")
                return False
                
            # Store the API information
            self.services["bioportal"] = {
                "api_key": api_key,
                "base_url": "https://data.bioontology.org/",
                "active": True
            }
            
            # Test the connection
            headers = {"Authorization": f"apikey token={api_key}"}
            test_url = urljoin(self.services["bioportal"]["base_url"], "ontologies")
            
            response = requests.get(test_url, headers=headers)
            response.raise_for_status()
            
            logger.info("BioPortal API service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing BioPortal API service: {e}")
            return False
    
    def _initialize_rxnav(self) -> bool:
        """
        Initialize the RxNav API service (no auth required).
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Store the API information
            self.services["rxnav"] = {
                "base_url": "https://rxnav.nlm.nih.gov/REST/",
                "active": True
            }
            
            # Test the connection
            test_url = urljoin(self.services["rxnav"]["base_url"], "version")
            
            response = requests.get(test_url)
            response.raise_for_status()
            
            logger.info("RxNav API service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing RxNav API service: {e}")
            return False
    
    def search_umls(self, term: str, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a term in the UMLS API.
        
        Args:
            term: The term to search for
            source: Optional source vocabulary (SNOMEDCT_US, LNC, RXNORM)
            
        Returns:
            Dictionary with mapping information or None if error/not found
        """
        # Check cache first if available
        cache_key = f"umls:{term}:{source or 'all'}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        
        if "umls" not in self.services or not self.services["umls"].get("active"):
            logger.warning("UMLS service not active")
            return None
            
        try:
            # Get a service ticket using the TGT
            tgt = self._get_umls_tgt()
            if not tgt:
                return None
                
            service = self.services["umls"]
            ticket_url = service["tgt"]
            
            ticket_response = requests.post(ticket_url, data={"service": "http://umlsks.nlm.nih.gov"})
            ticket_response.raise_for_status()
            service_ticket = ticket_response.text
            
            # Search for the term
            search_endpoint = urljoin(service["base_url"], "search/current")
            params = {
                "string": term,
                "ticket": service_ticket,
                "returnIdType": "concept",
                "sabs": source or ""
            }
            
            response = requests.get(search_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result", {}).get("results"):
                results = data["result"]["results"]
                if results:
                    # Get the best match
                    best_match = results[0]
                    
                    concept_info = {
                        "cui": best_match.get("ui"),
                        "name": best_match.get("name"),
                        "source": best_match.get("rootSource"),
                        "found": True
                    }
                    
                    # Cache the result if caching is enabled
                    if self.cache:
                        self.cache.setex(
                            cache_key,
                            3600,  # Cache for 1 hour
                            json.dumps(concept_info)
                        )
                    
                    return concept_info
            
            return None
        except Exception as e:
            logger.error(f"Error searching UMLS for term '{term}': {e}")
            return None
    
    def search_bioportal(self, term: str, ontology: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a term in BioPortal.
        
        Args:
            term: The term to search for
            ontology: Optional ontology ID (SNOMEDCT, LOINC, RXNORM)
            
        Returns:
            Dictionary with mapping information or None if error/not found
        """
        # Check cache first if available
        cache_key = f"bioportal:{term}:{ontology or 'all'}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        
        if "bioportal" not in self.services or not self.services["bioportal"].get("active"):
            logger.warning("BioPortal service not active")
            return None
            
        try:
            service = self.services["bioportal"]
            headers = {"Authorization": f"apikey token={service['api_key']}"}
            
            search_endpoint = urljoin(service["base_url"], "search")
            params = {
                "q": term,
                "ontologies": ontology or "",
                "include": "prefLabel,synonym,cui,semanticType",
                "require_exact_match": "false"
            }
            
            response = requests.get(search_endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("collection") and data["collection"]:
                results = data["collection"]
                if results:
                    # Get the best match
                    best_match = results[0]
                    
                    concept_info = {
                        "code": best_match.get("id", "").split("/")[-1],
                        "display": best_match.get("prefLabel"),
                        "ontology": best_match.get("links", {}).get("ontology", "").split("/")[-1],
                        "found": True
                    }
                    
                    # Cache the result if caching is enabled
                    if self.cache:
                        self.cache.setex(
                            cache_key,
                            3600,  # Cache for 1 hour
                            json.dumps(concept_info)
                        )
                    
                    return concept_info
            
            return None
        except Exception as e:
            logger.error(f"Error searching BioPortal for term '{term}': {e}")
            return None
    
    def search_rxnav(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Search for a medication term in RxNav.
        
        Args:
            term: The medication term to search for
            
        Returns:
            Dictionary with mapping information or None if error/not found
        """
        # Check cache first if available
        cache_key = f"rxnav:{term}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        
        if "rxnav" not in self.services or not self.services["rxnav"].get("active"):
            logger.warning("RxNav service not active")
            return None
            
        try:
            service = self.services["rxnav"]
            
            search_endpoint = urljoin(service["base_url"], "approximateTerm")
            params = {
                "term": term,
                "maxEntries": 1,
                "option": 0  # 0 = Current concepts
            }
            
            response = requests.get(search_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("approximateGroup", {}).get("candidate"):
                candidates = data["approximateGroup"]["candidate"]
                if candidates:
                    best_match = candidates[0]
                    
                    # Now get the RxCUI details
                    rxcui = best_match.get("rxcui")
                    if rxcui:
                        rxcui_endpoint = urljoin(service["base_url"], f"rxcui/{rxcui}/property")
                        rxcui_params = {"propName": "RxNorm Name"}
                        
                        rxcui_response = requests.get(rxcui_endpoint, params=rxcui_params)
                        rxcui_response.raise_for_status()
                        rxcui_data = rxcui_response.json()
                        
                        preferred_name = None
                        if rxcui_data.get("propConceptGroup", {}).get("propConcept"):
                            props = rxcui_data["propConceptGroup"]["propConcept"]
                            for prop in props:
                                if prop.get("propName") == "RxNorm Name":
                                    preferred_name = prop.get("propValue")
                                    break
                        
                        concept_info = {
                            "code": rxcui,
                            "display": preferred_name or best_match.get("name"),
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "score": best_match.get("score"),
                            "found": True
                        }
                        
                        # Cache the result if caching is enabled
                        if self.cache:
                            self.cache.setex(
                                cache_key,
                                3600,  # Cache for 1 hour
                                json.dumps(concept_info)
                            )
                        
                        return concept_info
            
            return None
        except Exception as e:
            logger.error(f"Error searching RxNav for term '{term}': {e}")
            return None
    
    def map_term(self, term: str, system: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Map a term to a standardized terminology using the best available service.
        
        Args:
            term: The term to map
            system: The target terminology system (snomed, loinc, rxnorm)
            context: Optional context to improve mapping
            
        Returns:
            Dictionary with mapping information or None if not found
        """
        result = None
        
        # Select appropriate service based on terminology system
        if system.lower() == "rxnorm":
            # For RxNorm, try RxNav first
            result = self.search_rxnav(term)
            
        if not result and system.lower() in ["snomed", "loinc", "rxnorm"]:
            # Map system to UMLS source abbreviation
            umls_source = {
                "snomed": "SNOMEDCT_US",
                "loinc": "LNC",
                "rxnorm": "RXNORM"
            }.get(system.lower())
            
            result = self.search_umls(term, umls_source)
            
        if not result:
            # Try BioPortal as fallback
            bioportal_ontology = {
                "snomed": "SNOMEDCT",
                "loinc": "LOINC",
                "rxnorm": "RXNORM"
            }.get(system.lower())
            
            result = self.search_bioportal(term, bioportal_ontology)
        
        if result and result.get("found"):
            # Transform to standard format if needed
            if "cui" in result:
                # Transform UMLS format
                system_uri = {
                    "snomed": "http://snomed.info/sct",
                    "loinc": "http://loinc.org",
                    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm"
                }.get(system.lower())
                
                return {
                    "code": result.get("cui"),
                    "display": result.get("name"),
                    "system": system_uri,
                    "found": True
                }
            
            # Make sure system URI is set correctly
            if "system" not in result:
                system_uri = {
                    "snomed": "http://snomed.info/sct",
                    "loinc": "http://loinc.org",
                    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm"
                }.get(system.lower())
                
                result["system"] = system_uri
            
            return result
        
        return None
    
    def is_available(self) -> bool:
        """
        Check if any external service is available.
        
        Returns:
            bool: True if at least one service is available
        """
        return any(service.get("active", False) for service in self.services.values())
    
    def close(self):
        """Close any open connections and resources."""
        # Close Redis connection if it exists
        if self.cache:
            try:
                self.cache.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")