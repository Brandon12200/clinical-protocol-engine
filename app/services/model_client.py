import requests
import os
import json
import logging
import time
from flask import current_app

logger = logging.getLogger(__name__)

class ModelClient:
    """Client for communicating with the model service API"""
    
    def __init__(self, base_url=None):
        """
        Initialize the model client
        
        Args:
            base_url (str, optional): URL of the model service. Defaults to None, which will
                                     use the MODEL_SERVICE_URL from Flask config or environment.
        """
        self.base_url = base_url or os.environ.get('MODEL_SERVICE_URL')
        self.timeout = int(os.environ.get('MODEL_REQUEST_TIMEOUT', 300))  # 5-minute default timeout
        
        # Log initialization
        logger.info(f"Initializing model client with base URL: {self.base_url}")
        
    def _get_base_url(self):
        """Get base URL, with fallback to Flask config if available"""
        if self.base_url:
            return self.base_url
            
        # Try to get from Flask config if available
        try:
            return current_app.config.get('MODEL_SERVICE_URL', 'http://localhost:5001')
        except RuntimeError:
            # Not in Flask context
            return os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5001')
    
    def health_check(self):
        """
        Check if model service is healthy
        
        Returns:
            bool: True if model service is healthy, False otherwise
        """
        try:
            base_url = self._get_base_url()
            response = requests.get(f"{base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                is_healthy = data.get('status') == 'healthy' and data.get('model_loaded', False)
                
                if not is_healthy:
                    logger.warning(f"Model service health check returned unhealthy status: {data}")
                
                return is_healthy
            else:
                logger.warning(f"Model service health check failed with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Model service health check failed: {str(e)}")
            return False
    
    def model_info(self):
        """
        Get information about the model
        
        Returns:
            dict: Model information or None if request failed
        """
        try:
            base_url = self._get_base_url()
            response = requests.get(f"{base_url}/model_info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def extract(self, text, threshold=0.7):
        """
        Send text to model service for extraction
        
        Args:
            text (str): Text to extract information from
            threshold (float, optional): Confidence threshold. Defaults to 0.7.
            
        Returns:
            dict: Extraction results with entities, sections, and relations
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            logger.info(f"Sending text for extraction (length: {len(text)})")
            
            base_url = self._get_base_url()
            response = requests.post(
                f"{base_url}/extract",
                json={"text": text, "threshold": threshold},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Extraction successful: {len(result.get('entities', []))} entities, "
                        f"{len(result.get('sections', []))} sections, "
                        f"{len(result.get('relations', []))} relations")
            
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling model service: {str(e)}")
            raise
    
    def extract_entities(self, text, threshold=0.7):
        """
        Extract only entities from text
        
        Args:
            text (str): Text to extract entities from
            threshold (float, optional): Confidence threshold. Defaults to 0.7.
            
        Returns:
            list: Extracted entities
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            base_url = self._get_base_url()
            response = requests.post(
                f"{base_url}/extract_entities",
                json={"text": text, "threshold": threshold},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            entities = result.get('entities', [])
            logger.info(f"Entity extraction successful: {len(entities)} entities")
            
            return entities
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling entity extraction: {str(e)}")
            raise
    
    def extract_sections(self, text):
        """
        Extract only sections from text
        
        Args:
            text (str): Text to extract sections from
            
        Returns:
            list: Extracted sections
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            base_url = self._get_base_url()
            response = requests.post(
                f"{base_url}/extract_sections",
                json={"text": text},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            sections = result.get('sections', [])
            logger.info(f"Section extraction successful: {len(sections)} sections")
            
            return sections
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling section extraction: {str(e)}")
            raise
    
    def extract_relations(self, text, entities):
        """
        Extract relations between entities
        
        Args:
            text (str): Text context
            entities (list): Entities to find relations between
            
        Returns:
            list: Extracted relations
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            base_url = self._get_base_url()
            response = requests.post(
                f"{base_url}/extract_relations",
                json={"text": text, "entities": entities},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            relations = result.get('relations', [])
            logger.info(f"Relation extraction successful: {len(relations)} relations")
            
            return relations
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling relation extraction: {str(e)}")
            raise
    
    def extract_with_retry(self, text, threshold=0.7, max_retries=3, backoff_factor=2):
        """
        Extract with retry logic for resilience
        
        Args:
            text (str): Text to extract information from
            threshold (float, optional): Confidence threshold. Defaults to 0.7.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            backoff_factor (int, optional): Exponential backoff factor. Defaults to 2.
            
        Returns:
            dict: Extraction results
            
        Raises:
            Exception: If all retries fail
        """
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                return self.extract(text, threshold)
            except Exception as e:
                last_exception = e
                retries += 1
                if retries >= max_retries:
                    logger.error(f"All {max_retries} extraction attempts failed")
                    break
                
                wait_time = backoff_factor ** retries
                logger.warning(f"Extraction attempt {retries} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        raise last_exception or Exception("Extraction failed after multiple attempts")
    
    def process_document(self, text, threshold=0.7):
        """
        Process a complete document with extraction and additional processing
        
        Args:
            text (str): Document text
            threshold (float, optional): Confidence threshold. Defaults to 0.7.
            
        Returns:
            dict: Processed document results
        """
        try:
            # Extract all information
            extraction_results = self.extract(text, threshold)
            
            # Additional processing could be added here, such as:
            # - Entity filtering
            # - Relation validation
            # - Section reorganization
            
            # For now, we'll just return the extraction results
            return {
                'success': True,
                'extraction': extraction_results,
                'document_length': len(text),
                'threshold': threshold,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'document_length': len(text),
                'threshold': threshold,
                'timestamp': time.time()
            }