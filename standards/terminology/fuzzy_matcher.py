"""
Fuzzy matching module for terminology mapping.

This module implements advanced fuzzy matching algorithms for mapping
clinical terms to standardized terminologies when exact matches are not found.
"""

import os
import re
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
import nltk
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Helper function to ensure NLTK resources are available
def ensure_nltk_resources():
    """Ensure that all required NLTK resources are downloaded."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

# Ensure resources are available
ensure_nltk_resources()


class FuzzyMatcher:
    """
    Implements fuzzy matching algorithms for terminology mapping.
    
    This class provides methods for fuzzy matching clinical terms to
    standardized terminologies when exact matches cannot be found.
    
    Attributes:
        config: Configuration dictionary with matcher settings
        db_manager: Reference to the embedded database manager
        stopwords: Set of common words to ignore during matching
        synonym_expander: Optional module for synonym expansion
        term_index: In-memory index for faster fuzzy matching
    """
    
    def __init__(self, db_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fuzzy matcher.
        
        Args:
            db_manager: Reference to the embedded database manager
            config: Optional configuration dictionary with settings
        """
        self.config = config or {}
        self.db_manager = db_manager
        self.stopwords = 'english'  # Use string instead of set for sklearn compatibility
        self.synonym_expander = None
        self.term_index = {
            "snomed": {},
            "loinc": {},
            "rxnorm": {}
        }
        self.vectorizer = None
        self.vector_matrices = {}
        self.term_lists = {
            "snomed": [],
            "loinc": [],
            "rxnorm": []
        }
        
        # Default thresholds for different matching algorithms
        self.thresholds = {
            "ratio": self.config.get("ratio_threshold", 90),
            "partial_ratio": self.config.get("partial_ratio_threshold", 95),
            "token_sort_ratio": self.config.get("token_sort_ratio_threshold", 85),
            "cosine": self.config.get("cosine_threshold", 0.7),
            "n_gram": self.config.get("n_gram_threshold", 0.6)
        }
        
        # Load custom synonyms if available
        self._load_synonyms()
    
    def _load_synonyms(self):
        """Load custom synonym mappings if available."""
        try:
            synonyms_path = self.config.get("synonyms_path")
            if synonyms_path and os.path.exists(synonyms_path):
                with open(synonyms_path, 'r') as f:
                    self.synonyms = json.load(f)
                logger.info(f"Loaded {len(self.synonyms)} synonym sets from {synonyms_path}")
            else:
                self.synonyms = {}
        except Exception as e:
            logger.error(f"Error loading synonyms: {e}")
            self.synonyms = {}
    
    def initialize(self) -> bool:
        """
        Initialize the fuzzy matcher.
        
        This builds the in-memory indexes needed for efficient fuzzy matching.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Ensure NLTK resources are available
            ensure_nltk_resources()
            
            # Build indexes for each terminology
            success = self._build_index("snomed")
            success = self._build_index("loinc") and success
            success = self._build_index("rxnorm") and success
            
            # Initialize TF-IDF vectorizer
            self._initialize_vectorizer()
            
            if success:
                logger.info("Fuzzy matcher initialized successfully")
            else:
                logger.warning("Fuzzy matcher initialization incomplete")
                
            return success
        except Exception as e:
            logger.error(f"Error initializing fuzzy matcher: {e}")
            return False
    
    def _build_index(self, system: str) -> bool:
        """
        Build an in-memory index for a terminology system.
        
        Args:
            system: The terminology system to index (snomed, loinc, rxnorm)
            
        Returns:
            bool: True if the index was built successfully
        """
        try:
            # Check if we have a database connection for this system
            if system not in self.db_manager.connections:
                logger.warning(f"No database connection for {system}")
                return False
                
            conn = self.db_manager.connections[system]
            cursor = conn.cursor()
            table_name = f"{system}_concepts"
            
            # Get the terms from the database
            cursor.execute(f"SELECT code, term, display FROM {table_name}")
            rows = cursor.fetchall()
            
            # Build the index
            for code, term, display in rows:
                # Skip empty terms
                if not term:
                    continue
                    
                # Convert to lowercase
                term_lower = term.lower()
                
                # Add to the term list for vectorization
                self.term_lists[system].append((code, term_lower, display))
                
                # Add to the term index
                self.term_index[system][term_lower] = {
                    "code": code,
                    "display": display
                }
                
                # Index variations of the term
                variations = self._generate_term_variations(term_lower)
                for var in variations:
                    if var != term_lower:
                        self.term_index[system][var] = {
                            "code": code,
                            "display": display
                        }
            
            logger.info(f"Built index for {system} with {len(self.term_lists[system])} terms")
            return True
        except Exception as e:
            logger.error(f"Error building index for {system}: {e}")
            return False
    
    def _initialize_vectorizer(self):
        """Initialize the TF-IDF vectorizer for cosine similarity matching."""
        try:
            # Create a TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=self._tokenize,
                lowercase=True,
                stop_words=self.stopwords,
                ngram_range=(1, 2)  # Use unigrams and bigrams
            )
            
            # Build vector matrices for each terminology
            for system in ["snomed", "loinc", "rxnorm"]:
                if not self.term_lists[system]:
                    continue
                    
                # Extract just the terms
                terms = [term for _, term, _ in self.term_lists[system]]
                
                # Build the document-term matrix
                try:
                    matrix = self.vectorizer.fit_transform(terms)
                    self.vector_matrices[system] = matrix
                    logger.info(f"Built TF-IDF matrix for {system} with shape {matrix.shape}")
                except Exception as e:
                    logger.error(f"Error building TF-IDF matrix for {system}: {e}")
        except Exception as e:
            logger.error(f"Error initializing vectorizer: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF vectorization.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Get stopwords as a set for filtering
        if isinstance(self.stopwords, str) and self.stopwords == 'english':
            stopwords_set = set(nltk.corpus.stopwords.words('english'))
        else:
            stopwords_set = set(self.stopwords) if self.stopwords else set()
        
        # Remove stopwords
        tokens = [token.lower() for token in tokens if token.lower() not in stopwords_set]
        
        return tokens
    
    def _generate_term_variations(self, term: str) -> List[str]:
        """
        Generate variations of a term for fuzzy matching.
        
        Args:
            term: The term to generate variations for
            
        Returns:
            List of term variations
        """
        variations = set([term])
        
        # Remove common prefixes
        prefixes = ["history of ", "chronic ", "acute ", "suspected "]
        for prefix in prefixes:
            if term.startswith(prefix):
                variations.add(term[len(prefix):])
        
        # Remove punctuation
        term_no_punct = re.sub(r'[^\w\s]', ' ', term)
        variations.add(term_no_punct)
        
        # Normalize whitespace
        term_norm = re.sub(r'\s+', ' ', term_no_punct).strip()
        variations.add(term_norm)
        
        # Handle common abbreviations
        abbrev_map = {
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "uti": "urinary tract infection",
            "ckd": "chronic kidney disease",
            "gerd": "gastroesophageal reflux disease",
            "afib": "atrial fibrillation",
            "hb a1c": "hemoglobin a1c"
        }
        
        for abbrev, expanded in abbrev_map.items():
            if term == abbrev:
                variations.add(expanded)
            elif term == expanded:
                variations.add(abbrev)
        
        # Add synonyms if available
        for syn_set in self.synonyms.values():
            if term in syn_set:
                variations.update(syn_set)
        
        return list(variations)
    
    def find_fuzzy_match(self, term: str, system: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best fuzzy match for a term.
        
        Args:
            term: The term to find a match for
            system: The terminology system to search (snomed, loinc, rxnorm)
            context: Optional context to improve matching accuracy
            
        Returns:
            Dictionary with mapping information or None if no good match
        """
        if not term:
            return None
            
        # Normalize the term
        clean_term = term.lower()
        clean_term = re.sub(r'\s+', ' ', clean_term).strip()
        
        # Generate variations of the term
        variations = self._generate_term_variations(clean_term)
        
        # Try direct match with variations first
        for var in variations:
            if var in self.term_index[system]:
                match_info = self.term_index[system][var]
                return {
                    "code": match_info["code"],
                    "display": match_info["display"],
                    "system": self._get_system_uri(system),
                    "found": True,
                    "match_type": "variation",
                    "score": 100
                }
        
        # 1. Try RapidFuzz string matching
        string_match = self._find_string_match(clean_term, system)
        
        # 2. Try TF-IDF cosine similarity matching
        cosine_match = self._find_cosine_match(clean_term, system)
        
        # 3. Choose the best match
        best_match = None
        
        if string_match and cosine_match:
            # If both found matches, pick the one with higher score
            if string_match.get("score", 0) >= cosine_match.get("score", 0):
                best_match = string_match
            else:
                best_match = cosine_match
        elif string_match:
            best_match = string_match
        elif cosine_match:
            best_match = cosine_match
        
        # 4. Apply context-specific adjustments
        if best_match and context:
            # Use context to influence match confidence and prioritize certain matches
            match_score = best_match.get("score", 0)
            match_type = best_match.get("match_type", "")
            
            # Extract keywords from context
            context_lower = context.lower()
            
            # Domain-specific context adjustments
            if system == "snomed":
                # For diseases, check if context contains relevant keywords
                disease_contexts = {
                    "diabetes": ["glucose", "sugar", "a1c", "metformin", "insulin", "glycemic"],
                    "hypertension": ["blood pressure", "bp", "systolic", "diastolic", "mmhg"],
                    "asthma": ["respiratory", "breathing", "wheeze", "inhaler", "bronchial"],
                    "pneumonia": ["lung", "respiratory", "cough", "infection", "fever"],
                    "heart": ["cardiac", "chest pain", "cardiovascular", "ecg", "ekg"]
                }
                
                # Adjust score based on contextual relevance
                for keyword, contextual_terms in disease_contexts.items():
                    if keyword in best_match.get("display", "").lower():
                        # Check if any contextual terms are in the context
                        for contextual_term in contextual_terms:
                            if contextual_term in context_lower:
                                # Increase score if context supports the match
                                if "score" in best_match:
                                    best_match["score"] = min(100, match_score + 10)
                                best_match["context_enhanced"] = True
                                best_match["context_term"] = contextual_term
                                break
            
            elif system == "loinc":
                # For lab tests, check if context contains relevant keywords
                lab_contexts = {
                    "hemoglobin": ["blood", "cbc", "anemia", "diabetes"],
                    "glucose": ["diabetes", "blood sugar", "fasting", "a1c"],
                    "cholesterol": ["lipid", "hdl", "ldl", "cardiovascular"],
                    "creatinine": ["kidney", "renal", "gfr", "bun"]
                }
                
                # Adjust score based on contextual relevance
                for keyword, contextual_terms in lab_contexts.items():
                    if keyword in best_match.get("display", "").lower():
                        # Check if any contextual terms are in the context
                        for contextual_term in contextual_terms:
                            if contextual_term in context_lower:
                                # Increase score if context supports the match
                                if "score" in best_match:
                                    best_match["score"] = min(100, match_score + 10)
                                best_match["context_enhanced"] = True
                                best_match["context_term"] = contextual_term
                                break
            
            elif system == "rxnorm":
                # For medications, check if context contains relevant keywords
                med_contexts = {
                    "metformin": ["diabetes", "hypoglycemic", "glucose", "a1c"],
                    "lisinopril": ["hypertension", "blood pressure", "ace inhibitor", "bp"],
                    "aspirin": ["antiplatelet", "pain", "blood thinner", "heart", "stroke"],
                    "atorvastatin": ["cholesterol", "statin", "lipid", "cardiovascular"]
                }
                
                # Adjust score based on contextual relevance
                for keyword, contextual_terms in med_contexts.items():
                    if keyword in best_match.get("display", "").lower():
                        # Check if any contextual terms are in the context
                        for contextual_term in contextual_terms:
                            if contextual_term in context_lower:
                                # Increase score if context supports the match
                                if "score" in best_match:
                                    best_match["score"] = min(100, match_score + 10)
                                best_match["context_enhanced"] = True
                                best_match["context_term"] = contextual_term
                                break
        
        return best_match
    
    def _find_string_match(self, term: str, system: str) -> Optional[Dict[str, Any]]:
        """
        Find the best string-based fuzzy match using RapidFuzz.
        
        Args:
            term: The term to match
            system: The terminology system to search
            
        Returns:
            Dictionary with mapping information or None if no good match
        """
        if not self.term_lists[system]:
            return None
            
        # Extract terms for matching
        terms = [(code, term_text, display) for code, term_text, display in self.term_lists[system]]
        
        # Try different fuzzy matching strategies
        
        # 1. Simple ratio (overall similarity)
        ratio_matches = process.extractOne(
            term,
            [t[1] for t in terms],
            scorer=fuzz.ratio,
            score_cutoff=self.thresholds["ratio"]
        )
        
        # 2. Partial ratio (best partial string alignment)
        partial_matches = process.extractOne(
            term,
            [t[1] for t in terms],
            scorer=fuzz.partial_ratio,
            score_cutoff=self.thresholds["partial_ratio"]
        )
        
        # 3. Token sort ratio (order-independent similarity)
        token_matches = process.extractOne(
            term,
            [t[1] for t in terms],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.thresholds["token_sort_ratio"]
        )
        
        # 4. Determine the best match
        best_match = None
        best_score = 0
        match_type = ""
        
        if ratio_matches and ratio_matches[1] > best_score:
            best_score = ratio_matches[1]
            best_match = ratio_matches[0]
            match_type = "ratio"
            
        if partial_matches and partial_matches[1] > best_score:
            best_score = partial_matches[1]
            best_match = partial_matches[0]
            match_type = "partial_ratio"
            
        if token_matches and token_matches[1] > best_score:
            best_score = token_matches[1]
            best_match = token_matches[0]
            match_type = "token_sort_ratio"
        
        if best_match:
            # Find the corresponding code and display
            index = [t[1] for t in terms].index(best_match)
            code, _, display = terms[index]
            
            return {
                "code": code,
                "display": display,
                "system": self._get_system_uri(system),
                "found": True,
                "match_type": match_type,
                "score": best_score
            }
            
        return None
    
    def _find_cosine_match(self, term: str, system: str) -> Optional[Dict[str, Any]]:
        """
        Find the best cosine similarity match using TF-IDF.
        
        Args:
            term: The term to match
            system: The terminology system to search
            
        Returns:
            Dictionary with mapping information or None if no good match
        """
        if system not in self.vector_matrices or not self.vectorizer:
            return None
            
        try:
            # Transform the query term
            term_vector = self.vectorizer.transform([term])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(term_vector, self.vector_matrices[system]).flatten()
            
            # Find the best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= self.thresholds["cosine"]:
                code, _, display = self.term_lists[system][best_idx]
                
                return {
                    "code": code,
                    "display": display,
                    "system": self._get_system_uri(system),
                    "found": True,
                    "match_type": "cosine",
                    "score": float(best_score * 100)  # Convert to percentage
                }
        except Exception as e:
            logger.error(f"Error finding cosine match for term '{term}': {e}")
            
        return None
    
    def add_synonym(self, term: str, synonyms: List[str]) -> bool:
        """
        Add synonyms for a term.
        
        Args:
            term: The primary term
            synonyms: List of synonyms for the term
            
        Returns:
            bool: True if synonyms were added successfully
        """
        try:
            # Create a unique set with the term and all synonyms
            syn_set = set([term.lower()] + [s.lower() for s in synonyms])
            
            # Check if this term is already in a synonym set
            for key, existing_set in self.synonyms.items():
                if term.lower() in existing_set:
                    # Add the new synonyms to the existing set
                    existing_set.update(syn_set)
                    self.synonyms[key] = list(existing_set)
                    
                    logger.info(f"Updated synonym set for '{term}' with {len(synonyms)} new synonyms")
                    
                    # Save the synonyms
                    self._save_synonyms()
                    return True
            
            # Create a new synonym set
            new_key = f"syn_set_{len(self.synonyms) + 1}"
            self.synonyms[new_key] = list(syn_set)
            
            logger.info(f"Created new synonym set for '{term}' with {len(synonyms)} synonyms")
            
            # Save the synonyms
            self._save_synonyms()
            return True
        except Exception as e:
            logger.error(f"Error adding synonyms for term '{term}': {e}")
            return False
    
    def _save_synonyms(self):
        """Save synonyms to the configured file."""
        try:
            synonyms_path = self.config.get("synonyms_path")
            if synonyms_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(synonyms_path), exist_ok=True)
                
                with open(synonyms_path, 'w') as f:
                    json.dump(self.synonyms, f, indent=2)
                    
                logger.info(f"Saved {len(self.synonyms)} synonym sets to {synonyms_path}")
        except Exception as e:
            logger.error(f"Error saving synonyms: {e}")
    
    def _get_system_uri(self, system: str) -> str:
        """Get the URI for a terminology system."""
        systems = {
            "snomed": "http://snomed.info/sct",
            "loinc": "http://loinc.org",
            "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm"
        }
        return systems.get(system.lower(), "unknown")