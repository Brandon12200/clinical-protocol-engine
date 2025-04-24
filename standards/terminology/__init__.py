"""
Terminology mapping module for clinical protocol extraction.

This module provides functionality for mapping extracted clinical terms
to standardized terminologies like SNOMED CT, LOINC, and RxNorm using
embedded databases, fuzzy matching, and external services.
"""

from standards.terminology.mapper import TerminologyMapper
from standards.terminology.embedded_db import EmbeddedDatabaseManager
from standards.terminology.fuzzy_matcher import FuzzyMatcher
from standards.terminology.external_service import ExternalTerminologyService

__all__ = [
    "TerminologyMapper",
    "EmbeddedDatabaseManager",
    "FuzzyMatcher",
    "ExternalTerminologyService"
]