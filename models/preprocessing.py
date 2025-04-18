import re
import nltk
from nltk.tokenize import sent_tokenize
import string
import logging
import unicodedata
import os
from typing import List, Dict, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)

# Medical abbreviations mapping dictionary
MEDICAL_ABBREVIATIONS = {
    r'\bpt\b': 'patient',
    r'\bpts\b': 'patients',
    r'\bDx\b': 'diagnosis',
    r'\bRx\b': 'treatment',
    r'\bHx\b': 'history',
    r'\bw/\b': 'with',
    r'\bw/o\b': 'without',
    r'\bs/p\b': 'status post',
    r'\bp\.o\.\b': 'by mouth',
    r'\bb\.i\.d\.\b': 'twice daily',
    r'\bt\.i\.d\.\b': 'three times daily',
    r'\bq\.i\.d\.\b': 'four times daily',
    r'\bq\.d\.\b': 'once daily',
    r'\bp\.r\.n\.\b': 'as needed',
    r'\bIV\b': 'intravenous',
    r'\bIM\b': 'intramuscular',
    r'\bSC\b': 'subcutaneous',
    r'\bq\b': 'every',
    r'\bh\b': 'hour',
    r'\bAM\b': 'morning',
    r'\bPM\b': 'evening',
    r'\byo\b': 'year old',
    r'\by\.o\.\b': 'year old'
}

# Common section headers in clinical protocols
SECTION_HEADERS = [
    "inclusion criteria",
    "exclusion criteria",
    "eligibility criteria",
    "study population",
    "patient selection",
    "study procedures",
    "intervention",
    "study design",
    "objectives",
    "endpoints",
    "primary endpoint",
    "secondary endpoints",
    "methodology",
    "statistical analysis",
    "adverse events",
    "dosage and administration",
    "background",
    "introduction",
    "study rationale",
    "conclusion",
    "references"
]


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    This function applies multiple normalization steps to prepare text for model processing:
    - Removes control characters
    - Normalizes Unicode (NFKC form)
    - Standardizes whitespace
    - Normalizes punctuation
    - Standardizes common medical abbreviations
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and normalized text
    """
    try:
        if not text:
            return ""
            
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize Unicode characters (NFKC form)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize newlines (convert multiple newlines to single newline)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove excessive whitespace at beginning and end of lines
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        # Standardize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Add space after periods if not present and followed by uppercase letter
        # This helps with sentence tokenization
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Normalize medical abbreviations
        text = normalize_medical_abbreviations(text)
        
        # Enhance section header detection
        text = highlight_section_headers(text)
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        # Return original text if an error occurs
        return text


def normalize_medical_abbreviations(text: str) -> str:
    """
    Standardize common medical abbreviations in text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with standardized abbreviations
    """
    try:
        # Apply all abbreviation replacements
        for pattern, replacement in MEDICAL_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    except Exception as e:
        logger.error(f"Error normalizing medical abbreviations: {str(e)}")
        return text


def highlight_section_headers(text: str) -> str:
    """
    Enhance section header detection by adding markers around likely headers.
    This helps the model identify document structure.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with enhanced section headers
    """
    try:
        # Process each line
        lines = text.split('\n')
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if line is likely a header (short, ends with colon, all caps, or matches known headers)
            is_header = False
            
            # Check if line is all uppercase or title case
            if line.isupper() or line.istitle():
                is_header = True
                
            # Check if line ends with colon
            if line.endswith(':'):
                is_header = True
                
            # Check if line is short (likely a header)
            if len(line) < 50 and len(line.split()) < 6:
                # Check against known section headers
                for header in SECTION_HEADERS:
                    if header in line.lower():
                        is_header = True
                        break
            
            # Add emphasis to headers if detected
            if is_header:
                lines[i] = f"{line}"
        
        return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Error highlighting section headers: {str(e)}")
        return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's sentence tokenizer.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of sentences
    """
    try:
        # Clean text before sentence splitting
        text = clean_text(text)
        
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Additional processing for better sentence boundaries
        processed_sentences = []
        for sentence in sentences:
            # Split sentences that might have been missed due to formatting issues
            if len(sentence) > 200 and '. ' in sentence:
                # Look for missed sentence boundaries (period followed by capital letter)
                sub_sentences = re.split(r'\.(?=\s+[A-Z])', sentence)
                for sub in sub_sentences:
                    if sub.strip():  # Only add non-empty sentences
                        processed_sentences.append(sub.strip() + '.' if not sub.strip().endswith('.') else sub.strip())
            else:
                processed_sentences.append(sentence)
        
        return processed_sentences
    except Exception as e:
        logger.error(f"Error splitting text into sentences: {str(e)}")
        # Fallback to simple splitting by periods
        return re.split(r'\.+', text)


def chunk_document(text: str, max_length: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split document into overlapping chunks for processing by the model.
    Tries to maintain sentence boundaries where possible.
    
    Args:
        text (str): Input text
        max_length (int): Maximum chunk length
        overlap (int): Overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of chunks with text and position offsets
    """
    try:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + max_length, text_length)
            
            # Try to find a sentence boundary for cleaner splitting
            if end < text_length:
                # First try to find sentence-ending punctuation followed by space or newline
                sentence_boundary_found = False
                for i in range(min(end + 30, text_length - 1), max(end - 30, start), -1):
                    if i >= text_length:
                        continue
                        
                    if text[i] in '.!?' and (i + 1 >= text_length or text[i + 1].isspace()):
                        end = i + 1
                        sentence_boundary_found = True
                        break
                
                # If no sentence boundary, try to find a newline
                if not sentence_boundary_found:
                    for i in range(min(end + 20, text_length - 1), max(end - 20, start), -1):
                        if i >= text_length:
                            continue
                            
                        if text[i] == '\n':
                            end = i + 1
                            break
            
            # Create chunk
            chunks.append({
                'text': text[start:end],
                'offset': start
            })
            
            # Move start position for next chunk, with overlap
            start = max(start + 1, end - overlap)
        
        logger.debug(f"Split document into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        # Fallback to simple chunking without considering sentence boundaries
        chunks = []
        for i in range(0, len(text), max_length - overlap):
            end = min(i + max_length, len(text))
            chunks.append({
                'text': text[i:end],
                'offset': i
            })
        return chunks


def detect_section_boundaries(text: str) -> List[Dict[str, Any]]:
    """
    Detect section boundaries in a document.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of sections with type, start and end positions
    """
    try:
        cleaned_text = clean_text(text)
        sections = []
        lines = cleaned_text.split('\n')
        current_position = 0
        current_section = None
        section_start = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for the newline
            
            # Skip empty lines
            if not line.strip():
                current_position += line_length
                continue
            
            # Check if line is a section header
            is_header = False
            section_type = None
            
            # Check against known section headers
            for header in SECTION_HEADERS:
                if header in line.lower():
                    is_header = True
                    section_type = header.upper().replace(' ', '_')
                    break
            
            # Check for formatting patterns of headers
            if not is_header:
                # All uppercase with few words
                if line.isupper() and len(line.split()) < 6:
                    is_header = True
                    section_type = "UNKNOWN_SECTION"
                # Numbered headers
                elif re.match(r'^[0-9]+\.[0-9]*\s+\w+', line):
                    is_header = True
                    section_type = "UNKNOWN_SECTION"
                # Headers ending with colon
                elif line.strip().endswith(':') and len(line) < 50:
                    is_header = True
                    section_type = "UNKNOWN_SECTION"
            
            # If found a new section header
            if is_header:
                # Save previous section if exists
                if current_section and current_position > section_start:
                    sections.append({
                        'type': current_section,
                        'start': section_start,
                        'end': current_position,
                        'text': text[section_start:current_position].strip()
                    })
                
                # Start new section
                current_section = section_type
                section_start = current_position
            
            current_position += line_length
        
        # Add final section
        if current_section and current_position > section_start:
            sections.append({
                'type': current_section,
                'start': section_start,
                'end': current_position,
                'text': text[section_start:current_position].strip()
            })
        
        logger.debug(f"Detected {len(sections)} sections in document")
        return sections
    except Exception as e:
        logger.error(f"Error detecting section boundaries: {str(e)}")
        # Return a single section for the whole document as fallback
        return [{
            'type': 'UNKNOWN_SECTION',
            'start': 0,
            'end': len(text),
            'text': text
        }]


def extract_eligibility_criteria(text: str) -> List[Dict[str, Any]]:
    """
    Extract eligibility criteria from text using rule-based methods.
    This is used as a fallback when the ML model fails.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of inclusion and exclusion criteria
    """
    try:
        criteria = []
        
        # Extract inclusion and exclusion sections
        inclusion_pattern = re.compile(r'(?i)inclusion criteria[:.\-\s]*?((?:[^\n]*?\n)+?)(?:\n\n|\Z|exclusion criteria)', re.DOTALL)
        exclusion_pattern = re.compile(r'(?i)exclusion criteria[:.\-\s]*?((?:[^\n]*?\n)+?)(?:\n\n|\Z)', re.DOTALL)
        
        # Find inclusion criteria
        inclusion_match = inclusion_pattern.search(text)
        if inclusion_match:
            inclusion_text = inclusion_match.group(1)
            # Extract bulleted or numbered criteria
            bullets = re.findall(r'(?:^|\n)[•\-*⦁○●\d+\.]+\s*([^\n]+)', inclusion_text)
            
            # If no bullets found, try splitting by newlines
            if not bullets:
                bullets = [line.strip() for line in inclusion_text.split('\n') if line.strip()]
                
            for criteria_text in bullets:
                criteria.append({
                    'type': 'inclusion',
                    'text': criteria_text.strip()
                })
        
        # Find exclusion criteria
        exclusion_match = exclusion_pattern.search(text)
        if exclusion_match:
            exclusion_text = exclusion_match.group(1)
            # Extract bulleted or numbered criteria
            bullets = re.findall(r'(?:^|\n)[•\-*⦁○●\d+\.]+\s*([^\n]+)', exclusion_text)
            
            # If no bullets found, try splitting by newlines
            if not bullets:
                bullets = [line.strip() for line in exclusion_text.split('\n') if line.strip()]
                
            for criteria_text in bullets:
                criteria.append({
                    'type': 'exclusion',
                    'text': criteria_text.strip()
                })
        
        logger.debug(f"Extracted {len(criteria)} eligibility criteria using rule-based method")
        return criteria
    except Exception as e:
        logger.error(f"Error extracting eligibility criteria: {str(e)}")
        return []


def extract_endpoints(text: str) -> List[Dict[str, Any]]:
    """
    Extract study endpoints from text using rule-based methods.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of primary and secondary endpoints
    """
    try:
        endpoints = []
        
        # Extract endpoints sections
        primary_pattern = re.compile(r'(?i)primary\s+endpoints?[:.\-\s]*?((?:[^\n]*?\n)+?)(?:\n\n|\Z|secondary\s+endpoints?)', re.DOTALL)
        secondary_pattern = re.compile(r'(?i)secondary\s+endpoints?[:.\-\s]*?((?:[^\n]*?\n)+?)(?:\n\n|\Z)', re.DOTALL)
        
        # Find primary endpoints
        primary_match = primary_pattern.search(text)
        if primary_match:
            primary_text = primary_match.group(1)
            # Extract bulleted or numbered endpoints
            bullets = re.findall(r'(?:^|\n)[•\-*⦁○●\d+\.]+\s*([^\n]+)', primary_text)
            
            # If no bullets found, use the whole text
            if not bullets:
                bullets = [primary_text.strip()]
                
            for endpoint_text in bullets:
                endpoints.append({
                    'type': 'primary',
                    'text': endpoint_text.strip()
                })
        
        # Find secondary endpoints
        secondary_match = secondary_pattern.search(text)
        if secondary_match:
            secondary_text = secondary_match.group(1)
            # Extract bulleted or numbered endpoints
            bullets = re.findall(r'(?:^|\n)[•\-*⦁○●\d+\.]+\s*([^\n]+)', secondary_text)
            
            # If no bullets found, use the whole text or split by newlines
            if not bullets:
                bullets = [line.strip() for line in secondary_text.split('\n') if line.strip()]
                
            for endpoint_text in bullets:
                endpoints.append({
                    'type': 'secondary',
                    'text': endpoint_text.strip()
                })
        
        logger.debug(f"Extracted {len(endpoints)} endpoints using rule-based method")
        return endpoints
    except Exception as e:
        logger.error(f"Error extracting endpoints: {str(e)}")
        return []


def extract_numeric_values(text: str) -> List[Dict[str, Any]]:
    """
    Extract numeric values and units from text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of numeric values with units
    """
    try:
        numeric_values = []
        
        # Pattern for numeric values with units
        # Matches numbers (integer or decimal) followed by optional units
        pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?)?'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            value = match.group(1)
            unit = match.group(2) if match.group(2) else None
            
            numeric_values.append({
                'value': float(value),
                'unit': unit,
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0)
            })
        
        return numeric_values
    except Exception as e:
        logger.error(f"Error extracting numeric values: {str(e)}")
        return []


def extract_time_expressions(text: str) -> List[Dict[str, Any]]:
    """
    Extract time-related expressions from text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of time expressions
    """
    try:
        time_expressions = []
        
        # Pattern for time durations
        duration_pattern = r'((?:\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?))'
        # Pattern for frequencies
        frequency_pattern = r'((?:once|twice|three times|four times)\s+(?:daily|weekly|monthly|yearly|a day|a week|a month|a year))'
        # Pattern for specific timepoints
        timepoint_pattern = r'((?:day|week|month|year)\s+\d+)'
        
        # Extract durations
        duration_matches = re.finditer(duration_pattern, text, re.IGNORECASE)
        for match in duration_matches:
            time_expressions.append({
                'type': 'duration',
                'text': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract frequencies
        frequency_matches = re.finditer(frequency_pattern, text, re.IGNORECASE)
        for match in frequency_matches:
            time_expressions.append({
                'type': 'frequency',
                'text': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract timepoints
        timepoint_matches = re.finditer(timepoint_pattern, text, re.IGNORECASE)
        for match in timepoint_matches:
            time_expressions.append({
                'type': 'timepoint',
                'text': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        return time_expressions
    except Exception as e:
        logger.error(f"Error extracting time expressions: {str(e)}")
        return []


def detect_language(text: str) -> str:
    """
    Detect language of text using character frequency analysis.
    Simplified method that works for major European languages.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected language code ('en', 'fr', 'de', 'es', etc.) or 'unknown'
    """
    try:
        # Common character trigrams for major languages
        language_profiles = {
            'en': ['the', 'and', 'ing', 'ion', 'to_', 'ed_', 'is_', 'at_', 'on_', 'as_'],
            'fr': ['les', 'de_', 'ent', 'et_', 'des', 'que', 'ons', 'ur_', 'est', 'ait'],
            'de': ['der', 'die', 'ein', 'und', 'den', 'sch', 'ich', 'cht', 'mit', 'gen'],
            'es': ['de_', 'la_', 'que', 'el_', 'en_', 'con', 'os_', 'ent', 'ado', 'as_'],
        }
        
        # Prepare text by lowercasing and padding
        text = text.lower()
        text = '_' + text + '_'
        
        # Count trigrams
        trigrams = {}
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            if trigram not in trigrams:
                trigrams[trigram] = 0
            trigrams[trigram] += 1
        
        # Get most common trigrams
        sorted_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)
        top_trigrams = [t[0] for t in sorted_trigrams[:20]]
        
        # Calculate scores for each language
        scores = {}
        for lang, profile in language_profiles.items():
            score = sum(3 - profile.index(t) if t in profile else 0 for t in top_trigrams[:10])
            scores[lang] = score
        
        # Get language with highest score
        if not scores:
            return 'unknown'
            
        max_lang = max(scores.items(), key=lambda x: x[1])
        if max_lang[1] > 0:
            return max_lang[0]
        return 'unknown'
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return 'unknown'


def detect_document_type(text: str) -> str:
    """
    Detect the type of clinical document based on content analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Document type (clinical_trial, practice_guideline, etc.)
    """
    try:
        text_lower = text.lower()
        
        # Clinical trial indicators
        clinical_trial_indicators = [
            'clinical trial', 'study protocol', 'inclusion criteria', 
            'exclusion criteria', 'randomization', 'placebo', 'double-blind',
            'endpoints', 'efficacy', 'participants', 'enrollment'
        ]
        
        # Practice guideline indicators
        practice_guideline_indicators = [
            'practice guideline', 'clinical practice', 'recommendation', 
            'guideline', 'standard of care', 'best practice', 'evidence-based',
            'grade of evidence', 'consensus', 'panel recommends'
        ]
        
        # Radiology report indicators
        radiology_report_indicators = [
            'radiograph', 'radiology', 'imaging', 'mri', 'ct scan', 'x-ray',
            'ultrasound', 'impression', 'findings', 'technique', 'contrast'
        ]
        
        # Count occurrences of indicators
        clinical_trial_count = sum(text_lower.count(indicator) for indicator in clinical_trial_indicators)
        practice_guideline_count = sum(text_lower.count(indicator) for indicator in practice_guideline_indicators)
        radiology_report_count = sum(text_lower.count(indicator) for indicator in radiology_report_indicators)
        
        # Determine document type based on highest count
        counts = {
            'clinical_trial': clinical_trial_count,
            'practice_guideline': practice_guideline_count,
            'radiology_report': radiology_report_count
        }
        
        document_type = max(counts, key=counts.get)
        if counts[document_type] == 0:
            return 'unknown'
            
        return document_type
    except Exception as e:
        logger.error(f"Error detecting document type: {str(e)}")
        return 'unknown'


def remove_headers_footers(text: str) -> str:
    """
    Remove page headers and footers from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with headers and footers removed
    """
    try:
        lines = text.split('\n')
        result_lines = []
        
        # Detect repeated lines that might be headers/footers
        line_counts = {}
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Count line occurrences
            if line_stripped not in line_counts:
                line_counts[line_stripped] = 0
            line_counts[line_stripped] += 1
        
        # Identify potential headers/footers (repeating lines)
        potential_headers_footers = {line for line, count in line_counts.items() 
                                    if count > 2 and len(line) < 80}
        
        # Remove headers/footers from text
        for line in lines:
            line_stripped = line.strip()
            if line_stripped not in potential_headers_footers:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    except Exception as e:
        logger.error(f"Error removing headers and footers: {str(e)}")
        return text


def handle_tables(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Detect and extract tabular data from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: Modified text and extracted tables
    """
    try:
        # Simple table detection heuristics
        lines = text.split('\n')
        tables = []
        current_table = None
        current_table_lines = []
        in_table = False
        
        for i, line in enumerate(lines):
            # Check if line has table-like structure
            cell_separators = ['|', '\t', '  ']
            is_table_row = any(separator in line for separator in cell_separators)
            
            # Detect table start
            if is_table_row and not in_table:
                in_table = True
                current_table_lines = [line]
                continue
                
            # Inside a table
            if in_table:
                if is_table_row:
                    current_table_lines.append(line)
                else:
                    # Table ended
                    if len(current_table_lines) > 1:
                        # Process table
                        separator = '|' if '|' in current_table_lines[0] else \
                                   '\t' if '\t' in current_table_lines[0] else '  '
                        
                        # Replace table with marker in text
                        table_id = f"TABLE_{len(tables)}"
                        processed_table = {
                            'id': table_id,
                            'rows': current_table_lines,
                            'separator': separator,
                            'start_line': i - len(current_table_lines),
                            'end_line': i
                        }
                        tables.append(processed_table)
                        
                        # Replace table with marker in text
                        lines[i - len(current_table_lines):i] = [f"[{table_id}]"]
                        
                    in_table = False
                    current_table_lines = []
        
        # Handle if text ends with a table
        if in_table and len(current_table_lines) > 1:
            table_id = f"TABLE_{len(tables)}"
            separator = '|' if '|' in current_table_lines[0] else \
                       '\t' if '\t' in current_table_lines[0] else '  '
            
            processed_table = {
                'id': table_id,
                'rows': current_table_lines,
                'separator': separator,
                'start_line': len(lines) - len(current_table_lines),
                'end_line': len(lines)
            }
            tables.append(processed_table)
            
            # Replace table with marker in text
            lines[len(lines) - len(current_table_lines):] = [f"[{table_id}]"]
        
        return '\n'.join(lines), tables
    except Exception as e:
        logger.error(f"Error handling tables: {str(e)}")
        return text, []


def extract_references(text: str) -> List[Dict[str, Any]]:
    """
    Extract references from text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Dict[str, Any]]: List of extracted references
    """
    try:
        references = []
        
        # Find references section
        ref_section_pattern = re.compile(r'(?i)(?:references|bibliography)[:.\-\s]*?((?:[^\n]*?\n)+?)(?:\n\n|\Z)', re.DOTALL)
        ref_section_match = ref_section_pattern.search(text)
        
        if ref_section_match:
            ref_section = ref_section_match.group(1)
            
            # Pattern for numbered references
            numbered_ref_pattern = re.compile(r'(?:^|\n)(?:\[(\d+)\]|\(?(\d+)[\.\)]) ([^\n]+)', re.MULTILINE)
            
            for match in numbered_ref_pattern.finditer(ref_section):
                # Get reference number
                ref_num = match.group(1) if match.group(1) else match.group(2)
                # Get reference text
                ref_text = match.group(3).strip()
                
                references.append({
                    'number': ref_num,
                    'text': ref_text
                })
            
            # If no numbered references found, try to split by lines
            if not references:
                lines = [line.strip() for line in ref_section.split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    # Skip lines that are likely not references
                    if len(line) < 10 or line.lower() in ['references', 'bibliography']:
                        continue
                    
                    references.append({
                        'number': str(i + 1),
                        'text': line
                    })
        
        return references
    except Exception as e:
        logger.error(f"Error extracting references: {str(e)}")
        return []


def detect_encoding(file_path: str) -> str:
    """
    Detect character encoding of a text file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected encoding
    """
    try:
        import chardet
        
        # Read a sample of the file
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB to detect encoding
        
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Fallback to utf-8 if detection failed
        if not encoding:
            encoding = 'utf-8'
            
        logger.debug(f"Detected encoding for {file_path}: {encoding}")
        return encoding
    except Exception as e:
        logger.error(f"Error detecting encoding: {str(e)}")
        return 'utf-8'  # Default to UTF-8


def segment_text_by_font(text: str) -> List[Dict[str, Any]]:
    """
    Segment text based on font markers (usually from PDF extraction).
    This is a heuristic approach that looks for patterns indicating font changes.
    
    Args:
        text (str): Input text with potential font markers
        
    Returns:
        List[Dict[str, Any]]: List of text segments with font information
    """
    try:
        segments = []
        
        # Pattern for font markers in PDF-extracted text
        font_pattern = re.compile(r'<font[\s="\w]+>')
        end_font_pattern = re.compile(r'</font>')
        
        # If no font markers, return single segment
        if not font_pattern.search(text) and not end_font_pattern.search(text):
            return [{
                'text': text,
                'font': 'default',
                'is_bold': False,
                'is_italic': False
            }]
        
        # Replace font markers with standardized tokens
        text = re.sub(r'<font[\s="\w]+bold[\s="\w]+>', '<bold>', text, flags=re.IGNORECASE)
        text = re.sub(r'<font[\s="\w]+italic[\s="\w]+>', '<italic>', text, flags=re.IGNORECASE)
        text = re.sub(r'<font[\s="\w]+>', '<font>', text)
        
        # Split text by font markers
        tokens = re.split(r'(</?(?:font|bold|italic)>)', text)
        
        current_font = 'default'
        is_bold = False
        is_italic = False
        current_text = ''
        
        for token in tokens:
            if token == '<bold>':
                # Save current segment if exists
                if current_text:
                    segments.append({
                        'text': current_text,
                        'font': current_font,
                        'is_bold': is_bold,
                        'is_italic': is_italic
                    })
                    current_text = ''
                
                is_bold = True
            elif token == '<italic>':
                # Save current segment if exists
                if current_text:
                    segments.append({
                        'text': current_text,
                        'font': current_font,
                        'is_bold': is_bold,
                        'is_italic': is_italic
                    })
                    current_text = ''
                
                is_italic = True
            elif token == '</bold>' or token == '</italic>' or token == '</font>':
                # Save current segment if exists
                if current_text:
                    segments.append({
                        'text': current_text,
                        'font': current_font,
                        'is_bold': is_bold,
                        'is_italic': is_italic
                    })
                    current_text = ''
                
                # Reset font attributes
                if token == '</bold>':
                    is_bold = False
                elif token == '</italic>':
                    is_italic = False
                else:  # </font>
                    current_font = 'default'
                    is_bold = False
                    is_italic = False
            elif token == '<font>':
                # Save current segment if exists
                if current_text:
                    segments.append({
                        'text': current_text,
                        'font': current_font,
                        'is_bold': is_bold,
                        'is_italic': is_italic
                    })
                    current_text = ''
                
                current_font = 'unknown'
            else:
                # Add text to current segment
                current_text += token
        
        # Add final segment if exists
        if current_text:
            segments.append({
                'text': current_text,
                'font': current_font,
                'is_bold': is_bold,
                'is_italic': is_italic
            })
        
        return segments
    except Exception as e:
        logger.error(f"Error segmenting text by font: {str(e)}")
        return [{
            'text': text,
            'font': 'default',
            'is_bold': False,
            'is_italic': False
        }]


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized whitespace
    """
    try:
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize newlines (convert multiple newlines to double newline)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove space at the beginning and end of lines
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        
        # Fix space before punctuation (but not after)
        text = re.sub(r' ([.,;:!?)])', r'\1', text)
        text = re.sub(r'(\() ', r'\1', text)
        
        # Ensure space after punctuation if followed by a letter
        text = re.sub(r'([.,;:!?)])([A-Za-z])', r'\1 \2', text)
        
        return text
    except Exception as e:
        logger.error(f"Error normalizing whitespace: {str(e)}")
        return text


def remove_duplicate_lines(text: str) -> str:
    """
    Remove consecutive duplicate lines from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with consecutive duplicates removed
    """
    try:
        lines = text.split('\n')
        result_lines = []
        prev_line = None
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped != prev_line:
                result_lines.append(line)
                prev_line = line_stripped
        
        return '\n'.join(result_lines)
    except Exception as e:
        logger.error(f"Error removing duplicate lines: {str(e)}")
        return text


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using statistical methods.
    
    Args:
        text (str): Input text
        max_phrases (int): Maximum number of phrases to extract
        
    Returns:
        List[str]: Extracted key phrases
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize text into sentences
        sentences = split_into_sentences(text)
        
        # Extract words (excluding stopwords)
        words = []
        for sentence in sentences:
            # Tokenize into words
            word_tokens = re.findall(r'\b\w+\b', sentence.lower())
            # Filter out stopwords and short words
            filtered_words = [word for word in word_tokens if word not in stop_words and len(word) > 2]
            words.extend(filtered_words)
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
        
        # Extract phrases (sequences of 1-3 words)
        phrases = []
        for sentence in sentences:
            # Extract n-grams
            for n in range(1, 4):
                tokens = re.findall(r'\b\w+\b', sentence.lower())
                for i in range(len(tokens) - n + 1):
                    phrase = ' '.join(tokens[i:i+n])
                    # Score phrase based on word frequencies
                    words_in_phrase = phrase.split()
                    if all(len(word) > 2 for word in words_in_phrase):
                        score = sum(word_freq.get(word, 0) for word in words_in_phrase) / len(words_in_phrase)
                        phrases.append((phrase, score))
        
        # Sort phrases by score and remove duplicates
        sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        unique_phrases = []
        for phrase, score in sorted_phrases:
            # Check if phrase is subset of already selected phrases
            if not any(phrase in selected for selected, _ in unique_phrases):
                unique_phrases.append((phrase, score))
                if len(unique_phrases) >= max_phrases:
                    break
        
        return [phrase for phrase, _ in unique_phrases]
    except Exception as e:
        logger.error(f"Error extracting key phrases: {str(e)}")
        return []


def convert_dosage_to_standard(text: str) -> str:
    """
    Convert medication dosages to standardized format.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with standardized dosages
    """
    try:
        # Standardize units
        standardizations = [
            # Weight units
            (r'\bkg\b', 'kg'),
            (r'\bkilograms?\b', 'kg'),
            (r'\bg\b', 'g'),
            (r'\bgrams?\b', 'g'),
            (r'\bmg\b', 'mg'),
            (r'\bmilligrams?\b', 'mg'),
            (r'\bµg\b', 'mcg'),
            (r'\bmcg\b', 'mcg'),
            (r'\bmicrograms?\b', 'mcg'),
            
            # Volume units
            (r'\bml\b', 'mL'),
            (r'\bmilliliters?\b', 'mL'),
            (r'\bl\b', 'L'),
            (r'\bliters?\b', 'L'),
            
            # Time units
            (r'\bhr\b', 'hour'),
            (r'\bhrs\b', 'hours'),
            (r'\bmin\b', 'minute'),
            (r'\bmins\b', 'minutes'),
            (r'\bsec\b', 'second'),
            (r'\bsecs\b', 'seconds'),
            
            # Frequency
            (r'\bq\.?d\.?\b', 'once daily'),
            (r'\bb\.?i\.?d\.?\b', 'twice daily'),
            (r'\bt\.?i\.?d\.?\b', 'three times daily'),
            (r'\bq\.?i\.?d\.?\b', 'four times daily'),
            (r'\bq\.?([0-9]+)h\.?\b', 'every \1 hours'),
            (r'\bq\.?([0-9]+)min\.?\b', 'every \1 minutes'),
            
            # Routes
            (r'\bp\.?o\.?\b', 'by mouth'),
            (r'\bper\.? os\b', 'by mouth'),
            (r'\bi\.?v\.?\b', 'intravenous'),
            (r'\bi\.?m\.?\b', 'intramuscular'),
            (r'\bs\.?c\.?\b', 'subcutaneous'),
            (r'\bsubcut\.?\b', 'subcutaneous'),
            (r'\bs\.?l\.?\b', 'sublingual'),
            (r'\bsubling\.?\b', 'sublingual'),
            
            # Special cases for common dosages
            (r'([0-9]+)\s*mg/kg', r'\1 mg/kg'),
            (r'([0-9]+)\s*g/day', r'\1 g/day'),
            (r'([0-9]+)\s*mg\s+daily', r'\1 mg once daily'),
            (r'([0-9]+)\s*mg\s+bid', r'\1 mg twice daily')
        ]
        
        # Apply standardizations
        result = text
        for pattern, replacement in standardizations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    except Exception as e:
        logger.error(f"Error converting dosages: {str(e)}")
        return text


def generate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Generate statistical information about the text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, Any]: Text statistics
    """
    try:
        if not text:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "average_words_per_sentence": 0,
                "average_word_length": 0,
                "character_count": 0
            }
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Count words
        words = re.findall(r'\b\w+\b', cleaned_text)
        word_count = len(words)
        
        # Count sentences
        sentences = split_into_sentences(cleaned_text)
        sentence_count = len(sentences)
        
        # Count characters (excluding whitespace)
        character_count = sum(1 for c in cleaned_text if not c.isspace())
        
        # Calculate averages
        average_word_length = sum(len(word) for word in words) / max(1, word_count)
        average_words_per_sentence = word_count / max(1, sentence_count)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_words_per_sentence": average_words_per_sentence,
            "average_word_length": average_word_length,
            "character_count": character_count
        }
    except Exception as e:
        logger.error(f"Error generating text statistics: {str(e)}")
        return {
            "word_count": 0,
            "sentence_count": 0,
            "average_words_per_sentence": 0,
            "average_word_length": 0,
            "character_count": 0,
            "error": str(e)
        }


def preprocess_for_model(text: str, model_max_length: int = 512) -> List[Dict[str, Any]]:
    """
    Comprehensive preprocessing for model input.
    Combines multiple preprocessing steps into a single pipeline.
    
    Args:
        text (str): Input text
        model_max_length (int): Maximum sequence length for model
        
    Returns:
        List[Dict[str, Any]]: Preprocessed text chunks ready for model input
    """
    try:
        # Step 1: Clean and normalize text
        clean_text_result = clean_text(text)
        
        # Step 2: Remove headers and footers
        text_no_headers = remove_headers_footers(clean_text_result)
        
        # Step 3: Normalize whitespace
        normalized_text = normalize_whitespace(text_no_headers)
        
        # Step 4: Remove duplicate lines
        deduplicated_text = remove_duplicate_lines(normalized_text)
        
        # Step 5: Extract and handle tables
        text_without_tables, tables = handle_tables(deduplicated_text)
        
        # Step 6: Split into overlapping chunks
        chunks = chunk_document(text_without_tables, max_length=model_max_length - 50, overlap=50)
        
        # Step 7: Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Convert dosages to standard format
            standardized_text = convert_dosage_to_standard(chunk['text'])
            
            # Add chunk metadata
            processed_chunk = {
                'text': standardized_text,
                'offset': chunk['offset'],
                'tables': [table for table in tables if 
                           table['start_line'] <= chunk['offset'] and 
                           table['end_line'] >= chunk['offset']],
                'statistics': generate_text_statistics(standardized_text)
            }
            
            processed_chunks.append(processed_chunk)
        
        logger.info(f"Preprocessed text into {len(processed_chunks)} chunks for model input")
        return processed_chunks
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        # Fallback to simple chunking
        chunks = chunk_document(text, max_length=model_max_length - 50, overlap=50)
        return [{'text': chunk['text'], 'offset': chunk['offset']} for chunk in chunks]


def find_section_context(text: str, entity_start: int, entity_end: int, window_size: int = 200) -> str:
    """
    Find the section context around an extracted entity.
    
    Args:
        text (str): Full document text
        entity_start (int): Start position of entity
        entity_end (int): End position of entity
        window_size (int): Size of context window
        
    Returns:
        str: Section context
    """
    try:
        # Get text before entity
        start_pos = max(0, entity_start - window_size)
        pre_text = text[start_pos:entity_start]
        
        # Get text after entity
        end_pos = min(len(text), entity_end + window_size)
        post_text = text[entity_end:end_pos]
        
        # Try to find section boundaries
        section_start = start_pos
        section_end = end_pos
        
        # Look for section header before entity
        section_patterns = [
            r'\n\s*([A-Z][A-Za-z\s]+:)\s*\n',
            r'\n\s*(\d+\.\d*\s+[A-Z][A-Za-z\s]+)\s*\n',
            r'\n\s*([A-Z][A-Z\s]+)\s*\n'
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, pre_text))
            if matches:
                last_match = matches[-1]
                section_start = start_pos + last_match.start()
                break
        
        # Look for next section header after entity
        for pattern in section_patterns:
            match = re.search(pattern, post_text)
            if match:
                section_end = entity_end + match.start()
                break
        
        # Extract section context
        section_context = text[section_start:section_end].strip()
        
        return section_context
    except Exception as e:
        logger.error(f"Error finding section context: {str(e)}")
        # Fallback to simple window
        start_pos = max(0, entity_start - window_size)
        end_pos = min(len(text), entity_end + window_size)
        return text[start_pos:end_pos].strip()


def normalize_dates(text: str) -> str:
    """
    Normalize date formats to ISO 8601 (YYYY-MM-DD).
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized dates
    """
    try:
        # Common date formats
        date_patterns = [
            # MM/DD/YYYY
            (r'(\b0?[1-9]|1[0-2])/(\b0?[1-9]|[12][0-9]|3[01])/(\d{4})\b', 
             lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
            
            # DD/MM/YYYY
            (r'(\b0?[1-9]|[12][0-9]|3[01])/(\b0?[1-9]|1[0-2])/(\d{4})\b',
             lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
            
            # Month DD, YYYY
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(\d{4})\b',
             lambda m: f"{m.group(3)}-{month_to_number(m.group(1)):02d}-{int(m.group(2)):02d}"),
            
            # DD Month YYYY
            (r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b',
             lambda m: f"{m.group(3)}-{month_to_number(m.group(2)):02d}-{int(m.group(1)):02d}"),
            
            # YYYY-MM-DD (already normalized, just for capturing)
            (r'\b(\d{4})-(\d{2})-(\d{2})\b',
             lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        ]
        
        result = text
        for pattern, replacement in date_patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    except Exception as e:
        logger.error(f"Error normalizing dates: {str(e)}")
        return text


def month_to_number(month_name: str) -> int:
    """
    Convert month name to month number.
    
    Args:
        month_name (str): Month name
        
    Returns:
        int: Month number (1-12)
    """
    months = {
        'january': 1,
        'february': 2,
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8,
        'september': 9,
        'october': 10,
        'november': 11,
        'december': 12
    }
    
    return months.get(month_name.lower(), 1)


# Main preprocessing function for direct use by other modules
def preprocess_document(doc_text: str, doc_type: str = None, normalize_dates_flag: bool = True) -> Dict[str, Any]:
    """
    Main entry point for document preprocessing.
    
    Args:
        doc_text (str): Document text
        doc_type (str, optional): Document type if known
        normalize_dates_flag (bool): Whether to normalize dates
        
    Returns:
        Dict[str, Any]: Preprocessed document with metadata
    """
    try:
        logger.info("Starting document preprocessing")
        
        # Clean and normalize text
        clean_text_result = clean_text(doc_text)
        
        # Detect document type if not provided
        if not doc_type:
            doc_type = detect_document_type(clean_text_result)
            
        # Detect language
        language = detect_language(clean_text_result)
        
        # Normalize dates if requested
        if normalize_dates_flag:
            clean_text_result = normalize_dates(clean_text_result)
            
        # Split into sentences
        sentences = split_into_sentences(clean_text_result)
        
        # Extract sections
        sections = detect_section_boundaries(clean_text_result)
        
        # Extract key phrases
        key_phrases = extract_key_phrases(clean_text_result)
        
        # Extract references
        references = extract_references(clean_text_result)
        
        # Generate statistics
        statistics = generate_text_statistics(clean_text_result)
        
        # Chunk document for model processing
        chunks = chunk_document(clean_text_result)
        
        # Prepare result
        result = {
            'text': clean_text_result,
            'document_type': doc_type,
            'language': language,
            'sentence_count': len(sentences),
            'section_count': len(sections),
            'sections': sections,
            'key_phrases': key_phrases,
            'references': references,
            'statistics': statistics,
            'chunks': chunks
        }
        
        logger.info("Document preprocessing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in document preprocessing: {str(e)}")
        # Return minimal result on error
        return {
            'text': doc_text,
            'error': str(e),
            'chunks': chunk_document(doc_text)
        }