"""
Section extractor for the Clinical Protocol Extraction Engine.
This module identifies and classifies document sections within clinical protocols
and other medical documents based on headings, formatting, and content patterns.
"""

import re
import logging
import numpy as np
from collections import defaultdict
from models.preprocessing import split_into_sentences

logger = logging.getLogger(__name__)

class SectionClassifier:
    """
    Classifies document sections based on their content and context.
    This is a helper class for the SectionExtractor.
    """
    
    def __init__(self, model_manager=None):
        """
        Initialize the section classifier.
        
        Args:
            model_manager: Optional ModelManager for ML-based classification
        """
        self.model_manager = model_manager
        
        # Common section types in clinical protocols with their keywords
        self.section_patterns = {
            'INTRODUCTION': [
                r'introduction',
                r'background',
                r'rationale',
                r'overview',
                r'summary'
            ],
            'OBJECTIVES': [
                r'objectives?',
                r'aims?',
                r'goals?',
                r'purpose',
                r'hypothesis'
            ],
            'ELIGIBILITY_SECTION': [
                r'eligibility',
                r'inclusion criteria',
                r'exclusion criteria',
                r'patient selection',
                r'study population'
            ],
            'PROCEDURES_SECTION': [
                r'procedures?',
                r'interventions?',
                r'methods?',
                r'study design',
                r'protocol',
                r'treatment'
            ],
            'STATISTICAL_ANALYSIS': [
                r'statistical analysis',
                r'statistics',
                r'data analysis',
                r'analysis plan',
                r'sample size'
            ],
            'ADVERSE_EVENTS_SECTION': [
                r'adverse events?',
                r'safety',
                r'side effects',
                r'toxicity',
                r'complications'
            ],
            'RESULTS': [
                r'results',
                r'findings',
                r'outcomes?',
                r'endpoints?'
            ],
            'DISCUSSION': [
                r'discussion',
                r'conclusions?',
                r'summary',
                r'interpretation'
            ],
            'RECOMMENDATIONS': [
                r'recommendations?',
                r'guidelines?',
                r'guidance',
                r'advisories?',
                r'best practices?'
            ]
        }
    
    def classify_section(self, title, content):
        """
        Classify a section based on its title and content.
        
        Args:
            title (str): Section title/heading
            content (str): Section content text
            
        Returns:
            tuple: (section_type, confidence)
        """
        # Model-based classification when available
        if self.model_manager and hasattr(self.model_manager, 'model'):
            try:
                return self._model_based_classification(title, content)
            except Exception as e:
                logger.warning(f"Model-based section classification failed: {str(e)}")
                # Fall back to rule-based approach
        
        return self._rule_based_classification(title, content)
    
    def _model_based_classification(self, title, content):
        """
        Classify a section using the model.
        
        Args:
            title (str): Section title/heading
            content (str): Section content text
            
        Returns:
            tuple: (section_type, confidence)
        """
        # This is a placeholder for model-based classification
        # Would be implemented when a section classifier model is available
        return None, 0.0
    
    def _rule_based_classification(self, title, content):
        """
        Classify a section using rule-based heuristics.
        
        Args:
            title (str): Section title/heading
            content (str): Section content text
            
        Returns:
            tuple: (section_type, confidence)
        """
        best_type = None
        best_confidence = 0.0
        
        # Normalize title and prepare content preview for matching
        norm_title = title.lower() if title else ""
        content_preview = content[:500].lower() if content else ""
        
        # First try to match section type based on title (higher confidence)
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                pattern_regex = r'\b' + re.escape(pattern) + r'\b'
                
                # Check title match (highest confidence)
                if norm_title and re.search(pattern_regex, norm_title):
                    confidence = 0.9
                    return section_type, confidence
            
            # Check first paragraph match (medium confidence)
            for pattern in patterns:
                pattern_regex = r'\b' + re.escape(pattern) + r'\b'
                if content_preview and re.search(pattern_regex, content_preview):
                    confidence = 0.7
                    if confidence > best_confidence:
                        best_type = section_type
                        best_confidence = confidence
        
        # If no good match found, analyze content for section-specific patterns
        if best_confidence < 0.5:
            section_scores = self._analyze_content_patterns(content)
            if section_scores:
                best_type, best_confidence = max(section_scores.items(), key=lambda x: x[1])
        
        # If still no good match, try to infer from document position
        if best_confidence < 0.5:
            # This would be expanded in a full implementation
            pass
        
        # If no match found, mark as UNKNOWN
        if best_type is None:
            best_type = "UNKNOWN"
            best_confidence = 0.3
        
        return best_type, best_confidence
    
    def _analyze_content_patterns(self, content):
        """
        Analyze content for section-specific patterns and keywords.
        
        Args:
            content (str): Section content text
            
        Returns:
            dict: Section type scores
        """
        if not content:
            return {}
        
        section_scores = defaultdict(float)
        content_lower = content.lower()
        
        # Additional content-specific patterns
        content_patterns = {
            'INTRODUCTION': [
                r'\bbackground\b.{1,50}\bdisease\b',
                r'\bcurrent.{1,30}\btreatment\b',
                r'\bproblem.{1,30}\baddressed\b'
            ],
            'OBJECTIVES': [
                r'\bprimary.{1,20}\bendpoint\b',
                r'\bsecondary.{1,20}\bendpoint\b',
                r'\baim\b.{1,20}\bstudy\b',
                r'\bgoal\b.{1,20}\bstudy\b'
            ],
            'ELIGIBILITY_SECTION': [
                r'\binclusion\b.{1,50}\bcriteria\b',
                r'\bexclusion\b.{1,50}\bcriteria\b',
                r'\bage\b.{1,20}\b(years|>\=|over)\b',
                r'\b(male|female)\b.{1,30}\bpatients\b'
            ],
            'PROCEDURES_SECTION': [
                r'\bdosage\b.{1,50}\badministration\b',
                r'\bfrequency\b.{1,30}\btreatment\b',
                r'\bprocedure\b.{1,50}\bperformed\b',
                r'\bmeasurement\b.{1,50}\btime\b'
            ],
            'STATISTICAL_ANALYSIS': [
                r'\bsample\b.{1,20}\bsize\b',
                r'\bp.{1,5}value\b',
                r'\bstatistical.{1,30}\banalysis\b',
                r'\bt.{1,5}test\b|\banova\b|\bregression\b'
            ],
            'ADVERSE_EVENTS_SECTION': [
                r'\badverse\b.{1,20}\bevent\b',
                r'\bside\b.{1,20}\beffect\b',
                r'\btoxicity\b',
                r'\bsafety\b.{1,30}\bmonitoring\b'
            ]
        }
        
        # Check for each pattern
        for section_type, patterns in content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    section_scores[section_type] += 0.2  # Accumulate scores
        
        # Normalize scores
        max_score = max(section_scores.values()) if section_scores else 0
        if max_score > 0:
            for section_type in section_scores:
                section_scores[section_type] = min(section_scores[section_type], 0.7)  # Cap at 0.7 confidence
        
        return section_scores


class SectionExtractor:
    """
    Extracts and classifies document sections from clinical protocols.
    Identifies section boundaries based on headings, formatting, and content.
    This class is used to divide documents into logical segments for more
    focused entity extraction and to provide structural context.
    """
    
    def __init__(self, model_manager=None):
        """
        Initialize the section extractor with an optional model manager.
        
        Args:
            model_manager: ModelManager instance for ML-based extraction
        """
        self.model_manager = model_manager
        self.section_classifier = SectionClassifier(model_manager)
        
        # Common heading patterns
        self.heading_patterns = [
            # Numbered headings
            r'^\s*(\d+\.)+\s+([A-Z][\w\s\-]+)$',
            # Capitalized headings
            r'^\s*([A-Z][A-Z\s\-]+)(?:\:|\.)?\s*$',
            # Underlined headings (dashes or equals below text)
            r'^([A-Za-z][\w\s\-]+)\s*\n\s*[-=]+\s*$',
            # Heading with colon
            r'^\s*([A-Z][\w\s\-]+)\:\s*$'
        ]
    
    def extract_sections(self, text):
        """
        Extract sections from document text.
        
        Args:
            text (str): Document text
            
        Returns:
            list: Extracted sections with metadata
        """
        try:
            logger.debug(f"Extracting sections from text (length: {len(text)})")
            
            # Identify section boundaries
            section_boundaries = self._identify_section_boundaries(text)
            
            # Extract sections using boundaries
            sections = []
            
            for i, (start, end, title) in enumerate(section_boundaries):
                section_id = f"sec{i+1}"
                section_text = text[start:end].strip()
                
                # Skip empty sections
                if not section_text:
                    continue
                
                # Classify section
                section_type, confidence = self.section_classifier.classify_section(title, section_text)
                
                # Create section object
                section = {
                    'id': section_id,
                    'title': title,
                    'type': section_type,
                    'start': start,
                    'end': end,
                    'text': section_text,
                    'confidence': confidence
                }
                
                sections.append(section)
            
            logger.info(f"Extracted {len(sections)} sections from document")
            return sections
            
        except Exception as e:
            logger.error(f"Error in section extraction: {str(e)}")
            # Fall back to basic paragraph-based extraction
            return self._fallback_extraction(text)
    
    def _identify_section_boundaries(self, text):
        """
        Identify section start and end positions.
        
        Args:
            text (str): Document text
            
        Returns:
            list: List of (start, end, title) tuples for each section
        """
        # Split text into lines
        lines = text.split('\n')
        
        # Find potential headings
        heading_positions = []
        
        for i, line in enumerate(lines):
            line_position = sum(len(l) + 1 for l in lines[:i])
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check for heading patterns
            for pattern in self.heading_patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract heading title from the match
                    title = match.group(1) if len(match.groups()) == 1 else match.group(2)
                    heading_positions.append((line_position, title.strip()))
                    break
            
            # Check for visual heading indicators (all caps, short line, etc.)
            if not heading_positions or heading_positions[-1][0] != line_position:
                line_stripped = line.strip()
                if (line_stripped.isupper() and 
                    len(line_stripped) > 3 and 
                    len(line_stripped) < 50 and
                    not line_stripped.endswith(':')):
                    heading_positions.append((line_position, line_stripped))
                    
                # Look for formatting hints like indentation changes
                elif (i > 0 and i < len(lines) - 1 and 
                      len(line_stripped) < 50 and
                      not lines[i-1].strip() and
                      lines[i+1].strip() and 
                      lines[i+1].startswith(' ')):
                    heading_positions.append((line_position, line_stripped))
        
        # Convert heading positions to section boundaries
        section_boundaries = []
        
        for i, (pos, title) in enumerate(heading_positions):
            start = pos
            end = len(text)
            
            # If this isn't the last heading, end at the next heading
            if i < len(heading_positions) - 1:
                end = heading_positions[i+1][0]
            
            section_boundaries.append((start, end, title))
        
        # If no headings found, try alternative methods
        if not section_boundaries:
            section_boundaries = self._extract_sections_by_spacing(text)
        
        # If still no sections, treat entire document as one section
        if not section_boundaries:
            section_boundaries = [(0, len(text), "Document")]
            
        return section_boundaries
    
    def _extract_sections_by_spacing(self, text):
        """
        Extract sections based on blank lines/spacing when headings aren't found.
        
        Args:
            text (str): Document text
            
        Returns:
            list: List of (start, end, title) tuples for each section
        """
        # Find paragraphs separated by blank lines
        paragraphs = re.split(r'\n\s*\n', text)
        
        section_boundaries = []
        current_pos = 0
        
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2  # +2 for the split newlines
                continue
            
            para_start = current_pos
            para_end = para_start + len(para)
            
            # Check if paragraph might be a heading (short, ends with colon, etc.)
            para_lines = para.split('\n')
            first_line = para_lines[0].strip()
            
            if (len(first_line) < 50 and 
                (first_line.endswith(':') or first_line.isupper() or 
                 (len(para_lines) > 1 and not para_lines[1].strip().startswith(' ')))):
                # This might be a section heading
                title = first_line
                content_start = para_start + len(first_line) + 1  # +1 for newline
                
                # Find where this section ends
                section_end = len(text)
                for next_para_start in [p[0] for p in section_boundaries if p[0] > content_start]:
                    section_end = next_para_start
                    break
                
                section_boundaries.append((content_start, section_end, title))
            
            current_pos = para_end + 2  # +2 for the split newlines
        
        return section_boundaries
    
    def _fallback_extraction(self, text):
        """
        Basic paragraph-based section extraction as fallback.
        
        Args:
            text (str): Document text
            
        Returns:
            list: List of extracted sections
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Group paragraphs into logical sections
        sections = []
        current_section = ""
        current_start = 0
        section_count = 0
        
        position = 0
        for i, para in enumerate(paragraphs):
            if not para.strip():
                position += len(para) + 2  # +2 for the split newlines
                continue
            
            para_start = position
            para_length = len(para)
            
            # Start a new section if:
            # 1. This is a potential heading (short, capitalized, etc.)
            # 2. We've accumulated a lot of text already
            is_heading = (len(para.strip()) < 50 and 
                         (para.strip().isupper() or 
                          para.strip().endswith(':') or
                          all(c.isupper() for c in para.strip() if c.isalpha())))
            
            if is_heading or (current_section and len(current_section) > 1000):
                # Save previous section if it exists
                if current_section:
                    section_count += 1
                    section_id = f"sec{section_count}"
                    
                    # Try to extract a title from the first line
                    lines = current_section.split('\n', 1)
                    title = lines[0].strip() if len(lines[0]) < 50 else "Untitled Section"
                    
                    # Classify section
                    section_type, confidence = self.section_classifier.classify_section(title, current_section)
                    
                    sections.append({
                        'id': section_id,
                        'title': title,
                        'type': section_type,
                        'start': current_start,
                        'end': para_start - 2,  # -2 for the split newlines
                        'text': current_section,
                        'confidence': confidence
                    })
                
                # Start new section
                current_section = para
                current_start = para_start
            else:
                # Continue current section
                if current_section:
                    current_section += "\n\n" + para
                else:
                    current_section = para
                    current_start = para_start
            
            position = para_start + para_length + 2  # +2 for the split newlines
        
        # Add the final section
        if current_section:
            section_count += 1
            section_id = f"sec{section_count}"
            
            # Try to extract a title
            lines = current_section.split('\n', 1)
            title = lines[0].strip() if len(lines[0]) < 50 else "Untitled Section"
            
            # Classify section
            section_type, confidence = self.section_classifier.classify_section(title, current_section)
            
            sections.append({
                'id': section_id,
                'title': title,
                'type': section_type,
                'start': current_start,
                'end': len(text),
                'text': current_section,
                'confidence': confidence
            })
        
        return sections
    
    def extract_with_fallback(self, text):
        """
        Extract sections with fallback mechanisms for error handling.
        
        Args:
            text (str): Document text
            
        Returns:
            list: Extracted sections
        """
        try:
            # Try primary extraction
            sections = self.extract_sections(text)
            
            # Check if extraction returned reasonable results
            if not sections:
                logger.warning("Section extraction failed, using fallback method")
                sections = self._fallback_extraction(text)
            
            return sections
        except Exception as e:
            logger.error(f"Error in section extraction: {str(e)}")
            return self._fallback_extraction(text)
    
    def get_section_by_type(self, sections, section_type):
        """
        Get all sections of a specific type.
        
        Args:
            sections (list): List of extracted sections
            section_type (str): Section type to filter by
            
        Returns:
            list: Sections matching the specified type
        """
        return [section for section in sections if section['type'] == section_type]
    
    def filter_sections_by_confidence(self, sections, threshold=0.5):
        """
        Filter sections by confidence score.
        
        Args:
            sections (list): List of extracted sections
            threshold (float): Confidence threshold
            
        Returns:
            list: Filtered section list
        """
        return [section for section in sections if section.get('confidence', 0) >= threshold]


# For easy testing
if __name__ == "__main__":
    import argparse
    import json
    import time
    from models.model_loader import ModelManager
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Extract sections from clinical documents')
    parser.add_argument('--text', help='Text file to process')
    parser.add_argument('--output', help='Output file for sections')
    args = parser.parse_args()
    
    try:
        # Initialize model manager (if available)
        try:
            model_manager = ModelManager()
            model_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize model manager: {str(e)}. Using rule-based extraction only.")
            model_manager = None
        
        # Initialize section extractor
        section_extractor = SectionExtractor(model_manager)
        
        # Load text
        if args.text:
            with open(args.text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # Sample text for testing
            text = """
            1. INTRODUCTION
            
            This clinical trial protocol establishes the procedures for testing the efficacy of 
            Drug XYZ in patients with Type 2 Diabetes Mellitus. The study aims to evaluate the 
            safety and effectiveness of the medication in reducing HbA1c levels.
            
            2. ELIGIBILITY CRITERIA
            
            2.1 Inclusion Criteria
            
            Patients meeting the following criteria are eligible for inclusion:
            - Age >= 18 years
            - Diagnosed with Type 2 Diabetes Mellitus for at least 6 months
            - HbA1c level between 7.5% and 10.0%
            - Body Mass Index (BMI) between 25 and 40 kg/m²
            
            2.2 Exclusion Criteria
            
            Patients with any of the following will be excluded:
            - Pregnancy or breastfeeding
            - History of ketoacidosis
            - Severe renal impairment (eGFR < 30 mL/min/1.73m²)
            - Current use of insulin therapy
            
            3. STUDY PROCEDURES
            
            3.1 Screening Visit
            
            All participants will undergo a comprehensive screening including:
            - Medical history review
            - Physical examination
            - Blood tests for HbA1c, fasting glucose, and renal function
            - Pregnancy test for women of childbearing potential
            
            3.2 Treatment Administration
            
            Participants will receive either Drug XYZ 10mg or placebo once daily for 12 weeks.
            Medication should be taken before breakfast with a full glass of water.
            """
        
        # Extract sections
        start_time = time.time()
        sections = section_extractor.extract_sections(text)
        end_time = time.time()
        
        # Display results
        print(f"\nExtracted {len(sections)} sections in {end_time - start_time:.2f} seconds")
        
        for section in sections:
            print(f"\nSection: {section['title']}")
            print(f"Type: {section['type']} (confidence: {section.get('confidence', 0):.2f})")
            print(f"Position: {section['start']} to {section['end']}")
            
            # Print preview of text
            text_preview = section['text'][:100]
            if len(section['text']) > 100:
                text_preview += "..."
            print(f"Content preview: {text_preview}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2)
            print(f"\nSections saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")