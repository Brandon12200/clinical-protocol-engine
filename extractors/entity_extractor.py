import torch
import numpy as np
import logging
import re
from models.preprocessing import clean_text, chunk_document

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts clinical entities from text using a biomedical language model.
    This class handles named entity recognition (NER) for clinical protocol elements.
    """
    
    def __init__(self, model_manager):
        """
        Initialize the entity extractor with a model manager.
        
        Args:
            model_manager: ModelManager instance that provides the NER model
        """
        self.model_manager = model_manager
        # Initialize model if not already done
        if not model_manager.is_initialized:
            logger.info("Initializing model manager for entity extraction")
            model_manager.initialize()
    
    def extract_entities(self, text, threshold=0.7):
        """
        Extract entities from text with confidence above the threshold.
        
        Args:
            text (str): Input text
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            list: Extracted entities with positions and confidence scores
        """
        try:
            logger.debug(f"Extracting entities from text (length: {len(text)})")
            
            # Clean and preprocess text
            cleaned_text = clean_text(text)
            
            # Split long text into processable chunks
            if len(cleaned_text) > 512:
                logger.debug("Text exceeds model context window, chunking document")
                chunks = chunk_document(cleaned_text, max_length=400, overlap=50)
                all_entities = []
                
                # Process each chunk
                for chunk in chunks:
                    chunk_text = chunk['text']
                    chunk_offset = chunk['offset']
                    
                    # Extract entities from chunk
                    chunk_entities = self._extract_from_chunk(chunk_text, threshold)
                    
                    # Adjust entity positions based on chunk offset
                    for entity in chunk_entities:
                        entity['start'] += chunk_offset
                        entity['end'] += chunk_offset
                        all_entities.append(entity)
                
                # Deduplicate overlapping entities from chunk overlap
                entities = self._resolve_overlapping_entities(all_entities)
            else:
                # Process short text directly
                entities = self._extract_from_chunk(cleaned_text, threshold)
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return []
    
    def _extract_from_chunk(self, text, threshold):
        """
        Extract entities from a single text chunk.
        
        Args:
            text (str): Text chunk
            threshold (float): Confidence threshold
            
        Returns:
            list: Extracted entities
        """
        try:
            # Ensure model is initialized
            if not self.model_manager.is_initialized:
                self.model_manager.initialize()
            
            # Tokenize input
            inputs = self.model_manager.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                padding="max_length"
            )
            
            # Get token offsets for mapping back to original text
            offset_mapping = inputs.pop("offset_mapping").numpy()[0]
            
            # Move inputs to device
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
            
            # Get predictions
            predictions = torch.nn.functional.softmax(outputs.logits, dim=2)
            predictions = predictions.cpu().numpy()[0]
            
            # Process predictions to get entities
            entities = self._process_predictions(
                text, 
                predictions, 
                offset_mapping, 
                threshold
            )
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting from chunk: {str(e)}")
            return []
    
    def _process_predictions(self, text, predictions, offsets, threshold):
        """
        Convert model predictions to entity objects.
        
        Args:
            text (str): Original text
            predictions (numpy.ndarray): Model predictions
            offsets (numpy.ndarray): Token offsets
            threshold (float): Confidence threshold
            
        Returns:
            list: Extracted entities
        """
        entities = []
        prev_entity = None
        entity_start = None
        entity_text = ""
        entity_confidences = []
        
        entity_labels = self.model_manager.get_entity_labels()
        
        for idx, (offset, pred) in enumerate(zip(offsets, predictions)):
            # Skip special tokens and padding
            if offset[0] == offset[1]:
                continue
                
            # Get predicted label and confidence
            label_id = pred.argmax()
            confidence = float(pred[label_id])
            label = entity_labels[label_id]
            
            # Skip 'O' (Outside) label or low confidence predictions
            if label == 'O' or confidence < threshold:
                if prev_entity:
                    # Add completed entity
                    avg_confidence = sum(entity_confidences) / len(entity_confidences)
                    entities.append({
                        'text': entity_text,
                        'label': prev_entity,
                        'type': prev_entity,
                        'start': entity_start,
                        'end': offset[0],
                        'confidence': avg_confidence
                    })
                    prev_entity = None
                    entity_text = ""
                    entity_confidences = []
                continue
            
            # Handle B- (Beginning) labels
            if label.startswith('B-'):
                if prev_entity:
                    # Add completed entity
                    avg_confidence = sum(entity_confidences) / len(entity_confidences)
                    entities.append({
                        'text': entity_text,
                        'label': prev_entity,
                        'type': prev_entity,
                        'start': entity_start,
                        'end': offset[0],
                        'confidence': avg_confidence
                    })
                
                # Start new entity
                entity_start = offset[0]
                entity_text = text[offset[0]:offset[1]]
                prev_entity = label[2:]  # Remove 'B-' prefix
                entity_confidences = [confidence]
                
            # Handle I- (Inside) labels
            elif label.startswith('I-') and prev_entity == label[2:]:
                # Continue current entity
                entity_text += text[offset[0]:offset[1]]
                entity_confidences.append(confidence)
            else:
                # Handle unexpected transition
                if prev_entity:
                    avg_confidence = sum(entity_confidences) / len(entity_confidences)
                    entities.append({
                        'text': entity_text,
                        'label': prev_entity,
                        'type': prev_entity,
                        'start': entity_start,
                        'end': offset[0],
                        'confidence': avg_confidence
                    })
                
                # Start new entity
                entity_start = offset[0]
                entity_text = text[offset[0]:offset[1]]
                prev_entity = label[2:] if (label.startswith('B-') or label.startswith('I-')) else label
                entity_confidences = [confidence]
        
        # Add final entity if exists
        if prev_entity and entity_start is not None:
            avg_confidence = sum(entity_confidences) / len(entity_confidences)
            entities.append({
                'text': entity_text,
                'label': prev_entity,
                'type': prev_entity,
                'start': entity_start,
                'end': len(text),
                'confidence': avg_confidence
            })
        
        return entities
    
    def _resolve_overlapping_entities(self, entities):
        """
        Resolve overlapping entity predictions by selecting the highest confidence predictions.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Non-overlapping entity list
        """
        if not entities:
            return []
            
        # Sort entities by confidence (descending) then by length (descending)
        sorted_entities = sorted(
            entities, 
            key=lambda e: (e['confidence'], e['end'] - e['start']), 
            reverse=True
        )
        
        # Keep track of non-overlapping entities
        result = []
        covered_ranges = []
        
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            
            # Check if this entity overlaps with any existing covered range
            overlapping = False
            for range_start, range_end in covered_ranges:
                # Check for overlap
                if not (end <= range_start or start >= range_end):
                    overlapping = True
                    break
            
            if not overlapping:
                # Add to results and mark range as covered
                result.append(entity)
                covered_ranges.append((start, end))
        
        # Sort final entities by position
        return sorted(result, key=lambda e: e['start'])
    
    def merge_entity_spans(self, entities):
        """
        Merge adjacent entity spans of the same type.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Merged entity list
        """
        if not entities:
            return []
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e['start'])
        
        merged = []
        current = sorted_entities[0]
        
        for i in range(1, len(sorted_entities)):
            next_entity = sorted_entities[i]
            
            # Check if adjacent and same type
            if (next_entity['start'] <= current['end'] + 3 and  # Allow small gaps
                next_entity['label'] == current['label']):
                
                # Merge entities
                current['end'] = next_entity['end']
                current['text'] = current['text'] + next_entity['text']
                
                # Average confidences
                current['confidence'] = (current['confidence'] + next_entity['confidence']) / 2
            else:
                # Add current entity to results and move to next
                merged.append(current)
                current = next_entity
        
        # Add final entity
        merged.append(current)
        
        return merged
    
    def filter_by_confidence(self, entities, threshold):
        """
        Filter entities by confidence score.
        
        Args:
            entities (list): List of extracted entities
            threshold (float): Confidence threshold
            
        Returns:
            list: Filtered entity list
        """
        return [entity for entity in entities if entity['confidence'] >= threshold]
    
    def extract_with_fallback(self, text, threshold=0.7):
        """
        Extract entities with fallback mechanisms for error handling.
        
        Args:
            text (str): Input text
            threshold (float): Confidence threshold
            
        Returns:
            list: Extracted entities
        """
        try:
            # Try primary extraction
            entities = self.extract_entities(text, threshold)
            
            # Check if extraction returned reasonable results
            if not entities or len(entities) < self._expected_minimum_entities(text):
                logger.warning("Extraction results below expected threshold, trying with lower confidence")
                # Retry with lower threshold
                entities = self.extract_entities(text, threshold * 0.8)
                
                # If still no results, try rule-based fallback
                if not entities:
                    logger.warning("Model extraction failed, using rule-based fallback")
                    entities = self._rule_based_extraction(text)
            
            return entities
        except Exception as e:
            logger.error(f"Error in extraction: {str(e)}")
            # Fall back to rule-based extraction
            return self._rule_based_extraction(text)
    
    def _expected_minimum_entities(self, text):
        """
        Estimate expected minimum number of entities based on text length.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Expected minimum entity count
        """
        # Simple heuristic: approximately 1 entity per 200 characters
        return max(1, len(text) // 200)
    
    def _rule_based_extraction(self, text):
        """
        Simple rule-based entity extraction as fallback.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Extracted entities
        """
        entities = []
        entity_count = 0
        
        # Simple patterns for eligibility criteria
        eligibility_patterns = [
            r"(?:inclusion|exclusion)[\s\-:]+([^\n.]+)",
            r"(?:include|exclude)[\s\-:]+([^\n.]+)",
            r"(?:eligible|ineligible)[\s\-:]+([^\n.]+)",
            r"(?:criteria|criterion)[\s\-:]+([^\n.]+)"
        ]
        
        for pattern in eligibility_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_count += 1
                entities.append({
                    'text': match.group(1).strip(),
                    'label': 'ELIGIBILITY',
                    'type': 'ELIGIBILITY',
                    'start': match.start(1),
                    'end': match.end(1),
                    'confidence': 0.5  # Fixed confidence for rule-based extraction
                })
        
        # Simple patterns for procedures
        procedure_patterns = [
            r"(?:procedure|protocol|intervention)[\s\-:]+([^\n.]+)",
            r"(?:administer|perform|conduct)[\s\-:]+([^\n.]+)"
        ]
        
        for pattern in procedure_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_count += 1
                entities.append({
                    'text': match.group(1).strip(),
                    'label': 'PROCEDURE',
                    'type': 'PROCEDURE',
                    'start': match.start(1),
                    'end': match.end(1),
                    'confidence': 0.5
                })
        
        logger.info(f"Rule-based fallback extraction found {len(entities)} entities")
        return entities


class PerformanceMonitor:
    """
    Tracks extraction performance for monitoring and optimization.
    """
    
    def __init__(self):
        """Initialize performance tracking"""
        self.extraction_times = []
        self.confidence_scores = []
        self.entity_counts = []
        self.error_counts = 0
    
    def record_extraction(self, start_time, end_time, entities):
        """
        Record performance metrics for an extraction.
        
        Args:
            start_time (float): Extraction start time
            end_time (float): Extraction end time
            entities (list): Extracted entities
        """
        duration = end_time - start_time
        self.extraction_times.append(duration)
        self.entity_counts.append(len(entities))
        
        if entities:
            avg_confidence = sum(e['confidence'] for e in entities) / len(entities)
            self.confidence_scores.append(avg_confidence)
    
    def record_error(self):
        """Record extraction error"""
        self.error_counts += 1
    
    def get_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        total_extractions = max(1, len(self.extraction_times))
        
        return {
            'avg_extraction_time': sum(self.extraction_times) / total_extractions,
            'avg_confidence': sum(self.confidence_scores) / max(1, len(self.confidence_scores)),
            'avg_entity_count': sum(self.entity_counts) / total_extractions,
            'error_rate': self.error_counts / (total_extractions + self.error_counts),
            'total_extractions': total_extractions,
            'total_errors': self.error_counts
        }
    
    def reset_metrics(self):
        """Reset performance counters"""
        self.extraction_times = []
        self.confidence_scores = []
        self.entity_counts = []
        self.error_counts = 0