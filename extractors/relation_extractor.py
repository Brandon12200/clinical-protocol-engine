"""
Relation extractor for the Clinical Protocol Extraction Engine.
This module identifies and extracts relationships between clinical entities
in medical protocols and other clinical documents.
"""

import torch
import numpy as np
import logging
from itertools import combinations
import re

logger = logging.getLogger(__name__)

class RelationExtractor:
    """
    Extracts relationships between clinical entities in text.
    This class analyzes entity pairs to identify meaningful connections
    such as inclusion/exclusion, treatment, causality, etc.
    """
    
    def __init__(self, model_manager):
        """
        Initialize the relation extractor with a model manager.
        
        Args:
            model_manager: ModelManager instance that provides the relation model
        """
        self.model_manager = model_manager
        # Initialize model if not already done
        if not model_manager.is_initialized:
            logger.info("Initializing model manager for relation extraction")
            model_manager.initialize()
        
        # Define relation types and their patterns
        self.relation_patterns = {
            'INCLUDES': [
                r'including',
                r'contains',
                r'consists of',
                r'comprises',
                r'with',
                r'having',
            ],
            'EXCLUDES': [
                r'excluding',
                r'without',
                r'except for',
                r'other than',
                r'not including',
            ],
            'TREATS': [
                r'treats',
                r'for treatment of',
                r'indicated for',
                r'used for',
                r'administered for',
            ],
            'CAUSES': [
                r'causes',
                r'results in',
                r'leads to',
                r'produces',
                r'associated with',
            ],
            'PRECEDES': [
                r'before',
                r'prior to',
                r'precedes',
                r'followed by',
                r'preceding',
            ],
            'FOLLOWS': [
                r'after',
                r'following',
                r'subsequent to',
                r'succeeded by',
            ],
            'MEASURES': [
                r'measures',
                r'assesses',
                r'evaluates',
                r'quantifies',
                r'determines',
            ],
            'RECOMMENDS': [
                r'recommended',
                r'advised',
                r'suggested',
                r'proposed',
                r'indicated',
            ]
        }
    
    def extract_relations(self, text, entities, threshold=0.5):
        """
        Extract relationships between entities in text.
        
        Args:
            text (str): Original document text
            entities (list): Extracted entities with positions
            threshold (float): Confidence threshold for relations
            
        Returns:
            list: Extracted relations between entities
        """
        if not entities or len(entities) < 2:
            logger.info("Not enough entities to extract relations")
            return []
        
        try:
            logger.debug(f"Extracting relations from {len(entities)} entities")
            
            # Create potential entity pairs
            entity_pairs = self._create_entity_pairs(entities)
            logger.debug(f"Created {len(entity_pairs)} potential entity pairs")
            
            relations = []
            relation_id = 0
            
            # Process each entity pair
            for pair in entity_pairs:
                source, target = pair
                
                # Skip if entities are too far apart (more than 500 chars)
                if abs(source['end'] - target['start']) > 500 and abs(target['end'] - source['start']) > 500:
                    continue
                
                # Extract context between entities
                context = self._extract_context(text, source, target)
                
                # Classify relation
                relation_type, confidence = self._classify_relation(context, source, target)
                
                # Add relation if confidence is above threshold
                if relation_type and confidence >= threshold:
                    relation_id += 1
                    relations.append({
                        'id': f'rel{relation_id}',
                        'source_id': source.get('id', f"entity-{entities.index(source)}"),
                        'target_id': target.get('id', f"entity-{entities.index(target)}"),
                        'type': relation_type,
                        'confidence': confidence
                    })
            
            logger.info(f"Extracted {len(relations)} relations between entities")
            return relations
            
        except Exception as e:
            logger.error(f"Error in relation extraction: {str(e)}")
            return []
    
    def _create_entity_pairs(self, entities):
        """
        Generate potential entity pairs for relation extraction.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Pairs of entities to check for relations
        """
        # Create all possible pairs
        all_pairs = list(combinations(entities, 2))
        
        # Filter for most relevant pairs (can be extended with more sophisticated logic)
        # For now, just create bidirectional pairs for all entities
        pairs = []
        for entity1, entity2 in all_pairs:
            pairs.append((entity1, entity2))
            pairs.append((entity2, entity1))  # Check both directions
        
        return pairs
    
    def _extract_context(self, text, entity1, entity2):
        """
        Extract text context between and around two entities.
        
        Args:
            text (str): Full document text
            entity1 (dict): First entity
            entity2 (dict): Second entity
            
        Returns:
            str: Context text
        """
        # Determine the start and end positions for context
        if entity1['start'] < entity2['start']:
            start = entity1['start']
            middle_start = entity1['end']
            middle_end = entity2['start']
            end = entity2['end']
            first = entity1
            second = entity2
        else:
            start = entity2['start']
            middle_start = entity2['end']
            middle_end = entity1['start']
            end = entity1['end']
            first = entity2
            second = entity1
        
        # Extract 50 characters before, between, and after the entities
        pre_context_start = max(0, start - 50)
        post_context_end = min(len(text), end + 50)
        
        pre_context = text[pre_context_start:start]
        entity1_text = text[start:middle_start]
        between_text = text[middle_start:middle_end]
        entity2_text = text[middle_end:end]
        post_context = text[end:post_context_end]
        
        # Create a formatted context string
        context = {
            'pre_context': pre_context,
            'first_entity': first['text'],
            'first_label': first['label'],
            'between_text': between_text,
            'second_entity': second['text'],
            'second_label': second['label'],
            'post_context': post_context,
            'full_text': text[pre_context_start:post_context_end]
        }
        
        return context
    
    def _classify_relation(self, context, entity1, entity2):
        """
        Determine relationship type between entities.
        
        Args:
            context (dict): Text context around and between entities
            entity1 (dict): First entity
            entity2 (dict): Second entity
            
        Returns:
            tuple: (relation_type, confidence)
        """
        # In a full implementation, this would use the model for classification
        # For now, use a rule-based approach with patterns
        
        # Model-based classification when integrated with a transformer model
        if hasattr(self.model_manager, 'relation_model') and self.model_manager.relation_model:
            try:
                return self._model_based_classification(context, entity1, entity2)
            except Exception as e:
                logger.warning(f"Model-based relation classification failed: {str(e)}")
                # Fall back to rule-based approach
        
        return self._rule_based_classification(context, entity1, entity2)
    
    def _model_based_classification(self, context, entity1, entity2):
        """
        Classify relationship using the model.
        
        Args:
            context (dict): Text context
            entity1 (dict): First entity
            entity2 (dict): Second entity
            
        Returns:
            tuple: (relation_type, confidence)
        """
        # This is a placeholder for model-based classification
        # To be implemented when a relation model is available
        
        # Example implementation:
        # inputs = self.model_manager.tokenizer(
        #     context['full_text'],
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=512
        # )
        # inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        # with torch.no_grad():
        #     outputs = self.model_manager.relation_model(**inputs)
        # predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        # predictions = predictions.cpu().numpy()[0]
        # relation_id = predictions.argmax()
        # confidence = float(predictions[relation_id])
        # relation_type = self.model_manager.get_relation_labels()[relation_id]
        
        # Return placeholder result
        return None, 0.0
    
    def _rule_based_classification(self, context, entity1, entity2):
        """
        Classify relationship using rule-based heuristics.
        
        Args:
            context (dict): Text context
            entity1 (dict): First entity
            entity2 (dict): Second entity
            
        Returns:
            tuple: (relation_type, confidence)
        """
        between_text = context['between_text'].lower()
        full_text = context['full_text'].lower()
        
        best_relation = None
        best_confidence = 0.0
        
        # Check patterns for each relation type
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                pattern_regex = r'\b' + re.escape(pattern) + r'\b'
                
                # Check if pattern appears in between text (higher confidence)
                if re.search(pattern_regex, between_text):
                    confidence = 0.8
                    return relation_type, confidence
                
                # Check if pattern appears in full context (lower confidence)
                elif re.search(pattern_regex, full_text):
                    confidence = 0.6
                    if confidence > best_confidence:
                        best_relation = relation_type
                        best_confidence = confidence
        
        # Use entity type pairs to determine likely relations
        entity_relation = self._infer_from_entity_types(entity1, entity2)
        if entity_relation and (best_relation is None or best_confidence < 0.7):
            relation_type, confidence = entity_relation
            if confidence > best_confidence:
                best_relation = relation_type
                best_confidence = confidence
        
        return best_relation, best_confidence
    
    def _infer_from_entity_types(self, entity1, entity2):
        """
        Infer potential relationship based on entity types.
        
        Args:
            entity1 (dict): First entity
            entity2 (dict): Second entity
            
        Returns:
            tuple: (relation_type, confidence) or (None, 0)
        """
        # Define common entity type relationships
        type_relations = {
            ('ELIGIBILITY', 'ELIGIBILITY'): ('INCLUDES', 0.5),
            ('ELIGIBILITY', 'CONDITION'): ('INCLUDES', 0.6),
            ('PROCEDURE', 'TIMEPOINT'): ('PRECEDES', 0.5),
            ('MEDICATION', 'CONDITION'): ('TREATS', 0.6),
            ('MEDICATION', 'ADVERSE_EVENT'): ('CAUSES', 0.6),
            ('ENDPOINT', 'PROCEDURE'): ('MEASURES', 0.6),
            ('RECOMMENDATION', 'PROCEDURE'): ('RECOMMENDS', 0.7),
            ('RECOMMENDATION', 'MEDICATION'): ('RECOMMENDS', 0.7),
        }
        
        # Check if entity type pair has a predefined relation
        type_pair = (entity1.get('label', ''), entity2.get('label', ''))
        if type_pair in type_relations:
            return type_relations[type_pair]
        
        # Check reverse order
        type_pair_reverse = (entity2.get('label', ''), entity1.get('label', ''))
        if type_pair_reverse in type_relations:
            return type_relations[type_pair_reverse]
        
        return None, 0.0
    
    def filter_relations(self, relations, threshold):
        """
        Filter relations by confidence score.
        
        Args:
            relations (list): List of extracted relations
            threshold (float): Confidence threshold
            
        Returns:
            list: Filtered relation list
        """
        return [relation for relation in relations if relation['confidence'] >= threshold]
    
    def extract_with_fallback(self, text, entities, threshold=0.5):
        """
        Extract relations with fallback mechanisms for error handling.
        
        Args:
            text (str): Input text
            entities (list): Extracted entities
            threshold (float): Confidence threshold
            
        Returns:
            list: Extracted relations
        """
        try:
            # Try primary extraction
            relations = self.extract_relations(text, entities, threshold)
            
            # If no relations are found but there are enough entities,
            # try with a lower threshold
            if not relations and len(entities) >= 2:
                logger.warning("No relations found, trying with lower threshold")
                relations = self.extract_relations(text, entities, threshold * 0.8)
            
            return relations
        except Exception as e:
            logger.error(f"Error in relation extraction: {str(e)}")
            return []


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
    
    parser = argparse.ArgumentParser(description='Extract relations from entities')
    parser.add_argument('--text', help='Text file to process')
    parser.add_argument('--entities', help='JSON file with entities')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', help='Output file for relations')
    args = parser.parse_args()
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        model_manager.initialize()
        
        # Initialize relation extractor
        relation_extractor = RelationExtractor(model_manager)
        
        # Load text
        if args.text:
            with open(args.text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = input("Enter text to analyze: ")
        
        # Load entities
        if args.entities:
            with open(args.entities, 'r', encoding='utf-8') as f:
                entities = json.load(f)
        else:
            # Sample entities for testing
            entities = [
                {
                    'id': 'ent1',
                    'text': 'Age >= 18 years',
                    'label': 'ELIGIBILITY',
                    'start': 100,
                    'end': 116,
                    'confidence': 0.95
                },
                {
                    'id': 'ent2',
                    'text': 'diabetes mellitus',
                    'label': 'CONDITION',
                    'start': 150,
                    'end': 167,
                    'confidence': 0.92
                }
            ]
        
        # Extract relations
        start_time = time.time()
        relations = relation_extractor.extract_relations(text, entities, args.threshold)
        end_time = time.time()
        
        # Display results
        print(f"\nExtracted {len(relations)} relations in {end_time - start_time:.2f} seconds")
        for relation in relations:
            source_id = relation['source_id']
            target_id = relation['target_id']
            source_entity = next((e for e in entities if e.get('id') == source_id), None)
            target_entity = next((e for e in entities if e.get('id') == target_id), None)
            
            if source_entity and target_entity:
                print(f"\nRelation: {relation['type']} (confidence: {relation['confidence']:.2f})")
                print(f"  {source_entity['text']} -> {target_entity['text']}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(relations, f, indent=2)
            print(f"\nRelations saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")