from flask import Flask, request, jsonify
import logging
import os
import torch
import numpy as np

# Import model components
from models.model_loader import ModelManager
from extractors.entity_extractor import EntityExtractor
from extractors.section_extractor import SectionExtractor
from extractors.relation_extractor import RelationExtractor

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'model_service.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize model components
logger.info("Initializing model components...")
model_manager = ModelManager()
initialization_success = model_manager.initialize()

if initialization_success:
    logger.info("Model loaded successfully")
    entity_extractor = EntityExtractor(model_manager)
    section_extractor = SectionExtractor(model_manager)
    relation_extractor = RelationExtractor(model_manager)
else:
    logger.error("Failed to initialize model")
    # We'll still create the extractors but they'll likely fail when used
    entity_extractor = EntityExtractor(model_manager)
    section_extractor = SectionExtractor(model_manager)
    relation_extractor = RelationExtractor(model_manager)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if model_manager.is_initialized else "unhealthy"
    return jsonify({"status": status, "model_loaded": model_manager.is_initialized})

@app.route('/extract', methods=['POST'])
def extract():
    """Extract information from text"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = request.json['text']
    threshold = request.json.get('threshold', 0.7)
    
    try:
        # Ensure model is initialized
        if not model_manager.is_initialized:
            logger.warning("Model not initialized, attempting to initialize")
            if not model_manager.initialize():
                return jsonify({"error": "Model not initialized and initialization failed"}), 500
        
        # Extract entities
        logger.info(f"Extracting entities from text (length: {len(text)})")
        entities = entity_extractor.extract_entities(text, threshold)
        
        # Extract sections
        logger.info("Extracting document sections")
        sections = section_extractor.extract_sections(text)
        
        # Extract relations if entities were found
        relations = []
        if entities:
            logger.info(f"Extracting relations between {len(entities)} entities")
            relations = relation_extractor.extract_relations(text, entities)
        
        logger.info(f"Processed text with {len(entities)} entities, {len(sections)} sections, and {len(relations)} relations")
        
        return jsonify({
            "entities": entities,
            "sections": sections,
            "relations": relations
        })
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/extract_entities', methods=['POST'])
def extract_entities_only():
    """Endpoint for entity extraction only"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = request.json['text']
    threshold = request.json.get('threshold', 0.7)
    
    try:
        # Ensure model is initialized
        if not model_manager.is_initialized:
            if not model_manager.initialize():
                return jsonify({"error": "Model not initialized and initialization failed"}), 500
        
        # Extract entities
        entities = entity_extractor.extract_entities(text, threshold)
        logger.info(f"Extracted {len(entities)} entities")
        
        return jsonify({"entities": entities})
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/extract_sections', methods=['POST'])
def extract_sections():
    """Endpoint for section extraction only"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = request.json['text']
    
    try:
        # Ensure model is initialized
        if not model_manager.is_initialized:
            if not model_manager.initialize():
                return jsonify({"error": "Model not initialized and initialization failed"}), 500
        
        # Extract sections
        sections = section_extractor.extract_sections(text)
        logger.info(f"Extract {len(sections)} sections")
        
        return jsonify({"sections": sections})
    except Exception as e:
        logger.error(f"Error in section Extractions: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/extract_relations', methods=['POST'])
def extract_relations():
    """Endpoint for relation extraction only"""
    if not request.json or 'text' not in request.json or 'entities' not in request.json:
        return jsonify({"error": "Missing 'text' or 'entities' in request"}), 400
    
    text = request.json['text']
    entities = request.json['entities']
    
    try:
        # Ensure model is initialized
        if not model_manager.is_initialized:
            if not model_manager.initialize():
                return jsonify({"error": "Model not initialized and initialization failed"}), 500
        
        # Extract relations
        relations = relation_extractor.extract_relations(text, entities)
        logger.info(f"Extracted {len(relations)} relations")
        
        return jsonify({"relations": relations})
    except Exception as e:
        logger.error(f"Error in relation extraction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the model"""
    try:
        info = {
            "initialized": model_manager.is_initialized,
            "device": model_manager.device if hasattr(model_manager, 'device') else "unknown",
            "entity_labels": model_manager.entity_labels if hasattr(model_manager, 'entity_labels') else [],
            "model_path": model_manager.model_path if hasattr(model_manager, 'model_path') else "unknown"
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def cleanup_resources():
    """Clean up model resources on shutdown"""
    logger.info("Cleaning up model resources")
    if model_manager.is_initialized:
        model_manager.cleanup()

if __name__ == '__main__':
    logger.info("Starting model service")
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup_resources)
    
    # Start Flask application
    app.run(host='0.0.0.0', port=5001)