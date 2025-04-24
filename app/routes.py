from flask import Blueprint, render_template, request, jsonify, current_app, send_file, url_for, redirect
import os
import uuid
import json
import logging
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

# Import service and utility modules
from app.services.model_client import ModelClient
from extractors.document_parser import DocumentParser
# Other imports will be uncommented as they are implemented
# from utils.file_handler import FileHandler
# from utils.results_visualizer import ResultsVisualizer
# from standards.fhir.converters import FHIRConverter
# from standards.omop.converters import OMOPConverter

# Create blueprint
main_bp = Blueprint('main', __name__)

# Setup logging
logger = logging.getLogger(__name__)

# Initialize services
# For services not yet implemented, use MockService
class MockService:
    def __init__(self):
        pass

# Initialize document parser with appropriate configuration
document_parser = DocumentParser(config={
    'enable_ocr': True,
    'ocr_language': 'eng',
    'normalize_unicode': True,
    'normalize_whitespace': True,
    'detect_sections': True,
    'extract_tables': True
})

# Initialize model client
try:
    model_client = ModelClient()
except Exception as e:
    logger.warning(f"Could not initialize ModelClient: {str(e)}. Using mock service instead.")
    model_client = MockService()

# Initialize placeholder services until they are implemented
file_handler = MockService()
visualizer = MockService()
fhir_converter = MockService()
omop_converter = MockService()

@main_bp.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@main_bp.route('/document_selection')
def document_selection():
    """Document selection page"""
    # Get sample documents
    sample_docs_dir = os.path.join(current_app.root_path, 'sample_docs')
    sample_docs = []
    
    if os.path.exists(sample_docs_dir):
        for filename in os.listdir(sample_docs_dir):
            if filename.lower().endswith(('.pdf', '.docx', '.txt')):
                sample_docs.append({
                    'filename': filename,
                    'path': os.path.join('sample_docs', filename),
                    'description': _get_sample_doc_description(filename)
                })
    
    return render_template('document_selection.html', sample_docs=sample_docs)

def _get_sample_doc_description(filename):
    """Get description for sample document"""
    descriptions = {
        'clinical_trial.pdf': 'Example clinical trial protocol with inclusion/exclusion criteria',
        'discharge_summary.pdf': 'Hospital discharge summary with medication and procedure details',
        'practice_guideline.pdf': 'Medical practice guideline with recommended procedures',
        'radiology_report.txt': 'Radiology report with findings and recommendations',
        'medication_list.pdf': 'Patient medication list with dosages and instructions'
    }
    return descriptions.get(filename, 'Sample document')

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle document upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'pdf', 'docx', 'txt'})
    if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename, extension = os.path.splitext(original_filename)
        unique_filename = f"{filename}_{uuid.uuid4().hex}{extension}"
        
        # Define file path
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        file_path = os.path.join(upload_folder, unique_filename)
        
        # Save the file
        file.save(file_path)
        logger.info(f"Uploaded file: {original_filename} as {unique_filename}")
        
        # Redirect to processing page
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'redirect': url_for('main.process_document', filename=unique_filename)
        })
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({'error': f"Upload failed: {str(e)}"}), 500

@main_bp.route('/sample/<filename>')
def use_sample(filename):
    """Use a sample document"""
    sample_docs_dir = os.path.join(current_app.root_path, 'sample_docs')
    source_path = os.path.join(sample_docs_dir, filename)
    
    if not os.path.exists(source_path):
        return render_template('error/404.html'), 404
    
    try:
        # Copy sample to uploads with unique name
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        name, extension = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex}{extension}"
        dest_path = os.path.join(upload_folder, unique_filename)
        
        # Copy file
        with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        
        logger.info(f"Copied sample document: {filename} as {unique_filename}")
        
        # Redirect to processing page
        return redirect(url_for('main.process_document', filename=unique_filename))
    except Exception as e:
        logger.error(f"Error copying sample document: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500

@main_bp.route('/process/<filename>')
def process_document(filename):
    """Process document page"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER')
    file_path = os.path.join(upload_folder, filename)
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found for processing: {file_path}")
        return render_template('error/404.html'), 404
    
    # Get original filename if available
    original_filename = filename
    if '_' in filename:
        original_filename = filename.split('_')[0] + os.path.splitext(filename)[1]
    
    return render_template('processing.html', 
                          filename=filename, 
                          original_filename=original_filename)

@main_bp.route('/api/process', methods=['POST'])
def api_process():
    """Process document API endpoint"""
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'Missing filename in request'}), 400
    
    filename = data['filename']
    upload_folder = current_app.config.get('UPLOAD_FOLDER')
    file_path = os.path.join(upload_folder, filename)
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found for API processing: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    
    try:
        logger.info(f"Starting processing of {filename}")
        
        # Generate a unique job ID
        job_id = uuid.uuid4().hex
        
        # Create results directory if it doesn't exist
        processed_folder = current_app.config.get('RESULTS_FOLDER')
        job_folder = os.path.join(processed_folder, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Save job metadata
        metadata = {
            'job_id': job_id,
            'original_filename': filename,
            'file_path': file_path,
            'start_time': datetime.now().isoformat(),
            'status': 'parsing'
        }
        with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # 1. Parse the document using document_parser
        logger.info(f"Parsing document {filename}")
        try:
            parsed_doc = document_parser.parse(file_path)
            
            # Save the parsed document
            with open(os.path.join(job_folder, 'parsed_document.json'), 'w') as f:
                # Convert parsed document to serializable JSON
                serializable_doc = {
                    'text': parsed_doc['text'],
                    'metadata': parsed_doc['metadata'],
                    'sections': parsed_doc.get('sections', [])
                }
                if 'tables' in parsed_doc and parsed_doc['tables']:
                    serializable_doc['tables'] = parsed_doc['tables']
                
                json.dump(serializable_doc, f, indent=2)
            
            # Update job status
            metadata['status'] = 'extracting_entities'
            metadata['parsing_completed'] = datetime.now().isoformat()
            metadata['document_metadata'] = parsed_doc['metadata']
            with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Document parsed successfully: {len(parsed_doc['text'])} chars, " +
                      f"{parsed_doc['metadata'].get('word_count', 0)} words")
        except Exception as parse_error:
            logger.error(f"Error parsing document: {str(parse_error)}", exc_info=True)
            metadata['status'] = 'error'
            metadata['error'] = f"Document parsing failed: {str(parse_error)}"
            with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            return jsonify({
                'error': f"Document parsing failed: {str(parse_error)}",
                'job_id': job_id,
                'status': 'error'
            }), 500
        
        # 2. Extract entities
        # If model_client is not a mock, use it to extract entities
        if not isinstance(model_client, MockService):
            try:
                logger.info(f"Extracting entities from document {filename}")
                extraction_result = model_client.extract_entities(parsed_doc['text'])
                
                with open(os.path.join(job_folder, 'entities.json'), 'w') as f:
                    json.dump(extraction_result, f, indent=2)
                
                # Update job status
                metadata['status'] = 'extracting_relations'
                metadata['entities_completed'] = datetime.now().isoformat()
                metadata['entities_found'] = len(extraction_result.get('entities', []))
                with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
                
                logger.info(f"Entity extraction completed: {len(extraction_result.get('entities', []))} entities found")
            except Exception as extract_error:
                logger.error(f"Error extracting entities: {str(extract_error)}", exc_info=True)
                # Continue with empty entities
                extraction_result = {'entities': []}
        else:
            # Create mock extraction results
            logger.info("Using mock entity extraction (model client not available)")
            extraction_result = {
                'entities': _generate_mock_entities(parsed_doc)
            }
            
            with open(os.path.join(job_folder, 'entities.json'), 'w') as f:
                json.dump(extraction_result, f, indent=2)
            
            # Update metadata
            metadata['status'] = 'extracting_relations'
            metadata['entities_completed'] = datetime.now().isoformat()
            metadata['entities_found'] = len(extraction_result['entities'])
            with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
        
        # Generate results JSON with actual data from the parsed document and extractions
        results = {
            'status': 'completed',
            'entities_found': len(extraction_result.get('entities', [])),
            'sections_found': len(parsed_doc.get('sections', [])),
            'relations_found': 0,  # Will be updated if relation extraction is implemented
            'document_length': len(parsed_doc['text']),
            'processing_time_seconds': (datetime.now() - datetime.fromisoformat(metadata['start_time'])).total_seconds()
        }
        
        # Add some example key findings based on document sections
        if parsed_doc.get('sections'):
            key_findings = []
            for section in parsed_doc['sections'][:3]:  # Use first 3 sections for samples
                if len(section.get('content', '')) > 30:
                    # Extract first sentence as a key finding
                    content = section.get('content', '')
                    first_sentence = content.split('.')[0] + '.' if '.' in content else content[:100] + '...'
                    key_findings.append(f"{section.get('name', 'Section')}: {first_sentence}")
            
            results['key_findings'] = key_findings
        
        with open(os.path.join(job_folder, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update metadata with completion
        metadata['status'] = 'completed'
        metadata['end_time'] = datetime.now().isoformat()
        with open(os.path.join(job_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Processing completed for job {job_id}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'completed',
            'redirect': url_for('main.view_results', job_id=job_id)
        })
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500

def _generate_mock_entities(parsed_doc):
    """Generate mock entities for demonstration purposes."""
    # Create some sample entity types
    entity_types = ['MEDICATION', 'CONDITION', 'PROCEDURE', 'TEST', 'TIME', 'VALUE']
    
    # Extract some random words from the document to use as entities
    words = parsed_doc['text'].split()
    entities = []
    
    # Only create entities if we have enough text
    if len(words) > 20:
        # Create about 1 entity per 50 words, with a minimum of 5 and max of 40
        num_entities = max(5, min(40, len(words) // 50))
        
        import random
        for i in range(num_entities):
            # Pick a random word from the document (skip short words)
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            while len(word) < 4:  # Skip short words
                word_idx = random.randint(0, len(words) - 1)
                word = words[word_idx]
            
            # Determine entity start position roughly
            # This is just an approximation since we don't have exact character positions
            start_pos = parsed_doc['text'].find(word)
            if start_pos == -1:  # If word not found (unlikely but possible)
                continue
                
            entity_type = random.choice(entity_types)
            
            entity = {
                'id': f'e{i+1}',
                'text': word,
                'type': entity_type,
                'start': start_pos,
                'end': start_pos + len(word),
                'confidence': round(random.uniform(0.7, 0.98), 2)
            }
            
            # If we have sections, assign the entity to a section
            if parsed_doc.get('sections'):
                # Find which section this entity belongs to
                for section in parsed_doc['sections']:
                    if section['start'] <= start_pos <= section['end']:
                        entity['section'] = section['name']
                        break
            
            entities.append(entity)
    
    return entities

@main_bp.route('/api/job_status/<job_id>')
def job_status(job_id):
    """Check status of a processing job"""
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    job_folder = os.path.join(processed_folder, job_id)
    
    if not os.path.exists(job_folder):
        return jsonify({'error': 'Job not found'}), 404
    
    try:
        # Read metadata file
        with open(os.path.join(job_folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        status = metadata.get('status', 'unknown')
        progress = 0
        
        # Calculate progress percentage based on status
        if status == 'parsing':
            progress = 15
        elif status == 'extracting_entities':
            progress = 40
        elif status == 'extracting_relations':
            progress = 65
        elif status == 'standardizing':
            progress = 85
        elif status == 'completed':
            progress = 100
        elif status == 'error':
            progress = 0
        
        # Basic response with required fields
        response = {
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'start_time': metadata.get('start_time'),
            'end_time': metadata.get('end_time') if status in ['completed', 'error'] else None
        }
        
        # Add additional metadata for better UI feedback
        if status == 'completed':
            # Check if results.json exists
            results_path = os.path.join(job_folder, 'results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    response.update({
                        'entities_found': results.get('entities_found', 0),
                        'sections_found': results.get('sections_found', 0),
                        'relations_found': results.get('relations_found', 0),
                        'processing_time': results.get('processing_time_seconds', 0)
                    })
        
        # Include error message if available
        if status == 'error' and 'error' in metadata:
            response['message'] = metadata['error']
        
        # Include current step description
        status_messages = {
            'parsing': 'Document parsing in progress...',
            'extracting_entities': 'Extracting clinical entities...',
            'extracting_relations': 'Detecting relationships between entities...',
            'standardizing': 'Converting to standard formats...',
            'completed': 'Processing completed successfully.',
            'error': 'An error occurred during processing.'
        }
        response['message'] = status_messages.get(status, 'Processing...')
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}", exc_info=True)
        return jsonify({
            'error': f"Status check failed: {str(e)}",
            'status': 'error',
            'progress': 0
        }), 500

@main_bp.route('/results/<job_id>')
def view_results(job_id):
    """Display results page"""
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    job_folder = os.path.join(processed_folder, job_id)
    
    if not os.path.exists(job_folder):
        logger.warning(f"Results not found for job: {job_id}")
        return render_template('error/404.html'), 404
    
    try:
        # Read metadata
        with open(os.path.join(job_folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Read results
        with open(os.path.join(job_folder, 'results.json'), 'r') as f:
            results = json.load(f)
        
        # In a real implementation, you would:
        # 1. Read the original document text
        # 2. Prepare visualization of extracted entities
        # 3. Format results for display
        
        return render_template('results.html', 
                             job_id=job_id,
                             metadata=metadata,
                             results=results)
    except Exception as e:
        logger.error(f"Error viewing results: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500

@main_bp.route('/details/<job_id>')
def view_details(job_id):
    """Display detailed results page"""
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    job_folder = os.path.join(processed_folder, job_id)
    
    if not os.path.exists(job_folder):
        return render_template('error/404.html'), 404
    
    try:
        # Read metadata
        with open(os.path.join(job_folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Read results
        with open(os.path.join(job_folder, 'results.json'), 'r') as f:
            results = json.load(f)
        
        # In a real implementation, you would:
        # 1. Format FHIR and OMOP results for display
        # 2. Prepare detailed visualization
        
        return render_template('details.html', 
                             job_id=job_id,
                             metadata=metadata,
                             results=results)
    except Exception as e:
        logger.error(f"Error viewing details: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500

@main_bp.route('/download/<job_id>')
def download_options(job_id):
    """Download options page"""
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    job_folder = os.path.join(processed_folder, job_id)
    
    if not os.path.exists(job_folder):
        return render_template('error/404.html'), 404
    
    try:
        # Read metadata
        with open(os.path.join(job_folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        return render_template('download.html', 
                             job_id=job_id,
                             metadata=metadata)
    except Exception as e:
        logger.error(f"Error displaying download options: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500

@main_bp.route('/api/download', methods=['POST'])
def api_download():
    """Generate and download results file"""
    data = request.json
    if not data or 'job_id' not in data or 'format' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    job_id = data['job_id']
    format_type = data['format']
    standard = data.get('standard', 'both')
    
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    job_folder = os.path.join(processed_folder, job_id)
    
    if not os.path.exists(job_folder):
        return jsonify({'error': 'Job not found'}), 404
    
    try:
        # In a real implementation, you would:
        # 1. Generate the requested format
        # 2. Create the appropriate file
        
        # For now, we'll create a dummy file for download
        download_folder = os.path.join(job_folder, 'downloads')
        os.makedirs(download_folder, exist_ok=True)
        
        # Determine filename based on format and standard
        if format_type == 'json':
            filename = f"{standard}_data.json"
            with open(os.path.join(download_folder, filename), 'w') as f:
                json.dump({'format': format_type, 'standard': standard, 'dummy': 'data'}, f)
        elif format_type == 'xml':
            filename = f"{standard}_data.xml"
            with open(os.path.join(download_folder, filename), 'w') as f:
                f.write(f"<{standard}><dummy>data</dummy></{standard}>")
        elif format_type == 'sql':
            filename = f"{standard}_data.sql"
            with open(os.path.join(download_folder, filename), 'w') as f:
                f.write("-- Example SQL script\nCREATE TABLE dummy (id INT);")
        elif format_type == 'csv':
            filename = f"{standard}_data.csv"
            with open(os.path.join(download_folder, filename), 'w') as f:
                f.write("header1,header2\nvalue1,value2")
        else:
            return jsonify({'error': 'Invalid format requested'}), 400
        
        download_path = os.path.join(download_folder, filename)
        logger.info(f"Created download file at {download_path}")
        
        # Return path for download
        return jsonify({
            'success': True,
            'download_url': url_for('main.download_file', job_id=job_id, filename=filename)
        })
    except Exception as e:
        logger.error(f"Error generating download: {str(e)}", exc_info=True)
        return jsonify({'error': f"Download generation failed: {str(e)}"}), 500

@main_bp.route('/download_file/<job_id>/<filename>')
def download_file(job_id, filename):
    """Serve download file"""
    processed_folder = current_app.config.get('RESULTS_FOLDER')
    download_path = os.path.join(processed_folder, job_id, 'downloads', filename)
    
    if not os.path.exists(download_path):
        return render_template('error/404.html'), 404
    
    try:
        # Create a download timestamp
        download_dir = os.path.dirname(download_path)
        with open(os.path.join(download_dir, 'last_download.txt'), 'w') as f:
            f.write(datetime.now().isoformat())
        
        return send_file(download_path, as_attachment=True, download_name=filename)
    except Exception as e:
        logger.error(f"Error serving download file: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500

@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

def cleanup_old_files():
    """Clean up old files that are no longer needed"""
    try:
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        processed_folder = current_app.config.get('RESULTS_FOLDER')
        max_age = timedelta(hours=24)  # Keep files for 24 hours
        cutoff_time = datetime.now() - max_age
        
        # Clean uploads
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Removed old upload: {file_path}")
        
        # Clean processed results
        for job_id in os.listdir(processed_folder):
            job_path = os.path.join(processed_folder, job_id)
            if os.path.isdir(job_path):
                # Check metadata for timestamp
                metadata_path = os.path.join(job_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    end_time_str = metadata.get('end_time')
                    if end_time_str:
                        end_time = datetime.fromisoformat(end_time_str)
                        if end_time < cutoff_time:
                            # Remove the directory
                            import shutil
                            shutil.rmtree(job_path)
                            logger.info(f"Removed old job folder: {job_path}")
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}", exc_info=True)