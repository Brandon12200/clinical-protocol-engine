import os
import logging
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
import uuid

def create_app():
    app = Flask(__name__)
    
    # Configure logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join('logs', 'app.log'))
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting application")
    
    # Load configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-for-development'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
        UPLOAD_FOLDER=os.path.join(app.root_path, 'uploads'),
        RESULTS_FOLDER=os.path.join(app.root_path, 'processed'),
        ALLOWED_EXTENSIONS={'pdf', 'docx', 'txt'},
        MODEL_SERVICE_URL=os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5001'),
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
        CONFIDENCE_THRESHOLD=0.7
    )
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Helper functions
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    # Basic routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/document_selection')
    def document_selection():
        # List sample documents if available
        sample_docs_dir = os.path.join(app.root_path, 'sample_docs')
        sample_docs = []
        
        if os.path.exists(sample_docs_dir):
            for filename in os.listdir(sample_docs_dir):
                if allowed_file(filename):
                    sample_docs.append({
                        'filename': filename,
                        'path': os.path.join('sample_docs', filename)
                    })
        
        return render_template('document_selection.html', sample_docs=sample_docs)
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            original_filename = secure_filename(file.filename)
            filename, extension = os.path.splitext(original_filename)
            unique_filename = f"{filename}_{uuid.uuid4().hex}{extension}"
            
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            logger.info(f"Uploaded file: {original_filename} as {unique_filename}")
            
            # Redirect to processing page
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'redirect': f"/process/{unique_filename}"
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/process/<filename>')
    def process_document(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return render_template('error/404.html'), 404
        
        # Pass the filename to the template for JavaScript to start processing
        return render_template('processing.html', filename=filename)
    
    @app.route('/api/process', methods=['POST'])
    def api_process():
        data = request.json
        if not data or 'filename' not in data:
            return jsonify({'error': 'Missing filename in request'}), 400
        
        filename = data['filename']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        try:
            # This is a placeholder - in the full implementation, you'd:
            # 1. Parse the document
            # 2. Send text to model service
            # 3. Process results
            # 4. Save results to processed folder
            
            # For now, just create a job ID
            job_id = uuid.uuid4().hex
            
            # Update this with actual processing logic
            logger.info(f"Started processing job {job_id} for file {filename}")
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'status': 'processing'
            })
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/results/<job_id>')
    def view_results(job_id):
        # This would be replaced with actual result retrieval logic
        return render_template('results.html', job_id=job_id)
    
    @app.route('/details/<job_id>')
    def view_details(job_id):
        # This would be replaced with actual detailed result retrieval
        return render_template('details.html', job_id=job_id)
    
    @app.route('/download/<job_id>')
    def download_options(job_id):
        # This would be replaced with download options logic
        return render_template('download.html', job_id=job_id)
    
    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"})
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error/404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return render_template('error/500.html'), 500
    
    # You'll register blueprints here in the future
    # from routes import main_bp
    # app.register_blueprint(main_bp)
    
    # Cleanup task for old uploads
    @app.before_request
    def cleanup_old_files():
        # Run cleanup infrequently - only 1% of requests
        import random
        if random.random() < 0.01:
            logger.info("Running cleanup task")
            # Placeholder - implement actual cleanup logic
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)