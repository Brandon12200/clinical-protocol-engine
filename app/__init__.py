from flask import Flask, render_template, request
import os
import logging
import random
from .config import get_config
from utils.logger import setup_logger


def create_app(config_override=None):
    """
    Application factory function
    
    Args:
        config_override: Configuration override dictionary (optional)
        
    Returns:
        app: Configured Flask application
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Override config if provided
    if config_override:
        app.config.update(config_override)
    
    # Setup logging
    log_dir = app.config.get('LOG_FOLDER')
    os.makedirs(log_dir, exist_ok=True)
    
    setup_logger('app', os.path.join(log_dir, 'app.log'))
    logger = logging.getLogger('app')
    logger.info(f"Starting application in {os.environ.get('FLASK_ENV', 'development')} mode")
    
    # Ensure upload and results directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 error: {request.path}")
        return render_template('error/404.html'), 404
        
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"500 error: {error}")
        return render_template('error/500.html'), 500
    
    @app.errorhandler(413)
    def too_large(error):
        logger.warning(f"413 error: File too large")
        return render_template('error/413.html', max_size=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)), 413
    
    # Register before_request handler to cleanup old files
    @app.before_request
    def before_request():
        # Import here to avoid circular import
        from flask import request
        
        # Log incoming request
        logger.debug(f"Request: {request.method} {request.path}")
        
        # Periodically clean up old files (limit frequency)
        if random.random() < 0.01:  # 1% chance to run on each request
            try:
                from utils.file_handler import FileHandler
                file_handler = FileHandler(
                    base_directory=os.path.dirname(app.config['UPLOAD_FOLDER'])
                )
                file_handler.cleanup_old_files(
                    max_age_hours=app.config['RESULT_EXPIRY'].total_seconds() / 3600
                )
            except Exception as e:
                logger.error(f"Error cleaning up old files: {str(e)}")
    
    # Register after_request handler for security headers in production
    @app.after_request
    def add_security_headers(response):
        # Add security headers for production
        if not app.config.get('DEBUG', False) and app.config.get('SECURITY_HEADERS'):
            for header, value in app.config.get('SECURITY_HEADERS').items():
                response.headers.set(header, value)
        return response
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        from flask import jsonify
        # Check model service connection if needed
        model_healthy = True
        try:
            if not app.config.get('TESTING', False):
                from app.services.model_client import ModelClient
                model_client = ModelClient()
                model_healthy = model_client.health_check()
        except Exception:
            model_healthy = False
            
        return jsonify({
            "status": "healthy",
            "model_service": "healthy" if model_healthy else "unhealthy"
        })
    
    # Setup context processors
    @app.context_processor
    def utility_processor():
        def format_file_size(size_bytes):
            """Format file size from bytes to human-readable format"""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        
        return dict(
            format_file_size=format_file_size,
            app_version=get_app_version()
        )
    
    logger.info("Application initialized successfully")
    return app

def get_app_version():
    """Get application version from version.txt file"""
    try:
        version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'version.txt')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                return f.read().strip()
        return "development"
    except Exception:
        return "unknown"