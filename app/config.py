import os
from datetime import timedelta

class Config:
    """Base configuration class for the Clinical Protocol Extraction application"""
    # Secret key for session signing
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-development')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/uploads'))
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    
    # Model service settings
    MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5001')
    MODEL_REQUEST_TIMEOUT = 300  # 5 minutes
    
    # Processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    CONFIDENCE_THRESHOLD = 0.7
    
    # Results settings
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed'))
    RESULT_EXPIRY = timedelta(hours=24)
    
    # Logging settings
    LOG_FOLDER = os.environ.get('LOG_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs'))
    
    # Docker container detection
    RUNNING_IN_CONTAINER = os.environ.get('CONTAINER', 'False').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Additional development settings
    MODEL_MOCK_ENABLED = os.environ.get('MODEL_MOCK_ENABLED', 'False').lower() == 'true'
    
    # Shorter request timeout for development
    MODEL_REQUEST_TIMEOUT = 120  # 2 minutes
    
    # Enable more verbose logging
    import logging
    LOG_LEVEL = logging.DEBUG

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    
    # Use temporary directories for testing
    import tempfile
    UPLOAD_FOLDER = tempfile.mkdtemp()
    RESULTS_FOLDER = tempfile.mkdtemp()
    
    # Use mock model for testing
    MODEL_MOCK_ENABLED = True
    
    # Faster timeouts for testing
    MODEL_REQUEST_TIMEOUT = 30  # 30 seconds
    
    # Test-specific settings
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    
    # Disable CSRF protection in testing
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Security enhancements for production
    SECRET_KEY = os.environ.get('SECRET_KEY_FILE', None)
    if SECRET_KEY and os.path.isfile(SECRET_KEY):
        with open(SECRET_KEY, 'r') as f:
            SECRET_KEY = f.read().strip()
    else:
        SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Ensure SECRET_KEY is set
    if not SECRET_KEY:
        import secrets
        SECRET_KEY = secrets.token_hex(32)
        print("WARNING: No SECRET_KEY provided. Generated random SECRET_KEY for this session.")
        
    # Production-specific settings
    MODEL_REQUEST_TIMEOUT = 600  # 10 minutes for larger documents
    
    # Increase maximum content length for production
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max upload size
    
    # Production logging level
    import logging
    LOG_LEVEL = logging.INFO
    
    # Security headers
    SECURITY_HEADERS = {
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; img-src 'self' data:;",
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block'
    }

class DockerConfig(ProductionConfig):
    """Configuration specific to Docker deployment"""
    # Docker-specific paths
    UPLOAD_FOLDER = '/app/data/uploads'
    RESULTS_FOLDER = '/app/data/processed'
    LOG_FOLDER = '/app/logs'
    
    # Assume running in container
    RUNNING_IN_CONTAINER = True
    
    # Docker health check settings
    HEALTH_CHECK_ENDPOINT = '/health'
    
    # Allow setting model service URL via environment
    MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://model:5001')

# Dictionary mapping environment names to config classes
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}

def get_config():
    """
    Get the active configuration based on environment
    
    Returns:
        config: Configuration class instance
    """
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Check for Docker environment
    if os.environ.get('CONTAINER', 'False').lower() == 'true':
        env = 'docker'
        
    return config_by_name.get(env, config_by_name['default'])