import os
from flask import Flask, render_template, jsonify

def create_app():
    app = Flask(__name__)
    
    # Load configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-for-development'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
        UPLOAD_FOLDER=os.path.join(app.root_path, 'uploads'),
        ALLOWED_EXTENSIONS={'pdf', 'docx', 'txt'},
        MODEL_SERVICE_URL=os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5001')
    )
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Basic routes for testing
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"})
    
    # Register blueprints here in the future
    # from app.routes import main
    # app.register_blueprint(main)
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error/404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('error/500.html'), 500
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)