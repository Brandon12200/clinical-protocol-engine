#!/bin/bash

# Exit on error
set -e

# Get the current directory to set up the project structure
PROJECT_ROOT=$(pwd)
echo "Setting up project structure in: $PROJECT_ROOT"

# Create the app directory and its subdirectories
echo "Creating app directory structure..."
mkdir -p app/services app/static/{css,js,images} app/templates/{components,error} app/sample_docs

# Create the models directory
echo "Creating models directory structure..."
mkdir -p models/biobert_model

# Create the data directory
echo "Creating data directory structure..."
mkdir -p data/{uploads,processed}
touch data/uploads/.gitkeep data/processed/.gitkeep

# Create the extractors directory
echo "Creating extractors directory structure..."
mkdir -p extractors

# Create the standards directory and its subdirectories
echo "Creating standards directory structure..."
mkdir -p standards/fhir/templates standards/omop/schemas

# Create the utils directory
echo "Creating utils directory structure..."
mkdir -p utils

# Create the secrets directory
echo "Creating secrets directory structure..."
mkdir -p secrets

# Create the tests directory
echo "Creating tests directory structure..."
mkdir -p tests/test_data

# Create the GitHub workflows directory
echo "Creating GitHub workflows directory structure..."
mkdir -p .github/workflows

# Create the root-level files
echo "Creating root-level files..."
touch README.md requirements.txt model-requirements.txt setup.py .gitignore .dockerignore
touch Dockerfile model.Dockerfile docker-compose.yml docker-compose.dev.yml
touch start-dev.sh deploy-prod.sh model_service.py
chmod +x start-dev.sh deploy-prod.sh

# Create the Python package __init__.py files
echo "Creating Python package __init__.py files..."
touch app/__init__.py app/services/__init__.py models/__init__.py extractors/__init__.py
touch standards/__init__.py standards/fhir/__init__.py standards/omop/__init__.py
touch utils/__init__.py tests/__init__.py

# Create the app files
echo "Creating app files..."
touch app/main.py app/config.py app/routes.py app/services/model_client.py

# Create other essential files
echo "Creating model files..."
touch models/model_loader.py models/preprocessing.py

echo "Creating extractor files..."
touch extractors/document_parser.py extractors/entity_extractor.py
touch extractors/section_extractor.py extractors/relation_extractor.py

echo "Creating standards files..."
touch standards/fhir/converters.py standards/fhir/validators.py
touch standards/omop/converters.py standards/omop/validators.py

echo "Creating FHIR template files..."
touch standards/fhir/templates/plan_definition.json
touch standards/fhir/templates/activity_definition.json
touch standards/fhir/templates/library.json

echo "Creating OMOP schema files..."
touch standards/omop/schemas/condition_occurrence.json
touch standards/omop/schemas/drug_exposure.json
touch standards/omop/schemas/procedure_occurrence.json

echo "Creating utils files..."
touch utils/logger.py utils/file_handler.py utils/visualization.py utils/sample_generator.py

echo "Creating secrets files..."
touch secrets/.gitignore secrets/app_secret_key.txt.example

echo "Creating test files..."
touch tests/conftest.py tests/test_extractors.py tests/test_models.py
touch tests/test_standards.py tests/test_docker.py

echo "Creating GitHub workflow files..."
touch .github/workflows/build-test.yml .github/workflows/docker-build.yml

# Create HTML templates
echo "Creating HTML templates..."
touch app/templates/base.html app/templates/index.html app/templates/results.html
touch app/templates/details.html app/templates/download.html
touch app/templates/components/document_viewer.html
touch app/templates/components/extraction_results.html
touch app/templates/components/loading_spinner.html
touch app/templates/error/404.html app/templates/error/500.html

# Create CSS and JS files
echo "Creating CSS and JS files..."
touch app/static/css/main.css app/static/css/bootstrap.min.css
touch app/static/js/main.js app/static/js/highlight.js app/static/js/bootstrap.bundle.min.js
touch app/static/images/logo.svg app/static/images/favicon.ico

# Create sample document README
echo "Creating sample document README..."
touch app/sample_docs/README.md

# Create a basic .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << 'EOL'
# Python bytecode
__pycache__/
*.py[cod]

# Model files
models/biobert_model/pytorch_model.bin

# Sample documents
app/sample_docs/*.pdf
!app/sample_docs/README.md

# Test outputs
tests/outputs/

# Virtual environments
venv/
.env/

# IDE files
.vscode/
.idea/

# Temporary files
*.tmp
*.log

# Docker volumes
data/uploads/*
!data/uploads/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Secrets
secrets/*
!secrets/.gitignore
!secrets/*.example
EOL

# Create a basic .dockerignore file
echo "Creating .dockerignore file..."
cat > .dockerignore << 'EOL'
.git
.github
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
venv/
ENV/
.idea
.vscode
*.swp
logs/
data/uploads/*
!data/uploads/.gitkeep
tests/
EOL

# Create a basic README.md
echo "Creating README.md file with basic content..."
cat > README.md << 'EOL'
# Clinical Protocol Extraction and Standardization Engine

This project implements an AI-powered system to extract clinical protocols from documents and convert them to standardized healthcare formats including FHIR resources and OMOP CDM tables.

## Features

- Document Processing: Handles PDF, DOCX, and TXT formats
- AI-Powered Extraction: Uses a lightweight biomedical language model
- Standards Conversion: Transforms extracted data to FHIR and OMOP formats
- Web Interface: User-friendly processing and visualization

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clinical-protocol-engine.git
cd clinical-protocol-engine

# Install dependencies
pip install -r requirements.txt

# Start the development environment
./start-dev.sh
```

## License

[Add license information]

## Acknowledgements

[Add acknowledgements]
EOL

# Create gitignore for secrets directory
echo "Creating secrets/.gitignore file..."
cat > secrets/.gitignore << 'EOL'
# Ignore all files in this directory
*
# Except this file and examples
!.gitignore
!*.example
EOL

# Initialize git repository
echo "Initializing git repository..."
git init

echo "Project structure setup complete!"
echo "Next steps:"
echo "1. Add your GitHub repository as remote: git remote add origin https://github.com/yourusername/clinical-protocol-engine.git"
echo "2. Make your initial commit: git add . && git commit -m 'Initial project structure setup'"
echo "3. Push to GitHub: git push -u origin main"
