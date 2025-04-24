# Clinical Protocol Extraction and Standardization Engine

An AI-powered system that transforms unstructured clinical protocol documents into standardized healthcare formats (FHIR resources and OMOP CDM tables), designed for healthcare institutions and medical researchers.

## Overview

The Clinical Protocol Extraction Engine uses NLP technology to extract structured data from clinical protocol documents, allowing for seamless integration with electronic health record systems, research databases, and clinical decision support tools.

## Key Features

- **Local Processing Architecture**: All document processing occurs locally, with no external data transmission
- **Medical Document Support**: Handles PDF, DOCX, and TXT formats commonly used in clinical settings
- **Biomedical NLP Model**: Utilizes a fine-tuned model optimized for clinical entity recognition
- **Dual Standard Support**: Converts extracted protocol elements to both FHIR R4 resources and OMOP CDM tables
- **Terminology Mapping**: Maps clinical terms to standard vocabularies (SNOMED CT, LOINC, RxNorm) using embedded databases
- **Docker Containerization**: Easily deployable in any environment with Docker support
- **Fallback Extraction**: Implements rule-based fallback extraction when model confidence is low
- **Performance Monitoring**: Tracks extraction metrics for continuous improvement
- **Context-Aware Mapping**: Enhances terminology mapping using document context for improved accuracy

## Technical Architecture

The system consists of four main processing layers:

1. **Document Processing Layer**
   - Converts PDFs, DOCX, and TXT files to normalized text
   - Handles document chunking for large files
   - Performs initial text preprocessing

2. **Protocol Extraction Layer**
   - Applies biomedical NLP model to identify clinical entities
   - Recognizes inclusion/exclusion criteria, procedures, endpoints
   - Extracts relationships between protocol elements
   - Identifies document sections and their hierarchies

3. **Standards Conversion Layer**
   - Maps extracted data to FHIR R4 resources (PlanDefinition, ActivityDefinition, Library, Questionnaire)
   - Creates OMOP CDM table representations
   - Provides terminology mapping to standard codes
   - Validates output against standards

4. **Web Presentation Layer**
   - Offers user interface for document upload and visualization
   - Displays extracted protocol elements with highlighting
   - Enables export to FHIR/OMOP formats
   - Provides detailed extraction results and confidence scores

This pipeline architecture processes documents sequentially through each layer, transforming unstructured text into standardized healthcare data formats.

## Getting Started

### Prerequisites
- **Hardware**: 
  - 8GB+ RAM for model loading and inference
  - 1GB disk space for application and model storage
  - CPU with 4+ cores recommended for reasonable processing time
- **Software**:
  - Docker and Docker Compose
  - Modern web browser

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Brandon12200/clinical-protocol-engine.git
cd clinical-protocol-engine

# Start the development environment
./start-dev.sh

# Access the web interface
# Open http://localhost:8080 in your browser
```

The development environment will automatically start and expose the application interface on port 8080.

### Development Workflow
```bash
# Stop the development environment
docker compose -f docker-compose.dev.yml down

# Rebuild containers after dependency changes
docker compose -f docker-compose.dev.yml build

# Access container shell
docker compose -f docker-compose.dev.yml exec web bash

# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_file.py::TestClass::test_function -v
```

## Project Structure
The project follows a modular organization:
- **app/**: Web application with Flask interface, routes, and templates
- **models/**: AI model management, preprocessing, and loading logic
- **extractors/**: Document processing, entity extraction, section and relation identification
- **standards/**: FHIR and OMOP conversion and validation components
- **utils/**: Common utilities for logging, file handling, and visualization
- **tests/**: Comprehensive test suite with test data

## Microservice Architecture
The application consists of two main services:
1. **Web Service**: Handles user interface, document upload, and result visualization
2. **Model Service**: Provides API endpoints for entity, section, and relation extraction

This separation allows for independent scaling and resource allocation based on workload.

## Data Storage
The project currently uses a simple file-based storage system:
- **data/uploads/**: Temporary storage for uploaded documents
- **data/processed/**: Storage for processed extraction results
- **Docker volumes**: Used for persistence between container restarts

## Dependencies

### Core Dependencies
- Flask 2.3.x - Web framework
- PyTorch 2.0.x - ML framework
- transformers 4.30.x - For NLP model
- Python-DOCX, PyPDF2 - Document processing
- FHIR.resources - FHIR standard support

## Features

### Entity Extraction
- Identifies clinical entities including:
  - Inclusion/exclusion criteria
  - Procedures and interventions
  - Medications and dosages
  - Endpoints and measurements
  - Timing information

### Relation Extraction
- Connects related entities within the protocol
- Identifies hierarchical relationships
- Links criteria to procedures and endpoints

### Section Identification
- Automatically identifies document sections
- Maintains hierarchical structure
- Provides context for extracted entities

### Standard Conversion
- Generates FHIR R4 resources:
  - PlanDefinition for overall protocol structure
  - ActivityDefinition for procedures and interventions
  - Library for logic and rules
  - Questionnaire for data collection
- Creates OMOP CDM tables for research use

## Current Status
The project has the following components implemented:
- Project structure and Docker setup
- Development environment configuration
- Web interface for document upload and processing
- Model service with API endpoints
- Entity extraction with fallback mechanisms
- Section and relation extraction
- Initial FHIR conversion implementation
- Comprehensive logging and monitoring

## Next Steps
Planned development priorities:
- Enhanced relation extraction
- Improved FHIR/OMOP conversion
- User authentication and access control
- Results visualization improvements
- Performance optimization
- Additional document format support