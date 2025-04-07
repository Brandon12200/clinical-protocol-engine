# Clinical Protocol Extraction and Standardization Engine
An AI-powered system that transforms unstructured clinical protocol documents into standardized healthcare formats (FHIR resources and OMOP CDM tables), designed for healthcare institutions and medical researchers.

## Overview
The Clinical Protocol Extraction Engine uses NLP technology to extract structured data from clinical protocol documents, allowing for seamless integration with electronic health record systems, research databases, and clinical decision support tools.

## Key Features
- **Local Processing Architecture**: All document processing occurs locally, with no external data transmission
- **Medical Document Support**: Handles PDF, DOCX, and TXT formats commonly used in clinical settings
- **Biomedical NLP Model**: Utilizes a fine-tuned DistilBERT model (330MB) optimized for clinical entity recognition
- **Dual Standard Support**: Converts extracted protocol elements to both FHIR R4 resources and OMOP CDM tables
- **Docker Containerization**: Easily deployable in any environment with Docker support

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
   
3. **Standards Conversion Layer**
   - Maps extracted data to FHIR R4 resources (PlanDefinition, ActivityDefinition)
   - Creates OMOP CDM table representations
   - Provides terminology mapping to standard codes

4. **Web Presentation Layer**
   - Offers user interface for document upload and visualization
   - Displays extracted protocol elements with highlighting
   - Enables export to FHIR/OMOP formats

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
```
The development environment will automatically start and open your browser to the application interface.

### Development Workflow
```bash
# Stop the development environment
docker compose -f docker-compose.dev.yml down
# Rebuild containers after dependency changes
docker compose -f docker-compose.dev.yml build
# Access container shell
docker compose -f docker-compose.dev.yml exec web bash
```

## Project Structure
The project follows a modular organization:
- **app/**: Web application with Flask interface
- **models/**: AI model management and preprocessing
- **extractors/**: Document processing and entity extraction
- **standards/**: FHIR and OMOP conversion components
- **utils/**: Common utilities and helpers

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

## Current Status
This project is in early development with the following components implemented:
- Project structure and Docker setup
- Development environment configuration
- Web interface for document upload
- Initial application architecture
- File-based storage system

## Next Steps
Planned development priorities:
- Document parsing and text extraction implementation
- AI model integration for entity recognition
- FHIR/OMOP conversion development
- Results visualization interface