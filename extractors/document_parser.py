"""
Document parser for the Clinical Protocol Extraction Engine.
This module handles the conversion of various file formats (PDF, DOCX, TXT) 
to standardized plain text for processing.
"""

import os
import tempfile
import logging
import re
from pathlib import Path
import PyPDF2
import docx
import magic

# Set up logging
logger = logging.getLogger(__name__)

class DocumentParser:
    """Parse various document formats into plain text for extraction."""
    
    def __init__(self, config=None):
        """
        Initialize document parser.
        
        Args:
            config (dict, optional): Configuration options for parsing.
        """
        self.config = config or {}
        self.mime_detector = magic.Magic(mime=True)
    
    def parse(self, file_path):
        """
        Parse document based on file extension.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            dict: Parsed document with text and metadata.
            
        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        file_path = os.path.abspath(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Parsing document: {file_path}")
        
        try:
            # Determine parser based on file extension
            if file_ext == '.pdf':
                parsed_doc = self.parse_pdf(file_path)
            elif file_ext == '.docx':
                parsed_doc = self.parse_docx(file_path)
            elif file_ext == '.txt':
                parsed_doc = self.parse_txt(file_path)
            else:
                # Try to determine type by MIME
                mime_type = self.mime_detector.from_file(file_path)
                logger.info(f"Detected MIME type: {mime_type}")
                
                if 'pdf' in mime_type.lower():
                    parsed_doc = self.parse_pdf(file_path)
                elif 'officedocument.wordprocessingml' in mime_type.lower():
                    parsed_doc = self.parse_docx(file_path)
                elif 'text/' in mime_type.lower():
                    parsed_doc = self.parse_txt(file_path)
                else:
                    error_msg = f"Unsupported file format: {file_ext} (MIME: {mime_type})"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Add file metadata
            parsed_doc['metadata']['filename'] = os.path.basename(file_path)
            parsed_doc['metadata']['file_size'] = os.path.getsize(file_path)
            parsed_doc['metadata']['file_path'] = file_path
            parsed_doc['metadata']['file_extension'] = file_ext
            
            # Basic text stats
            parsed_doc['metadata']['char_count'] = len(parsed_doc['text'])
            parsed_doc['metadata']['word_count'] = len(parsed_doc['text'].split())
            parsed_doc['metadata']['line_count'] = parsed_doc['text'].count('\n') + 1
            
            logger.info(f"Successfully parsed document ({parsed_doc['metadata']['char_count']} chars, "
                       f"{parsed_doc['metadata']['word_count']} words)")
            
            return parsed_doc
        
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}", exc_info=True)
            raise
    
    def parse_pdf(self, file_path):
        """
        Extract text from PDF files using PyPDF2.
        
        Args:
            file_path (str): Path to PDF file.
            
        Returns:
            dict: Parsed document with text and metadata.
        """
        text = ""
        metadata = {
            'page_count': 0,
            'pages': [],
            'title': None,
            'author': None,
            'creation_date': None
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                
                # Extract document metadata
                if pdf.metadata:
                    metadata['title'] = pdf.metadata.get('/Title')
                    metadata['author'] = pdf.metadata.get('/Author')
                    metadata['creation_date'] = pdf.metadata.get('/CreationDate')
                
                # Extract text from each page
                metadata['page_count'] = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    
                    # Store page info
                    metadata['pages'].append({
                        'page_number': i + 1,
                        'char_count': len(page_text),
                        'word_count': len(page_text.split())
                    })
                    
                    # Add page text with separator
                    if page_text:
                        text += page_text + "\n\n"
            
            # Check if we extracted any text
            if not text.strip():
                logger.warning(f"No text extracted from PDF, may be image-based: {file_path}")
                
                # If pdf2image is available, we could add OCR here in a future update
                # But for now, just provide the warning
            
            return {
                'text': text.strip(),
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error parsing PDF with PyPDF2: {str(e)}", exc_info=True)
            raise
    
    def parse_docx(self, file_path):
        """
        Extract text from DOCX files using python-docx.
        
        Args:
            file_path (str): Path to DOCX file.
            
        Returns:
            dict: Parsed document with text and metadata.
        """
        text = ""
        metadata = {
            'title': None,
            'author': None,
            'paragraph_count': 0,
            'has_tables': False,
            'has_images': False
        }
        
        try:
            doc = docx.Document(file_path)
            
            # Extract document properties
            core_properties = doc.core_properties
            metadata['title'] = core_properties.title
            metadata['author'] = core_properties.author
            metadata['created'] = core_properties.created.isoformat() if core_properties.created else None
            metadata['modified'] = core_properties.modified.isoformat() if core_properties.modified else None
            
            # Extract paragraphs
            metadata['paragraph_count'] = len(doc.paragraphs)
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text.strip() + "\n"
            
            # Check for tables
            metadata['has_tables'] = len(doc.tables) > 0
            metadata['table_count'] = len(doc.tables)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text += row_text + "\n"
            
            # Check for images (simple detection, not extraction)
            metadata['has_images'] = bool(len(doc.inline_shapes))
            metadata['image_count'] = len(doc.inline_shapes)
            
            return {
                'text': text.strip(),
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error parsing DOCX with python-docx: {str(e)}", exc_info=True)
            raise
    
    def parse_txt(self, file_path):
        """
        Read and parse plain text files.
        
        Args:
            file_path (str): Path to text file.
            
        Returns:
            dict: Parsed document with text and metadata.
        """
        try:
            # Detect encoding
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            
            # Normalize line endings
            text = re.sub(r'\r\n?', '\n', text)
            
            metadata = {
                'encoding': encoding,
                'line_count': text.count('\n') + 1
            }
            
            return {
                'text': text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {str(e)}", exc_info=True)
            raise
    
    def detect_encoding(self, file_path):
        """
        Detect character encoding of text files.
        
        Args:
            file_path (str): Path to text file.
            
        Returns:
            str: Detected encoding or utf-8 as fallback.
        """
        try:
            # Use python-magic to detect mime type
            mime_type = self.mime_detector.from_file(file_path)
            
            # Check if mime type contains charset information
            if 'charset=' in mime_type:
                encoding = mime_type.split('charset=')[1].strip()
                return encoding
            
            # If no charset in mime type, try checking for BOM markers
            with open(file_path, 'rb') as file:
                raw_data = file.read(4)  # Read first 4 bytes for BOM detection
                
                if raw_data.startswith(b'\xef\xbb\xbf'):
                    return 'utf-8-sig'
                elif raw_data.startswith(b'\xff\xfe'):
                    return 'utf-16-le'
                elif raw_data.startswith(b'\xfe\xff'):
                    return 'utf-16-be'
            
            # Default to UTF-8 if no specific encoding detected
            return 'utf-8'
            
        except Exception as e:
            logger.warning(f"Error detecting encoding, falling back to utf-8: {str(e)}")
            return 'utf-8'
    
    def extract_metadata(self, file_path):
        """
        Extract metadata from document without full text extraction.
        
        Args:
            file_path (str): Path to document file.
            
        Returns:
            dict: Document metadata.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            'filename': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'file_path': file_path,
            'file_extension': file_ext,
            'last_modified': os.path.getmtime(file_path),
            'mime_type': self.mime_detector.from_file(file_path)
        }
        
        try:
            if file_ext == '.pdf':
                # Extract just PDF metadata
                with open(file_path, 'rb') as file:
                    pdf = PyPDF2.PdfReader(file)
                    if pdf.metadata:
                        metadata.update({
                            'title': pdf.metadata.get('/Title'),
                            'author': pdf.metadata.get('/Author'),
                            'creation_date': pdf.metadata.get('/CreationDate'),
                            'page_count': len(pdf.pages)
                        })
            
            elif file_ext == '.docx':
                # Extract just DOCX metadata
                doc = docx.Document(file_path)
                core_properties = doc.core_properties
                metadata.update({
                    'title': core_properties.title,
                    'author': core_properties.author,
                    'created': core_properties.created.isoformat() if core_properties.created else None,
                    'modified': core_properties.modified.isoformat() if core_properties.modified else None,
                    'paragraph_count': len(doc.paragraphs),
                    'table_count': len(doc.tables),
                    'image_count': len(doc.inline_shapes)
                })
            
        except Exception as e:
            logger.warning(f"Error extracting detailed metadata: {str(e)}")
        
        return metadata
    
    def get_supported_formats(self):
        """Return list of supported document formats."""
        return ['.pdf', '.docx', '.txt']


# For easy testing
if __name__ == "__main__":
    import argparse
    import json
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Parse document to text')
    parser.add_argument('file_path', help='Path to document file')
    parser.add_argument('--metadata-only', action='store_true', help='Extract only metadata')
    parser.add_argument('--output', help='Output file for extracted text')
    args = parser.parse_args()
    
    try:
        parser = DocumentParser()
        
        if args.metadata_only:
            result = parser.extract_metadata(args.file_path)
            print(json.dumps(result, indent=2))
        else:
            result = parser.parse(args.file_path)
            
            # Print metadata
            print("\nDocument Metadata:")
            print(json.dumps(result['metadata'], indent=2))
            
            # Handle output
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                print(f"\nText saved to {args.output}")
            else:
                # Print preview of text
                text_preview = result['text'][:500]
                if len(result['text']) > 500:
                    text_preview += "..."
                print("\nText Preview:")
                print(text_preview)
                print(f"\nFull text length: {len(result['text'])} characters")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")