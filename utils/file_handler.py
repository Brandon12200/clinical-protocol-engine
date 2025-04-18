import os
import uuid
import shutil
import datetime
import logging
import re
import json

logger = logging.getLogger(__name__)

class FileHandler:
    """File operation utilities for document management"""
    
    def __init__(self, base_directory=None):
        """Initialize with base directory"""
        if base_directory:
            self.base_directory = base_directory
        else:
            # Default to app data directory
            self.base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        
        # Ensure directories exist
        self.uploads_dir = os.path.join(self.base_directory, 'uploads')
        self.processed_dir = os.path.join(self.base_directory, 'processed')
        
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        logger.info(f"FileHandler initialized with base directory: {self.base_directory}")
    
    def save_file(self, file, directory=None):
        """
        Save uploaded file with unique name
        
        Args:
            file: File object to save
            directory: Target directory (defaults to uploads directory)
            
        Returns:
            dict: File information including original filename, saved filename, and file path
        """
        try:
            if directory is None:
                directory = self.uploads_dir
                
            # Generate unique filename
            original_filename = file.filename
            filename, extension = os.path.splitext(original_filename)
            safe_filename = self._sanitize_filename(filename)
            unique_filename = f"{safe_filename}_{uuid.uuid4().hex}{extension}"
            
            file_path = os.path.join(directory, unique_filename)
            
            # Save file
            file.save(file_path)
            
            logger.info(f"Saved file {original_filename} as {unique_filename}")
            
            return {
                "original_filename": original_filename,
                "saved_filename": unique_filename,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            raise
    
    def get_file_path(self, filename, directory=None):
        """
        Get full path to file
        
        Args:
            filename: Name of the file to locate
            directory: Specific directory to check (defaults to checking all directories)
            
        Returns:
            str: Full path to file if found, None otherwise
        """
        if directory is None:
            # Check both uploads and processed directories
            uploads_path = os.path.join(self.uploads_dir, filename)
            if os.path.exists(uploads_path):
                return uploads_path
                
            processed_path = os.path.join(self.processed_dir, filename)
            if os.path.exists(processed_path):
                return processed_path
                
            return None
        else:
            # Check specific directory
            file_path = os.path.join(directory, filename)
            return file_path if os.path.exists(file_path) else None
    
    def remove_file(self, filename, directory=None):
        """
        Delete file
        
        Args:
            filename: Name of the file to delete
            directory: Specific directory (defaults to checking all directories)
            
        Returns:
            bool: True if file was deleted, False otherwise
        """
        try:
            if directory is None:
                # Try both directories
                file_path = self.get_file_path(filename)
                if file_path is None:
                    logger.warning(f"File not found for deletion: {filename}")
                    return False
            else:
                file_path = os.path.join(directory, filename)
                if not os.path.exists(file_path):
                    logger.warning(f"File not found for deletion: {file_path}")
                    return False
            
            # Delete file
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            return False
    
    def cleanup_old_files(self, max_age_hours=24):
        """
        Remove files older than max age
        
        Args:
            max_age_hours: Maximum age of files in hours before deletion
            
        Returns:
            int: Number of files deleted
        """
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
            
            # Check uploads directory
            cleaned_count = 0
            for filename in os.listdir(self.uploads_dir):
                file_path = os.path.join(self.uploads_dir, filename)
                if os.path.isfile(file_path):
                    file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mod_time < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
            
            # Check processed directory
            for filename in os.listdir(self.processed_dir):
                file_path = os.path.join(self.processed_dir, filename)
                if os.path.isfile(file_path):
                    file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mod_time < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} files older than {max_age_hours} hours")
            return cleaned_count
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")
            return 0
    
    def list_files(self, directory=None, extensions=None):
        """
        List files in directory with optional filtering
        
        Args:
            directory: Directory to list files from (defaults to uploads directory)
            extensions: List of file extensions to filter by (e.g., ['.pdf', '.docx'])
            
        Returns:
            list: List of file information dictionaries
        """
        try:
            if directory is None:
                directory = self.uploads_dir
                
            files = []
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                # Skip directories
                if not os.path.isfile(file_path):
                    continue
                    
                # Check extension if specified
                if extensions:
                    _, file_ext = os.path.splitext(filename)
                    if file_ext.lower() not in extensions:
                        continue
                
                # Get file info
                file_size = os.path.getsize(file_path)
                file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                files.append({
                    "filename": filename,
                    "path": file_path,
                    "size": file_size,
                    "modified": file_modified,
                    "extension": os.path.splitext(filename)[1].lower()
                })
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x["modified"], reverse=True)
            
            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {str(e)}")
            return []
    
    def copy_file(self, filename, source_dir=None, target_dir=None):
        """
        Copy file between directories
        
        Args:
            filename: Name of the file to copy
            source_dir: Source directory (defaults to uploads directory)
            target_dir: Target directory (defaults to processed directory)
            
        Returns:
            str: Path to copied file if successful, None otherwise
        """
        try:
            if source_dir is None:
                source_dir = self.uploads_dir
                
            if target_dir is None:
                target_dir = self.processed_dir
                
            source_path = os.path.join(source_dir, filename)
            
            if not os.path.exists(source_path):
                logger.warning(f"Source file not found for copying: {source_path}")
                return None
            
            # Generate target path, using unique name if file exists
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_dir, f"{name}_{uuid.uuid4().hex[:8]}{ext}")
            
            # Copy file
            shutil.copy2(source_path, target_path)
            logger.info(f"Copied file from {source_path} to {target_path}")
            
            return target_path
        except Exception as e:
            logger.error(f"Error copying file {filename}: {str(e)}")
            return None
    
    def save_result(self, result_data, filename=None):
        """
        Save processing result to processed directory
        
        Args:
            result_data: Result data to save (serializable object)
            filename: Custom filename (defaults to generated UUID)
            
        Returns:
            str: Path to saved result file
        """
        try:
            if filename is None:
                filename = f"result_{uuid.uuid4().hex}.json"
                
            if not filename.endswith('.json'):
                filename += '.json'
                
            file_path = os.path.join(self.processed_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save JSON data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved result data to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving result data: {str(e)}")
            raise
    
    def load_result(self, filename):
        """
        Load processing result from processed directory
        
        Args:
            filename: Name of the result file
            
        Returns:
            dict: Loaded result data
        """
        try:
            file_path = self.get_file_path(filename, self.processed_dir)
            
            if not file_path:
                logger.warning(f"Result file not found: {filename}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
            return result_data
        except Exception as e:
            logger.error(f"Error loading result data from {filename}: {str(e)}")
            return None
    
    def create_temp_directory(self):
        """
        Create a temporary directory
        
        Returns:
            str: Path to created temporary directory
        """
        try:
            temp_dir = os.path.join(self.base_directory, 'temp', uuid.uuid4().hex)
            os.makedirs(temp_dir, exist_ok=True)
            logger.debug(f"Created temporary directory: {temp_dir}")
            return temp_dir
        except Exception as e:
            logger.error(f"Error creating temporary directory: {str(e)}")
            raise
    
    def _sanitize_filename(self, filename):
        """
        Sanitize filename to remove invalid characters
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            str: Sanitized filename
        """
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Remove any non-alphanumeric characters except underscores and hyphens
        filename = re.sub(r'[^\w\-]', '', filename)
        # Truncate if too long
        if len(filename) > 50:
            filename = filename[:50]
        return filename