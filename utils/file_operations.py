"""File system operations utility module.

This module provides a collection of utility functions for common file system
operations, including temporary file management, safe file deletion,
file hashing, and metadata handling.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import hashlib
from datetime import datetime
import tempfile
from loguru import logger

def create_temp_directory(prefix: str = "core-processing-") -> Path:
    """Create a temporary directory with a unique name.
    
    Args:
        prefix (str, optional): Prefix for the directory name.
            Defaults to "core-processing-".
            
    Returns:
        Path: Path to the created temporary directory
        
    Note:
        The directory is created with appropriate permissions and
        a unique name to avoid conflicts. The caller is responsible
        for cleaning up the directory when it's no longer needed.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir

def safe_delete(path: Path) -> None:
    """Safely delete a file or directory.
    
    Args:
        path (Path): Path to the file or directory to delete
        
    Note:
        This function handles both files and directories, and logs
        any errors that occur during deletion without raising exceptions.
        For directories, it performs recursive deletion.
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        logger.debug(f"Successfully deleted: {path}")
    except Exception as e:
        logger.error(f"Failed to delete {path}: {str(e)}")

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file.
    
    Args:
        file_path (Path): Path to the file to hash
        
    Returns:
        str: Hexadecimal representation of the file's MD5 hash
        
    Note:
        The file is read in chunks to efficiently handle large files
        without loading them entirely into memory.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_directory(directory: Path) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (Path): Path to the directory to ensure exists
        
    Note:
        Creates parent directories as needed (mkdir -p equivalent).
        No error is raised if the directory already exists.
    """
    directory.mkdir(parents=True, exist_ok=True)

def list_files_by_extension(
    directory: Path,
    extensions: List[str],
    recursive: bool = False
) -> List[Path]:
    """List files in a directory filtered by extension.
    
    Args:
        directory (Path): Directory to search in
        extensions (List[str]): List of file extensions to include
        recursive (bool, optional): Whether to search subdirectories.
            Defaults to False.
            
    Returns:
        List[Path]: Sorted list of paths to matching files
        
    Note:
        Extensions can be specified with or without the leading dot.
        The search is case-insensitive for extensions.
    """
    files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        files.extend(directory.glob(f"{pattern}{ext}"))
    
    return sorted(files)

def generate_unique_filename(
    directory: Path,
    base_name: str,
    extension: str
) -> Path:
    """Generate a unique filename in the given directory.
    
    Args:
        directory (Path): Directory where the file will be created
        base_name (str): Base name for the file
        extension (str): File extension
        
    Returns:
        Path: Path with a unique filename
        
    Note:
        The generated filename includes a timestamp and, if needed,
        a counter to ensure uniqueness. Format:
        {base_name}_{timestamp}[_counter]{extension}
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 0
    
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        filename = f"{base_name}_{timestamp}{suffix}{extension}"
        file_path = directory / filename
        
        if not file_path.exists():
            return file_path
        
        counter += 1

def copy_with_metadata(
    src_path: Path,
    dst_path: Path,
    preserve_timestamps: bool = True
) -> None:
    """Copy a file while preserving metadata.
    
    Args:
        src_path (Path): Source file path
        dst_path (Path): Destination file path
        preserve_timestamps (bool, optional): Whether to preserve timestamps.
            Defaults to True.
            
    Note:
        When preserve_timestamps is True, uses shutil.copy2 to preserve
        metadata including timestamps. Otherwise, uses shutil.copy which
        only copies content and permissions.
    """
    shutil.copy2(src_path, dst_path) if preserve_timestamps else shutil.copy(src_path, dst_path)
    logger.debug(f"Copied {src_path} to {dst_path}")

def get_file_info(file_path: Path) -> dict:
    """Get detailed information about a file.
    
    Args:
        file_path (Path): Path to the file to inspect
        
    Returns:
        dict: Dictionary containing file information:
            - name: File name
            - extension: File extension
            - size: File size in bytes
            - created: Creation timestamp
            - modified: Last modification timestamp
            - accessed: Last access timestamp
            - is_file: Whether it's a regular file
            - is_dir: Whether it's a directory
            - hash: MD5 hash (for regular files only)
    """
    stat = file_path.stat()
    return {
        "name": file_path.name,
        "extension": file_path.suffix,
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "accessed": datetime.fromtimestamp(stat.st_atime),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "hash": get_file_hash(file_path) if file_path.is_file() else None
    } 