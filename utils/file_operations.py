import os
import shutil
from pathlib import Path
from typing import List, Optional
import hashlib
from datetime import datetime
import tempfile
from loguru import logger

def create_temp_directory(prefix: str = "core-processing-") -> Path:
    """Create a temporary directory with the given prefix"""
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir

def safe_delete(path: Path) -> None:
    """Safely delete a file or directory"""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        logger.debug(f"Successfully deleted: {path}")
    except Exception as e:
        logger.error(f"Failed to delete {path}: {str(e)}")

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_directory(directory: Path) -> None:
    """Ensure a directory exists, create if it doesn't"""
    directory.mkdir(parents=True, exist_ok=True)

def list_files_by_extension(
    directory: Path,
    extensions: List[str],
    recursive: bool = False
) -> List[Path]:
    """List all files in a directory with specified extensions"""
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
    """Generate a unique filename in the given directory"""
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
    """Copy a file while preserving metadata"""
    shutil.copy2(src_path, dst_path) if preserve_timestamps else shutil.copy(src_path, dst_path)
    logger.debug(f"Copied {src_path} to {dst_path}")

def get_file_info(file_path: Path) -> dict:
    """Get detailed information about a file"""
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