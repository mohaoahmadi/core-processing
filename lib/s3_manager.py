"""AWS S3 operations manager module.

This module provides asynchronous operations for interacting with AWS S3,
including file upload, download, and presigned URL generation.
All operations are performed in a non-blocking manner using asyncio.
"""

import boto3
from typing import Optional, BinaryIO
from botocore.client import BaseClient
import asyncio
from config import get_settings

settings = get_settings()
_s3_client: Optional[BaseClient] = None

def init_s3_client() -> None:
    """Initialize the AWS S3 client.
    
    Creates a new boto3 S3 client instance using credentials from settings.
    The client is stored in a module-level variable for reuse.
    
    Note:
        This function should be called during application startup.
        It uses AWS credentials from settings (access key, secret key, region).
    """
    global _s3_client
    _s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION
    )

def get_s3_client() -> BaseClient:
    """Get the AWS S3 client instance.
    
    Returns:
        BaseClient: The initialized boto3 S3 client
        
    Note:
        If the client hasn't been initialized, this function will
        automatically initialize it before returning.
    """
    if _s3_client is None:
        init_s3_client()
    return _s3_client

async def upload_file(file_path: str, s3_key: str) -> str:
    """Upload a file to S3 asynchronously.
    
    Args:
        file_path (str): Local path to the file to upload
        s3_key (str): Destination key in S3 bucket
        
    Returns:
        str: S3 URI of the uploaded file (s3://bucket/key)
        
    Note:
        The upload is performed in a separate thread to avoid blocking
        the event loop. The function returns when the upload is complete.
    """
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _upload_file():
        client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
        return f"s3://{settings.S3_BUCKET_NAME}/{s3_key}"
    
    return await asyncio.to_thread(_upload_file)

async def download_file(s3_key: str, local_path: str) -> None:
    """Download a file from S3 asynchronously.
    
    Args:
        s3_key (str): Source key in S3 bucket
        local_path (str): Local path where the file should be saved
        
    Note:
        The download is performed in a separate thread to avoid blocking
        the event loop. The function returns when the download is complete.
    """
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _download_file():
        client.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
    
    return await asyncio.to_thread(_download_file)

async def get_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for an S3 object asynchronously.
    
    Args:
        s3_key (str): Key of the object in S3 bucket
        expires_in (int, optional): URL expiration time in seconds. Defaults to 3600.
        
    Returns:
        str: Presigned URL for the S3 object
        
    Note:
        The URL generation is performed in a separate thread to avoid blocking
        the event loop. The URL will be valid for the specified expiration time.
    """
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _get_presigned_url():
        return client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=expires_in
        )
    
    return await asyncio.to_thread(_get_presigned_url)