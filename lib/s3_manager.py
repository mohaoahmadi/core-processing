import boto3
from typing import Optional, BinaryIO
from botocore.client import BaseClient
import asyncio
from config import get_settings

settings = get_settings()
_s3_client: Optional[BaseClient] = None

def init_s3_client() -> None:
    """Initialize S3 client"""
    global _s3_client
    _s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION
    )

def get_s3_client() -> BaseClient:
    """Get S3 client instance"""
    if _s3_client is None:
        init_s3_client()
    return _s3_client

async def upload_file(file_path: str, s3_key: str) -> str:
    """Upload a file to S3 bucket"""
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _upload_file():
        client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
        return f"s3://{settings.S3_BUCKET_NAME}/{s3_key}"
    
    return await asyncio.to_thread(_upload_file)

async def download_file(s3_key: str, local_path: str) -> None:
    """Download a file from S3 bucket"""
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _download_file():
        client.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
    
    return await asyncio.to_thread(_download_file)

async def get_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for an S3 object"""
    client = get_s3_client()
    
    # Run in a separate thread to avoid blocking
    def _get_presigned_url():
        return client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=expires_in
        )
    
    return await asyncio.to_thread(_get_presigned_url)