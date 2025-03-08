"""AWS S3 operations manager module.

This module provides asynchronous operations for interacting with AWS S3,
including file upload, download, and presigned URL generation.
All operations are performed in a non-blocking manner using asyncio.
"""

import boto3
from typing import Optional, BinaryIO
from botocore.client import BaseClient
import asyncio
from loguru import logger
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
    logger.info("Initializing AWS S3 client")
    try:
        global _s3_client
        
        # Log credentials being used (mask sensitive parts)
        access_key = settings.AWS_ACCESS_KEY_ID
        masked_key = f"{access_key[:4]}...{access_key[-4:]}" if access_key else "Not Set"
        logger.debug(f"Using AWS Access Key ID: {masked_key}")
        logger.debug(f"Using AWS Region: {settings.AWS_DEFAULT_REGION}")
        logger.debug(f"Target S3 Bucket: {settings.S3_BUCKET_NAME}")
        
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION
        )
        
        # Test bucket access
        try:
            _s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
            logger.info(f"Successfully connected to bucket: {settings.S3_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Failed to access bucket {settings.S3_BUCKET_NAME}: {str(e)}")
            raise
            
        logger.info("AWS S3 client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise

def get_s3_client() -> BaseClient:
    """Get the AWS S3 client instance.
    
    Returns:
        BaseClient: The initialized boto3 S3 client
        
    Note:
        If the client hasn't been initialized, this function will
        automatically initialize it before returning.
    """
    if _s3_client is None:
        logger.debug("S3 client not initialized, initializing now")
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
    logger.info(f"Starting S3 upload - File: {file_path}, Key: {s3_key}")
    client = get_s3_client()
    
    def _upload_file():
        try:
            logger.debug(f"Uploading to bucket: {settings.S3_BUCKET_NAME}")
            client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
            s3_uri = f"s3://{settings.S3_BUCKET_NAME}/{s3_key}"
            logger.info(f"Upload completed successfully - URI: {s3_uri}")
            return s3_uri
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            raise
    
    try:
        return await asyncio.to_thread(_upload_file)
    except Exception as e:
        logger.error(f"Error in upload thread: {str(e)}")
        raise

async def download_file(s3_key: str, local_path: str) -> None:
    """Download a file from S3 asynchronously.
    
    Args:
        s3_key (str): Source key in S3 bucket
        local_path (str): Local path where the file should be saved
        
    Note:
        The download is performed in a separate thread to avoid blocking
        the event loop. The function returns when the download is complete.
    """
    logger.info(f"Starting S3 download - Key: {s3_key}, Local path: {local_path}")
    client = get_s3_client()
    
    def _download_file():
        try:
            # Add check if file exists first
            logger.debug(f"Checking if file exists in bucket: {settings.S3_BUCKET_NAME}")
            try:
                client.head_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
                logger.debug(f"File found in S3: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
            except client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.error(f"File not found in S3: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
                    logger.debug("Available path components:")
                    # List the prefix to see what's available
                    prefix = '/'.join(s3_key.split('/')[:-1]) + '/'
                    try:
                        response = client.list_objects_v2(
                            Bucket=settings.S3_BUCKET_NAME,
                            Prefix=prefix,
                            Delimiter='/'
                        )
                        if 'CommonPrefixes' in response:
                            logger.debug(f"Available directories: {[p['Prefix'] for p in response['CommonPrefixes']]}")
                        if 'Contents' in response:
                            logger.debug(f"Available files: {[obj['Key'] for obj in response['Contents']]}")
                    except Exception as list_err:
                        logger.error(f"Error listing S3 contents: {str(list_err)}")
                    raise FileNotFoundError(f"{s3_key}: No such file or directory")
                else:
                    raise

            logger.debug(f"Downloading from bucket: {settings.S3_BUCKET_NAME}")
            client.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
            logger.info("Download completed successfully")
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}", exc_info=True)
            raise
    
    try:
        return await asyncio.to_thread(_download_file)
    except Exception as e:
        logger.error(f"Error in download thread: {str(e)}")
        raise

async def get_presigned_url(s3_key: str, expires_in: int = 3600, http_method: str = "GET") -> str:
    """Generate a presigned URL for an S3 object asynchronously.
    
    Args:
        s3_key (str): Key of the object in S3 bucket
        expires_in (int, optional): URL expiration time in seconds. Defaults to 3600.
        http_method (str, optional): HTTP method for the URL. Defaults to "GET".
        
    Returns:
        str: Presigned URL for the S3 object
        
    Note:
        The URL generation is performed in a separate thread to avoid blocking
        the event loop. The URL will be valid for the specified expiration time.
    """
    logger.info(f"Generating presigned URL - Key: {s3_key}, Method: {http_method}, Expiration: {expires_in}s")
    client = get_s3_client()
    
    def _get_presigned_url():
        try:
            logger.debug(f"Generating URL for bucket: {settings.S3_BUCKET_NAME}")
            
            # For PUT requests, we need to use the client.generate_presigned_post method
            if http_method == "PUT":
                # For PUT operations, we'll use presigned POST which is more reliable for uploads
                response = client.generate_presigned_url(
                    ClientMethod='put_object',
                    Params={
                        'Bucket': settings.S3_BUCKET_NAME,
                        'Key': s3_key,
                        'ContentType': 'image/tiff'  # Set the content type
                    },
                    ExpiresIn=expires_in,
                    HttpMethod='PUT'
                )
                logger.info("Presigned PUT URL generated successfully")
                return response
            else:
                # For GET requests, use the standard method
                url = client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': s3_key},
                    ExpiresIn=expires_in
                )
                logger.info("Presigned GET URL generated successfully")
                return url
                
        except Exception as e:
            logger.error(f"URL generation failed: {str(e)}", exc_info=True)
            raise
    
    try:
        return await asyncio.to_thread(_get_presigned_url)
    except Exception as e:
        logger.error(f"Error in URL generation thread: {str(e)}")
        raise

def download_file_sync(s3_key: str, local_path: str) -> None:
    """Download a file from S3 to a local path (synchronous version).
    
    Args:
        s3_key (str): S3 key of the file to download
        local_path (str): Local path to save the file
        
    Raises:
        Exception: If the download fails
    """
    try:
        logger.info(f"Downloading {s3_key} to {local_path}")
        s3_client = get_s3_client()
        s3_client.download_file(
            Bucket=get_settings().S3_BUCKET_NAME,
            Key=s3_key,
            Filename=local_path
        )
        logger.info(f"Download completed: {local_path}")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {str(e)}")
        raise Exception(f"Failed to download file from S3: {str(e)}")

async def delete_file(file_path: str, bucket_name: str = None) -> bool:
    """
    Delete a file from S3.
    
    Args:
        file_path: The path to the file within the bucket
        bucket_name: The name of the S3 bucket. If None, uses the default from settings.
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        from config import get_settings
        settings = get_settings()
        
        # Use provided bucket_name or default from settings
        bucket = bucket_name if bucket_name else settings.S3_BUCKET_NAME
        
        logger.debug(f"Attempting S3 deletion - Bucket: {bucket}, Key: {file_path}")
        
        # Get the S3 client
        s3_client = get_s3_client()
        
        # Try to delete the object
        try:
            # First check if the object exists
            logger.debug(f"Checking if object exists in bucket {bucket}")
            s3_client.head_object(Bucket=bucket, Key=file_path)
            
            # Object exists, delete it
            logger.debug(f"Object found, proceeding with deletion")
            s3_client.delete_object(Bucket=bucket, Key=file_path)
            logger.info(f"Successfully deleted file from S3 bucket {bucket}: {file_path}")
            return True
            
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Try to list objects with this prefix to help debug
                try:
                    prefix = '/'.join(file_path.split('/')[:-1]) + '/'
                    response = s3_client.list_objects_v2(
                        Bucket=bucket,
                        Prefix=prefix,
                        MaxKeys=10
                    )
                    if 'Contents' in response:
                        logger.debug(f"Files found with similar prefix: {[obj['Key'] for obj in response['Contents']]}")
                    else:
                        logger.debug(f"No files found with prefix: {prefix}")
                except Exception as list_err:
                    logger.debug(f"Error listing similar files: {str(list_err)}")
                
                logger.warning(f"File not found in S3 bucket {bucket}: {file_path}")
                return False
            else:
                logger.error(f"S3 error: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Error deleting file from S3: {e}")
        return False

async def delete_raster_file(raster_id: str):
    try:
        # Assuming you have a way to get the bucket_name and file_path
        # You might need to fetch these from your database first
        raster_info = await get_raster_info(raster_id)  # This function would need to exist
        bucket_name = "your-bucket-name"  # Replace with actual bucket name or get from config
        file_path = raster_info.get("file_path")  # Get the file path from the raster info
        
        # Now call delete_file with both required parameters
        await delete_file(file_path, bucket_name)
        
        # ... rest of the function ...
    except Exception as e:
        logger.warning(f"Error deleting file from S3: {e}")