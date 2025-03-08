"""Supabase client module for database operations.

This module provides a singleton client for interacting with Supabase,
ensuring a single connection is maintained throughout the application lifecycle.
"""

from supabase import create_client, Client
from typing import Optional
from config import get_settings
from loguru import logger

settings = get_settings()
_supabase_client: Optional[Client] = None
_supabase_service_client: Optional[Client] = None

def init_supabase() -> None:
    """Initialize the Supabase client.
    
    Creates a new Supabase client instance using credentials from settings.
    The client is stored in a module-level variable for reuse.
    
    Note:
        This function should be called during application startup.
        It uses the SUPABASE_URL and SUPABASE_KEY from settings.
        If SUPABASE_SERVICE_KEY is available, it also initializes a service role client.
    """
    global _supabase_client, _supabase_service_client
    
    # Initialize regular client
    _supabase_client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )
    
    # Initialize service role client if credentials are available
    if hasattr(settings, 'SUPABASE_SERVICE_KEY') and settings.SUPABASE_SERVICE_KEY:
        try:
            _supabase_service_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY
            )
            logger.info("Initialized Supabase service role client")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase service role client: {str(e)}")
            _supabase_service_client = None
    else:
        logger.warning("SUPABASE_SERVICE_KEY not found in settings, service role client not initialized")
        _supabase_service_client = None

def get_supabase() -> Client:
    """Get the Supabase client instance.
    
    Returns:
        Client: The initialized Supabase client
        
    Note:
        If the client hasn't been initialized, this function will
        automatically initialize it before returning.
    """
    if _supabase_client is None:
        init_supabase()
    return _supabase_client

def get_supabase_service_client() -> Optional[Client]:
    """Get the Supabase service role client instance.
    
    Returns:
        Optional[Client]: The initialized Supabase service role client, or None if not available
        
    Note:
        If the clients haven't been initialized, this function will
        automatically initialize them before returning.
        The service role client may be None if the SUPABASE_SERVICE_KEY is not available.
    """
    if _supabase_client is None:
        # This will initialize both clients
        init_supabase()
    return _supabase_service_client 