"""Supabase client module for database operations.

This module provides a singleton client for interacting with Supabase,
ensuring a single connection is maintained throughout the application lifecycle.
"""

from supabase import create_client, Client
from typing import Optional
from config import get_settings

settings = get_settings()
_supabase_client: Optional[Client] = None

def init_supabase() -> None:
    """Initialize the Supabase client.
    
    Creates a new Supabase client instance using credentials from settings.
    The client is stored in a module-level variable for reuse.
    
    Note:
        This function should be called during application startup.
        It uses the SUPABASE_URL and SUPABASE_KEY from settings.
    """
    global _supabase_client
    _supabase_client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )

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