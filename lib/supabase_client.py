from supabase import create_client, Client
from typing import Optional
from config import get_settings

settings = get_settings()
_supabase_client: Optional[Client] = None

def init_supabase() -> None:
    """Initialize Supabase client"""
    global _supabase_client
    _supabase_client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )

def get_supabase() -> Client:
    """Get Supabase client instance"""
    if _supabase_client is None:
        init_supabase()
    return _supabase_client 