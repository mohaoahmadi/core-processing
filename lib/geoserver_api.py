"""GeoServer REST API client module.

This module provides asynchronous operations for interacting with GeoServer,
including workspace management and layer publishing capabilities.
All operations are performed in a non-blocking manner using asyncio.
"""

from typing import Optional, Dict, Any
import requests
from requests.auth import HTTPBasicAuth
import asyncio
from config import get_settings

settings = get_settings()
_geoserver_session: Optional[requests.Session] = None

def init_geoserver_client() -> None:
    """Initialize the GeoServer client session.
    
    Creates a new requests Session with basic authentication using credentials
    from settings. The session is stored in a module-level variable for reuse.
    
    Note:
        This function should be called during application startup.
        It uses GeoServer credentials from settings (username, password).
    """
    global _geoserver_session
    _geoserver_session = requests.Session()
    _geoserver_session.auth = HTTPBasicAuth(
        settings.GEOSERVER_USERNAME,
        settings.GEOSERVER_PASSWORD
    )

def get_geoserver_session() -> requests.Session:
    """Get the GeoServer session instance.
    
    Returns:
        requests.Session: The initialized GeoServer session with authentication
        
    Note:
        If the session hasn't been initialized, this function will
        automatically initialize it before returning.
    """
    if _geoserver_session is None:
        init_geoserver_client()
    return _geoserver_session

async def create_workspace(workspace: str = None) -> Dict[str, Any]:
    """Create a new workspace in GeoServer asynchronously.
    
    Args:
        workspace (str, optional): Name of the workspace to create.
            If None, uses the default workspace from settings.
            
    Returns:
        Dict[str, Any]: Response containing workspace creation status
        
    Note:
        The workspace creation is performed in a separate thread to avoid
        blocking the event loop. Raises HTTPError for failed requests.
    """
    session = get_geoserver_session()
    workspace = workspace or settings.GEOSERVER_WORKSPACE
    
    # Run in a separate thread to avoid blocking
    def _create_workspace():
        response = session.post(
            f"{settings.GEOSERVER_URL}/rest/workspaces",
            json={"workspace": {"name": workspace}},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return {"status": "success", "workspace": workspace}
    
    return await asyncio.to_thread(_create_workspace)

async def publish_geotiff(
    layer_name: str,
    file_path: str,
    workspace: str = None
) -> Dict[str, Any]:
    """Publish a GeoTIFF file as a new layer in GeoServer asynchronously.
    
    Args:
        layer_name (str): Name for the new layer
        file_path (str): Path to the GeoTIFF file to publish
        workspace (str, optional): Target workspace. If None, uses default workspace.
        
    Returns:
        Dict[str, Any]: Response containing layer publication status
        
    Note:
        The file upload and layer creation are performed in a separate thread
        to avoid blocking the event loop. Raises HTTPError for failed requests.
    """
    session = get_geoserver_session()
    workspace = workspace or settings.GEOSERVER_WORKSPACE
    
    # Run in a separate thread to avoid blocking
    def _publish_geotiff():
        with open(file_path, 'rb') as f:
            response = session.put(
                f"{settings.GEOSERVER_URL}/rest/workspaces/{workspace}/coveragestores/{layer_name}/file.geotiff",
                data=f,
                headers={"Content-type": "image/tiff"}
            )
        response.raise_for_status()
        return {
            "status": "success",
            "layer": layer_name,
            "workspace": workspace
        }
    
    return await asyncio.to_thread(_publish_geotiff)

async def delete_layer(
    layer_name: str,
    workspace: str = None,
    recurse: bool = True
) -> Dict[str, Any]:
    """Delete a layer from GeoServer asynchronously.
    
    Args:
        layer_name (str): Name of the layer to delete
        workspace (str, optional): Workspace containing the layer.
            If None, uses default workspace.
        recurse (bool, optional): Whether to recursively delete all resources.
            Defaults to True.
            
    Returns:
        Dict[str, Any]: Response containing layer deletion status
        
    Note:
        The layer deletion is performed in a separate thread to avoid
        blocking the event loop. Raises HTTPError for failed requests.
    """
    session = get_geoserver_session()
    workspace = workspace or settings.GEOSERVER_WORKSPACE
    
    # Run in a separate thread to avoid blocking
    def _delete_layer():
        response = session.delete(
            f"{settings.GEOSERVER_URL}/rest/workspaces/{workspace}/coveragestores/{layer_name}",
            params={"recurse": str(recurse).lower()}
        )
        response.raise_for_status()
        return {"status": "success", "layer": layer_name}
    
    return await asyncio.to_thread(_delete_layer)