from typing import Optional, Dict, Any
import requests
from requests.auth import HTTPBasicAuth
import asyncio
from config import get_settings

settings = get_settings()
_geoserver_session: Optional[requests.Session] = None

def init_geoserver_client() -> None:
    """Initialize GeoServer client session"""
    global _geoserver_session
    _geoserver_session = requests.Session()
    _geoserver_session.auth = HTTPBasicAuth(
        settings.GEOSERVER_USERNAME,
        settings.GEOSERVER_PASSWORD
    )

def get_geoserver_session() -> requests.Session:
    """Get GeoServer session"""
    if _geoserver_session is None:
        init_geoserver_client()
    return _geoserver_session

async def create_workspace(workspace: str = None) -> Dict[str, Any]:
    """Create a new workspace in GeoServer"""
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
    """Publish a GeoTIFF file as a new layer"""
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
    """Delete a layer from GeoServer"""
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