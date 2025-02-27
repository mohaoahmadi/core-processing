from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ProcessingResult(BaseModel):
    """Base model for processing results"""
    status: str
    message: str
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseProcessor(ABC):
    """Base class for all processors"""
    
    def __init__(self):
        self.result = None

    @abstractmethod
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        pass

    @abstractmethod
    async def process(self, **kwargs) -> ProcessingResult:
        """Execute the processing logic"""
        pass

    async def cleanup(self) -> None:
        """Clean up temporary files and resources"""
        pass

    async def __call__(self, **kwargs) -> ProcessingResult:
        """Main entry point for processing"""
        try:
            if not await self.validate_input(**kwargs):
                return ProcessingResult(
                    status="error",
                    message="Invalid input parameters"
                )
            
            self.result = await self.process(**kwargs)
            return self.result
        except Exception as e:
            return ProcessingResult(
                status="error",
                message=str(e)
            )
        finally:
            await self.cleanup() 