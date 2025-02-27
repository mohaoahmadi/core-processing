"""Base processor module for geospatial data processing.

This module defines the base interfaces and common functionality for all
processors in the system. It provides abstract base classes that enforce
a consistent interface across different processing implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ProcessingResult(BaseModel):
    """Model for standardized processing results.
    
    This class defines the structure for results returned by all processors.
    It uses Pydantic for validation and serialization.
    
    Attributes:
        status (str): Processing status ("success" or "error")
        message (str): Human-readable description of the result
        output_path (Optional[str]): Path to the output file, if any
        metadata (Dict[str, Any]): Additional processing metadata
    """
    status: str
    message: str
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseProcessor(ABC):
    """Abstract base class for all processors.
    
    This class defines the interface that all processors must implement.
    It provides a common structure for input validation, processing,
    and cleanup operations.
    
    Attributes:
        result (Optional[ProcessingResult]): The result of the last processing operation
    """
    
    def __init__(self):
        """Initialize the processor.
        
        Sets up the initial state with no result.
        """
        self.result = None

    @abstractmethod
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters before processing.
        
        Args:
            **kwargs: Arbitrary keyword arguments specific to each processor
            
        Returns:
            bool: True if inputs are valid, False otherwise
            
        Note:
            This method must be implemented by all processor subclasses.
            It should verify all required parameters are present and valid.
        """
        pass

    @abstractmethod
    async def process(self, **kwargs) -> ProcessingResult:
        """Execute the processing logic.
        
        Args:
            **kwargs: Arbitrary keyword arguments specific to each processor
            
        Returns:
            ProcessingResult: The result of the processing operation
            
        Note:
            This method must be implemented by all processor subclasses.
            It should contain the core processing logic for the specific
            type of operation being performed.
        """
        pass

    async def cleanup(self) -> None:
        """Clean up temporary files and resources.
        
        This method is called after processing completes (success or failure).
        It should remove any temporary files and free any resources.
        
        Note:
            This method can be overridden by subclasses if cleanup is needed.
            The default implementation does nothing.
        """
        pass

    async def __call__(self, **kwargs) -> ProcessingResult:
        """Main entry point for processing.
        
        This method orchestrates the processing workflow:
        1. Validates inputs
        2. Executes processing
        3. Performs cleanup
        
        Args:
            **kwargs: Arbitrary keyword arguments passed to validate_input and process
            
        Returns:
            ProcessingResult: The result of the processing operation
            
        Note:
            This method handles the high-level flow and error handling.
            Subclasses should not override this method; instead, they should
            implement validate_input, process, and optionally cleanup.
        """
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