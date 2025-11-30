"""Base classes for Chinese word segmentation models."""

from abc import ABC, abstractmethod
from typing import List, Union
import os


class BaseSegmenter(ABC):
    """
    Abstract base class for all segmentation models.
    
    All segmenters must implement the `segment` method.
    """

    def __init__(self):
        self.trained = False
        self.model_name = self.__class__.__name__

    @abstractmethod
    def segment(self, text: str) -> List[str]:
        """
        Segment text into words.
        
        Args:
            text: Input Chinese text
            
        Returns:
            List of segmented words
        """
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        Alias for segment() for backward compatibility.
        
        Args:
            text: Input Chinese text
            
        Returns:
            List of segmented words
        """
        return self.segment(text)

    def train(self, data: Union[str, List[str]]) -> None:
        """
        Train the model on data.
        
        Args:
            data: Training data path(s)
        """
        raise NotImplementedError(f"{self.model_name} does not support training")

    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving {self.model_name} to {path}")

    def load(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"Loading {self.model_name} from {path}")

    def __repr__(self) -> str:
        return f"{self.model_name}(trained={self.trained})"
