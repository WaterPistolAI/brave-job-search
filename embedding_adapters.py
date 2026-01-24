"""
Embedding adapters for different embedding providers.
Supports local PyTorch embeddings and OpenAI-compatible endpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingAdapter(ABC):
    """Base class for embedding adapters."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class LocalEmbeddingAdapter(BaseEmbeddingAdapter):
    """Local PyTorch embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the local model."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension


class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """OpenAI-compatible embeddings using ChromaDB's OpenAIEmbeddingFunction."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "text-embedding-3-small",
        base_url: str = None,
        dimension: int = None,
    ):
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        except ImportError:
            raise ImportError(
                "chromadb is required for OpenAI embeddings. "
                "Install it with: pip install chromadb"
            )

        # Check for OPENAI_EMBEDDING_API_KEY first, then fall back to OPENAI_API_KEY
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Check for OPENAI_EMBEDDING_BASE_URL first, then fall back to OPENAI_BASE_URL
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_EMBEDDING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )

        # Check for OPENAI_EMBEDDING_MODEL first, then fall back to OPENAI_MODEL, then use model_name parameter
        self.model_name = (
            model_name
            or os.environ.get("OPENAI_EMBEDDING_MODEL")
            or os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
        )

        # Initialize the embedding function
        kwargs = {
            "api_key": self.api_key,
            "model_name": self.model_name,
        }

        # Try to use openai_api_base if supported, otherwise ignore base_url
        # Some versions of ChromaDB don't support custom base URLs
        if self.base_url:
            try:
                # Try with openai_api_base parameter
                test_kwargs = kwargs.copy()
                test_kwargs["openai_api_base"] = self.base_url
                self.embedding_function = OpenAIEmbeddingFunction(**test_kwargs)
            except TypeError:
                # If openai_api_base is not supported, initialize without it
                # and log a warning
                import logging

                logging.warning(
                    "OpenAIEmbeddingFunction does not support custom base_url in this version. "
                    "Using default OpenAI endpoint. To use a custom endpoint, set OPENAI_API_BASE_URL "
                    "environment variable and use the OpenAI client directly."
                )
                self.embedding_function = OpenAIEmbeddingFunction(**kwargs)
        else:
            self.embedding_function = OpenAIEmbeddingFunction(**kwargs)

        # Set dimension based on model or custom dimension
        if dimension is not None:
            self._dimension = dimension
        else:
            self._dimension = self._get_model_dimension(model_name)

    def _get_model_dimension(self, model_name: str) -> int:
        """Get the dimension for a given OpenAI model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model_name, 1536)  # Default to 1536

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API."""
        embeddings = self.embedding_function(texts)
        return embeddings

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension


class EmbeddingAdapterFactory:
    """Factory for creating embedding adapters."""

    @staticmethod
    def create_adapter(provider: str = None, **kwargs) -> BaseEmbeddingAdapter:
        """
        Create an embedding adapter based on the provider.

        Args:
            provider: The embedding provider ('local' or 'openai').
                     If None, reads from EMBEDDING_PROVIDER environment variable.
            **kwargs: Additional arguments for the adapter.

        Returns:
            An instance of BaseEmbeddingAdapter.
        """
        provider = provider or os.environ.get("EMBEDDING_PROVIDER", "local")

        if provider.lower() == "openai":
            return OpenAIEmbeddingAdapter(**kwargs)
        elif provider.lower() == "local":
            return LocalEmbeddingAdapter(**kwargs)
        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                "Supported providers: 'local', 'openai'"
            )


def get_embedding_adapter(**kwargs) -> BaseEmbeddingAdapter:
    """
    Convenience function to get an embedding adapter.

    Args:
        **kwargs: Additional arguments for the adapter.

    Returns:
        An instance of BaseEmbeddingAdapter.
    """
    return EmbeddingAdapterFactory.create_adapter(**kwargs)
