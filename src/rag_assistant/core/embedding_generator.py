import logging
from langchain_community.embeddings import HuggingFaceEmbeddings # Use the specific HF wrapper
from langchain_core.documents import Document # For type hinting
from typing import List # For explicit List type hint

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for a given text using a specified HuggingFace model
    """

    def __init__(self, config: dict):
        """Load the configuration for embedding and initializes the HF model.

        Args:
            config (dict): configuration details for the embedding generator
        Returns:
        """
        self.config = config
        
        embeddings_params = self.config.get("embedding", {})
        self.model_name = embeddings_params.get('model_name')
        device = embeddings_params.get('device', 'cpu') # Default to 'cpu' if missing
        normalize = embeddings_params.get('normalize_embeddings', False) # Default to False if missing

        
        if not self.model_name:
            logger.error("❌ Embedding model name ('embedding_model.model_name') not specified in configuration.")
            raise ValueError("Embedding model name is required in config.")

        self.model_kwargs = {'device': device}

        # encode_kwargs expects normalization information
        self.encode_kwargs = {'normalize_embeddings': normalize}

        logger.info(f"Initializing embedding model: {self.model_name}")
        logger.info(f"Model kwargs: {self.model_kwargs}")
        logger.info(f"Encode kwargs: {self.encode_kwargs}")

        try:
            # initializing huggingface embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = self.model_name,
                model_kwargs = self.model_kwargs,
                encode_kwargs = self.encode_kwargs
            )
        except ImportError as IE:
            logger.error(f" Install command might be: pip install sentence-transformers torch")
            raise IE # Re-raise the import error
        
        except Exception as e:
            logger.error(f"Exception {e} occurred while initializing embedding model {self.model_name}.")
            raise RuntimeError(f"Failed to load embedding model: {self.model_name}) from {e}")
    
    def generate(self, chunks: List[Document]) -> List[List[float]]:
        """Generates embeddings for a list of document chunks.

        Args:
            chunks (List[Document]): A list of LangChain Document objects (the text chunks).

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
                               Returns an empty list if input is empty or an error occurs.
        """
        if not chunks:
            logger.warning(f"No chunks of documents provided")
            return []

        texts_to_embed = [chunk.page_content for chunk in chunks]
        logger.info(f"Generating chunks for {len(texts_to_embed)} texts using model {self.model_name}")

        try:
            embeddings = self.embedding_model.embed_documents(texts_to_embed)            
            if not embeddings or len(embeddings) != len(texts_to_embed):
                 logger.error(f"❌ Embedding generation failed or returned unexpected number of results. "
                              f"Expected {len(texts_to_embed)}, Got: {len(embeddings) if embeddings else 0}")
                 return []
            embedding_dim = len(embeddings[0]) if embeddings else 0
            logger.info(f"✅ Successfully generated {len(embeddings)} embeddings.")
            logger.info(f"   Embedding dimension: {embedding_dim}")

            return embeddings

        except Exception as e:
            logger.error(f"❌ An error occurred during embedding generation: {e}", exc_info=True) # Log traceback
            return [] # Return empty list on failure 