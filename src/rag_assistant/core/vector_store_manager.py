import logging
import os
import sys
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import chromadb
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    handles the creation, persistence and loading of a vector store.
    Currently supports CHROMADB
    """
    def __init__(self, config: dict):
        """Initializes the VectorStoreManager with configurations.

        Args:
            config (dict): configuration info regarding vector store in dict format.
        Raises:
            ValueError: if vector store configuration is missing or invalid.
        """
        self.config = config
        vector_store_config = self.config.get('vector_store', {})
        self.store_type = vector_store_config.get('type') # Now "ChromaClient"
        self.chroma_server_url = vector_store_config.get('chroma_server_url') # New config param
        
        # Validate Configuration
        # --- Validate Configuration ---
        if not self.store_type:
            logger.error("❌ Vector store type ('vector_store.type') not specified in config.")
            raise ValueError("Vector store type is required.")
        if self.store_type.lower() != 'chromaclient': # Now checks for "ChromaClient"
             logger.error(f"❌ Unsupported vector store type: '{self.store_type}'. Only 'ChromaClient' is currently supported.")
             raise ValueError(f"Unsupported vector store type: {self.store_type}")
        if not self.chroma_server_url:
            logger.error("❌ Chroma server URL ('vector_store.chroma_server_url') not specified.")
            raise ValueError("Chroma server URL is required for ChromaClient type.")

        logger.info(f"VectorStoreManager initialized. Type: {self.store_type}, Server URL: {self.chroma_server_url}")
        # No need to create persist_directory here, as it's for the server.

        # Initialize the Chroma client, connecting to the server
        try:
            self._chroma_client = chromadb.HttpClient(host=self.chroma_server_url.split(':')[1].strip('/'), port=int(self.chroma_server_url.split(':')[-1]))
            # Check if client can connect (optional but good practice)
            # self._chroma_client.heartbeat() # A simple way to check connection
            logger.info(f"✅ ChromaDB client connected to server: {self.chroma_server_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB server at {self.chroma_server_url}: {e}")
            raise RuntimeError(f"Could not connect to ChromaDB server: {e}")

    def create_and_persist(self, chunks: List[Document], embedding_function: Embeddings) -> bool:
        """Creates a new vector store from documents and an embedding function,
        and persists it to the configured directory. Overwrites if exists.

        Args:
            chunks (List[Document]): The list of document chunks to store.
            embedding_function (Embeddings): The LangChain embedding function instance
                                              used to generate embeddings (needed by Chroma).

        Returns:
            bool: True if the store was created and persisted successfully, False otherwise.
        """
        if not chunks:
            logger.warning(f"⚠️ No chunks provided to create vector store. Skipping.")
            return False
        if not embedding_function:
            logger.error("❌ Embedding function instance not provided to VectorStoreManager. Cannot create store.")
            return False
        logger.info(f"Creating new {self.store_type} vector store.")
        logger.info(f"Total input chunks {len(chunks)}.")
        # logger.info(f"Persistent Dir: {self.persistent_dir}")

        try:
            # Get or create the collection (like a table in a traditional DB)
            collection_name = "my_rag_collection"
            vector_store = Chroma(
                client=self._chroma_client, # Pass the client instance
                collection_name=collection_name,
                embedding_function=embedding_function # Pass the embedding function
            )

            # Add documents to the collection.
            # This will internally embed and add them to the collection managed by the server.
            vector_store.add_documents(chunks)

            logger.info(f"✅ Successfully added {len(chunks)} entries to Chroma collection '{collection_name}' on server.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add chunks to ChromaDB server: {e}", exc_info=True)
            return False
    
    def load_vector_store(self, embedding_function: Embeddings):
        """
        Returns a LangChain Chroma vector store instance connected to the server.
        """
        if not embedding_function:
             logger.error("❌ Embedding function instance not provided. Cannot load store for query.")
             return None

        logger.info(f"🔄 Preparing Chroma vector store for query operations (connecting to server).")
        try:
            collection_name = "my_rag_collection"
            vector_store = Chroma(
                client=self._chroma_client, # Use the already connected client
                collection_name=collection_name,
                embedding_function=embedding_function # Must be the same model used for creation
            )
            logger.info(f"✅ Successfully connected to Chroma collection '{collection_name}' on server for query.")
            return vector_store
        except Exception as e:
            logger.error(f"❌ Failed to connect to Chroma collection '{collection_name}' on server: {e}", exc_info=True)
            return None