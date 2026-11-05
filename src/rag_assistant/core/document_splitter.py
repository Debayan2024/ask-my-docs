import os
import sys
import logging

# let add dir to the python path


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# setting up module level logger using __name__
logger = logging.getLogger(__name__)

class DocumentSplitter:
    "Splits documents into smaller chunks"

    def __init__(self, config):
        self.config = config

        ingestion_config = self.config.get("ingestion", {})

        if not ingestion_config:
            raise ValueError(f"ingestion required in config.yaml")
        self.chunk_size = ingestion_config.get("chunk_size", {})
        self.chunk_overlap = ingestion_config.get("chunk_overlap", {})

        if not isinstance(self.chunk_size, int) or not isinstance(self.chunk_overlap, int):
            logger.error(f"❌ Chunk size and chunk overlap must be integers in config.")
            raise TypeError(f"Invalid chunk size and chunk overlap type")
        
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(f"⚠️ Chunk overlap ({self.chunk_overlap}) >= chunk size ({self.chunk_size}). Adjusting overlap to 10% of size.")
            self.chunk_overlap = int(self.chunk_size * 0.1)

        # initialize splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            add_start_index = True
        )
        logger.info(f"Document splitter initialized. Chunk size: {self.chunk_size}, Chunk Overlap: {self.chunk_overlap}")

    
    def split(self, documents:list[Document]) -> list[Document]:
        """Splits a list of documents into chunks

        Args:
            documents (list[Document]): the list of LangChain DOcument objects to split.

        Returns:
            list[Document]: A list of smaller chunk Documents, or empty list on failure/no input.
        """

        if not documents:
            logger.warning("No documents provided to split")
            return []
        logger.info(f"🧠 Splitting {len(documents)} documents into chunks...")

        try:
            chunks = self._splitter.split_documents(documents)
            logger.info(f"\n Splitting Summary:")
            logger.info(f"Processed {len(documents)} original documents.")
            logger.info(f"Generated {len(chunks)} chunks of text.")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ An error occurred during document splitting: {e}")
            return [] # Return empty list on failure