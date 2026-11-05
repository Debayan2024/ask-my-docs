import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.rag_assistant.core.document_loader import DocumentLoader
from src.rag_assistant.core.document_splitter import DocumentSplitter
from src.rag_assistant.core.embedding_generator import EmbeddingGenerator
from src.rag_assistant.core.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Orchestrate the document ingestion process
    """
    def __init__(self, config):
        """Initialize all the components necessary to run the ingestion process
        based on the config
        """
        self.config = config
        try:
            self.doc_loader = DocumentLoader(self.config)
            self.doc_splitter = DocumentSplitter(self.config)
            # Initialize other components later
            self.embedder = EmbeddingGenerator(self.config)
            self.vector_store = VectorStoreManager(config)
            logger.info("IngestionPipeline initialized successfully.")
        except (ValueError, FileNotFoundError) as e:
             logger.error(f"❌ Failed to initialize IngestionPipeline components: {e}")
             # Option 1: Re-raise the exception to halt execution
             raise
             # Option 2: Set a flag to indicate failure (less common for init)
             # self.initialization_failed = True


    def run(self):
        """
        Executes the full ingestion pipeline.
        """
        logger.info("🚀 Starting Ingestion Pipeline Run...")

        # Step 1: Load Documents (using the loader instance)
        logger.info("\n===== Stage 1: Loading Documents =====")
        documents = self.doc_loader.load()
        if not documents:
            logger.error("❌ Document loading failed or yielded no documents. Halting pipeline.")
            return # Stop the pipeline run

        # Step 2: Split Documents (using the splitter instance)
        logger.info("\n===== Stage 2: Splitting Documents =====")
        chunks = self.doc_splitter.split(documents)
        if not chunks:
            logger.error("❌ Document splitting failed or yielded no chunks. Halting pipeline.")
            return # Stop the pipeline run

        # Step 3: Generate Embeddings (Placeholder)
        logger.info("\n===== Stage 3: Generating Embeddings=====")
        embeddings = self.embedder.generate(chunks)
        if not embeddings:
            logger.error("❌ Embedding generation failed. Halting pipeline.")
            return
        logger.info("\n Embeddings generated")
        if embeddings:
            logger.info(f"   Sample embedding (first 5 values of first vector): {embeddings[0][:5]}")
            logger.info(f"   Embedding dimension: {len(embeddings[0])}")

        # Step 4: Store in Vector DB (Placeholder)
        logger.info("\n===== Stage 4: Storing in Vector DB =====")
        success = self.vector_store.create_and_persist(chunks=chunks, embedding_function=self.embedder.embedding_model)
        if not success:
            logger.error("❌ Failed to store chunks in vector store. Halting pipeline.")
            return # Stop the pipeline run
        logger.info("\n✅ Chunks successfully stored in vector database.")

        logger.info("\n🏁 Ingestion Pipeline Run Completed (up to implemented steps).")
