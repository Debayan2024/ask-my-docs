from rag_assistant import bootstrap
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from .embedding_generator import EmbeddingGenerator
from .vector_store_manager import VectorStoreManager
from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class RAGQueryHandler:
    """
    Handles the RAG (Retrieval-Augmented Generation) query process.
    """
    def __init__(self, config: dict):
        """
        Initializes RAG with necessary components.
        Args:
            config (dict): configuration dict created from config.yaml
        """

        self.config = config
        logger.info("Initializing RAG Handler.......")

        try:
             # 1. Initialize the embedding generator (needed for query embedding AND loading store)
            self.embedder = EmbeddingGenerator(config)
            self.embedding_function = self.embedder.embedding_model # The actual LangChain embedding model instance

            # 2. Initialize VectorStoreManager and load the store
            self.vector_store_mgr = VectorStoreManager(config)
            self.vector_store = self.vector_store_mgr.load_vector_store(
                embedding_function=self.embedding_function
            )
            if not self.vector_store:
                # Error already logged by VectorStoreManager
                raise RuntimeError("Failed to load vector store. RAGQueryHandler cannot operate.")

            # 3. Initialize LLM Handler 
            self.llm_handler = LLMHandler(config)

            logger.info("✅ RAGQueryHandler initialized successfully.")

        except (ValueError, RuntimeError, ImportError) as e: # Catch init errors from components
            logger.error(f"❌ Failed to initialize RAGQueryHandler: {e}")
            raise # Re-raise to prevent app from starting in a bad state

    
    def _retrieve_relevant_documents(self, query: str, k: int=3) -> List[Document]:
        """
        Retrieves the top k relevant documents from the vector store for a given query.
        
        """
        if not self.vector_store:
            logger.error("Vector store not loaded. Cannot retrieve documents.")
            return []

        logger.debug(f"Retrieving top {k} relevant documents for query: {query[:50]}...")
        try:
            # The vector_store object (e.g., Chroma instance) has a similarity_search method
            # It internally embeds the query using its configured embedding_function
            retrieved_docs = self.vector_store.similarity_search(query=query, k=k)
            logger.debug(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            return []
        
    def _construct_prompt(self, query: str, context_documents: List[Document]) -> str:
        """
        Constructs a prompt for the LLM using the query and retrieved context.
        """
        if not context_documents:
            # If no context, just use the query, or a modified prompt
            context_str = "No specific context found."
            logger.warning("No context documents to include in prompt.")
        else:
            context_str = "\n\n".join([doc.page_content for doc in context_documents])
        # Basic prompt template - can be made more sophisticated and configurable
        prompt_template = f"""
                You are a helpful AI assistant. Answer the following question based on the provided context.
                If the answer is not found in the context, say "I cannot find the answer in the provided documents."
                Context:
                {context_str}
                Question: {query}
                Answer:"""
        return prompt_template
        
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a user query through the RAG pipeline.

        Args:
            query (str): The user's question.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and source documents.
        """
        logger.info(f"Processing query: {query}")

        # 1. Retrieve relevant documents
        retrieved_docs = self._retrieve_relevant_documents(query)

        # 2. Construct the prompt
        prompt = self._construct_prompt(query, retrieved_docs)

        # 3. Generate response from LLM
        answer = self.llm_handler.generate_response(prompt)

        # Prepare source documents metadata for the response
        source_documents_metadata = []
        if retrieved_docs:
            for doc in retrieved_docs:
                source_documents_metadata.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page_content_preview': doc.page_content[:200] + "..." # Preview
                    # Add other metadata like 'start_index' if useful
                })

        return {
            "query": query,
            "answer": answer,
            "source_documents": source_documents_metadata
        }
    

if __name__ == "__main__":
    
    import os, sys
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, project_root)

    from src.rag_assistant.utils.config_loader import ConfigParser

   

    config = ConfigParser().config_loader()
    rag = RAGQueryHandler(config)

    print(rag.answer_query("what are some application of rag?"))