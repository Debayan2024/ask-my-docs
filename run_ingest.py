import os
import sys
import logging
import json

# let add dir to the python path
# this will help in loading modules from src/rag_assistant
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)


from src.rag_assistant.pipeline.ingestion import IngestionPipeline
from src.rag_assistant.utils.config_loader import ConfigParser

# --basic logging setup
# TODO: config for logger
# messages at the INFO level and above (WARNING, ERROR, CRITICAL) will be output. DEBUG messages will be ignored
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# setting up module level logger using __name__
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=================================================")
    logger.info("         RAG Assistant - Ingestion Runner        ")
    logger.info("=================================================")

    # 1. Load Configuration for pipeline
    logger.info("Attempting to load configuration...")
    try:
        config = ConfigParser().config_loader()
        if not config:
            logger.critical("❌ Configuration loading returned None. Exiting.")
            sys.exit(1)
        logger.info("✅ Configuration loaded successfully.")
        # Optional: Print config in debug mode
        if logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"Loaded configuration:\n{json.dumps(config, indent=2)}")
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.critical(f"❌ Failed to load or validate configuration: {e}.")
        sys.exit(1)


    # 2. Initialize and run Pipeline
    logger.info("Attempting to initialize and run pipeline...")
    try:
        logger.info("\nInitializing ingestion pipeline...")
        pipeline = IngestionPipeline(config)
        logger.info("✅ Pipeline initialized.")

        logger.info("\nStarting pipeline execution...")

        pipeline.run()
        
        logger.info("✅ Pipeline execution finished.")

    except (ValueError, FileNotFoundError) as e: # Catch init errors
        logger.critical(f"❌ Pipeline initialization failed: {e}. Exiting.")
        sys.exit(1)
    except Exception as e: # Catch runtime errors during pipeline.run()
         logger.critical(f"❌ An unexpected error occurred during pipeline execution: {e}", exc_info=True) # Log traceback
         sys.exit(1)

    logger.info("\n=================================================")
    logger.info("          Ingestion Runner Finished          ")
    logger.info("=================================================")
    sys.exit(0) # Explicitly exit with success code


