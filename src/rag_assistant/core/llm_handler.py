import logging
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

class LLMHandler:
    """ Handles interaction with LLM. Currently supports Ollama.
    """
    def __init__(self, config:dict):
        """Initializes Large Language Model specified in config.yaml

        Args:
            config (dict): configurations for llm
        Raises:
            ValueError: For invalid LLM
            ImportError: if any required libs are missing.
        """
        self.config = config
        llm_config = self.config.get("llm", {})
        self.provider = llm_config.get("provider")
        self.model_name = llm_config.get("model_name")
        self.base_url = llm_config.get("model_base_url")

        # validate configs
        # if not self.provider:
        #     logger.error("❌ LLM provider ('llm.provider') not specified in config.")
        #     raise ValueError("LLM provider is required.")
        # if self.provider.lower() != 'ollama':
        #     logger.error(f"❌ Unsupported LLM provider: '{self.provider}'. Only 'Ollama' is currently supported.")
        #     raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not self.model_name:
            logger.error("❌ Ollama model name ('llm.model_name') not specified.")
            raise ValueError("Ollama model name is required.")
        if not self.base_url:
            logger.error("❌ Ollama base URL ('llm.model_base_url') not specified.")
            raise ValueError("Ollama base URL is required.")

        logger.info(f"Initializing LLM Handler for provider: {self.provider}")
        logger.info(f"  Model: {self.model_name}, Base URL: {self.base_url}")

        # initialize Langchain model
        try:
            self.llm = Ollama(
                model=self.model_name,
                # provider=self.provider,
                base_url=self.base_url
            )
        # print(self.llm.invoke("Hello"))
        except ImportError as ie:
            logger.error(f"❌ Failed to import necessary libraries for Ollama. Ensure 'langchain_community' is installed correctly.")
            raise ie
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama LLM '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to initialize Ollama LLM: {e}")
        
        logger.info(f"quick test to see if it connects (can be slow if model needs to load)")
        try:
            print(self.llm.invoke("Hello"))
            logger.info("✅ Successfully connected to Ollama LLM and invoked a test prompt.")
        except Exception as e:
            logger.warning(f"⚠️ Could not invoke test prompt on Ollama LLM: {e}. Ensure Ollama is running and model '{self.model_name}' is pulled.")

        logger.info(f"✅ LLM '{self.model_name}' via Ollama initialized.")

    def generate_response(self, prompt:str) -> str:
        """
        Generates a response from the LLM given a prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM's generated response, or an error message string if fails.
        """
        if not prompt:
            logger.warning("⚠️ Empty prompt provided to LLM.")
            return "Error: Empty prompt received."

        logger.debug(f"LLM Pinging with prompt (first 100 chars): {prompt[:100]}...")
        try:
            # get the response from the model
            response = self.llm.invoke(prompt)
            logger.debug(f"LLM Response (first 100 chars): {str(response)[:100]}...")
            return str(response) # Ensure it's a string
        
        except Exception as e:
            logger.error(f"❌ Error during LLM response generation: {e}", exc_info=True)
            return f"Error: Could not generate response from LLM. Details: {e}"



if __name__ == "__main__":
    config = {
        "llm": {
            "provider": "Ollama",
            "model_name": "tinydolphin",  # Use the actual model name
            "model_base_url": "http://localhost:11434"
            # Add "ollama_temperature": 0.7 if your class handles it
        }
    }
    LLMHandler(config)