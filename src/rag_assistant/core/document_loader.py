import os
import sys
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --basic logging setup
# TODO: config for logger
# messages at the INFO level and above (WARNING, ERROR, CRITICAL) will be output. DEBUG messages will be ignored
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# setting up module level logger using __name__
logger = logging.getLogger(__name__)

class DocumentLoader:
    """load documents from a specificed directiory in config
    """

    def __init__(self, config):
        """Initializes DocumentLoader with configuration
        """
        self.config = config

        # if value is not present variables will be set to {}
        self.data_dir = self.config.get('paths', {}).get('data_dir')
        self.supported_extns = self.config.get('ingestion', {}).get('supported_extensions')

        if not self.data_dir:
            logger.error(f"❌ Source directory not specified in configuration (paths.data_dir).")
            raise ValueError("Source directory required in config")
        if not self.supported_extns:
            logger.error(f"❌ Supported extensions not specified in configuration (ingestion.supported_extensions).")
            raise ValueError("Supported extensions required in config")
        
        logger.info(
            f"DocumentLoader initialized. Data Path {os.path.abspath(self.data_dir)}. Extensions: {self.supported_extns}")
        
    
    def _get_loader(self, filepath: str, file_extension:str):
        """Based on the extension type loader returns an appropriate LangChain loader

        Args:
            filepath (str): file path
            file_extension (str): extension type of the file
        """
        try:
            if file_extension == '.txt':
                return TextLoader(filepath)
            elif file_extension == '.pdf':
                return PyPDFLoader(filepath)
            else:
                logger.warning(f"Loader not present for the extension type: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Error initializing loader for extension type: {file_extension} with error {e}")
            return None

    
    def load(self):
        """load all the supported documents from the source directory

        Returns:
            list: A list of all the LangChain Document objects, or an empty list if none found.loaded
        """
        logger.info(f"📁 Loading documents from: {os.path.abspath(self.data_dir)}")
        documents = []
        files_processed = 0
        files_skipped = 0
        files_error = 0

        data_full_path = os.path.abspath(self.data_dir)
        for filename in os.listdir(data_full_path):
            src_filepath = os.path.join(data_full_path, filename)
            if os.path.isfile(src_filepath):
                file_extn = os.path.splitext(src_filepath)[1].lower()

                if file_extn in self.supported_extns:
                    logger.info(f"Attempting to load {filename} (Type: {file_extn})")

                    # get loader --> LangChain Loader for documents
                    loader = self._get_loader(src_filepath, file_extn)

                    if loader:
                        try:
                            # performing actual load
                            loaded_docs = loader.load()
                            # PyPDFLoader creates one separate LangChain Document object for each page in the PDF
                            for doc in loaded_docs:
                                doc.metadata["source"] = filename
                            documents.extend(loaded_docs)
                            logger.info(f" ✅ Successfully loaded file {filename}")
                            files_processed += 1
                                
                        except FileNotFoundError:
                            logger.error(f"File {filename} not present during loading")
                            files_error +=1
                        except Exception as exc:
                            logger.error(f"File {filename} cannot be loaded due to {exc}")
                            files_error +=1

                    else:
                        logger.warning(f"Skipping {filename}: Issue in loader initialization or no specific loader")
                        files_skipped +=1

        logger.info(f"\n📄 Document Loading Summary:")
        logger.info(f"   Processed {files_processed} supported files.")
        logger.info(f"   Skipped {files_skipped} files (no specific loader/init error).")
        logger.info(f"   Encountered errors in {files_error} files during load.")
        logger.info(f"   Total LangChain documents loaded: {len(documents)}")

        # if files but cannot be loaded despite finding the files
        if not documents and (files_error + files_processed + files_skipped)>0:
            logger.warning(f"⚠️ No documents were successfully loaded despite finding files. Check logs")
        elif not documents:
            logger.warning(f"Source directory is empty or contains no supported file types.")


        return documents
    
if __name__ == "__main__":
    print(DocumentLoader().config)