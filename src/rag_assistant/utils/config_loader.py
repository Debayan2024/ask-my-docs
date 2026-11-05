import yaml
import os

class ConfigParser:
    """
    Class that handles any configuration related functionality
    """
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.config_loader()       
        

    def config_loader(self):
        """load configuration details from a YAML file

        Args:
            config_path (str, optional): path to config file. Defaults to "config/config.yaml".
        """
        # construct abs path relative to the project
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
        abs_config_path = os.path.join(base_dir, self.config_path)
        
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"Configuration file not found at the path: {abs_config_path}")
        
        try:
            with open(abs_config_path, 'r') as stream:
                config = yaml.safe_load(stream)
            print(f"Configuration loaded successfully from {abs_config_path}")
            return config
        
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file :{exc}")
            raise
        except Exception as e:
            print(f"Unexpected exception occurred : {e}")
            raise

if __name__ =="__main__":
    
    try:
        config = ConfigParser()
        import json

        print(json.dumps(config))
    except Exception as e:
        print(f"Exception occurred while loading config {e}")

