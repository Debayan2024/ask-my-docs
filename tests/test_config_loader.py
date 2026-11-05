import pytest
import os
import yaml
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.rag_assistant.utils.config_loader import ConfigParser

# Create a temporary config file for testing
@pytest.fixture
def temp_config_file(tmp_path):
    config_content = {
        "data": {"source_directory": "./test_data"},
        "llm": {"provider": "Ollama", "ollama_model_name": "test_model"}
    }
    config_dir = tmp_path / "config"
    # print(f"tmp_path is: {tmp_path}")
    # print(f"Type of tmp_path is: {type(tmp_path)}")
    config_dir.mkdir()
    config_file = config_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    
    return str(config_file)


@pytest.fixture
def mock_project_root(monkeypatch, tmp_path):
    """
    Mocks the base_dir calculation in ConfigParser to point to tmp_path,
    allowing config_path="config/test_config.yaml" to resolve correctly within tmp_path.
    """
    def mock_abspath_join(*args):
        # This simulates the ConfigParser's base_dir calculation being tmp_path
        # when it goes up three levels from a hypothetical utils directory.
        # So, if config_path is "config/test_config.yaml", it joins with tmp_path.
        return os.path.join(tmp_path, args[-1]) # Simplified: just join with last part of path

    # This is a bit tricky as it depends on how ConfigParser resolves paths.
    # A more robust way is to make ConfigParser accept an absolute path for testing.
    # For now, let's assume we can make ConfigParser find the config at tmp_path/config/test_config.yaml

    # Let's try another approach: make the ConfigParser use a root_dir relative to tmp_path
    # This requires modifying ConfigParser or testing it differently.
    # The easiest for now: pass the full path of temp_config_file to ConfigParser in the test.
    pass


def test_config_parser_loads_valid_config(temp_config_file):
    # For this test to work easily, ConfigParser should be able to take a full path
    # or we must ensure its relative path resolution logic aligns with temp_config_file location.
    # Let's assume ConfigParser can be instantiated with a full path to the config.
    try:
        # If ConfigParser takes a full path:
        parser = ConfigParser(config_path=temp_config_file)
        assert parser.config is not None
        assert parser.config["data"]["source_directory"] == "./test_data"
        assert parser.config["llm"]["provider"] == "Ollama"
    except FileNotFoundError:
        # This might happen if ConfigParser's relative path logic doesn't work with tmp_path
        # without more sophisticated mocking of its path resolution.
        pytest.fail(f"Config file not found at {temp_config_file}, check ConfigParser path logic or test setup.")


def test_config_parser_raises_file_not_found(tmp_path):
    non_existent_path = str(tmp_path / "config" / "non_existent_config.yaml")
    # non_existent_path = "config/config.yaml" # --> tests fail here
    with pytest.raises(FileNotFoundError):
        # Assuming ConfigParser can take a full path here too for testing
        ConfigParser(config_path=non_existent_path)