import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
from eva_core.services.prompt_master import PromptMaster

class TestPromptMaster:

    def test_init_sets_templates_dir(self):
        """Verify that templates_dir is set correctly during initialization."""
        pm = PromptMaster(templates_dir="/custom/path")
        assert pm.templates_dir == "/custom/path"

    def test_load_template_success(self):
        """Verify that a template is loaded correctly when it exists."""
        # Setup mocks
        with patch("os.path.exists") as mock_exists, \
             patch("builtins.open", mock_open(read_data="Template Content")) as mock_file:

            mock_exists.return_value = True
            pm = PromptMaster(templates_dir="/docs")

            # Action
            content = pm._load_template("test.md")

            # Assertions
            expected_path = os.path.join("/docs", "test.md")
            assert content == "Template Content"
            mock_exists.assert_called_with(expected_path)
            mock_file.assert_called_with(expected_path, "r", encoding="utf-8")

    def test_load_template_not_found(self):
        """Verify that empty string is returned when template does not exist."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            pm = PromptMaster(templates_dir="/docs")

            content = pm._load_template("missing.md")

            assert content == ""

    def test_wrap_with_method_found(self):
        """Verify wrapping when method template is found."""
        # Use patch.object on the instance method for clearer isolation if needed,
        # or patch the class method. Here patching class method is fine.
        with patch.object(PromptMaster, "_load_template", return_value="Method Template Content"):
            pm = PromptMaster(templates_dir="/docs")

            result = pm.wrap_with_method("User Query", method="react")

            assert "### PROTOCOLE REACT ORIGINEL" in result
            assert "Method Template Content" in result
            assert "### MISSION" in result
            assert "User Query" in result

    def test_wrap_with_method_not_found_or_invalid(self):
        """Verify fallback when method is invalid or template not found."""
        with patch.object(PromptMaster, "_load_template", return_value=""):
            pm = PromptMaster(templates_dir="/docs")

            # Case 1: Method name not in map (returns text directly)
            # "unknown_method" is not in the map, so it returns text immediately without calling _load_template
            # Wait, let's check code: method_path = self.methods_map.get(method.lower())
            # If not found, method_path is None.
            # method_template = self._load_template(method_path) if method_path else ""

            result_unknown = pm.wrap_with_method("Query", method="unknown_method")
            assert result_unknown == "Query"

            # Case 2: Method in map but template load fails (returns text directly)
            # "react" is in map. _load_template returns "" (mocked).
            result_empty = pm.wrap_with_method("Query", method="react")
            assert result_empty == "Query"

    def test_get_expert_injector(self):
        """Verify expert injector retrieval."""
        with patch.object(PromptMaster, "_load_template") as mock_load:
            pm = PromptMaster(templates_dir="/docs")

            # Case 1: Found
            mock_load.return_value = "Expert Prompt"
            result = pm.get_expert_injector("banker")
            assert result == "Expert Prompt"

            # Case 2: Not found (fallback)
            mock_load.return_value = ""
            result_fallback = pm.get_expert_injector("banker")
            assert "Tu es Expert banker" in result_fallback
