import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Mock eva_core.main to prevent heavy imports and initialization
sys.modules["eva_core.main"] = MagicMock()

# Now we can import PromptMaster safely
from eva_core.services.prompt_master import PromptMaster

class TestPromptMaster(unittest.TestCase):
    def setUp(self):
        self.pm = PromptMaster(templates_dir="dummy/dir")

    def test_init(self):
        self.assertEqual(self.pm.templates_dir, "dummy/dir")
        self.assertIn("react", self.pm.methods_map)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Mock Template Content")
    def test_load_template(self, mock_file, mock_exists):
        mock_exists.return_value = True

        content = self.pm._load_template("some/path.md")

        self.assertEqual(content, "Mock Template Content")
        mock_exists.assert_called_with("dummy/dir/some/path.md")
        mock_file.assert_called_with("dummy/dir/some/path.md", "r", encoding="utf-8")

    @patch("os.path.exists")
    def test_load_template_not_found(self, mock_exists):
        mock_exists.return_value = False

        content = self.pm._load_template("some/path.md")
        self.assertEqual(content, "")

    @patch("os.path.exists")
    def test_load_template_error(self, mock_exists):
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=IOError("Boom")):
            content = self.pm._load_template("some/path.md")
            self.assertEqual(content, "")

    @patch.object(PromptMaster, "_load_template")
    def test_wrap_with_method(self, mock_load):
        mock_load.return_value = "TEMPLATE_CONTENT"

        result = self.pm.wrap_with_method("My Task", method="react")

        self.assertIn("### PROTOCOLE REACT ORIGINEL", result)
        self.assertIn("TEMPLATE_CONTENT", result)
        self.assertIn("My Task", result)

        # Verify it tried to load the correct path
        expected_path = self.pm.methods_map["react"]
        mock_load.assert_called_with(expected_path)

    @patch.object(PromptMaster, "_load_template")
    def test_wrap_with_method_unknown(self, mock_load):
        # If method is unknown, _load_template is not called
        result = self.pm.wrap_with_method("My Task", method="unknown_method")

        self.assertEqual(result, "My Task")
        mock_load.assert_not_called()

    @patch.object(PromptMaster, "_load_template")
    def test_wrap_with_method_template_not_found(self, mock_load):
        mock_load.return_value = ""

        result = self.pm.wrap_with_method("My Task", method="react")

        # If template not found, it returns original text
        self.assertEqual(result, "My Task")

    @patch.object(PromptMaster, "_load_template")
    def test_get_expert_injector(self, mock_load):
        mock_load.return_value = "EXPERT_TEMPLATE"

        result = self.pm.get_expert_injector("banker")
        self.assertEqual(result, "EXPERT_TEMPLATE")

    @patch.object(PromptMaster, "_load_template")
    def test_get_expert_injector_fallback(self, mock_load):
        mock_load.return_value = ""

        result = self.pm.get_expert_injector("unknown_expert")
        self.assertIn("Expert unknown_expert", result)
