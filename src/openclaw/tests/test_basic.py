import os
import unittest
from unittest.mock import patch, MagicMock
from openclaw.skills.basic import web_search

class TestWebSearch(unittest.TestCase):

    @patch("openclaw.skills.basic.httpx.Client")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_tavily_key"}, clear=True)
    def test_web_search_tavily(self, mock_client_cls):
        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Test Result 1", "url": "http://example.com/1", "content": "Content 1"},
                {"title": "Test Result 2", "url": "http://example.com/2", "content": "Content 2"}
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        result = web_search("test query")

        # Verify Tavily API was called
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://api.tavily.com/search")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_tavily_key")

        # Verify result formatting
        self.assertIn("Résultats Tavily pour 'test query'", result)
        self.assertIn("Test Result 1", result)
        self.assertIn("http://example.com/1", result)

    @patch("openclaw.skills.basic.httpx.Client")
    @patch.dict(os.environ, {"EXA_API_KEY": "test_exa_key"}, clear=True)
    def test_web_search_exa(self, mock_client_cls):
        # Ensure TAVILY is not set (clear=True handles it but let's be safe)
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Exa Result 1", "url": "http://exa.com/1", "text": "Exa Content 1"}
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        result = web_search("exa query")

        # Verify Exa API was called
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://api.exa.ai/search")
        self.assertEqual(kwargs["headers"]["x-api-key"], "test_exa_key")

        # Verify result formatting
        self.assertIn("Résultats Exa pour 'exa query'", result)
        self.assertIn("Exa Result 1", result)

    @patch.dict(os.environ, {}, clear=True)
    def test_web_search_fallback(self):
        # Ensure no keys
        result = web_search("mock query")
        self.assertIn("[MOCK] Résultats de recherche", result)
        self.assertIn("mock query", result)

    @patch("openclaw.skills.basic.httpx.Client")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"}, clear=True)
    def test_web_search_error(self, mock_client_cls):
        # Mock error
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        result = web_search("error query")
        self.assertIn("Error performing Tavily search", result)
        self.assertIn("Connection error", result)

if __name__ == "__main__":
    unittest.main()
