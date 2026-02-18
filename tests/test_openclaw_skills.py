import os
import unittest
from unittest.mock import patch, MagicMock
import httpx
from src.openclaw.skills.basic import web_search

class TestOpenClawSkills(unittest.TestCase):

    @patch("src.openclaw.skills.basic.os.getenv")
    @patch("src.openclaw.skills.basic.httpx.post")
    def test_web_search_with_key_success(self, mock_post, mock_getenv):
        # Mock environment variable
        mock_getenv.return_value = "fake_key"

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Example Result",
                    "url": "http://example.com",
                    "content": "This is an example content."
                }
            ]
        }
        mock_post.return_value = mock_response

        # Execute
        result = web_search("test query")

        # Assertions
        self.assertIn("Search results for 'test query':", result)
        self.assertIn("1. Example Result", result)
        self.assertIn("URL: http://example.com", result)
        self.assertIn("Content: This is an example content.", result)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['api_key'], "fake_key")
        self.assertEqual(kwargs['json']['query'], "test query")

    @patch("src.openclaw.skills.basic.os.getenv")
    def test_web_search_without_key(self, mock_getenv):
        # Mock environment variable returning None
        mock_getenv.return_value = None

        # Execute
        result = web_search("test query")

        # Assertions
        self.assertIn("[MOCK]", result)
        self.assertIn("API Key missing", result)

    @patch("src.openclaw.skills.basic.os.getenv")
    @patch("src.openclaw.skills.basic.httpx.post")
    def test_web_search_api_failure(self, mock_post, mock_getenv):
        # Mock environment variable
        mock_getenv.return_value = "fake_key"

        # Mock API failure
        mock_post.side_effect = httpx.RequestError("Network error", request=MagicMock())

        # Execute
        result = web_search("test query")

        # Assertions
        self.assertIn("Error performing web search", result)
