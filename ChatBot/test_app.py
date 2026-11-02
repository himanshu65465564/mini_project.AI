import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
import threading
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

import app


class ParseMessagesTests(unittest.TestCase):
    def test_accepts_user_and_assistant_history(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        self.assertEqual(app.parse_messages({"messages": messages}), messages)

    def test_rejects_assistant_as_last_message(self):
        with self.assertRaisesRegex(ValueError, "last message must be from the user"):
            app.parse_messages({"messages": [{"role": "assistant", "content": "Hello"}]})

    def test_rejects_non_alternating_roles(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Again"},
        ]
        with self.assertRaisesRegex(ValueError, "roles must alternate"):
            app.parse_messages({"messages": messages})

    def test_rejects_oversized_message(self):
        with self.assertRaisesRegex(ValueError, "1 to 8000 characters"):
            app.parse_messages({"messages": [{"role": "user", "content": "x" * 8001}]})


class LocalDevelopmentCorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.ChatHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def request(self, method, path, headers=None, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response_body = response.read()
        result = response.status, response.headers, response_body
        connection.close()
        return result

    def test_allows_live_server_loopback_origins(self):
        self.assertTrue(app.is_local_development_origin("http://127.0.0.1:5500"))
        self.assertTrue(app.is_local_development_origin("http://localhost:5501"))
        self.assertTrue(app.is_local_development_origin("http://[::1]:5500"))

    def test_rejects_non_local_or_non_http_origins(self):
        self.assertFalse(app.is_local_development_origin("https://localhost:5500"))
        self.assertFalse(app.is_local_development_origin("http://example.com:5500"))
        self.assertFalse(app.is_local_development_origin("null"))

    def test_handles_live_server_preflight(self):
        status, headers, body = self.request(
            "OPTIONS",
            "/api/chat",
            {
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:5500")
        self.assertIn("POST", headers["Access-Control-Allow-Methods"])
        self.assertEqual(body, b"")

    def test_api_errors_are_readable_from_live_server(self):
        with patch.dict("os.environ", {"HF_TOKEN": ""}):
            status, headers, body = self.request(
                "POST",
                "/api/chat",
                {
                    "Origin": "http://localhost:5500",
                    "Content-Type": "application/json",
                },
                json.dumps({"messages": [{"role": "user", "content": "Hello"}]}),
            )
        self.assertEqual(status, 503)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:5500")
        self.assertIn("HF_TOKEN", json.loads(body)["error"])

    def test_payment_required_is_returned_as_402_with_billing_guidance(self):
        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}), patch(
            "app.generate_reply",
            side_effect=app.ProviderError("Hugging Face needs credits.", status_code=402),
        ):
            status, headers, body = self.request(
                "POST",
                "/api/chat",
                {"Content-Type": "application/json"},
                json.dumps({"messages": [{"role": "user", "content": "Hello"}]}),
            )
        self.assertEqual(status, 402)
        self.assertIn("Hugging Face needs credits", json.loads(body)["error"])


class HuggingFaceRequestTests(unittest.TestCase):
    @patch("app.urlopen")
    def test_explains_payment_required(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            app.API_URL,
            402,
            "Payment Required",
            {},
            BytesIO(b'{"error":{"message":"Payment required"}}'),
        )

        with self.assertRaises(app.ProviderError) as context:
            app.generate_reply([{"role": "user", "content": "Hello"}], "test-token")

        self.assertEqual(context.exception.status_code, 402)
        self.assertIn("no available Inference Providers credits", str(context.exception))
        self.assertIn("settings/billing", str(context.exception))

    @patch("app.urlopen")
    def test_sends_conversation_and_returns_text(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": "A reply."}
                    }]
                }).encode("utf-8")

        mock_urlopen.return_value = FakeResponse()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Again"},
        ]

        reply = app.generate_reply(messages, "test-token-not-real")

        self.assertEqual(reply, "A reply.")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            [content["role"] for content in payload["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(payload["model"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertNotIn("test-token-not-real", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token-not-real")
        self.assertEqual(request.full_url, app.API_URL)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 60)

    @patch("app.urlopen")
    def test_explains_unauthorized_token(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            app.API_URL,
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error":{"message":"Invalid token"}}'),
        )

        with self.assertRaisesRegex(RuntimeError, "Hugging Face rejected the token \\(HTTP 401\\)"):
            app.generate_reply([{"role": "user", "content": "Hello"}], "invalid-test-token")

    @patch("app.urlopen")
    def test_explains_unsupported_chat_method(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            app.API_URL,
            405,
            "Method Not Allowed",
            {},
            BytesIO(b""),
        )

        with self.assertRaisesRegex(RuntimeError, "chat-completions method \\(HTTP 405\\)"):
            app.generate_reply([{"role": "user", "content": "Hello"}], "test-token")

    @patch("app.urlopen")
    def test_accepts_content_block_list(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": [{"type": "text", "text": "A reply."}]}
                    }]
                }).encode("utf-8")

        mock_urlopen.return_value = FakeResponse()
        self.assertEqual(
            app.generate_reply([{"role": "user", "content": "Hello"}], "test-token"),
            "A reply.",
        )

    @patch("app.urlopen")
    def test_reports_empty_provider_body_without_json_error(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b""

        mock_urlopen.return_value = FakeResponse()
        with self.assertRaisesRegex(RuntimeError, "unreadable response"):
            app.generate_reply([{"role": "user", "content": "Hello"}], "test-token")


if __name__ == "__main__":
    unittest.main()
