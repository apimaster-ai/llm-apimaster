"""Tests run against a local mock endpoint — no key, no network, no tokens spent.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

import llm_apimaster as plugin
from llm.cli import cli

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-key":
            self._json(401, {"error": {"message": "invalid api key"}})
            return
        if self.path.startswith("/v1/models"):
            self._json(
                200,
                {
                    "data": [
                        {"id": "gpt-5.5"},
                        {"id": "claude-sonnet-4-6"},
                        {"id": "gpt-image-2"},
                        {"id": "sora-2"},
                        {"id": "text-embedding-3-large"},
                    ]
                },
            )
            return
        if self.path.startswith("/img/"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(TINY_PNG)))
            self.end_headers()
            self.wfile.write(TINY_PNG)
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        # Consume the body first: closing with unread data makes Windows send a TCP RST.
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if self.headers.get("Authorization") != "Bearer test-key":
            self._json(401, {"error": {"message": "invalid api key"}})
            return
        payload = json.loads(raw or b"{}")
        if self.path.startswith("/v1/images/generations"):
            if not payload.get("prompt"):
                self._json(400, {"error": {"message": "prompt is required"}})
                return
            port = self.server.server_port
            self._json(
                200,
                {"created": int(time.time()), "data": [{"url": f"http://127.0.0.1:{port}/img/a.png"}]},
            )
            return
        self._json(404, {"error": {"message": "not found"}})


@pytest.fixture(scope="module")
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture(autouse=True)
def isolated_user_dir(tmp_path, monkeypatch):
    """Keep the real ~/.io.datasette.llm directory out of the tests."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv(plugin.KEY_ENV_VAR, "test-key")
    yield


class TestClassification:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("gpt-5.5", "chat"),
            ("claude-sonnet-4-6", "chat"),
            ("gpt-image-2", "image"),
            ("doubao-seedream-5-0-pro-260628", "image"),
            ("midjourney-v8.2", "image"),
            ("MiniMax-H3", "video"),
            ("sora-2-pro", "video"),
            ("kling-v3-motion-control", "video"),
            ("text-embedding-3-large", "embedding"),
            ("whisper-1", "audio"),
        ],
    )
    def test_kinds(self, model_id, expected):
        assert plugin.classify(model_id) == expected

    def test_capability_heuristics(self):
        assert plugin.describe("claude-sonnet-4-6")["vision"] is True
        assert plugin.describe("claude-sonnet-4-6")["schema"] is True
        assert plugin.describe("some-tiny-local-model")["vision"] is False


class TestRegistration:
    def test_seed_models_register_without_a_cache(self):
        registered = []
        plugin.register_models(lambda model, async_model=None, aliases=None: registered.append(model))
        ids = [model.model_id for model in registered]
        assert "apimaster/gpt-5.5" in ids
        assert all(model_id.startswith("apimaster/") for model_id in ids)

    def test_registration_never_hits_the_network(self, monkeypatch):
        """register_models runs on every llm invocation — a network call there is a bug."""

        def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
            raise AssertionError("register_models must not make HTTP requests")

        monkeypatch.setattr(plugin.httpx, "get", explode)
        monkeypatch.setattr(plugin.httpx, "Client", explode)
        plugin.register_models(lambda model, async_model=None, aliases=None: None)

    def test_only_chat_models_are_registered(self, endpoint, monkeypatch):
        plugin.write_cache(
            {
                "base_url": endpoint,
                "models": [plugin.describe(m) for m in ["gpt-5.5", "gpt-image-2", "sora-2"]],
            }
        )
        registered = []
        plugin.register_models(lambda model, async_model=None, aliases=None: registered.append(model))
        ids = [model.model_id for model in registered]
        assert ids == ["apimaster/gpt-5.5"]
        assert registered[0].api_base == endpoint

    def test_registered_models_carry_the_key_binding(self):
        registered = []
        plugin.register_models(lambda model, async_model=None, aliases=None: registered.append(model))
        assert registered[0].needs_key == "apimaster"
        assert registered[0].key_env_var == "APIMASTER_API_KEY"


class TestRefresh:
    def test_refresh_caches_the_catalog(self, endpoint):
        result = CliRunner().invoke(cli, ["apimaster", "refresh", "--base-url", endpoint])
        assert result.exit_code == 0, result.output
        assert "Cached 5 models" in result.output
        cached = plugin.read_cache()
        assert cached["base_url"] == endpoint
        assert {m["id"] for m in cached["models"]} >= {"gpt-5.5", "sora-2"}

    def test_bad_key_gives_a_useful_message(self, endpoint, monkeypatch):
        monkeypatch.setenv(plugin.KEY_ENV_VAR, "wrong-key")
        result = CliRunner().invoke(cli, ["apimaster", "refresh", "--base-url", endpoint])
        assert result.exit_code != 0
        assert "copied with whitespace" in result.output

    def test_models_command_requires_a_refresh_first(self):
        result = CliRunner().invoke(cli, ["apimaster", "models"])
        assert result.exit_code != 0
        assert "llm apimaster refresh" in result.output

    def test_models_command_filters_by_kind(self, endpoint):
        CliRunner().invoke(cli, ["apimaster", "refresh", "--base-url", endpoint])
        result = CliRunner().invoke(cli, ["apimaster", "models", "--kind", "video", "--json"])
        assert result.exit_code == 0, result.output
        assert [m["id"] for m in json.loads(result.output)] == ["sora-2"]


class TestImage:
    def test_image_downloads_the_result(self, endpoint, tmp_path):
        output = tmp_path / "out.png"
        result = CliRunner().invoke(
            cli,
            ["apimaster", "image", "a red circle", "--base-url", endpoint, "-o", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.read_bytes().startswith(b"\x89PNG")

    def test_upstream_400_is_surfaced(self, endpoint, tmp_path):
        result = CliRunner().invoke(
            cli,
            ["apimaster", "image", "", "--base-url", endpoint, "-o", str(tmp_path / "x.png")],
        )
        assert result.exit_code != 0
        assert "prompt is required" in result.output
