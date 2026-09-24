"""llm plugin for APIMaster and other OpenAI-compatible gateways.

Design notes, because they are easy to get wrong in an `llm` plugin:

* `register_models` runs on **every** `llm` invocation. It must never touch the network,
  or the whole CLI becomes slow and breaks offline. Models come from a cache file that
  `llm apimaster refresh` writes; a small seed list keeps the plugin useful before the
  first refresh.
* Capability flags (vision, schema, tools) are heuristics: an OpenAI-compatible catalog
  does not report them. The cache is plain JSON so a user can correct it by hand.
* HTTP reuses whichever httpx `llm` already ships, so this plugin adds no dependency of
  its own. llm 0.35 pins `httpx2`; older releases pinned plain `httpx`. Both expose the
  same API, so a two-line shim covers each without pinning either ourselves.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import click
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

try:  # llm >= 0.35
    import httpx2 as httpx
except ImportError:  # older llm releases
    import httpx

DEFAULT_BASE_URL = "https://apimaster.ai/v1"
KEY_ALIAS = "apimaster"
KEY_ENV_VAR = "APIMASTER_API_KEY"
CACHE_FILENAME = "apimaster_models.json"

# Used until `llm apimaster refresh` caches the live catalog. Kept deliberately short:
# a long guessed list produces confusing 400s for ids that do not exist.
SEED_MODELS = ["gpt-5.5", "claude-sonnet-4-6"]

MEDIA_RE = re.compile(
    r"(image|video|sora|seedance|kling|banana|seedream|dall|imagen|veo|whisper|tts|embedding|rerank)",
    re.I,
)
VISION_RE = re.compile(r"(gpt-[5-9]|gpt-4o|claude|gemini|qwen-vl|glm-4v|pixtral|llava)", re.I)
STRUCTURED_RE = re.compile(r"(gpt-[4-9]|o[1-9]|claude|gemini|deepseek|glm|qwen|kimi)", re.I)


# --------------------------------------------------------------------------- cache


def cache_path() -> Path:
    return llm.user_dir() / CACHE_FILENAME


def read_cache() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_cache(payload: Dict[str, Any]) -> Path:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def classify(model_id: str) -> str:
    lowered = model_id.lower()
    if re.search(r"(sora|video|kling|seedance|minimax-h|veo|wan)", lowered):
        return "video"
    if re.search(r"(image|banana|seedream|flux|dall|imagen|midjourney|mj_)", lowered):
        return "image"
    if re.search(r"(embedding|embed|rerank|bge|gte)", lowered):
        return "embedding"
    if re.search(r"(whisper|tts|audio|speech)", lowered):
        return "audio"
    return "chat"


def describe(model_id: str) -> Dict[str, Any]:
    """Heuristic capability flags. Wrong guesses are correctable in the cache file."""
    return {
        "id": model_id,
        "kind": classify(model_id),
        "vision": bool(VISION_RE.search(model_id)),
        "schema": bool(STRUCTURED_RE.search(model_id)),
        "tools": bool(STRUCTURED_RE.search(model_id)),
    }


# ----------------------------------------------------------------------- http calls


def base_url(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    cached = read_cache() or {}
    return str(cached.get("base_url") or DEFAULT_BASE_URL).rstrip("/")


def api_key(explicit: Optional[str] = None) -> str:
    key = llm.get_key(explicit, KEY_ALIAS, KEY_ENV_VAR)
    if not key:
        raise click.ClickException(
            "No API key. Set one with:\n"
            f"  llm keys set {KEY_ALIAS}\n"
            f"or export {KEY_ENV_VAR}=...\n"
            "Get a key: https://apimaster.ai/docs/getting-started/api-key"
        )
    return key.strip()


def explain(response: httpx.Response) -> str:
    hints = {
        400: "Bad request: check the model id and parameters.",
        401: "Unauthorized: the key is wrong, expired, or was copied with whitespace.",
        402: "Insufficient balance.",
        404: "Not found: the base URL should end with /v1.",
        408: "Generation timed out: lower the resolution, or use --async.",
        429: "Rate limited.",
    }
    hint = hints.get(response.status_code, "Request failed.")
    return f"HTTP {response.status_code}: {hint} {response.text[:200]}"


def fetch_models(key: str, url: str, timeout: float = 30.0) -> List[str]:
    response = httpx.get(
        f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout
    )
    if response.status_code != 200:
        raise click.ClickException(explain(response))
    return [item["id"] for item in response.json().get("data", []) if item.get("id")]


# --------------------------------------------------------------------------- models


class ApiMasterChat(Chat):
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR

    def __str__(self) -> str:
        return f"APIMaster: {self.model_id}"


class ApiMasterAsyncChat(AsyncChat):
    needs_key = KEY_ALIAS
    key_env_var = KEY_ENV_VAR

    def __str__(self) -> str:
        return f"APIMaster: {self.model_id}"


def chat_models_from_cache() -> List[Dict[str, Any]]:
    cached = read_cache()
    if cached and cached.get("models"):
        return [m for m in cached["models"] if m.get("kind") == "chat"]
    return [describe(model_id) for model_id in SEED_MODELS]


@llm.hookimpl
def register_models(register) -> None:
    url = base_url()
    for spec in chat_models_from_cache():
        model_id = spec["id"]
        shared = {
            "model_name": model_id,
            "api_base": url,
            "vision": bool(spec.get("vision")),
            "supports_schema": bool(spec.get("schema")),
            "supports_tools": bool(spec.get("tools")),
        }
        register(
            ApiMasterChat(model_id=f"apimaster/{model_id}", **shared),
            ApiMasterAsyncChat(model_id=f"apimaster/{model_id}", **shared),
        )


# ------------------------------------------------------------------------- commands


def _poll_image(client: httpx.Client, url: str, key: str, task_id: str, model: str) -> List[str]:
    time.sleep(12)
    for _ in range(200):
        response = client.get(
            f"{url}/tasks/{task_id}",
            params={"model": model},
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if response.status_code != 200:
            raise click.ClickException(explain(response))
        payload = response.json().get("data", response.json())
        status = payload.get("status")
        if status == "completed":
            urls: List[str] = []
            for image in (payload.get("result") or {}).get("images", []):
                value = image.get("url")
                urls.extend(value if isinstance(value, list) else [value] if value else [])
            return urls
        if status in ("failed", "error", "cancelled"):
            raise click.ClickException(f"Task {status}: {json.dumps(payload)[:300]}")
        time.sleep(4)
    raise click.ClickException("Gave up polling the image task.")


def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def _save(client: httpx.Client, media_url: str, key: str, path: Path, url: str) -> int:
    # Compare hosts, not URL prefixes: generated images come back from the same domain
    # but outside /v1 (/imgs/x.png), while upstream CDNs reject an unknown Authorization
    # header. Prefix matching silently drops the token on the first case.
    headers = {"Authorization": f"Bearer {key}"} if _same_host(media_url, url) else {}
    response = client.get(media_url, headers=headers, timeout=600, follow_redirects=True)
    if response.status_code != 200:
        raise click.ClickException(f"Download failed: HTTP {response.status_code}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return len(response.content)


@llm.hookimpl
def register_commands(cli) -> None:
    @cli.group()
    def apimaster():
        "Commands for APIMaster and other OpenAI-compatible gateways"

    @apimaster.command()
    @click.option("--key", help="API key to use for this call")
    @click.option("--base-url", "url", help=f"Override the base URL (default {DEFAULT_BASE_URL})")
    def refresh(key, url):
        """Fetch the live model catalog and cache it.

        Run this after installing, and whenever a model id stops working, because aggregator
        catalogs change.
        """
        resolved_url = base_url(url)
        ids = fetch_models(api_key(key), resolved_url)
        payload = {
            "base_url": resolved_url,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": [describe(model_id) for model_id in ids],
        }
        path = write_cache(payload)
        kinds: Dict[str, int] = {}
        for model in payload["models"]:
            kinds[model["kind"]] = kinds.get(model["kind"], 0) + 1
        summary = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        click.echo(f"Cached {len(ids)} models ({summary}) to {path}")
        click.echo("Chat models are now available as apimaster/<id>. Try: llm models | grep apimaster")

    @apimaster.command(name="models")
    @click.option("--kind", type=click.Choice(["chat", "image", "video", "embedding", "audio"]))
    @click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
    def list_models(kind, as_json):
        """List cached models, including the image and video ones llm cannot run directly."""
        cached = read_cache()
        if not cached:
            raise click.ClickException("No cache yet. Run: llm apimaster refresh")
        models = cached["models"]
        if kind:
            models = [m for m in models if m["kind"] == kind]
        if as_json:
            click.echo(json.dumps(models, indent=2))
            return
        click.echo(f"{len(models)} models (fetched {cached.get('fetched_at', '?')})")
        for model in models:
            flags = " ".join(
                name for name in ("vision", "schema", "tools") if model.get(name)
            )
            prefix = "apimaster/" if model["kind"] == "chat" else " " * 10
            click.echo(f"  {prefix}{model['id']:<34} {model['kind']:<10} {flags}")

    @apimaster.command()
    @click.argument("prompt")
    @click.option("-m", "--model", default="gpt-image-2", help="Image model id")
    @click.option("--size", help="1:1, 16:9, 9:16, 4:3 ... or pixels like 1881x836")
    @click.option("--resolution", type=click.Choice(["1k", "2k", "4k"]), default="1k")
    @click.option("--ref", multiple=True, help="Reference image URL for image-to-image")
    @click.option("-o", "--output", default="image.png", help="Where to write the result")
    @click.option("--async", "is_async", is_flag=True, help="Submit and poll (use for 2k/4k)")
    @click.option("--key", help="API key to use for this call")
    @click.option("--base-url", "url", help="Override the base URL")
    def image(prompt, model, size, resolution, ref, output, is_async, key, url):
        """Generate an image. `llm` has no image models, so this is a plain command."""
        resolved_url = base_url(url)
        resolved_key = api_key(key)
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "resolution": resolution}
        if size:
            body["size"] = size
        if ref:
            body["image_urls"] = list(ref)

        # 1k renders can take minutes; 4k can take ten. A short timeout aborts jobs you
        # have already been charged for.
        timeout = {"1k": 200.0, "2k": 320.0, "4k": 620.0}[resolution]
        headers = {"Authorization": f"Bearer {resolved_key}"}

        with httpx.Client() as client:
            if is_async:
                response = client.post(
                    f"{resolved_url}/images/generations/async", json=body, headers=headers, timeout=60
                )
                if response.status_code != 200:
                    raise click.ClickException(explain(response))
                items = response.json().get("data") or [{}]
                task_id = items[0].get("task_id")
                if not task_id:
                    raise click.ClickException(f"No task_id: {response.text[:200]}")
                click.echo(f"task {task_id}", err=True)
                urls = _poll_image(client, resolved_url, resolved_key, task_id, model)
            else:
                response = client.post(
                    f"{resolved_url}/images/generations", json=body, headers=headers, timeout=timeout
                )
                if response.status_code == 408:
                    raise click.ClickException(
                        "Generation timed out. Re-run the same command with --async."
                    )
                if response.status_code != 200:
                    raise click.ClickException(explain(response))
                urls = [item["url"] for item in response.json().get("data", []) if item.get("url")]

            if not urls:
                raise click.ClickException("The endpoint returned no image URL.")
            target = Path(output)
            for index, media_url in enumerate(urls):
                path = target if index == 0 else target.with_name(f"{target.stem}-{index + 1}{target.suffix}")
                written = _save(client, media_url, resolved_key, path, resolved_url)
                click.echo(f"{path} ({written // 1024} KB)")

    @apimaster.command()
    @click.argument("prompt")
    @click.option("-m", "--model", default="sora-2", help="Video model id")
    @click.option("--duration", type=int, default=4)
    @click.option("--resolution", default="720p")
    @click.option("--aspect", default="16:9", type=click.Choice(["16:9", "9:16"]))
    @click.option("--ref", help="Reference image URL for image-to-video")
    @click.option("-o", "--output", default="video.mp4")
    @click.option("--key", help="API key to use for this call")
    @click.option("--base-url", "url", help="Override the base URL")
    def video(prompt, model, duration, resolution, aspect, ref, output, key, url):
        """Generate a video, wait for the job, download the MP4."""
        resolved_url = base_url(url)
        resolved_key = api_key(key)
        headers = {"Authorization": f"Bearer {resolved_key}"}
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            # Always explicit: a portrait reference with no aspect comes back 16:9.
            "aspect_ratio": aspect,
        }
        if ref:
            body["image_urls"] = [ref]

        with httpx.Client() as client:
            response = client.post(
                f"{resolved_url}/videos/generations", json=body, headers=headers, timeout=60
            )
            if response.status_code != 200:
                raise click.ClickException(explain(response))
            payload = response.json()
            items = payload.get("data") or [{}]
            task_id = items[0].get("task_id") or payload.get("id")
            if not task_id:
                raise click.ClickException(f"No task id: {response.text[:200]}")
            click.echo(f"task {task_id} - typically 1-3 minutes", err=True)

            time.sleep(15)
            for _ in range(240):
                status_response = client.get(
                    f"{resolved_url}/videos/{task_id}", headers=headers, timeout=30
                )
                if status_response.status_code != 200:
                    raise click.ClickException(explain(status_response))
                status_payload = status_response.json()
                status = status_payload.get("status")
                if status == "completed":
                    media_url = status_payload.get("url") or f"{resolved_url}/videos/{task_id}/content"
                    written = _save(client, media_url, resolved_key, Path(output), resolved_url)
                    click.echo(f"{output} ({written / 1e6:.1f} MB)")
                    return
                if status in ("failed", "error", "cancelled"):
                    raise click.ClickException(f"Task {status}: {json.dumps(status_payload)[:300]}")
                time.sleep(4)
            raise click.ClickException("Gave up polling the video task.")
