# llm-apimaster

[![PyPI](https://img.shields.io/pypi/v/llm-apimaster.svg)](https://pypi.org/project/llm-apimaster/)
[![Tests](https://github.com/apimaster-ai/llm-apimaster/actions/workflows/test.yml/badge.svg)](https://github.com/apimaster-ai/llm-apimaster/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding [APIMaster](https://apimaster.ai/docs)
and any other OpenAI-compatible gateway — plus the image and video generation that `llm`
has no model type for.

## Install

```bash
llm install llm-apimaster
```

## Configure

```bash
llm keys set apimaster
# <paste your key>

llm apimaster refresh
```

`refresh` reads the live catalog and caches it. Chat models then show up as
`apimaster/<id>`:

```bash
llm models | grep apimaster
# APIMaster: apimaster/gpt-5.5
# APIMaster: apimaster/claude-sonnet-4-6
```

Run `refresh` again whenever a model id stops working — aggregator catalogs change, and
a removed id comes back as a 400 that looks like a plugin bug.

## Use

```bash
llm -m apimaster/gpt-5.5 "Explain time-to-first-token in one paragraph"

llm -m apimaster/claude-sonnet-4-6 "Summarise this" < notes.md

# conversations, schemas, tools and everything else llm does, unchanged
llm -m apimaster/gpt-5.5 --schema 'name, age int' "Invent a character"
```

### Images

`llm` has no image models, so this ships as a command:

```bash
llm apimaster image "a corgi astronaut on the moon" --size 16:9 -o corgi.png
llm apimaster image "replace the background with a desert sunset" --ref https://example.com/a.png
llm apimaster image "a detailed matte painting" --resolution 4k --async -o matte.png
```

Use `--async` for 2k and 4k. Synchronous generation at those sizes can exceed the
gateway's own timeout and return a 408; the command tells you to switch when that happens.

### Video

```bash
llm apimaster video "a waterfall forming a rainbow, cinematic" --duration 4 -o clip.mp4
llm apimaster video "slow push-in, hair in the breeze" --ref https://example.com/face.jpg --aspect 9:16
```

Always pass `--aspect` for image-to-video: a portrait reference with no aspect is treated
as 16:9 by the gateway and comes back letterboxed.

### Seeing everything the endpoint serves

```bash
llm apimaster models                 # all cached models with capability flags
llm apimaster models --kind image
llm apimaster models --kind video --json
```

## Another gateway

Nothing here is specific to one vendor:

```bash
llm apimaster refresh --base-url https://your-gateway/v1
llm apimaster image "..." --base-url https://your-gateway/v1
```

The base URL from the last `refresh` is remembered in the cache and used for model
registration.

## How it decides capabilities

An OpenAI-compatible `/models` response says nothing about vision, schema or tool
support, so the plugin guesses from the model id and writes the result into the cache:

```json
{ "id": "claude-sonnet-4-6", "kind": "chat", "vision": true, "schema": true, "tools": true }
```

If a guess is wrong, edit the cache file directly — `llm apimaster models` prints its
location, and the format is stable.

## Design notes

Two things worth knowing if you read the source:

- **`register_models` never makes a network call.** It runs on every single `llm`
  invocation; a request there would slow the whole CLI down and break it offline. The
  catalog comes from the cache, with a two-model seed list so the plugin works before the
  first `refresh`.
- **No new dependencies.** It reuses whichever httpx `llm` already ships (0.35 pins
  `httpx2`, older releases pin `httpx`) behind a two-line import shim.

## Development

```bash
pip install -e '.[test]'
python -m pytest tests/ -q
```

Tests run against a mock HTTP server built into the test file — no key, no network,
nothing spent.

## License

MIT
