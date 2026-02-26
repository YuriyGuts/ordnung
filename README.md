# Ordnung

An agentic AI-powered file organizer that looks at a messy directory and sorts everything into neatly categorized directories according to file contents.

[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

The primary goal of this project is to build an LLM agent completely from scratch: no LangChain / CrewAI, no cloud APIs, no vibecoding.
Just a plain agentic loop on top of an OpenAI-compatible API, running fully locally with [Ollama](https://ollama.com/).

It is a learning exercise in understanding how AI agents actually work under the hood.

The project was live-coded completely by hand across two [«Шо по коду?»](https://www.youtube.com/@shopokodu) podcast episodes ([1](https://www.youtube.com/watch?v=lJaUUX38eZY), [2](https://www.youtube.com/watch?v=AKxkyt1aeuk)), with some minor offline cleanup afterward.

## How It Works

You point it at a directory. The agent analyzes the files (including their contents when filenames are ambiguous), decides on categories, creates subdirectories, and moves everything into place.
It asks for your approval before each action.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```shell
uv sync
```

## Running Locally with Ollama

Install [Ollama](https://ollama.com/) and pull a model:

```shell
ollama pull gpt-oss:20b
```

Then run:

```shell
uv run ordnung ~/Desktop
```

This uses the defaults (Ollama at `localhost:11434`, model `gpt-oss:20b`).

## Running with a Cloud LLM

Any OpenAI-compatible API works, as long as it supports the [Responses API](https://developers.openai.com/api/reference/responses/overview).

The API key is read from the `OPENAI_API_KEY` environment variable, or can be passed explicitly with `--llm-api-key`.

For example, with GPT-5.2:

```shell
export OPENAI_API_KEY=sk-...

uv run ordnung ~/Downloads \
    --llm-api-base-url https://api.openai.com/v1 \
    --llm-name gpt-5.2
```

## Developer Tools

```shell
make lint           # Format check + linting + type checking
make lint-fix       # Auto-fix formatting and linter issues
make test           # Run tests
make test-coverage  # Run tests with coverage report
make check          # Lint + test
```

## License

The source code is licensed under the [BSD-3-Clause License](LICENSE).
