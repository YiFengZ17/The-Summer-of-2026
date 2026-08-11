# CS224n Assignment 4 - LLM Evals

In this assignment, you will evaluate the properties of various different LLMs using
standard benchmarking techniques, implement an LLM-as-a-judge evaluation, and explore
red-teaming approaches.

## Install

First, create a new conda environment with Python 3.10:

```bash
conda create -n cs224n-A4 python=3.10
```

Activate the environment:

```bash
conda activate cs224n-A4
```

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Set up your environment:**

This assignment uses **Google Vertex AI (Gemini)** via your local Google Cloud credentials (e.g., `gcloud auth application-default login`).

Create a local `.env` file (or export env vars in your shell) with:

- `GCP_PROJECT_NAME`: your GCP project ID (e.g., `hellow-world-485923-x5`) (used to initialize the Vertex AI client)
- `STUDENT_EMAIL`: only needed for models **G/H/I** (used to deterministically seed the per-student password)

For an example see `example_usage.py`.

## Self-study providers (no GCP required)

The query client supports three backends selected with `LLM_PROVIDER`:

- `mock` (default): deterministic, zero-cost responses for testing code paths.
- `ollama`: a real local model served by Ollama.
- `gemini`: the original Vertex AI course backend.

Start with the offline mock backend:

```bash
export LLM_PROVIDER=mock
python example_usage.py
```

For local model experiments, install Ollama separately, start its service, then pull
an instruction model appropriate for your hardware. The default is `qwen2.5:3b`:

```bash
ollama pull qwen2.5:3b
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:3b
python example_usage.py
```

Optional settings:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export STUDENT_EMAIL=you@example.com  # required only for G/H/I
```

Local and mock calls report `cost=0.0`. Their scores are useful for learning the
evaluation workflow but are not directly comparable with official Vertex AI results.

**Run the example:**

```bash
python example_usage.py
```
