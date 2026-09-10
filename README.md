# tongflow-modal-spark-x25

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text generation with **Spark-X2.5-4B** (`XHToken/Spark-X2.5-4B`, Apache-2.0) — iFlytek's on-device general LLM with a native 1M-token context window and 200+ languages — running on an A10G via [Modal](https://modal.com).

The model is **text-only** (no image / audio / video input); use `tongflow-modal-qwen38` or `tongflow-modal-gemma4` for multimodal understanding.

## Capabilities

- **Text generation** (`gen-text`) — generate or rewrite text from a prompt.
- **Combine text** (`combine-text`) — merge several text inputs under one instruction.

Thinking mode is **off** by default (the model's chat template enables it by default; the plugin passes `enable_thinking=False`). Set the `SPARK_X25_ENABLE_THINKING=1` env var on the Modal app to turn it on. Sampling follows the official recommendation (`temperature 1.0 / top_p 0.95 / top_k off`).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The `XHToken/Spark-X2.5-4B` weights (~8 GB, BF16) are public — no Hugging Face token required; fetch them into the `models` volume once with `modal run download.py::download`.
