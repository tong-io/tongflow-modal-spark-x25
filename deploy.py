"""Modal deploy entry for spark-x25 (Spark-X2.5-4B).

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import os
import modal
from tongflow import deploy
from pathlib import Path
from typing import Any, Optional

_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "XHToken/Spark-X2.5-4B")
MODEL_DIR = f"/models/{REPO_ID}"
_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot

# ── plugin-internal knobs (not ABI fields) ───────────────────────────────────

# Spark-X2.5 enables thinking by default in its chat template. A canvas text
# node wants a direct answer, so thinking is off unless overridden via env.
ENABLE_THINKING = os.environ.get("SPARK_X25_ENABLE_THINKING", "0") == "1"
DEFAULT_MAX_NEW_TOKENS = 1024
# Thinking-mode answers need headroom for the <think> block itself.
THINKING_MAX_NEW_TOKENS = 8192
# Official recommendation: temperature 1.0 / top_p 0.95 / top_k disabled.
TEMPERATURE = 1.0
TOP_P = 0.95

# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .pip_install(
        "tongflow==0.3.2", "fastapi[standard]",
        # The checkpoint ships custom modeling code targeting this exact
        # transformers release (config.json "transformers_version").
        "transformers==4.57.1",
        "accelerate==1.13.0",
    )
)

with image.imports():
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="A10G",
    memory=8192,
    volumes={"/models": volume},
    timeout=1200,
)
class Inference:
    @modal.enter()
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    # ── core ─────────────────────────────────────────────────────────────

    def _generate(
        self,
        prompt: str,
        system: str = "",
        max_new_tokens: Optional[int] = None,
        enable_thinking: bool = ENABLE_THINKING,
    ) -> dict:
        """Chat generation. Returns {"text", "thinking"}."""
        if max_new_tokens is None:
            max_new_tokens = THINKING_MAX_NEW_TOKENS if enable_thinking else DEFAULT_MAX_NEW_TOKENS

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=0,
                do_sample=True,
            )
        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        thinking = ""
        m = re.search(r"<think>(.*?)</think>", response, flags=re.S)
        if m:
            thinking = m.group(1).strip()
            response = response[m.end():]
        elif "</think>" in response:
            # The opening tag is sometimes emitted by the template, not the model.
            head, _, response = response.partition("</think>")
            thinking = head.strip()
        return {"text": response.strip(), "thinking": thinking}

    # ── slots ────────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.GEN_TEXT)
    def gen_text(self, input: GenTextInput) -> GenTextOutput:
        user_message = (
            f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
            "Note: output only the requested answer. Do not include any other content."
        )
        out = self._generate(prompt=user_message)
        return GenTextOutput(success=True, text=str(out.get("text", "")))

    @modal.method()
    @node_slot(NodeSlots.COMBINE_TEXT)
    def combine_text(self, input: CombineTextInput) -> CombineTextOutput:
        joined = "\n\n".join(input.texts)
        user_message = (
            f"{input.userPrompt or ''}\n\nUser input: {joined}\n\n"
            "Note: output only the requested answer. Do not include any other content."
        )
        out = self._generate(prompt=user_message)
        return CombineTextOutput(success=True, text=str(out.get("text", "")))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
