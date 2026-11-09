"""Serve one locally staged auxiliary model over a bounded HTTP JSON API."""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast

MAX_BODY = 65_536
MAX_TEXT = 4_096
MAX_BATCH = 16
MAX_TOKENS = 512
MAX_HISTORY = 2_048
MAX_FORECAST = 64
MAX_QUANTILES = 9
RETRIEVAL_TASK = "Given a web search query, retrieve relevant passages that answer the query"
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

Payload = dict[str, object]
Inference = Callable[[Payload], Payload]


class RequestError(ValueError):
    """Report a bounded, client-safe HTTP request error."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        """Keep the response status alongside the public error message."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Model:
    """Associate one model ID and endpoint with the loaded inference function."""

    name: str
    endpoint: str
    infer: Inference


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        msg = f"{field} must be a nonempty string of at most {MAX_TEXT} characters"
        raise RequestError(msg)
    return value


def _texts(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_BATCH:
        msg = f"{field} must contain 1 to {MAX_BATCH} strings"
        raise RequestError(msg)
    return [_text(item, field) for item in value]


def _integer(value: object, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        msg = f"{field} must be an integer between 1 and {maximum}"
        raise RequestError(msg)
    return value


def _numbers(value: object, field: str, maximum: int) -> list[float]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        msg = f"{field} must contain 1 to {maximum} numbers"
        raise RequestError(msg)
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (float, int)):
            msg = f"{field} must contain only finite numbers"
            raise RequestError(msg)
        try:
            number = float(item)
        except OverflowError:
            msg = f"{field} contains a number outside the supported range"
            raise RequestError(msg) from None
        if not math.isfinite(number):
            msg = f"{field} must contain only finite numbers"
            raise RequestError(msg)
        numbers.append(number)
    return numbers


def _load_embedding(directory: Path) -> Inference:
    import torch
    from torch.nn.functional import normalize
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(directory, use_safetensors=True, local_files_only=True).eval()

    @torch.inference_mode()
    def infer(payload: Payload) -> Payload:
        value = payload.get("input")
        texts = [_text(value, "input")] if isinstance(value, str) else _texts(value, "input")
        if payload.get("encoding_format", "float") != "float":
            msg = "encoding_format must be float"
            raise RequestError(msg)
        if payload.get("dimensions", 1024) != 1024:
            msg = "dimensions must be 1024"
            raise RequestError(msg)
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        hidden_states = model(**inputs).last_hidden_state[:, -1]
        vectors = normalize(hidden_states.float(), dim=-1).tolist()
        tokens = int(inputs.attention_mask.sum().item())
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return infer


def _load_reranker(directory: Path) -> Inference:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, padding_side="left", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(directory, use_safetensors=True, local_files_only=True).eval()
    label_ids = [tokenizer.convert_tokens_to_ids(label) for label in ("no", "yes")]
    prefix_ids = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
    context_limit = MAX_TOKENS - len(prefix_ids) - len(suffix_ids)

    @torch.inference_mode()
    def infer(payload: Payload) -> Payload:
        query = _text(payload.get("query"), "query")
        documents = _texts(payload.get("documents"), "documents")
        limit = _integer(payload.get("top_n", len(documents)), "top_n", len(documents))
        prompts = [f"<Instruct>: {RETRIEVAL_TASK}\n<Query>: {query}\n<Document>: {document}" for document in documents]
        prompt_ids = tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=context_limit).input_ids
        tokens = [prefix_ids + prompt + suffix_ids for prompt in prompt_ids]
        inputs = tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt")
        logits = model(**inputs).logits[:, -1, label_ids].float()
        scores = logits.softmax(-1)[:, 1].tolist()
        ranking = sorted(range(len(documents)), key=scores.__getitem__, reverse=True)
        return {"results": [{"index": index, "relevance_score": scores[index]} for index in ranking[:limit]]}

    return infer


def _load_forecaster(directory: Path) -> Inference:
    import torch
    from chronos import Chronos2Pipeline

    model = Chronos2Pipeline.from_pretrained(directory, device_map="cpu", local_files_only=True)

    @torch.inference_mode()
    def infer(payload: Payload) -> Payload:
        history = _numbers(payload.get("series"), "series", MAX_HISTORY)
        if any(abs(item) > 1e12 for item in history):
            msg = "series must be scaled to values between -1e12 and 1e12"
            raise RequestError(msg)
        horizon = _integer(payload.get("prediction_length"), "prediction_length", MAX_FORECAST)
        levels = _numbers(payload.get("quantile_levels", [0.1, 0.5, 0.9]), "quantile_levels", MAX_QUANTILES)
        if levels != sorted(set(levels)) or not all(0 < level < 1 for level in levels):
            msg = "quantile_levels must be unique, ascending, and strictly between 0 and 1"
            raise RequestError(msg)
        quantiles, points = model.predict_quantiles(
            [torch.tensor(history, dtype=torch.float32)], prediction_length=horizon, quantile_levels=levels
        )
        values = quantiles[0][0].transpose(0, 1).tolist()
        return {
            "point": points[0][0].tolist(),
            "quantiles": [{"level": level, "values": series} for level, series in zip(levels, values, strict=True)],
        }

    return infer


def _load_model(mode: str, directory: Path, threads: int) -> Model:
    import torch

    torch.set_num_threads(threads)
    name, endpoint, loader = {
        "embedding": ("qwen3-embedding-0.6b", "/v1/embeddings", _load_embedding),
        "reranker": ("qwen3-reranker-0.6b", "/v1/rerank", _load_reranker),
        "chronos": ("chronos-2", "/v1/forecast", _load_forecaster),
    }[mode]
    return Model(name, endpoint, loader(directory))


def _get_token(file: str | None) -> bytes:
    token = os.environ.get("MODEL_API_KEY")
    if file is not None and token is not None:
        msg = "configure either a model API key file or MODEL_API_KEY"
        raise ValueError(msg)
    if file is not None:
        token = Path(file).read_text(encoding="utf-8").rstrip()
    if token is None:
        return b""
    if not token or not token.isascii() or any(not 33 <= ord(char) <= 126 for char in token):
        msg = "invalid model API key"
        raise ValueError(msg)
    return token.encode("ascii")


class InferenceServer(HTTPServer):
    """Serve model inference serially with a bounded listen backlog."""

    request_queue_size = 8

    def __init__(self, address: tuple[str, int], model: Model, token: bytes) -> None:
        """Bind an address only after the model and its authentication are ready."""
        self.model = model
        self.token = token
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    """Dispatch authenticated HTTP requests and emit JSON responses."""

    server: InferenceServer
    timeout = 10

    def log_message(self, format: str, *args: object) -> None:
        """Omit request paths and headers from service logs."""

    def do_GET(self) -> None:
        """Return readiness or an authenticated model inventory."""
        if self.path == "/health":
            self._reply(HTTPStatus.OK, {"status": "ok"})
        elif self._authorized():
            if self.path != "/v1/models":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            model = {"id": self.server.model.name, "object": "model", "owned_by": "local"}
            self._reply(HTTPStatus.OK, {"object": "list", "data": [model]})

    def do_POST(self) -> None:
        """Validate one bounded request and execute local inference."""
        if not self._authorized():
            return
        if self.path != self.server.model.endpoint:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_payload()
            if payload.get("model", self.server.model.name) != self.server.model.name:
                msg = "model is not served by this endpoint"
                raise RequestError(msg, HTTPStatus.NOT_FOUND)
            result = self.server.model.infer(payload)
            self._reply(HTTPStatus.OK, {"model": self.server.model.name, **result})
        except RequestError as error:
            self._error(error.status, str(error))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            logging.error("Inference error in %s (%s)", self.server.model.name, type(error).__name__)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Replace the standard HTML response with a generic JSON error."""
        status = HTTPStatus(code)
        self._error(status, status.phrase)

    def _authorized(self) -> bool:
        if not self.server.token:
            return True
        headers = self.headers.get_all("Authorization", [])
        if len(headers) == 1:
            scheme, _, provided = headers[0].partition(" ")
            if scheme.lower() == "bearer" and hmac.compare_digest(
                provided.lstrip(" ").encode("utf-8"), self.server.token
            ):
                return True
        self._error(HTTPStatus.UNAUTHORIZED, "Valid Bearer authentication is required")
        return False

    def _read_payload(self) -> Payload:
        if self.headers.get("Transfer-Encoding") is not None:
            msg = "Transfer-Encoding is not supported"
            raise RequestError(msg)
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 8:
            msg = "A valid Content-Length is required"
            raise RequestError(msg, HTTPStatus.LENGTH_REQUIRED)
        size = int(lengths[0])
        if size > MAX_BODY:
            msg = f"Request body exceeds {MAX_BODY} bytes"
            raise RequestError(msg, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if self.headers.get_content_type() != "application/json":
            msg = "Content-Type must be application/json"
            raise RequestError(msg, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            body = self.rfile.read(size)
        except TimeoutError:
            msg = "Request body timed out"
            raise RequestError(msg, HTTPStatus.REQUEST_TIMEOUT) from None
        if len(body) != size:
            msg = "Incomplete request body"
            raise RequestError(msg)
        try:
            value = json.loads(body)
        except (ValueError, RecursionError):
            msg = "Request body must be valid JSON"
            raise RequestError(msg) from None
        if not isinstance(value, dict):
            msg = "Request body must be a JSON object"
            raise RequestError(msg)
        return cast("Payload", value)

    def _error(self, status: HTTPStatus, message: str) -> None:
        kind = "server_error" if status >= 500 else "invalid_request_error"
        self._reply(status, {"error": {"message": message, "type": kind}})

    def _reply(self, status: HTTPStatus, payload: Payload) -> None:
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Load a local CPU model and serve it on the configured IPv4 address."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("embedding", "reranker", "chronos"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--api-key-file", default=os.environ.get("MODEL_API_KEY_FILE"))
    args = parser.parse_args()
    try:
        host = ipaddress.IPv4Address(args.host)
        token = _get_token(args.api_key_file)
    except (ValueError, OSError):
        parser.error("invalid IPv4 host or model API key configuration")
    if not host.is_loopback and not token:
        parser.error("a model API key is required for a non-loopback host")
    if not 1 <= args.threads <= 32 or not 1 <= args.port <= 65_535:
        parser.error("invalid thread count or port")
    directory = args.model_dir.resolve()
    if not all((directory / name).is_file() for name in ("config.json", "model.safetensors")):
        parser.error("model-dir must contain the local config.json and model.safetensors")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    model = _load_model(args.mode, directory, args.threads)
    with InferenceServer((str(host), args.port), model, token) as server:
        print(f"Ready: {model.name} at {host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
