"""Recover K2-Horizon learned routing biases as a standalone BF16 safetensors overlay.

Public, pinned Hugging Face endpoints only. No credentials or implicit model
directory writes; no full-shard downloads. BF16 bytes are copied as-is.
Run from anywhere: python extract_k2_routing_bias_overlay.py --output-dir <directory>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import stat
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCE_REPO = "IFM/K2-Horizon-MoVA-36B-A4B"
SOURCE_REV = "7730b92d1b574e04663b04023d5d6fa83475432f"
QUANT_REPO = "vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3"
QUANT_REV = "c88277ce7f6b90b723f79b5188c0f1b951732099"
QUANT_SUBDIR = "6.50bpw/"
PATTERN = re.compile(r"^model\.layers\.(\d+)\.(self_attn\.v_router|mlp\.gate)\.bias$")
NONE_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.None$")
SIZES = {"self_attn.v_router": 64, "mlp.gate": 100}
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_REDIRECTS = 5
EXPECTED_OVERLAY_SHA256 = "8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c"


def url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def session() -> requests.Session:
    client = requests.Session()
    client.trust_env = False  # Never import .netrc auth (or proxy settings) for public endpoints.
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def _check_url(address: str) -> None:
    try:
        parsed = urlsplit(address)
        hostname = parsed.hostname
        allowed = hostname is not None and any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in ("huggingface.co", "hf.co")
        )
        if (parsed.scheme != "https" or not allowed or parsed.username is not None
                or parsed.password is not None or parsed.port not in (None, 443)
                or parsed.fragment):
            raise ValueError("Unsafe Hugging Face HTTPS URL")
    except ValueError as exc:
        raise ValueError("Unsafe Hugging Face HTTPS URL") from exc


def _get_pinned(client: requests.Session, address: str, *, headers: dict | None = None) -> requests.Response:
    for hop in range(MAX_REDIRECTS + 1):
        _check_url(address)  # Validate BEFORE each network request, including reused resolved URLs.
        response = client.get(address, headers=headers, timeout=90, stream=True, allow_redirects=False)
        try:
            _check_url(response.url)
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                if not location or hop == MAX_REDIRECTS:
                    raise ValueError("Missing or excessive Hugging Face redirect")
                address = urljoin(response.url, location)
                _check_url(address)
            else:
                return response
        except BaseException:
            response.close()
            raise
        response.close()
    raise AssertionError("Unreachable redirect loop")


def _bounded_body(response: requests.Response, limit: int) -> bytes:
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Unsupported Content-Encoding on bounded HTTP response")
    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=65536):
        size += len(chunk)
        if size > limit:
            raise ValueError("HTTP response body exceeds byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def range_bytes(client: requests.Session, address: str, start: int, end: int,
                *, total: int | None = None, etag: str | None = None) -> tuple[bytes, int, str | None, str]:
    # Inspect status/headers before buffering: a CDN ignoring Range may send a full shard.
    if start < 0 or end < start or end - start + 1 > MAX_HEADER_BYTES:
        raise ValueError("Invalid or oversized range request")
    response = _get_pinned(client, address, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    try:
        response.raise_for_status()
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match:
            raise ValueError(f"HTTP range not honored: {response.status_code} {address.split('?')[0]}")
        lo, hi, size = map(int, match.groups())
        if (lo, hi) != (start, end) or int(response.headers.get("Content-Length", end - start + 1)) != end - start + 1:
            raise ValueError(f"Truncated/wrong range {lo}-{hi} for {start}-{end}")
        if total is not None and total != size:
            raise ValueError("Shard size changed between range requests")
        got_etag = response.headers.get("ETag")
        if etag is not None and etag != got_etag:
            raise ValueError("Shard ETag missing or changed between range requests")
        raw = _bounded_body(response, end - start + 1)
        if len(raw) != end - start + 1:
            raise ValueError(f"Truncated/wrong range {lo}-{hi} for {start}-{end}")
        return raw, size, got_etag, response.url
    finally:
        response.close()


def shard_header(client: requests.Session, address: str) -> tuple[dict, int, int, str | None, str]:
    # Source shard headers fit in 64 KiB; large quant headers use the fallback.
    first, total, etag, resolved = range_bytes(client, address, 0, 65535)
    if not etag:
        raise ValueError("Shard ETag missing on initial range request")
    length = struct.unpack("<Q", first[:8])[0]
    if not 0 < length <= MAX_HEADER_BYTES or length >= total - 8:
        raise ValueError(f"Invalid safetensors header length: {length}")
    if length + 8 > len(first):
        header, _, _, resolved = range_bytes(client, resolved, 8, 7 + length, total=total, etag=etag)
    else:
        header = first[8:8 + length]
    parsed = json.loads(header)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid safetensors header")
    return parsed, length, total, etag, resolved


def index(client: requests.Session, repo: str, revision: str, filename: str) -> tuple[dict, str]:
    response = _get_pinned(client, url(repo, revision, filename), headers={"Accept-Encoding": "identity"})
    try:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) > MAX_INDEX_BYTES:
            raise ValueError("Index body exceeds byte limit")
        raw = _bounded_body(response, MAX_INDEX_BYTES)
        return json.loads(raw)["weight_map"], hashlib.sha256(raw).hexdigest()
    finally:
        response.close()


def inspect_quant(shard: str, keys: list[str]) -> dict[str, bytes]:
    with session() as client:
        header, length, total, etag, resolved = shard_header(client, url(QUANT_REPO, QUANT_REV, QUANT_SUBDIR + shard))
        if any(PATTERN.fullmatch(k) for k in header):
            raise ValueError(f"Quant shard {shard} contains a routing bias despite index absence")
        if {k for k in header if NONE_PATTERN.fullmatch(k)} != set(keys):
            raise ValueError(f"Quant index/header .mlp.None discrepancy in {shard}")
        results = {}
        for key in keys:
            info = header[key]
            if info["dtype"] != "F32" or info["shape"] != [100]:
                raise ValueError(f"Unexpected bogus tensor metadata: {key}: {info}")
            begin, end = info["data_offsets"]
            if end - begin != 400:
                raise ValueError(f"Wrong .mlp.None data length: {key}")
            raw, _, _, _ = range_bytes(client, resolved, 8 + length + begin, 7 + length + end,
                                       total=total, etag=etag)
            results[key] = raw
    return results


def extract_shard(shard: str, keys: list[str]) -> list[tuple[str, bytes, dict]]:
    with session() as client:
        header, length, total, etag, resolved = shard_header(client, url(SOURCE_REPO, SOURCE_REV, shard))
        result = []
        for key in keys:
            if key not in header:
                raise ValueError(f"Source index/header mismatch: {shard}: {key}")
            info = header[key]
            match = PATTERN.fullmatch(key)
            assert match is not None
            expected_len = SIZES[match.group(2)] * 2
            begin, end = info["data_offsets"]
            if info["dtype"] != "BF16" or info["shape"] != [SIZES[match.group(2)]] or end - begin != expected_len:
                raise ValueError(f"Wrong source dtype/shape/size: {key}: {info}")
            raw, _, _, _ = range_bytes(client, resolved, 8 + length + begin, 7 + length + end,
                                       total=total, etag=etag)
            result.append((key, raw, {"source_shard": shard, "source_data_offsets": [begin, end],
                                      "source_file_byte_range": [8 + length + begin, 7 + length + end],
                                      "source_shard_etag": etag, "dtype": "BF16", "shape": info["shape"],
                                      "sha256": hashlib.sha256(raw).hexdigest()}))
        return result


def make_overlay(tensors: dict[str, bytes], metadata: dict[str, dict]) -> bytes:
    offset = 0
    header = {}
    for key in sorted(tensors):
        size = len(tensors[key])
        header[key] = {"dtype": "BF16", "shape": metadata[key]["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-(8 + len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors[k] for k in sorted(tensors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for the generated .safetensors overlay and JSON manifest (outside the checkout recommended)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = parse_args(argv).output_dir
    overlay_path = output_dir / "k2-routing-bias-overlay.safetensors"
    manifest_path = output_dir / "k2-routing-bias-overlay.manifest.json"
    with session() as client:
        source, source_sha = index(client, SOURCE_REPO, SOURCE_REV, "model.safetensors.index.json")
        quant, quant_sha = index(client, QUANT_REPO, QUANT_REV, QUANT_SUBDIR + "model.safetensors.index.json")
    expected = {f"model.layers.{layer}.{part}.bias" for layer in range(3, 48) for part in SIZES}
    actual = {key for key in source if PATTERN.fullmatch(key)}
    quant_actual = {key for key in quant if PATTERN.fullmatch(key)}
    none = {key for key in quant if NONE_PATTERN.fullmatch(key)}
    expected_none = {f"model.layers.{layer}.mlp.None" for layer in range(3, 48)}
    if actual != expected or quant_actual or none != expected_none:
        raise ValueError(f"Index discrepancy changed: source missing={sorted(expected - actual)}, "
                         f"source extras={sorted(actual - expected)}, quant biases={sorted(quant_actual)}, "
                         f"None missing={sorted(expected_none - none)}, None extras={sorted(none - expected_none)}")
    source_groups: dict[str, list[str]] = {}
    for key in sorted(expected):
        source_groups.setdefault(source[key], []).append(key)
    quant_groups: dict[str, list[str]] = {}
    for key in sorted(none):
        quant_groups.setdefault(quant[key], []).append(key)
    # Fetch in parallel by shard; each worker checks headers, ranges and immutable object identity.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        source_jobs = [pool.submit(extract_shard, shard, keys) for shard, keys in sorted(source_groups.items())]
        quant_jobs = [pool.submit(inspect_quant, shard, keys) for shard, keys in sorted(quant_groups.items())]
        records = [item for job in source_jobs for item in job.result()]
        quant_none_raw = {k: v for job in quant_jobs for k, v in job.result().items()}
    if len(records) != 90 or len(quant_none_raw) != 45:
        raise ValueError("Incomplete extraction")
    tensors = {key: raw for key, raw, _ in records}
    details = {key: detail for key, _, detail in records}
    if len(tensors) != 90:
        raise ValueError("Duplicate source tensor name")
    none_classifications = {}
    for key, raw in sorted(quant_none_raw.items()):
        source_key = key.removesuffix(".None") + ".gate.bias"
        expected_f32 = b"".join(b"\x00\x00" + tensors[source_key][i:i + 2]
                                for i in range(0, len(tensors[source_key]), 2))
        status = "source_gate_bias_widened_f32" if raw == expected_f32 else "all_zero" if not any(raw) else "different_nonzero"
        none_classifications[key] = {"classification": status, "sha256": hashlib.sha256(raw).hexdigest()}
    overlay = make_overlay(tensors, details)
    overlay_sha = hashlib.sha256(overlay).hexdigest()
    if overlay_sha != EXPECTED_OVERLAY_SHA256:
        raise ValueError(f"Overlay SHA-256 mismatch: {overlay_sha} != {EXPECTED_OVERLAY_SHA256}")
    manifest = {
        "schema": "k2-routing-bias-overlay-v1", "source_repo": SOURCE_REPO, "source_revision": SOURCE_REV,
        "source_index_sha256": source_sha, "quant_repo": QUANT_REPO, "quant_revision": QUANT_REV,
        "quant_subdir": QUANT_SUBDIR.rstrip("/"), "quant_index_sha256": quant_sha,
        "tensor_count": len(tensors),
        "quant_mlp_None": none_classifications,
        "quant_mlp_None_classification_counts": {status: sum(v["classification"] == status for v in none_classifications.values())
                                                   for status in ("source_gate_bias_widened_f32", "all_zero", "different_nonzero")},
        "overlay_file": overlay_path.name, "overlay_sha256": overlay_sha,
        "tensors": {key: details[key] for key in sorted(details)},
        "note": "All 90 bias names are absent from the 6.50bpw quant index and shard headers. 45 quant .mlp.None tensors are F32 and misnamed; their content classification is recorded separately. No equivalent v_router.bias tensor name was found. This overlay is not installed into the quant model."
    }
    # Complete all remote checks and both temporary writes before replacing outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for target, payload in (
            (overlay_path, overlay),
            (manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")),
        ):
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=output_dir)
            staged = Path(name)
            pending.append((staged, target))
            try:
                output = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                raise
            with output:
                output.write(payload)
        for staged, target in pending:
            if not stat.S_ISREG(staged.lstat().st_mode):
                raise ValueError(f"Staged output is not a regular file: {staged}")
            os.replace(staged, target)
    finally:
        for staged, _ in pending:
            staged.unlink(missing_ok=True)
    print(json.dumps({"tensor_count": len(tensors), "quant_None_classifications": manifest["quant_mlp_None_classification_counts"],
                      "overlay_sha256": manifest["overlay_sha256"], "overlay": str(overlay_path),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
