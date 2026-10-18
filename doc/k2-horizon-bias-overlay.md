# K2-Horizon MoVA: recover missing routing biases (6.50bpw only)

The **current** `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/6.50bpw` revision
`c88277ce7f6b90b723f79b5188c0f1b951732099` omits 90 learned selection biases:
45 `model.layers.{3..47}.self_attn.v_router.bias` (BF16 `[64]`) and
45 `model.layers.{3..47}.mlp.gate.bias` (BF16 `[100]`). Its 45 misnamed
`mlp.None` F32 tensors are not replacements (40 zero, five different nonzero).
The extractor copies the exact BF16 bytes from public source
`IFM/K2-Horizon-MoVA-36B-A4B` revision
`7730b92d1b574e04663b04023d5d6fa83475432f`. **Do not assume this
is needed or correct for other quant revisions, bitrates, or models.**

From the fork checkout, using Python 3.10+ and `requests`:

```sh
python -m pip install requests
python tools/extract_k2_routing_bias_overlay.py --output-dir /path/outside-repo/k2-biases
sha256sum /path/outside-repo/k2-biases/k2-routing-bias-overlay.safetensors
```

The expected SHA-256 for these pinned revisions is
`8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c`.
The script checks both indices, relevant shard headers, byte ranges, shapes and
45 quant `.None` payloads, then verifies the **entire overlay SHA-256 against the
pinned digest above before writing either file**. It requires a stable ETag on
each shard range, refuses safetensors headers over 16 MiB before fetching them,
and disables `requests` environment trust (including implicit `.netrc` auth and
environment proxies). Redirects are checked **before** each follow-up request:
only HTTPS on port 443 (or the implicit HTTPS default) to `huggingface.co` / its
subdomains or `hf.co` / its subdomains is accepted. URL credentials, other hosts,
HTTP downgrades and more than five redirects are rejected. The observed
`us.aws.cdn.hf.co` shard redirect is allowed; unrelated HTTPS hosts are not.
Both index and range requests require identity encoding; responses declaring
another Content-Encoding are rejected before streaming to prevent a compressed
chunk from inflating past the byte cap. Uncompressed index and range responses
are streamed with byte limits (32 MiB per index, 16 MiB maximum requested range),
even if a server lies about Content-Length or ignores Range; the exact expected
range length and ETag are still checked.
It uses HTTP ranges rather than fetching whole model shards.
It produces the overlay and a local JSON manifest
with revisions, index hashes, per-tensor provenance and hashes, and `.None`
classifications. Reruns stage both complete files in temporary files **only in
the output dir**, writing through the original open file descriptors rather than
reopening their paths. A non-regular staged path is rejected before replacement.
A failure while writing either staged file leaves a previously verified
overlay/manifest intact and cleans up temporary files. The two final replacements
are not a filesystem transaction if a replace operation itself fails. The
pre-replacement path check is not a guarantee against concurrent swaps in an
output directory writable by another actor.
Keep the output outside the checkout and do not commit the generated weights
or manifest to this repository.

Pass the overlay explicitly when loading a local **6.50bpw model directory**:

```python
from exllamav3 import Config
config = Config.from_directory(
    "/path/to/K2-Horizon-MoVA-36B-A4B-EXL3/6.50bpw",
    routing_bias_overlay="/path/outside-repo/k2-biases/k2-routing-bias-overlay.safetensors",
)
```

Do not modify or upload the quant repository; do not rename `.mlp.None` tensors.
K2 loading fails closed on missing required biases or a missing overlay path.
The router bias participates in **selection only**, not the projection logits.
Run offline regression checks with
`python -m pytest -q tests/test_k2_bias_overlay_extractor.py tests/test_k2_horizon.py`.
At integrated source `18dd328` on Spark2, the full 6.50bpw model loaded with
the overlay, produced the cached decode ` Paris. The capital of`, completed a
2-row CUDA calibration probe, and unloaded. This is a limited verified smoke,
**not** a BF16 or steady-state performance benchmark; the extractor hardening
here was separately verified by a pinned-source end-to-end extraction and
the overlay digest above.
