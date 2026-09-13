# Local π0.5 assets

The user authorized downloading missing model/tokenizer files on 2026-09-13.
Downloads run on an x86 host into `/scratch`; Thor remains offline and receives
no package installation. Version pins and checksums are in `sources.lock.json`.

## Reproduce the download

Use existing Python >= 3.9 and curl on the download host:

```bash
bash scripts/download_assets.sh --direct-cdn
```

The helper installs nothing. It streams each file to a scratch partial file,
resumes an interrupted HTTP transfer, checks size and checksum, then renames it
into place. It downloads one file at a time and hashes with a 4 MiB buffer.

On `deep-space`, the configured local proxy was slow, while a bounded direct
transfer from the official CDN succeeded. `--direct-cdn` bypasses that proxy only
for `us.aws.cdn.hf.co` in this process; Hugging Face metadata still uses the
existing network configuration. Omit the flag if direct CDN access is unavailable.
No system network configuration is modified.

The checkpoint is `lerobot/pi05_base` at
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`. The safetensors file is
14,467,165,872 bytes including its header. SHA256:
`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`.
Config and processor metadata are checked against their pinned Git blob IDs;
every downloaded file receives a SHA256 in the final provenance report.

## Public tokenizer source

The official [OpenPI tokenizer implementation](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)
uses `gs://big_vision/paligemma_tokenizer.model` with anonymous access.
Its public HTTPS equivalent is downloaded here. Its SHA256 exactly matches the
`tokenizer.model` metadata in Google's `paligemma-3b-pt-224` repository at
`35e4f46485b4d07967e7e9935bc3786aad50687c`:
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.

Only the local Gemma text-tokenizer configuration is reconstructed: BOS enabled,
EOS disabled, right padding with ID 0. The runtime's Transformers reads the
SentencePiece file locally. No PaliGemma model weights are needed. The verifier
compares IDs, masks, BOS, padding and truncation with the original SentencePiece
implementation on 14 π0.5 task/state cases, including Unicode and long prompts.

```bash
# Run on Thor, after the files are ready:
bash scripts/thor_python.sh scripts/verify_tokenizer.py \
  --tokenizer /scratch/$USER/vla-vllm-profiling/assets/paligemma_tokenizer \
  --report results/processed/pi05_assets/tokenizer_validation.json
```

Model assets retain their upstream license. Weights and tokenizer files are
disposable downloads and are not committed to GitHub. The public repository
contains the downloader, exact sources, hashes and validation records.
