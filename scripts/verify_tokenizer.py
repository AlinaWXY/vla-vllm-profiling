#!/usr/bin/env python3
"""Verify the reconstructed local tokenizer against the official SentencePiece file."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    import sentencepiece as spm
    from transformers import AutoTokenizer
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer / "tokenizer.model"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    prompts = ["pick up the red block and place it in the bin", "", "open_drawer\nand place cup",
               "拿起红色积木", "move 0.125 m; rotate -90°", "café\t cup", "pick up the block " * 100]
    cases = []
    for prompt in prompts:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        for values in ([127] * 32, [0, 255, 1, 128] * 8):
            state = " ".join(map(str, values))
            text = f"Task: {cleaned}, State: {state};\nAction: "
            expected = sp.encode(text, add_bos=True)[:200]
            actual = tokenizer(text, padding="max_length", max_length=200, truncation=True,
                               add_special_tokens=True)
            ids = actual["input_ids"]
            mask = actual["attention_mask"]
            assert [i for i, m in zip(ids, mask) if m] == expected, (prompt, expected, ids)
            assert ids[len(expected):] == [0] * (200 - len(expected))
            assert mask == [1] * len(expected) + [0] * (200 - len(expected))
            cases.append({"prompt": prompt, "tokens": len(expected),
                          "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"status": "passed", "cases": cases,
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_token_id, "bos_id": tokenizer.bos_token_id,
        "source_sha256": hashlib.sha256((args.tokenizer / "tokenizer.model").read_bytes()).hexdigest(),
        "scope": f"{len(cases)} pi05 prompt/state cases: token IDs, BOS, truncation, right padding against official SentencePiece"},
        indent=2, ensure_ascii=False) + "\n")
    print(f"Tokenizer parity passed: {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()
