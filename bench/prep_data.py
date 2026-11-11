"""Train a 32k byte-level BPE tokenizer on FineWeb-Edu and tokenize the shard to uint16 .bin files.

Output (in --out): tokenizer.json, train.bin, val.bin, meta.json.
Validation = the last --val_docs documents of the shard; the tokenizer never sees them.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="/workspace/data/fineweb-edu/sample/10BT/000_00000.parquet")
ap.add_argument("--out", default="/workspace/data")
ap.add_argument("--vocab", type=int, default=32000)
ap.add_argument("--tok_docs", type=int, default=150_000)
ap.add_argument("--val_docs", type=int, default=5_000)
ap.add_argument("--max_train_tokens", type=float, default=800e6)
args = ap.parse_args()

pf = pq.ParquetFile(args.parquet)
n_docs = pf.metadata.num_rows
n_train_docs = n_docs - args.val_docs
print(f"{n_docs} documents; train {n_train_docs}, val {args.val_docs}", flush=True)


def iter_docs(start=0, stop=None):
    i = 0
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for t in batch.column(0).to_pylist():
            if i >= start and (stop is None or i < stop):
                yield t
            i += 1
            if stop is not None and i >= stop:
                return


tok_path = os.path.join(args.out, "tokenizer.json")
if not os.path.exists(tok_path):
    t0 = time.time()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(iter_docs(0, args.tok_docs), trainer=trainer, length=args.tok_docs)
    tok.save(tok_path)
    print(f"tokenizer trained in {time.time() - t0:.0f}s, vocab={tok.get_vocab_size()}", flush=True)
tok = Tokenizer.from_file(tok_path)
eot = tok.token_to_id("<|endoftext|>")
assert tok.get_vocab_size() <= 65535


def write_bin(path, start, stop, max_tokens):
    t0, total = time.time(), 0
    with open(path, "wb") as f:
        buf = []
        for doc in iter_docs(start, stop):
            buf.append(doc)
            if len(buf) == 5000:
                total += flush(f, buf)
                buf = []
                print(f"  {path}: {total / 1e6:.0f}M tokens, {time.time() - t0:.0f}s", flush=True)
                if total >= max_tokens:
                    break
        if buf and total < max_tokens:
            total += flush(f, buf)
    return total


def flush(f, docs):
    ids = []
    for enc in tok.encode_batch(docs):
        ids.extend(enc.ids)
        ids.append(eot)
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


n_val = write_bin(os.path.join(args.out, "val.bin"), n_train_docs, n_docs, float("inf"))
n_train = write_bin(os.path.join(args.out, "train.bin"), 0, n_train_docs, args.max_train_tokens)
meta = dict(vocab=tok.get_vocab_size(), eot=eot, train_tokens=n_train, val_tokens=n_val,
            tok_docs=args.tok_docs, val_docs=args.val_docs, parquet=args.parquet)
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("done", meta, flush=True)
