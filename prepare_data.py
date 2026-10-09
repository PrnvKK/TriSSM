"""Download TinyStories (V2, GPT-4), train an 8k SentencePiece BPE tokenizer and
write the tokenized train/validation splits as uint16 binaries.

Output (in --data_dir, default data/):
    TinyStoriesV2-GPT4-train.txt   raw text
    tokenizer_8k.model/.vocab      tokenizer (trained on the first 10 MB)
    train.bin, val.bin             first 95% / last 5% of the text, tokenized

Example:
    python prepare_data.py
"""
import argparse
import os
import urllib.request

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
       "TinyStoriesV2-GPT4-train.txt")


def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists.")
        return
    print(f"Downloading {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded {filepath}.")


def tokenize_stream(input_filepath, out_filename, model_file, start_byte, end_byte,
                    chunk_lines=50000):
    """Tokenize the byte range [start_byte, end_byte) of a text file line by
    line and write the token ids to out_filename as uint16."""
    print(f"Processing {out_filename}...")
    sp = spm.SentencePieceProcessor(model_file=model_file)

    def flush(lines, fout):
        tokens = [tok for sublist in sp.encode(lines) for tok in sublist]
        fout.write(np.array(tokens, dtype=np.uint16).tobytes())

    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_filename, 'wb') as fout:

        if start_byte > 0:
            fin.seek(start_byte)
            fin.readline()  # align to a line boundary

        buffer_lines = []
        with tqdm(total=end_byte - start_byte, unit='B', unit_scale=True,
                  desc=os.path.basename(out_filename)) as pbar:
            last_tell = fin.tell()
            while fin.tell() < end_byte:
                line = fin.readline()
                if not line:
                    break
                buffer_lines.append(line)

                current_tell = fin.tell()
                pbar.update(current_tell - last_tell)
                last_tell = current_tell

                if len(buffer_lines) >= chunk_lines:
                    flush(buffer_lines, fout)
                    buffer_lines = []

            if buffer_lines:
                flush(buffer_lines, fout)

    print(f"Saved {out_filename}.\n")


def main():
    p = argparse.ArgumentParser(description="Prepare TinyStories train/val binaries.")
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    filepath = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-train.txt")
    tok_prefix = os.path.join(args.data_dir, "tokenizer_8k")
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")

    download_file(URL, filepath)

    total_bytes = os.path.getsize(filepath)
    split_byte = int(total_bytes * 0.95)

    if not os.path.exists(tok_prefix + ".model"):
        slice_path = os.path.join(args.data_dir, "train_tokenizer_slice.txt")
        print("Extracting 10MB slice for tokenizer training...")
        with open(filepath, 'rb') as fin, open(slice_path, 'wb') as fout:
            fout.write(fin.read(10_000_000))

        print("Training SentencePiece tokenizer (8k vocab)...")
        spm.SentencePieceTrainer.train(
            input=slice_path,
            model_prefix=tok_prefix,
            vocab_size=8000,
            model_type='bpe',
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>',
            character_coverage=0.9995,
        )
    else:
        print(f"{tok_prefix}.model already exists, skipping training.\n")

    if not os.path.exists(train_bin):
        tokenize_stream(filepath, train_bin, tok_prefix + ".model", 0, split_byte)
    else:
        print(f"{train_bin} already exists, skipping.")

    if not os.path.exists(val_bin):
        tokenize_stream(filepath, val_bin, tok_prefix + ".model", split_byte, total_bytes)
    else:
        print(f"{val_bin} already exists, skipping.")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
