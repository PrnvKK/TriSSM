# TriSSM

Code for "Associative Recall Under Ternary Quantization: A Controlled Comparison of State Space Models and Transformers at 5M Parameters".

TriSSM is a selective state space model whose input-dependent sequence-mixing projections (dt, B, C) are BitNet b1.58-style `BitLinear` layers: ternary weights in {-1, 0, +1} and int8 activations, trained with straight-through estimators. The repository supports a controlled 2x2 comparison at about 5M parameters on TinyStories: SSM vs. Transformer, each in full precision and in ternary form. A Transformer with ternary Q/K/V/O projections and a full-precision MLP serves as an additional control. Models are trained with a matched token budget, then adapted and evaluated on a key-value associative recall task.

## Layout

```
bitlinear.py         BitLinear layer (ternary weights, int8 activations, STE)
bit_ssm.py           selective SSM block with parallel scan (ternary or full precision)
model.py             TriSSM language model
baselines.py         Transformer baselines (ternary, full precision, QKVO-only)
configs.py           model registry, CSV helpers, shared driver utilities
prepare_data.py      download TinyStories, train tokenizer, write train.bin / val.bin
train.py             train one (model, seed) cell
recall.py            associative recall adaptation and evaluation
eval_perplexity.py   validation perplexity across seeds
run_seeds.py         driver: multi-seed training and recall
run_capacity.py      driver: memory-capacity sweep
run_configs.py       driver: parameter-matched Transformer configurations
run_variance.py      driver: per-step variance probe
```

Registry keys: `trissm`, `fp_ssm`, `bitnet_transformer`, `fp_transformer`, `qkvo_transformer`, and `v2t_{deep,mid,shallow,widehead}_{fp,tern}`.

## Setup

```
pip install -r requirements.txt
```

Requires PyTorch 2.4 or newer. A CUDA GPU is used if available; otherwise training runs on CPU.

## Data

```
python prepare_data.py
```

Writes the tokenizer and `train.bin` / `val.bin` to `data/`.

## Training

```
python train.py --arch trissm --seed 42
python train.py --arch fp_transformer --seed 42
```

Checkpoints are written to `out/<arch>_s<seed>/model.pt`. Training rows are appended to `logs/train_results.csv`. Use `--resume` to continue an interrupted run.

## Associative recall

```
python recall.py --model trissm --seed 42 --n_pairs 10
python recall.py --model fp_ssm --seed 42 --n_pairs 50 \
    --distance_mode stratified --dist_edges 0,1,2,4,8,16,32,50 --experiment distance
```

Loads `out/<model>_s<seed>/model.pt`, adapts it on the recall task, and appends results to `logs/recall_results.csv`.

## Validation perplexity

```
python eval_perplexity.py --seeds 42,43,44
```

## Experiment drivers

Each driver runs the underlying scripts sequentially, skips cells already recorded in `logs/`, and accepts `--dry_run` to print the commands.

```
python run_seeds.py          # train headline models and run recall for each seed
python run_capacity.py       # recall with varying number of key-value pairs
python run_configs.py        # parameter-matched Transformer configurations
python run_variance.py       # per-step variance probe
```

## Citation

```bibtex
@article{krishnakumar2026trissm,
  title={Associative Recall Under Ternary Quantization: A Controlled Comparison of State Space Models and Transformers at 5M Parameters},
  author={Krishnakumar, Pranav and Pandey, Garima and Murthy Y, Vishnu Srinivasa and Koolagudi, Shashidhar G.},
  note={Under review},
  year={2026}
}
```
