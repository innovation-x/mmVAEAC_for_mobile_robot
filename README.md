# Situation-Dependent Motion Generation with mmVAEAC

Implementation accompanying **“A Study on a Situation-Dependent Motion
Generation Framework Using a Generative Model of Motion and Observation.”**

The project learns a multimodal variational autoencoder with arbitrary
conditioning (mmVAEAC) for mobile-robot motion generation.  Given an observed
scene and/or a partial motion sequence, it generates plausible robot motion and
supports experiments on the relationship between generated motion and presence
intensity.

<p align="center">
  <a href="https://github.com/innovation-x/mmVAEAC_for_mobile_robot/blob/main/origin.mp4">Input scene</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/innovation-x/mmVAEAC_for_mobile_robot/blob/main/demo.mp4">Generated-motion demo</a>
</p>

## Repository layout

```text
.
├── checkpoints/              # Pretrained weights (new training outputs are ignored)
├── datasets/                 # Video + motion data; see the dataset format below
├── eval/                     # Evaluation helpers (generated results stay local)
├── videovae/                 # VAEAC and mmVAEAC model implementations
├── train.py                  # Training entry point
├── mmVAEAC_train_and_test.ipynb
├── eval.ipynb
├── eval_action.ipynb
├── requirements.txt          # Reproducible pip dependencies
└── environment.yml           # Conda environment definition
```

## Quick start

The original experiments were run with **Python 3.8**, **PyTorch 1.10.1**, and
**CUDA 11.3**.  A CUDA-capable NVIDIA GPU is recommended for training.

```bash
git clone https://github.com/innovation-x/mmVAEAC_for_mobile_robot.git
cd mmVAEAC_for_mobile_robot

conda env create -f environment.yml
conda activate mmvaeac-mobile-robot
```

Alternatively, create a virtual environment and install the pinned packages:

```bash
python3.8 -m venv mmvaeac
source mmvaeac/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cu113.txt
```

For a CPU-only setup, use the matching CPU PyTorch wheels:

```bash
pip install -r requirements-cpu.txt
```

## Dataset format

Pass `--data_path` a directory with the following structure.  Video and motion
files are paired by their numeric filename within each recording folder.

```text
<data_path>/
├── train/
│   └── <recording-id>/
│       ├── 1.mp4
│       ├── 1.csv
│       └── …
└── test/
    └── <recording-id>/
        ├── 1.mp4
        ├── 1.csv
        └── …
```

`VideoDataset` builds `metadata_*.pkl` indexes on first use.  They are cached
locally and intentionally excluded from Git.  The large ATR datasets, model
checkpoints, W&B logs, and locally created environments are likewise ignored;
use a dataset or artifact store for sharing them.

## Training

```bash
python train.py \
  --data_path ./datasets/ATR_1F_3F_2024-12-25_withtest \
  --sequence_length 16 \
  --resolution 64 \
  --batch_size 16 \
  --num_workers 8 \
  --LR_RATE 5e-5 \
  --NUM_EPOCHS 100
```

Training logs to [Weights & Biases](https://wandb.ai/) by default.  To run
without synchronizing results, set `WANDB_MODE=offline` before the command.
Checkpoints are written to `checkpoints/pretrained_mmVAEAC/` every five epochs.

## Evaluation and demos

The notebooks document the original experimental flow:

| Notebook | Purpose |
| --- | --- |
| `mmVAEAC_train_and_test.ipynb` | Model development, training, and qualitative generation |
| `eval.ipynb` | Distance-based generated-vs-real motion evaluation |
| `eval_action.ipynb` | Guided and unguided action evaluation |

Open them after activating the environment:

```bash
jupyter notebook
```

## Reproducibility notes

- The tracked `livingroom` data is a small example dataset.  The full ATR data
  and experiment artifacts are deliberately not versioned because they are
  large and can exceed GitHub's file-size limit.
- Two pretrained checkpoint files are retained for compatibility with the
  original experiments.  New checkpoints are excluded by `.gitignore`.
- Evaluation tensors saved as `eval/*.pt` are local outputs of the evaluation
  notebooks and are also excluded from version control.
- The current entry point exposes all training parameters as command-line
  options; run `python train.py --help` to see the defaults.

## Citation

If you use this code, please cite the accompanying paper:

```text
A Study on a Situation-Dependent Motion Generation Framework Using a
Generative Model of Motion and Observation.
```
