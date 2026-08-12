# jepa

Cross-modal **Joint-Embedding Predictive Architecture** for Xenium spatial
multi-omics: it learns aligned per-cell embeddings by predicting one modality's
latent from the other.

- **Image/shape view** — the `[DAPI, protein]` morphology crop, encoded by a
  **Qwen2.5-VL** vision tower finetuned with **LoRA**.
- **RNA view** — the per-cell 405-gene transcript vector, encoded by an MLP.
- **Objective** — Head A predicts the RNA embedding from the image embedding,
  Head B predicts the image embedding from the RNA embedding, against
  **EMA target** encoders (BYOL/I-JEPA style), with an optional VICReg
  variance/covariance collapse guard.

See [`design.md`](design.md) for the full rationale and
[`scripts/plan.md`](scripts/plan.md) for the build plan.

## Setup

```bash
micromamba create -f environment.yml -y
micromamba activate jepa
```

Data and crops are reused from the sibling `../aaran` project (crop cache +
Xenium `cell_feature_matrix.h5`); nothing is re-cropped.

## Run

```bash
cd scripts
./run.sh smoke     # sanity run on 512 cells (data + train + eval)
./run.sh full      # full train -> eval -> export embeddings
```

Or step by step (run from `scripts/`):

```bash
python config.py                          # verify paths
python data.py                            # build index + a small split, sanity batch
python train.py --num-workers 8           # train (writes out/ckpt/{last,best}.pt)
python eval.py                            # recall@k, alignment, out/umap.png
python embed.py --split all               # out/embeddings.parquet
```

## Outputs (`out/`)

| file | contents |
|------|----------|
| `pairs_index.parquet` | every `(cell_id, channel, crop_path)` |
| `rna_log1p.npz` | cached log1p 405-gene matrix |
| `train.parquet` / `val.parquet` | split-by-cell indices |
| `ckpt/best.pt`, `ckpt/last.pt` | checkpoints |
| `umap.png` | UMAP of fused embeddings by channel |
| `embeddings.parquet` | per-`(cell,channel)` `img_*`, `rna_*`, `emb_*` vectors |

## Layout

```
scripts/
  config.py              Config dataclass + aaran bridge
  data.py                paired dataset, split-by-cell, loaders
  models/
    rna_encoder.py       MLP RNA encoder
    image_encoder.py     Qwen2.5-VL vision tower + LoRA + projection
    heads.py             Head A / Head B predictors
    jepa.py              online+EMA encoders, loss, EMA update
  train.py               AMP, cosine LR, EMA ramp, ckpt, val
  eval.py                retrieval recall@k, alignment, UMAP
  embed.py               export aligned embeddings
  run.sh                 smoke / full pipeline
```

## Annotation Web App

You can run the annotation notebook workflow as a small Streamlit app:

```bash
micromamba activate jepa
pip install -r assess/requirements-webapp.txt
streamlit run assess/annotation_webapp.py
```

For GitHub deployment, push this repository and point your app host
(for example Streamlit Community Cloud) at:

- App file: `assess/annotation_webapp.py`
- Requirements: `assess/requirements-webapp.txt`

To enable in-app commits of annotations back to GitHub, set a token in
Streamlit secrets (with `repo` scope for private repos, or `contents:write`
for fine-grained tokens):

```toml
[github]
token = "ghp_xxx"
```

Then use the app sidebar `GitHub Sync` controls. You can either click
`Commit CSV to GitHub` manually, or enable `Auto-commit to GitHub` and choose
`Auto-commit every N saves` to push annotations in batches while you label.
