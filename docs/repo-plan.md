# Repo Plan

## Directory tree
```
.
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── model/{regbridge_oai.yaml, regbridge_gait.yaml}
│   ├── data/{oai.yaml, most.yaml, gaitoa.yaml}
│   ├── train/{adamw_cosine.yaml}
│   └── experiment/{main.yaml, ablation_*.yaml, supplementary_*.yaml, _smoke.yaml}
├── src/damascus/
│   ├── blade.py                 assembled RegBridge + DamascusConfig + build_regbridge
│   ├── stock/                   data: cohorts, charge (loaders), quench_tank (transforms), work_order (types)
│   ├── billets/                 encoders: motion_billet (TME), imaging_billet (IFE)
│   ├── forge/                   RBM: heat (confidence+gate), weld (bilinear fusion), bevel (heads)
│   ├── tongs/                   losses: stresses
│   ├── hardening/               training: heat_treat (trainer), cycle (optim/sched/CV)
│   ├── proof/                   eval: edge_tests (metrics), bend_tests (loops), assay (stats)
│   ├── smithy/                  runtime: hearth (seed/device/io/log), settings (config loader)
│   └── anvil/                   CLI: __main__, commands (Typer verbs)
├── scripts/{forge_train.sh, forge_eval.sh, forge_prepare.sh}
├── tests/
└── docs/{project-context.md, implementation-map.md, repo-plan.md, deviations.md}
```

## Module responsibilities
- **stock** — deterministic synthetic cohorts realising Definition 1; torch `Dataset`/`DataLoader`
  assembly; imaging + pseudo-temporal motion transforms; typed batch contracts.
- **billets** — Temporal Motion Encoder (dilated causal conv stack + multi-head self-attention) and
  Imaging Feature Encoder (DenseNet-201 + channel/spatial attention).
- **forge** — Regulatory Bridge Module: confidence heads, quality-aware gating, bidirectional gated
  bilinear fusion with residual LayerNorm, classification and auxiliary unimodal heads.
- **tongs** — label-smoothed cross-entropy, relative-target confidence calibration, total objective.
- **hardening** — trainer implementing Algorithm 1 (AMP, EMA, atomic checkpoints, DDP-ready,
  seed restore on resume), optimizer/scheduler builders, 5-fold cross-validation.
- **proof** — metrics (AUROC, accuracy, quadratic kappa, F1, sensitivity, specificity, per-class,
  confusion), task evaluation loops for every reported table, statistics (bootstrap CI, paired
  permutation, DeLong).
- **smithy** — seeding/device/logging/atomic IO; msgspec config schema + YAML loader.
- **anvil** — Typer CLI with verbs `draw`, `weld`, `proof`, `edge`, `export`.

## Config / CLI stack
`msgspec.Struct` frozen schema; YAML config files decoded with `msgspec.yaml`; composition via a
`base:` key resolved by a deep-merge include chain; CLI overrides as `section.key=value`. Commands
through Typer subcommands.

## Dependencies (pinned ranges)
torch==2.1.0, torchvision==0.16.0, numpy>=1.24, scipy>=1.10, scikit-learn>=1.3, pandas>=2.0,
msgspec>=0.18, typer>=0.9, PyYAML>=6.0. Dev: pytest, ruff, black, isort, mypy, pre-commit.

## Test coverage (varied kinds)
shape; missing-modality invariance (Thm. 3); single-batch overfit; metric agreement vs scikit-learn and
a DeLong reference; gradient flow over all parameters; determinism under fixed seed; config base-merge;
fusion-variant equivalence (ablation toggles); calibration-loss direction; gating optimality direction
(Thm. 1); style guard (no comments / no docstrings / forbidden phrases / no emoji across src + tests +
README + Makefile); end-to-end training smoke on `_smoke.yaml`.
