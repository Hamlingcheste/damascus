```
=============================================================================
  MASTERSMITH PERFORMANCE TEST  --  SCORE CARD
  Test piece ... damascus  (RegBridge: confidence-gated multimodal fusion)
  Maker ........ The RegBridge authors
  Discipline ... early knee-osteoarthritis assessment from movement + imaging
  Venue ........ npj Digital Medicine
  Bar stock .... OAI  /  MOST  /  GaitOA
  Quenchant .... AdamW + cosine warm restarts, FP16
=============================================================================
```

A bladesmith is judged on one question: do two billets become one blade that holds an edge?
`damascus` forge-welds a **movement billet** and an **imaging billet** into a single fused
representation under a quality-aware gate, then grades the result on five-class Kellgren-Lawrence
grading, early-OA detection, and 48-month progression. This card tells a reviewer how to set up the
forge, run the graded cuts, and read the score against the manuscript.

## The test piece

Two encoders feed one regulatory bridge:

```
   movement billet (TME) --.                         .--> auxiliary head (motion)
   dilated causal conv      \                        /
   + 8-head attention        >-- regulatory bridge --+--> fused --> classification edge
   imaging billet (IFE)     /    confidence c_M,c_I  \
   DenseNet-201 + CSA  ----'     gate -> bilinear      `--> auxiliary head (imaging)
                                 fusion -> residual
```

The bridge estimates a per-sample confidence for each modality, scales each billet by its confidence,
and fuses them with a bidirectional gated bilinear interaction plus a residual. A confidence
calibration loss drives the more reliable modality to carry the higher weight, which is what lets the
movement billet dominate at early grades and the imaging billet dominate at late grades.

## Forge layout

| Station | Package | Role on the bench |
|---|---|---|
| Bar stock | `damascus/stock` | cohorts (Definition 1 generator), loaders, transforms, batch types |
| Billets | `damascus/billets` | temporal motion encoder, imaging feature encoder |
| Weld | `damascus/forge` | confidence heads, quality gate, gated bilinear fusion, heads |
| Tongs | `damascus/tongs` | label-smoothed CE, confidence calibration, total objective |
| Heat treatment | `damascus/hardening` | trainer (Algorithm 1), optimizer/scheduler, cross-validation |
| Proof | `damascus/proof` | metrics, task evaluation loops, bootstrap / permutation / DeLong |
| Hearth | `damascus/smithy` | seeding, device, atomic checkpoints, config loader |
| Anvil | `damascus/anvil` | command line: `draw`, `weld`, `proof`, `edge`, `export` |

## Setting up the forge

pip:
```
python -m pip install .
```
conda:
```
conda env create -f environment.yml
conda activate damascus
python -m pip install .
```
Docker:
```
docker build -t damascus:latest .
docker run --rm damascus:latest --help
```

## The graded cuts

Each station is one command; `--set section.key=value` overrides any field.

| Cut | Command |
|---|---|
| Inspect the stock | `python -m damascus.anvil draw --config configs/experiment/main.yaml` |
| Forge-weld (train) | `python -m damascus.anvil weld --config configs/experiment/main.yaml` |
| Proof the edge (evaluate) | `python -m damascus.anvil proof --config configs/experiment/main.yaml` |
| Single strike (infer) | `python -m damascus.anvil edge --config configs/experiment/main.yaml` |
| Stamp the maker's mark (ONNX) | `python -m damascus.anvil export --config configs/experiment/main.yaml` |

Ablation variants A-G and the cross-dataset / gait studies are selected by config:
```
python -m damascus.anvil weld --config configs/experiment/ablation_a_concat.yaml
python -m damascus.anvil weld --config configs/experiment/supplementary_gait.yaml
python -m damascus.anvil weld --config configs/experiment/supplementary_sensitivity.yaml --set model.tme.num_layers=8
```

## Score card (manuscript targets)

Run on the real cohorts, a passing blade meets these marks (mean over 5 folds; tolerance is the
reported standard deviation). The shipped synthetic generator exercises every code path but does not
reproduce these values, which require the access-gated cohorts.

| Station | Command | Mark | Tolerance |
|---|---|---|---|
| OAI 5-class KL accuracy | `proof --config .../main.yaml` | 74.8% | +/- 0.6 |
| OAI 5-class KL AUROC | `proof --config .../main.yaml` | 0.917 | +/- 0.008 |
| OAI 5-class quadratic kappa | `proof --config .../main.yaml` | 0.659 | +/- 0.009 |
| OAI early-OA AUROC | `proof --config .../main.yaml --set data.task=early_oa` | 0.893 | +/- 0.011 |
| OAI progression AUROC | `proof --config .../main.yaml --set data.task=progression` | 0.821 | +/- 0.015 |
| MOST zero-shot AUROC | `proof --config .../supplementary_cross_dataset.yaml` | 0.886 | +/- 0.010 |
| MOST 10%-fine-tuned AUROC | `proof --config .../supplementary_cross_dataset.yaml` | 0.908 | +/- 0.010 |
| GaitOA 3-class accuracy | `proof --config .../supplementary_gait.yaml` | 78.3% | +/- 1.4 |
| GaitOA binary AUROC | `proof --config .../supplementary_gait.yaml --set data.task=gaitbin` | 0.874 | +/- 0.013 |

## Fuel and time at the forge

| Item | Value |
|---|---|
| Hardware | 4x NVIDIA A100 80GB |
| Effective batch | 64 |
| Schedule | 120 epochs, cosine warm restarts (T0=10, Tmult=2), early stop patience 15 |
| Precision | FP16 mixed |
| Train wall-clock (OAI) | ~4.8 h |
| Inference | 18.4 ms / sample |
| Parameters | 47.3M |
| FLOPs | 9.4 GFLOPs |

## Bar stock

Datasets are access-gated or open at the sources below; none is redistributed here.

| Cohort | Subjects | Access | Source |
|---|---|---|---|
| OAI | 4,796 | registration + DUA | https://nda.nih.gov/oai |
| MOST | 3,026 | registration + DUA | https://mostonline.ucsf.edu |
| GaitOA | 260 (166 used) | open | https://springernature.figshare.com/articles/dataset/A_Dataset_of_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological_and_Orthopedic_Cohorts/28806086 |

The manuscript prints `mostudy.org` for MOST; that host does not serve the study, so the UCSF portal
above is used instead (see `docs/deviations.md`).

## Bench inspection

```
make test     # pytest: shape, missing-modality, single-batch overfit, metrics, gradients,
              #          determinism, config merge, fusion variants, calibration, style guard, smoke
make lint     # ruff + isort
make type     # mypy --strict
make smoke    # two training steps on the smoke config
```

## Maker's notes

- No public cohort pairs raw IMU with imaging for the same subjects; the three cohorts are
  triangulated, and the GaitOA second modality is a spectrogram of the same IMU rather than an
  independent view. Numbers are retrospective and not a clearance tool.
- The data is de-identified secondary use of public benchmarks; no IRB was required. Confidence
  channels expose per-modality reliability to support expert referral of conflicting cases.
- The imaging backbone defaults to random initialisation per the implementation-details section; set
  `model.ife.pretrained=true` to enable ImageNet weights where available.
```
                                                              ____________
   pass / fail .......... see score card                    /            \
   judged by ........... the bench inspection above        |   damascus   |
                                                             \____________/
```
