# Implementation Map

Every numbered equation, theorem, algorithm box, and reported table is traced to the module that
realises it. Source files carry no comments or docstrings by project convention; this table is the
single provenance record.

| paper item | location | file | symbol | notes |
|---|---|---|---|---|
| Heteroscedastic multimodal model | Def. 1, §4.1 | `stock/cohorts.py` | `forge_split`, `_modality_sigmas` | latent z; h_M, h_I as noisy views; sigma_M large at KL0-1, sigma_I large at KL3-4 |
| Predictor objective | §4.1 Eq. caption | `blade.py` | `RegBridge.forward` | f: X_I x X_M -> simplex with explicit modality-weight heads |
| TME residual dilated block | Eq. (1), §4.3 | `billets/motion_billet.py` | `DilatedCausalBlock` | dilation 2^(l-1), k=3, BN, ReLU, residual |
| TME self-attention | Eq. (2), §4.3 | `billets/motion_billet.py` | `TemporalSelfAttention` | N_h=8, scaled dot-product, time-pooled output |
| IFE channel attention | Eq. (3), §4.4 | `billets/imaging_billet.py` | `ChannelGate` | squeeze-excitation, reduction r=16 |
| IFE spatial attention | Eq. (4), §4.4 | `billets/imaging_billet.py` | `SpatialGate` | 7x7 conv over avg/max pooled channels |
| Confidence heads | Eq. (5), §4.5 | `forge/heat.py` | `ConfidenceHead` | MLP d->d/4->1, sigmoid |
| Quality-aware gating | Eq. (6), §4.5 | `forge/heat.py` | `quality_gate` | h~ = c * h |
| Gated bilinear fusion | Eq. (7)-(8), §4.5 | `forge/weld.py` | `BilinearBridge` | bidirectional M->I, I->M elementwise interaction |
| Residual fusion | Eq. (9), §4.5 | `forge/weld.py` | `RegulatoryBridge.forward` | LayerNorm(h~_M+h~_I+h_{M->I}+h_{I->M}) |
| Classification head | Eq. (11), §4.7 | `forge/bevel.py` | `ClassificationEdge` | FC->GELU->dropout(0.3)->FC->softmax |
| Total loss | Eq. (12), §4.8 | `tongs/stresses.py` | `total_loss` | L_cls + lambda_c * L_conf |
| Label-smoothed CE | Eq. (13), §4.8 | `tongs/stresses.py` | `smoothed_cross_entropy` | eps=0.1 |
| Confidence calibration loss | Eq. (14), §4.8 | `tongs/stresses.py` | `confidence_calibration` | relative soft targets from unimodal losses |
| Precision-weighted MVUE | Thm. 1, Eq. (10) | `forge/heat.py` (gating) + `tests/test_calibration_and_gating.py` | `quality_gate` | gating recovers inverse-variance weights (see deviations D5) |
| Stage-dependent margin | Thm. 2, §4.6 | `proof/bend_tests.py` | `stagewise_auroc` | per-stage Delta reported (Table 6) |
| Graceful missing-modality | Thm. 3, §4.6 | `stock/work_order.py` + `tests/test_missing_modality.py` | `mask_modality` | zeroed stream -> single-modality behaviour |
| Training procedure | Alg. 1, §4.8 | `hardening/heat_treat.py` | `Smith.step` | TME/IFE/confidence/gate/fuse/aux/loss/AdamW |
| Complexity | §4.9 | (analysis only) | — | no code item |
| Imaging preprocessing | §4.10 L563-566 | `stock/quench_tank.py` | `radiograph_augment`, `normalise_image` | 224x224, ImageNet norm, flip/jitter/erase |
| Pseudo-temporal motion encoding | §4.10 L567-579 | `stock/quench_tank.py` | `sinusoidal_motion_sequence` | T=32 sinusoidal bases over 14 measures |
| GaitOA IMU processing | §4.10 L582-590 | `stock/cohorts.py` | `gait_signal_block` | T=256 stream + spectrogram proxy via config |
| Metrics (AUROC/ACC/kappa/F1) | Tables 2-4 | `proof/edge_tests.py` | `classification_metrics` | quadratic kappa, macro-F1, sens/spec |
| Bootstrap CI / permutation / DeLong | §2.1, §2.9, Table 9 | `proof/assay.py` | `bootstrap_ci`, `paired_permutation`, `delong_auc_test` | 10,000 permutations |
| OAI / MOST / GaitOA task results | Tables 2-4 | `proof/bend_tests.py` | `evaluate_model`, `collect_outputs` | per-task via configs (kl5/early_oa/gait3) |
| Ablation (variants A-G) | Table 5 | `configs/experiment/ablation_*.yaml` | model toggles | RBM/conf-heads/TME/positional toggles |
| Stage-wise AUROC | Table 6 | `proof/bend_tests.py` | `stagewise_auroc` | KL boundary tasks |
| Hyperparameter sensitivity | Table 7 | `configs/experiment/supplementary_sensitivity.yaml` | `--set` L / lambda_c / d_ca sweeps | optimal recorded; others via overrides |
| Computational cost | Table 8 | `proof/edge_tests.py` | `count_parameters` | parameter count (M) |
| Robustness | Table 10 | `proof/bend_tests.py` | `evaluate_robustness` | missing modality + motion SNR |
| Confusion matrix | Table 12 | `proof/edge_tests.py` | `normalised_confusion` | normalised 5-class |
| Per-fold / training dynamics | Tables 13/15 | `hardening/heat_treat.py` | `cross_validate` | 5-fold AUROC |

## Verbatim manuscript statements (retained here; omitted from README by convention)

**Data Availability (verbatim).** "The datasets are all available to the public: OAI at
https://nda.nih.gov/oai, MOST at https://mostudy.org, and GaitOA (see [44]). None of the currently
existing datasets has paired raw IMU or imaging from a common set of subjects, thus the current
evaluation has made use of proxy combinations as described in Methods."

**Code Availability (verbatim).** "Full source code for the entire RegBridge system (including
preprocessing, all components of the TME/IFE/RBM in PyTorch, scripts for training and testing; per-fold
predictions; pretrained checkpoints) will be made available upon request from editors and reviewers
during the peer review process through the editorial system and, if accepted, will be released via
permissive open source license."

**Citation.**
```bibtex
@article{regbridge_npjdm,
  title   = {Neural Regulation Meets Imaging: A Multimodal Framework Combining Movement Data and
             Radiographic Imaging for Early Osteoarthritis Assessment in Active and Athletic Populations},
  journal = {npj Digital Medicine},
  year    = {2025}
}
```
