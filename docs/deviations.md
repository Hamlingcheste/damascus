# Deviations

Intentional departures from the manuscript, each with a paper-section link and justification.

## D1 — MOST access URL
- **Where:** §2.1 (Datasets); Data Availability statement.
- **Manuscript value:** MOST is listed at `https://mostudy.org`.
- **Deviation:** that host does not resolve to the Multicenter Osteoarthritis Study (it serves an
  unrelated test-preparation site). The canonical access point `https://mostonline.ucsf.edu`
  (San Francisco Coordinating Center, UCSF) is used in `configs/data/most.yaml`, `project-context.md`,
  and the README.
- **Justification:** the printed link is non-functional for its stated purpose; the substitute is the
  recognised MOST data portal. No scientific change.

## D2 — Synthetic cohorts in place of restricted data
- **Where:** §2.1; §4.1 Definition 1; §4.10.
- **Manuscript value:** experiments run on OAI, MOST, and GaitOA, all access-gated or external.
- **Deviation:** the package ships a deterministic synthetic cohort generator
  (`stock/cohorts.py`) that realises the heteroscedastic-Gaussian model of Definition 1 (latent
  severity with stage-dependent per-modality noise). It is used by tests, the smoke configuration, and
  for demonstration; no dataset is redistributed.
- **Justification:** the datasets cannot be embedded; a generator faithful to Definition 1 lets the
  full pipeline and the gating behaviour (Theorems 1-3) be exercised offline. Real-data adapters read
  from the documented sources once a user supplies them.

## D3 — Pretrained DenseNet-201 weights optional
- **Where:** §4.4; §4.11 ("There was no pre-trained DenseNet-201 weighting set used in IFE
  pre-calculated using ImageNet").
- **Manuscript value:** the imaging encoder is described as ImageNet-pretrained end-to-end fine-tuned
  in §4.4, while §4.11 states no ImageNet pre-training was used for the IFE.
- **Deviation:** the backbone defaults to randomly initialised weights (matching §4.11); ImageNet
  weights can be enabled by config when available. The two manuscript statements conflict; §4.11 (the
  implementation-details section) is treated as authoritative.
- **Justification:** resolves an internal manuscript inconsistency toward the more specific source and
  keeps the build self-contained offline.

## D4 — Auxiliary unimodal supervision term
- **Where:** §4.8 Eq. (12); Fig. 7 caption.
- **Manuscript value:** Eq. (12) writes the objective as `L = L_cls + lambda_c * L_conf`; the unimodal
  losses `l_M`, `l_I` appear only as targets inside `L_conf` (Eq. 14).
- **Deviation:** the auxiliary unimodal cross-entropies `l_M + l_I` are added to the total objective
  with weight `train.aux_weight` (default 1.0). The confidence-calibration targets remain detached.
- **Justification:** Fig. 7 states the auxiliary classifier of each unimodal branch "is used to
  provide additional training supervision signals to assist in the confidence-calibration process."
  Without this term the auxiliary heads receive no gradient (their losses are detached targets), so the
  reliabilities feeding Eq. 14 would be uninformative. Setting `aux_weight=0` recovers Eq. (12) exactly.

## D5 — Precision-weight assignment in Theorem 1
- **Where:** §4.6 Eq. (10).
- **Manuscript value:** the printed MVUE writes the weight on `h_M` as
  `sigma_I^-2 / (sigma_M^-2 + sigma_I^-2)` and the weight on `h_I` as `sigma_M^-2 / (...)`.
- **Deviation:** the gating direction asserted in tests follows standard inverse-variance combination,
  i.e. the more reliable (lower-variance) modality receives the larger weight
  (`sigma_M^-2 / (sigma_M^-2 + sigma_I^-2)` on `h_M`). The two numerators in Eq. (10) appear
  transposed relative to that convention and to the paper's own narrative (motion dominant when its
  noise is small at KL0-1).
- **Justification:** the learned confidence gate is supervised by Eq. (14), not by Eq. (10) directly,
  so no code path implements the printed weights; the test encodes the physically correct direction
  that Theorem 1 and §2.6 describe.
