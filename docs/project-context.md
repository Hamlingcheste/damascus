# Project Context

    project_name       : damascus                                    [HIGH]
    domain             : multimodal medical AI — early knee OA        [HIGH]
    framework          : PyTorch 2.1.0 + torchvision DenseNet-201     [HIGH]
    venue              : npj Digital Medicine                         [MED]
    primary_datasets   : 3 datasets (see §6)                          [HIGH]
    compute_target     : 4x A100 80GB, ~4.8 h (OAI)                   [HIGH]
    hparams_reference  : §2.1, §4.11, Tables 7/13                     [HIGH]
    supp_path          : none on disk (SI §S3/§S8 referenced only)
    extra_signals      : Alg.1, Thm.1-3, no released checkpoints

    NEEDS_USER_DECISION: 0
    Status             : resolved; build proceeds.

The model is **RegBridge**, a confidence-gated multimodal fusion network that links continuous
movement characteristics to radiographic knee imaging for early osteoarthritis (OA) assessment. This
package, `damascus`, is one implementation of that method.

## 1. project_name — `damascus`
Derived value: package import name `damascus`; distribution name `damascus`.
Reasoning: the method forge-welds two modality "billets" (movement, imaging) into a single fused
representation under a quality-aware gate — the pattern-welded-steel framing names the package and its
module layout. The science identifier RegBridge is retained in prose and docs.
Paper location: title; Abstract L14-17; §4.2.
Confidence: HIGH.

## 2. supp_path — none
No supplementary file accompanies the manuscript on disk. The text cross-references Supplementary
§S3 (cohort-length sensitivity) and §S8 (full Theorem-1 derivation), but those sections are not
present as a separate artifact. No values below depend on an unavailable SI item.
Confidence: HIGH (absence verified by directory inspection).

## 3. domain — multimodal medical AI for early knee OA
Specific sub-field: per-sample reliability-weighted fusion of wearable/clinical movement data and
radiographic knee imaging, evaluated on Kellgren-Lawrence (KL) grading, early-OA detection, and
48-month progression.
Paper location: title; Abstract; §1; §2.1.
Confidence: HIGH.

## 4. framework — PyTorch 2.1.0
Derived value: PyTorch 2.1.0; torchvision DenseNet-201 backbone (ImageNet weights optional).
Reasoning: §2.1 states "PyTorch 2.1.0"; §4.4 names a DenseNet-201 backbone; §4.3 a dilated-causal-conv
+ multi-head-attention temporal encoder; all expressible in plain `torch.nn` + torchvision. The pinned
2.1.0 is recorded in configs and the Dockerfile per the compute-profile rule.
Paper location: §2.1 L122; §4.4 L489.
Confidence: HIGH.

## 5. venue — npj Digital Medicine
Reasoning: the manuscript naming and structured presentation (graphical abstract, three retrospective
cohorts, Data/Code-availability statements) match an npj Digital Medicine submission; references [24],
[25], [31] are npj Digital Medicine items in the same area.
Paper location: manuscript framing; reference list.
Confidence: MED.

## 6. primary_datasets — 3 public cohorts
Each entry: {name, version, license/access, URL}. Links checked live during build.

1. **OAI — Osteoarthritis Initiative.** n=4,796 participants (ages 45-79), knee radiographs + 14
   physical-performance measures. KL counts 1847/1203/986/534/226. Access: registration + Data Use
   Agreement via the NIMH Data Archive. URL: https://nda.nih.gov/oai (verified live). License: public,
   DUA-gated. Paper location: §2.1 L128-130; Table 1; reference [46].
2. **MOST — Multicenter Osteoarthritis Study.** n=3,026 participants (ages 50-79), knee radiographs +
   clinical measures; cross-dataset cohort. Access: registration + Data Use Agreement at the San
   Francisco Coordinating Center (UCSF). URL: https://mostonline.ucsf.edu (verified live). License:
   public, DUA-gated. NOTE: the manuscript prints `mostudy.org`, which does not resolve to MOST; the
   canonical access point above is used instead (see `deviations.md` D1). Paper location: §2.1 L130-132;
   reference [47].
3. **GaitOA — clinical gait signals with wearable sensors.** 1,356 trials from 260 subjects, four IMUs
   at 100 Hz; the 166-subject healthy/hip-OA/knee-OA subset is used. Access: open on figshare. URL:
   https://springernature.figshare.com/articles/dataset/A_Dataset_of_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological_and_Orthopedic_Cohorts/28806086
   (DOI 10.6084/m9.figshare.28806086; dataset paper DOI 10.1038/s41597-025-05959-w). License: open.
   Paper location: §2.1 L132-135; §4.10 L582-587; reference [44].

No paired raw-IMU + imaging public cohort exists; the study triangulates the three above. This package
ships a deterministic synthetic cohort generator for tests and demonstration; it does not redistribute
any dataset.

## 7. compute_target — 4x A100 80GB
Derived value: 4x NVIDIA A100 80GB; effective batch 64; 120 epochs; ~4.8 h wall-clock on OAI;
inference 18.4 ms/sample; model 47.3M params, 9.4 GFLOPs.
Paper location: §2.1 L122-126; §2.8 L227-229; Table 8.
Confidence: HIGH.

## 8. hparams_reference — §2.1, §4.11, Tables 7/13
- optimizer AdamW; weight decay 1e-2; peak lr ~3e-4 (Table 13 cosine-restart trajectory).
- scheduler CosineAnnealingWarmRestarts, T0=10, Tmult=2; 120 epochs; early stopping on validation
  AUROC, patience 15.
- batch size 64; precision FP16 mixed; 5-fold CV; significance by paired permutation test (10,000).
- label smoothing eps=0.1; confidence-loss weight lambda_c=0.1.
- embedding d=512 (OAI/MOST), d=256 (GaitOA); TME L=6, d_m=256 (OAI), L=4, d_m=128 (GaitOA);
  attention heads N_h=8; SE reduction r=16; cross-attention dim d_ca=256; dropout 0.2 (encoders),
  0.3 (classification head).
- seeds {42, 123, 456, 789, 1024}.
Paper location: §2.1 L122-126; §4.7 L543; §4.8 L547-551; §4.11 L593-598; Tables 7 and 13.
Confidence: HIGH.

## 9. extra_signals
- Algorithm 1 (RegBridge training procedure) — implemented in the trainer step.
- Theorems 1-3 (precision-weighted gating optimality, stage-dependent margin, graceful
  missing-modality) — exercised by tests (missing-modality invariance, gating direction).
- No proprietary tokenizer; no released checkpoints ("available upon reasonable request").
- Verbatim Code-Availability, Data-Availability, and Citation text are retained in
  `implementation-map.md` only; the README omits those three sections by project convention.
- Ethics: secondary use of de-identified public benchmarks; no IRB required; built-in
  confidence-based fairness indicators noted, no specific quantitative fairness demand.
