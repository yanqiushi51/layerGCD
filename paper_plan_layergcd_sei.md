# LayerGCD-SEI WCL Paper Plan

## Working Title

LayerGCD-SEI: Hierarchical Generalized Category Discovery for Open-World Radar Specific Emitter Identification

## Central Claim

Conventional open-set SEI mainly answers whether a signal belongs to a known emitter or a single unknown bin. LayerGCD-SEI targets a stricter radar surveillance setting: the system must preserve known-emitter recognition while discovering multiple unseen emitters without being given the number of novel classes. The paper should therefore emphasize K+N discovery first, and report K+1 OS-ACC only as a fair projection for comparison with existing OSR-SEI methods.

## Contributions

1. We formulate radar SEI as generalized category discovery, where unlabeled data may include both known emitters and multiple novel emitters.
2. We adapt LayerGCD to RF fingerprints through a dual-stream temporal-fractal encoder, reconstruction pre-training, and decoupled SupCon/SimCLR learning.
3. We replace fixed-K discovery with cosine-distance hierarchical agglomerative discovery, where the merge threshold is calibrated from labeled known emitters instead of requiring the number of novel emitters.
4. We evaluate both K+N discovery metrics and K+1 projected OS-ACC, making the work comparable with existing open-set SEI while showing the extra value of novel-emitter separation.

## Experimental Design

### Table 1: Protocol

| Experiment | Dataset | Split | Purpose | Report |
| :--- | :--- | :--- | :--- | :--- |
| A1 SNR robustness | LFM 30/35/40/45/50 dB | 7+3 | Noise robustness under fixed openness | OS-ACC, All, Old, New, AUROC, NMI, ARI, #Pred |
| A2 openness | LFM 40 dB | 9+1, 8+2, 7+3, 6+4, 5+5 | Novel-emitter ratio sensitivity | Same as A1 |
| A3 waveform transfer | BPSK, FMCW, Frank | 4+2 or 7+3 | Robustness across radar/communication waveforms | Same as A1 |
| B ablation | LFM 30 dB | 7+3, 6+4, 5+5 | Validate fractal branch and reconstruction pre-training | Full vs no_fractal vs no_recon vs pure_base |
| C discovery dynamics | LFM 40 dB | 7+3 | Show pseudo-label promotion behavior | Iteration, selected purity, promoted count, #Pred |

### Main Metrics

| Metric | Meaning | Why it matters |
| :--- | :--- | :--- |
| OS-ACC | K+1 projected accuracy after merging all non-known discoveries into unknown | Fair comparison with open-set SEI |
| All ACC | Hungarian-aligned accuracy across all known and novel emitter identities | Overall K+N discovery quality |
| Old ACC | Accuracy on known emitters | Whether known identities remain stable |
| New ACC | Accuracy on novel emitters after cluster-label alignment | Whether unknown emitters are separated, not merely rejected |
| AUROC | Known-vs-unknown separability using known prototype similarity | Threshold-free open-set separability |
| NMI/ARI | Clustering agreement | Discovery structure quality |
| #Pred | Number of clusters discovered by HAC | Whether the method estimates novel structure reasonably |
| tau | Automatically estimated cosine-distance threshold | Reproducibility of the parameter-free discovery step |

## Paper Structure

### Abstract

State the limitation of K+1 open-set SEI, then introduce K+N radar emitter discovery without known novel-class number. Mention temporal-fractal encoder, reconstruction pre-training, decoupled contrastive learning, and hierarchical discovery. End with the strongest result once the clean experiment table is available.

### Introduction

Paragraph 1: SEI matters for physical-layer security and radar spectrum surveillance.

Paragraph 2: Existing open-set SEI rejects unknown emitters but collapses all unknowns into one bin, which is insufficient for emitter inventory update, threat triage, and later re-identification.

Paragraph 3: GCD is a natural but underexplored formulation for radar SEI because unlabeled monitoring data contains both known and unseen emitters, and the number of unseen emitters is unknown.

Paragraph 4: Contributions, matching the four items above.

### System Model and Problem Formulation

Define received RF signal, hardware fingerprint distortion, labeled known set, unlabeled mixed set, and target assignment in `Y_known union Y_novel`. Make the distinction between K+N and K+1 explicit:

```tex
\hat{y}^{K+1} =
\begin{cases}
\hat{y}, & \hat{y}\in\mathcal{Y}_K,\\
\mathrm{unknown}, & \mathrm{otherwise}.
\end{cases}
```

This equation justifies why OS-ACC can be reported while the actual method still discovers multiple novel emitters.

### Method

Subsection 1: Dual-stream temporal-fractal encoder.

Subsection 2: Reconstruction pre-training and decoupled contrastive learning.

Subsection 3: Hierarchical category discovery. Describe L2-normalized embeddings, known-emitter distance calibration, and average-linkage agglomeration. Emphasize that the method does not require ground-truth total class count or novel-emitter labels.

### Experiments

Use `paper_layergcd_outputs/experiment_results_layergcd.md` as the source of truth. Do not reuse numbers from the old IEEE template. Every table should cite the clean runner settings and whether the row is `final_iter` or `best_iter`; main paper should use `final_iter`.

### Discussion

Keep it short for WCL. The key interpretation is that K+1 OS-ACC shows competitive rejection ability, while New ACC and #Pred reveal the additional discovery capability that K+1 baselines cannot measure.

### Conclusion

One paragraph: LayerGCD-SEI moves radar SEI from rejection-only open-set recognition toward open-world emitter discovery. Mention limitations: channel drift, real over-the-air captures, and online cluster-number adaptation.

## Text Snippets

### Metric Protocol

For fair comparison with conventional open-set SEI, we report OS-ACC under the K+1 projection. Specifically, after hierarchical discovery, clusters aligned to known emitters retain their known labels, whereas all remaining clusters are merged into a single unknown label. This projection evaluates whether the model can reject unseen emitters. In contrast, All ACC, Old ACC, New ACC, NMI, ARI, and the discovered cluster count evaluate the stricter K+N generalized discovery task.

### Method Summary

LayerGCD-SEI first learns RF fingerprint embeddings through reconstruction-guided initialization and decoupled contrastive refinement. The temporal branch captures waveform morphology, while the differentiable fractal branch emphasizes multi-scale residual roughness caused by hardware imperfections. In the discovery stage, L2-normalized embeddings are grouped by cosine-distance hierarchical agglomeration. The merge threshold is calibrated from labeled known-emitter embeddings, avoiding the need to specify the number of novel emitters or use novel-emitter labels.

## Immediate Execution Plan

1. Run `pilot` to verify the clean pipeline and table generator.
2. Run `core` with at least three seeds for A1/A2/A3 main claims.
3. Run `ablation` on LFM 30 dB to validate the two method modules.
4. Run `iteration` only if WCL space allows a compact diagnostic table.
5. Freeze `paper_layergcd_outputs/experiment_results_layergcd.md` as the only numeric source for the manuscript.

For the detailed executable matrix, use `paper_layergcd_outputs/complete_experiment_design.md`.
