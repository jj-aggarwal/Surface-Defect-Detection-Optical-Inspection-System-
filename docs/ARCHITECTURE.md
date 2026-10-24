# Architecture & design notes

```
 image ─► [optics] undistort ─► [preprocess] grayscale + robust flat-field
       ─► [detect] fine / coarse / ridge z-maps ─► hysteresis + scale selection ─► regions
       ─► [classify] geometric+photometric features ─► rules or Random Forest
       ─► [pipeline] mm metrology ─► accept/reject rules ─► verdict, overlay, reports
```

## Why these choices

**Robust flat-fielding (`preprocess.py`).** Uneven illumination is the #1 cause of false rejects.
A low-order polynomial surface is fitted to a down-sampled image with iterative sigma-clipping, so
defects behave as outliers instead of biasing the fit. A large Gaussian blur would partly absorb
big stains, which is why it was not used.

**Three detection scales (`detect.py`).**
| Map | Responds to | How |
|---|---|---|
| fine | pits, dents | Gaussian σ≈1.2 residual |
| coarse | faint stains, discolouration | Gaussian σ≈5 residual |
| ridge | scratches, cracks | multi-scale Hessian, elongated (|λ₁|−|λ₂|) |

Each map is converted to a **robust z-score** (median/MAD), so thresholds are expressed in "noise
units" and transfer across cameras and exposures. The ridge map has heavier noise tails and is
down-weighted (`ridge_weight`). A region needs one pixel above `high_threshold` and grows to
`low_threshold` (hysteresis). Regions dominated by the coarse scale are kept whole (stains); all
others are tightened to a fraction of their sharp peak so cracks and pits keep their real outline.

**Skeleton metrology (`classify.py`).** For elongated defects, length is the skeleton path length and
width is `area / length`, not the blurred bounding box. Tortuosity (path/chord) and lateral wiggle
separate straight scratches from meandering cracks.

**Classifier.** Rule-based by default (transparent, zero training data). An optional Random Forest
on the same 11 features is trained with `sdd train`; on synthetic data it raises classification
accuracy from ~89 % to ~99 %. On real parts, train on your own labelled regions.

**Optics (`optics.py`).** Zhang calibration from checkerboard views gives intrinsics/distortion and a
mm/pixel estimate. For fronto-parallel telecentric setups, a single gauge feature
(`pixel_size_from_reference`) is usually more accurate.

## Optical setup guidance (real deployments)
* **Brushed / machined metal:** low-angle bar or dark-field lighting makes scratches bright.
* **Glossy / painted surfaces:** coaxial bright-field with a diffuse dome to suppress glare.
* **Lock exposure and white balance;** the pipeline normalises illumination *gradients*, not flicker.
* Keep ≥ 3 px across the smallest defect you must catch (pixel size ≤ ⅓ of the minimum defect width).
* Re-calibrate scale whenever the lens, working distance or focus changes.

## Extending
* **Deep learning:** add a detector (e.g. an anomaly model such as PatchCore/PaDiM) behind the same
  `Inspector.inspect` contract: return a label image + score map and reuse classification/metrology.
* **Real-time line:** the pipeline is ~70 ms for 512×512 on one CPU core; tile or down-sample for
  larger sensors, and run one `Inspector` per worker process.
* **PLC / MES integration:** `InspectionResult.to_dict()` is JSON-ready for REST/MQTT/OPC-UA bridges.
