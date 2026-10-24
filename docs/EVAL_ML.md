# Evaluation report

| Metric | Value |
|---|---|
| Images | 200 |
| Defect precision | 1.000 |
| Defect recall | 0.997 |
| Defect F1 | 0.998 |
| Pixel IoU | 0.758 |
| Classification accuracy (matched defects) | 0.997 |
| Image-level accuracy | 1.000 (TP 160 / FP 0 / FN 0 / TN 40) |
| Latency mean / p95 | 88.6 ms / 129.4 ms |

**Confusion matrix** (rows = ground truth, columns = predicted)

| | scratch | crack | pit | stain |
|---|---|---|---|---|
| scratch | 71 | 0 | 0 | 0 |
| crack | 1 | 88 | 0 | 0 |
| pit | 0 | 0 | 78 | 0 |
| stain | 0 | 0 | 0 | 73 |
