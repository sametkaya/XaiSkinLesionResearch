# HAM10000 ABC-Scored Dermoscopy Dataset

## Overview
This dataset extends the [HAM10000](https://doi.org/10.1038/sdata.2018.161) 
dermoscopy dataset (10,015 images, 7 diagnostic classes) with 
pseudo-ABC (Asymmetry, Border irregularity, Color variegation) scores 
for each image.

## ABC Scoring Methodology

### Why ABC (not ABCD)?
The diameter (D) criterion is excluded because HAM10000 aggregates images 
from multiple devices at varying magnifications. After standard 224×224 px 
resizing, no pixel-to-physical-scale conversion is possible. See: 
Choi et al. (2024), *Applied Sciences* 14(22), 10294.

### Scoring methods

#### Method 1: DL Regressor (abc_dl)
EfficientNet-B0 backbone (pre-trained on HAM10000) + multi-output 
regression head trained on PH2 + Derm7pt with ground-truth ABC annotations.

#### Method 2: Image Processing (abc_ip)
- **A (Asymmetry):** Principal-axis overlap ratio after reflection
- **B (Border):** Compactness index (4π·Area/Perimeter²), inverted
- **C (Color):** Count of detected standard dermoscopic colors (Argenziano 1998)

### Normalisation
All scores normalised to [0.0, 1.0]:
| Score | 0.0 | 1.0 |
|-------|-----|-----|
| A | Symmetric | Fully asymmetric |
| B | Circular border | Maximally irregular |
| C | Monochromatic | 6 colors present |

## File Structure
```
ham10000_abc_scores.csv      — per-image scores
ham10000_abc_scored.h5       — images + scores (HDF5)
dataset-metadata.json        — Kaggle card
README.md                    — this file
```

## CSV Columns
`image_id, dx, A_dl, B_dl, C_dl, A_ip, B_ip, C_ip, A_mean, B_mean, C_mean`

## HDF5 Groups
`/images (N,3,224,224)  /labels (N,)  /abc_dl (N,3)  /abc_ip (N,3)  /abc_mean (N,3)`

## License
CC BY-NC 4.0 — compatible with the HAM10000 source license.

## Citation
If you use this dataset, please cite:
1. Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
2. Mendonça, T., et al. (2013). PH2. *IEEE EMBC*, 5437–5440.
3. Kawahara, J., et al. (2019). Seven-point checklist. *IEEE JBHI*, 23(2), 538–546.
