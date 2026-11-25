# Fall Armyworm Damage Segmentation in Maize (RGB + Thermal)

Pixel-level segmentation and quantification of fall armyworm (*Spodoptera frugiperda*) feeding damage in maize leaves, using a MobileNetV2 U-Net on RGB images, an RGB–thermal fusion model (FiLM layers) and a local Flask web application.

This repository contains the code of my M.Sc. Data Science dissertation (VIT Vellore, 2026): *Pixel-Level Segmentation and Quantification of Fall Armyworm Feeding Damage in Maize Leaves Using RGB and Registered Thermal Imaging*.

## What it does
- Marks every damaged pixel on a maize leaf photo (holes, scraped "window-pane" tissue, ragged edges).
- Reports the damage as a percentage of the visible leaf and of the image.
- Tests whether a registered thermal image improves the segmentation.
- Runs as a local web app: upload a photo, get the damage overlay, mask, damage % and scouting guidance.

The system measures visible feeding damage; it does not identify the insect species.

## Results (held-out test set: 53 damaged + 46 healthy images, used once)
| Metric | Main RGB model |
|---|---|
| Mean per-image Dice (95% CI) | 0.446 (0.404–0.487) |
| Boundary-tolerant Dice (3 px) | 0.604 |
| Damage regions detected | 77.7% (310 of 399) |
| Damage % of image: mean absolute error / bias | 0.77 / +0.05 percentage points |
| Healthy images with false alarm > 1% | 0 of 46 |

RGB–thermal fusion gave no Dice gain (−0.006, p = 0.23) but reduced the false-alarm area on healthy leaves by 48% in cross-validation (p < 0.001).

## Method
- **Data:** 633 paired RGB (640 × 480) and radiometric thermal (320 × 240) FLIR E8 images (325 damaged, 308 healthy).
- **Annotation:** damage polygons drawn in Labelme and converted to binary masks.
- **Splitting:** 301 capture blocks so near-identical frames never cross splits; grouped 5-fold cross-validation plus a locked test set.
- **Model:** U-Net with a frozen ImageNet MobileNetV2 encoder; loss 0.5 BCE + 0.5 Dice; 5-fold ensemble with overlapping 320 px patches.
- **Damage %:** damaged pixels ÷ (leaf ∪ damage), with the leaf found by an Otsu threshold on the CIELAB a* channel.
- **Thermal:** raw values converted to °C, registered to the RGB image with the camera's calibration, and fused at five scales with FiLM layers.

## Repository layout
```
faw_seg/
  src/faw/       core code: data pipeline, segmentation, leaf mask, FLIR reading, thermal registration, fusion model
  scripts/       numbered pipeline: audit, splits, training, evaluation, post-processing, figures
  configs/       experiment settings (e00 RGB baseline, e10 RGB–thermal FiLM)
  webapp/        Flask web application
src/phase1/experiment_06/
  model_experiment_06_mobilenetv2_unet.py   MobileNetV2 U-Net definition
```

## Dataset
The images are from the public dataset of Sandhya et al. (2025), *Frontiers in Agronomy*, doi:10.3389/fagro.2025.1629681. They are not included here. Place them under `dataset/rgb/` and `dataset/flir/` (see `faw_seg/configs/config.yaml`).

## Trained models
The trained weights (5 folds × 2 models, about 800 MB) are not stored in this repository. The web app expects them at
`faw_seg/outputs/experiments/<experiment name>/fold<k>/best.weights.h5`.

## Run the web app (Windows, from the repository root)
```
pip install -r requirements.txt
python faw_seg\webapp\app.py
```
Then open http://127.0.0.1:5000.

## Tools
Python, TensorFlow/Keras, OpenCV, NumPy, pandas, Matplotlib, Labelme, Flask

## Author
Sinha Anushka Arvindkumar, M.Sc. Data Science, VIT Vellore. Guide: Dr. Shubhangi Dwivedi.
