# Maize Leaf Damage Analyzer (local web app)

Upload an RGB photo or a FLIR E8 radiometric JPG; the app returns
- pixel-level damage segmentation (overlay + binary mask, downloadable at the original resolution),
- % of visible leaf area damaged and % of image damaged,
- a numbered list of damaged regions with location (upper-left … lower-right) and size,
- a descriptive damage level and management guidance (CIMMYT/USAID FAW IPM guide, 2018),
- a JSON report.

Models: RGB photo -> main RGB model e00 (5-fold ensemble, thr 0.35, min region 100 px);
FLIR radiometric JPG -> built-in photo + registered thermal -> RGB-thermal FiLM model e10
(5-fold ensemble, thr 0.30, min region 50 px). Thermal-only pictures without the built-in photo
are rejected (the models need the colour image).

## Run (Windows, from the project root)
    pip install flask
    python faw_seg\webapp\app.py
Then open http://127.0.0.1:5000 . Loading the 10 models takes about a minute.

Files: app.py (server, model loading), inference.py (analysis), advice.py (levels + guidance),
templates/index.html (page). Uses faw_seg/src/faw (flir, thermal, leaf, seg, data, fusion).
