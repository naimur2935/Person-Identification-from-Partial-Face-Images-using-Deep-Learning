Person Identification from Partial Face Images using Deep Learning
=================================================================

This repository reconstructs masked faces using a U-Net and matches the reconstruction against a detailed embedding database built with FaceNet.

Quick start
-----------
- Create/update embeddings:

  ./venv311/bin/python build_embeddings.py

- Identify a masked image (default shows top-3 matches using avg cosine):

  ./venv311/bin/python identification.py custom_000161.jpg

- Run identification without opening a GUI and save metadata json:

  ./venv311/bin/python identification.py custom_000161.jpg 3 --method ensemble_max --no-show --save-metadata

Matching methods
----------------
- avg_cosine: use the averaged embedding for each image (fast).
- ensemble_max: compare query to all per-image variant embeddings and use the maximum similarity.
- ensemble_mean: mean similarity across variants.
- mahalanobis: convert Mahalanobis distance (wrt per-image covariance) to a similarity score.

Flags
-----
- top_k (positional): number of top matches to show/save (default: 3)
- --method: matching method (choices: avg_cosine, ensemble_max, ensemble_mean, mahalanobis)
- --threshold: recognition threshold (if omitted, use --auto-threshold to load recommended per-method thresholds)
- --auto-threshold: use recommended thresholds from `thresholds_recommended.json` (or specify `--thresholds-file <path>`)- --no-show: save PNG but do not open a display window
- --save-metadata: save a JSON with per-match metadata beside the PNG

Notes: For headless environments, use `--no-show` and open the saved PNG manually (macOS: Preview).

Testing
-------
- Run smoke tests (runs identification for each method and checks outputs):

  ./venv311/bin/python tests/run_identification_tests.py

- Run unit tests for matching utilities:

  ./venv311/bin/python tests/run_unit_tests.py

Calibration & next steps
------------------------
- Thresholds are dataset dependent. To calibrate, run the evaluation script on a labeled validation split to compute ROC/PR curves and select operating points (TODO: add calibration script).

Contributing
------------
- Please open issues or submit PRs for improvements such as GPU support, indexing with Faiss for speed, or identity-level aggregation.
