# Research dataset credits and sample provenance

- Voice recordings and labels: Fred Prior, Tuhin Virmani, Anu Iyer, Linda Larson-Prior, Aaron Kemp, Yasir Rahmatallah, Lakshmi Pillai, and Aliyah Glover, “Voice Samples for Patients with Parkinson’s Disease and Healthy Controls,” Figshare (2023), CC BY 4.0. https://doi.org/10.6084/m9.figshare.23849127
- Gait recordings and labels: Jeffrey Hausdorff, “Gait in Parkinson’s Disease,” PhysioNet GaitPDB v1.0.0, Open Data Commons Attribution License 1.0. https://doi.org/10.13026/C24H3N
- Spiral examples and saved ResNet18/HOG+SVM model artifacts came from files already present in the user-provided project. Its supplied files did not include a dataset source/license record, so no further provenance is asserted here.

The release includes one selected real sample per class and modality under `static/demo_samples`. Demo histories use correctly classified held-out examples selected from the available evaluation predictions. The examples are intentionally curated to make a classroom comparison easy; they are not representative estimates of model performance. Three display dates are illustrative and combine different participants across modalities. No one-participant longitudinal histories are implied.

The full voice and gait source archives are not included. See `README.md` for the dataset pages and training steps.
