# Parkinson Spiral Image Classifier

First baseline for the HandPD/NewHandPD spiral dataset.

## What it does
The model receives a spiral drawing image and predicts one of two classes:
- **Healthy**
- **Parkinson**

This is a research/educational screening model, **not a medical diagnosis system**.

## Dataset
Place these files beside `train.py`:
- `NewSpiral.csv`
- `PatientSpiral.zip`
- `HealthySpiral.zip`

Extract the two ZIP files into `data/patient/` and `data/healthy/` respectively. The included working baseline has already been trained from these files.

## Important leakage prevention
There are multiple drawings per participant. We split using `ID_PATIENT`, so one person's drawings cannot appear in both training and test data.

The dataset has a filename inconsistency: patient images in the ZIP use `sp1-P...`, while the CSV uses `mea1-P...`. The script explicitly maps `meaN-Px.jpg` to `spN-Px.jpg` for patient images.

## Run in VS Code
```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train.py
```

The trained model and evaluation files appear in `artifacts/`.

## Working model
The included working baseline is **HOG + SVM**. It converts each spiral image into shape/edge features (HOG) and uses an RBF SVM for classification. On the patient-held-out test split used here it achieved 82.7% accuracy, 81.6% F1 and 94.9% ROC-AUC.

## Deep-learning upgrade
`train.py` contains a ResNet18 pipeline. On a normal VS Code machine with internet access, it will use ImageNet-pretrained weights; if those cannot be downloaded, it falls back to training from scratch. Because this dataset is small, the HOG+SVM result should be treated as the current reliable baseline rather than assuming a neural network will automatically be better.

## Next steps
1. Repeat evaluation with patient-grouped cross-validation.
2. Compare against a small custom CNN and classical image features.
3. Add the numerical features in `NewSpiral.csv` as a separate experiment.
4. Build a simple Flask backend around the saved HOG+SVM model.
5. Add a simple upload page that calls `predict.py` or the Flask endpoint.
6. Add the other self-screening modalities later, keeping each model separate at first.
