# NeuroCare classroom demo

A mobile-responsive Flask website that runs three research classifiers: spiral drawings (ResNet18 or HOG+SVM), sustained-vowel voice, and shoe-insole gait sensor recordings. It also includes two comparison profiles with real, correctly classified held-out research examples. The sample histories are curated demonstrations, not representative patients or longitudinal studies.

## Run in VS Code on Windows

Open this folder in VS Code. In the terminal, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:5000`. Stop the app with **Ctrl+C** in the terminal. To restart after changing code, run the last command again.

To view on a phone connected to the same Wi-Fi, use the computer's local IPv4 address, such as `http://192.168.x.x:5000`; allow Python through Windows Firewall on your private network if prompted. Phone microphone access generally requires HTTPS, so use the supplied WAV samples or a hosted HTTPS site for recording. This demo has no real authentication: do not expose it to the public internet or enter real patient data.

## Demo profiles and sample data

Choose **Healthy control demo** or **Parkinson participant demo** on the sign-in screen. Each has three sample dates spaced two days apart, and each date contains spiral, voice, and gait model results. Each model’s score is weighted by its own reported validation accuracy in the combined research index. The dates are illustrative; examples across modalities are from different research participants and do not represent one person's clinical history.

The sample uploads are in `static/demo_samples`. Use them for a quick demonstration:

- Spiral: JPG; select either **ResNet18** or **HOG + SVM**. ResNet18 held-out test accuracy is 69.2%; HOG+SVM validation accuracy is 82.7%. These scores use different evaluation procedures/splits and are not directly comparable. The saved ResNet checkpoint overcalls Parkinson's patterns among healthy drawings, so HOG+SVM is available as an optional alternative.
- Voice: WAV sustained-vowel `/a/`; nested participant-level 5-fold evaluation on 81 people (41 healthy, 40 Parkinson), accuracy 76.5%, ROC-AUC 84.5%.
- Gait: TXT from PhysioNet GaitPDB with 100 Hz insole force data (time plus 18 channels); nested participant-level 5-fold evaluation on 165 people (72 healthy, 93 Parkinson), accuracy 70.9%, ROC-AUC 81.2%. Ordinary walking videos and phone accelerometer data are not compatible.

These metrics are research estimates, not calibrated probabilities, medical validation, or diagnosis. Histories are stored in browser local storage and can be cleared from Settings.

## Research sources and retraining

- Voice: [Voice Samples for Patients with Parkinson's Disease and Healthy Controls](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127), CC BY 4.0.
- Gait: [PhysioNet Gait in Parkinson's Disease (GaitPDB)](https://physionet.org/content/gaitpdb/1.0.0/), Open Data Commons Attribution License 1.0.
- Spiral images and ResNet/HOG model files were already present in the project provided by the user. No dataset provenance/license record for those spiral images was included in the supplied project; the demo does not add a new source attribution.

Training and download scripts are included for the voice and gait models. Full source datasets are not included in this release; the six small demonstration inputs are included under `static/demo_samples`.
