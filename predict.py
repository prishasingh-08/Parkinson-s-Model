from pathlib import Path
import sys
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib
ROOT=Path(__file__).resolve().parent
MODEL=joblib.load(ROOT/'artifacts'/'hog_svm.joblib')

def extract(path):
 im=np.array(Image.open(path).convert('L').resize((256,256)))/255.0
 return hog(im,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys')
if len(sys.argv)!=2:
 print('Usage: python predict.py path/to/spiral.jpg'); raise SystemExit(1)
x=extract(sys.argv[1]).reshape(1,-1); p=float(MODEL.predict_proba(x)[0,1]); label='Parkinson' if p>=.5 else 'Healthy'
print(f'Prediction: {label}')
print(f'Parkinson probability: {p:.3f}')
print('NOTE: Research/educational screening only. Not a medical diagnosis.')
