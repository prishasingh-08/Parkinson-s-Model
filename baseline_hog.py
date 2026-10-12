from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
ROOT=Path(__file__).resolve().parent; ART=ROOT/'artifacts'
df=pd.read_csv(ROOT/'NewSpiral.csv',dtype={'ID_PATIENT':str}); df['label']=(df.CLASS_TYPE.astype(int)==2).astype(int)

def locate(name,label):
    c=[name]
    if label==2 and name.startswith('mea'): c.append('sp'+name[3:])
    base=ROOT/'data'/('patient' if label==2 else 'healthy')
    for x in c:
      hits=list(base.rglob(x))
      if hits:return hits[0]
    raise FileNotFoundError(name)
df['path']=[locate(r.IMAGE_NAME,int(r.CLASS_TYPE)) for _,r in df.iterrows()]
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); folds=list(sg.split(df,df.label,groups=df.ID_PATIENT)); _,te=folds[0]; _,va=folds[1]; tr=np.concatenate([folds[i][1] for i in range(2,5)])

def feat(path):
 im=np.array(Image.open(path).convert('L').resize((256,256)))/255.0
 return hog(im,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys')
X=np.vstack([feat(p) for p in df.path]); y=df.label.to_numpy()
model=Pipeline([('scale',StandardScaler()),('svc',SVC(kernel='rbf',C=2,gamma='scale',probability=True,class_weight='balanced',random_state=42))]); model.fit(X[tr],y[tr]); prob=model.predict_proba(X[te])[:,1]; pred=(prob>=.5).astype(int)
metrics={'accuracy':accuracy_score(y[te],pred),'precision':precision_score(y[te],pred,zero_division=0),'recall':recall_score(y[te],pred,zero_division=0),'f1':f1_score(y[te],pred,zero_division=0),'roc_auc':roc_auc_score(y[te],prob)}
print('HOG+SVM TEST',metrics); print(confusion_matrix(y[te],pred)); print(classification_report(y[te],pred,target_names=['Healthy','Parkinson'],zero_division=0)); joblib.dump(model,ART/'hog_svm.joblib'); pd.DataFrame([metrics]).to_csv(ART/'hog_svm_metrics.csv',index=False)
