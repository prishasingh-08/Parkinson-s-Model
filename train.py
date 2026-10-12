import os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODEL_DIR = ROOT / 'artifacts'
MODEL_DIR.mkdir(exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# ---------- Build dataframe ----------
CSV=ROOT/'NewSpiral.csv'
df=pd.read_csv(CSV,dtype={'ID_PATIENT':str})

def find_image(name, label):
    # Search inside the extracted HealthySpiral / PatientSpiral folders
    roots = [
        Path("HealthySpiral"),
        Path("PatientSpiral")
    ]

    # Try the filename exactly as it appears in the CSV
    candidates = [name]

    # Patient images may use mea1-P... in the CSV but sp1-P... on disk
    if label == 2 and name.startswith("mea"):
        candidates.append("sp" + name[3:])

    for root in roots:
        if not root.exists():
            continue

        for candidate in candidates:
            hits = list(root.rglob(candidate))
            if hits:
                return str(hits[0])

    return None

df['label'] = (df['CLASS_TYPE'].astype(int)==2).astype(int) # 0 healthy, 1 Parkinson
# locate actual images
paths=[]
for _,r in df.iterrows(): paths.append(find_image(r.IMAGE_NAME, int(r.CLASS_TYPE)))
df['path']=paths
missing=df[df.path.isna()]
if len(missing):
    raise RuntimeError(f'Missing {len(missing)} images, e.g. {missing.IMAGE_NAME.head().tolist()}')
print('Rows:',len(df),'Patients:',df.ID_PATIENT.nunique())
print('Class counts:',df.label.value_counts().sort_index().to_dict())

# ---------- Patient-safe 70/15/15 split ----------
# 5-fold stratified group split: one fold test, one fold validation, remaining folds train
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(df,df.label,groups=df.ID_PATIENT))
_,te_idx=folds[0]
_,va_idx=folds[1]
train_idx=np.concatenate([folds[i][1] for i in range(2,5)])
train=df.iloc[train_idx].reset_index(drop=True)
val=df.iloc[va_idx].reset_index(drop=True)
test=df.iloc[te_idx].reset_index(drop=True)
print('Split patients:',train.ID_PATIENT.nunique(),val.ID_PATIENT.nunique(),test.ID_PATIENT.nunique())
print('Split rows:',len(train),len(val),len(test))
assert set(train.ID_PATIENT).isdisjoint(val.ID_PATIENT)
assert set(train.ID_PATIENT).isdisjoint(test.ID_PATIENT)
assert set(val.ID_PATIENT).isdisjoint(test.ID_PATIENT)
for name,x in [('train',train),('val',val),('test',test)]: print(name,'classes',x.label.value_counts().sort_index().to_dict())

train.to_csv(MODEL_DIR/'train_split.csv',index=False); val.to_csv(MODEL_DIR/'val_split.csv',index=False); test.to_csv(MODEL_DIR/'test_split.csv',index=False)

# ---------- Transforms ----------
mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
train_tf=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0,translate=(0.05,0.05),scale=(0.95,1.05)),
    transforms.ToTensor(), transforms.Normalize(mean,std)
])
eval_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])

class SpiralDataset(Dataset):
    def __init__(self, frame, tf): self.frame=frame.reset_index(drop=True); self.tf=tf
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        im=Image.open(r.path).convert('RGB')
        return self.tf(im), torch.tensor(int(r.label),dtype=torch.long)

tr_ds=SpiralDataset(train,train_tf); va_ds=SpiralDataset(val,eval_tf); te_ds=SpiralDataset(test,eval_tf)
counts=np.bincount(train.label.values,minlength=2)
weights=np.array([1/max(counts[int(y)],1) for y in train.label])
sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(weights),replacement=True)
tr_loader=DataLoader(tr_ds,batch_size=16,sampler=sampler,num_workers=0)
va_loader=DataLoader(va_ds,batch_size=32,shuffle=False,num_workers=0)
te_loader=DataLoader(te_ds,batch_size=32,shuffle=False,num_workers=0)

# ---------- Model ----------
try:
    weights_enum=models.ResNet18_Weights.DEFAULT
    model=models.resnet18(weights=weights_enum)
    pretrained=True
    print('Using ImageNet-pretrained ResNet18.')
except Exception as e:
    print('Pretrained weights unavailable, using ResNet18 from scratch:', e)
    model=models.resnet18(weights=None)
    pretrained=False
if pretrained:
    for p in model.parameters(): p.requires_grad=False
model.fc=nn.Sequential(nn.Dropout(0.35),nn.Linear(model.fc.in_features,2))
model=model.to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.fc.parameters() if pretrained else model.parameters(),lr=1e-3,weight_decay=1e-4)

@torch.no_grad()
def predict(loader):
    model.eval(); ys=[]; ps=[]; probs=[]
    for x,y in loader:
        x=x.to(DEVICE); out=model(x); pr=torch.softmax(out,1)[:,1]
        ys.extend(y.numpy()); ps.extend(out.argmax(1).cpu().numpy()); probs.extend(pr.cpu().numpy())
    return np.array(ys),np.array(ps),np.array(probs)

def metrics(y,p,prob):
    return {'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'roc_auc':float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None}

best=-1; best_state=None; patience=5; stale=0; history=[]
for epoch in range(1,11):
    model.train(); running=0
    for x,y in tr_loader:
        x,y=x.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step(); running+=loss.item()*len(y)
    vy,vp,vprob=predict(va_loader); m=metrics(vy,vp,vprob); history.append({'epoch':epoch,'train_loss':running/len(tr_ds),**{f'val_{k}':v for k,v in m.items()}})
    print(f"Epoch {epoch:02d} loss={running/len(tr_ds):.4f} val_acc={m['accuracy']:.3f} val_f1={m['f1']:.3f} val_auc={m['roc_auc']:.3f}")
    if m['f1']>best:
        best=m['f1']; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; stale=0
    else: stale+=1
    if stale>=patience: print('Early stopping.'); break

model.load_state_dict(best_state)
torch.save({'model_state_dict':model.state_dict(),'class_names':['Healthy','Parkinson'],'architecture':'resnet18','pretrained':pretrained,'input_size':224,'mean':mean,'std':std},MODEL_DIR/'resnet18_spiral.pth')

ty,yp,tprob=predict(te_loader); tm=metrics(ty,yp,tprob)
print('\nTEST METRICS:',json.dumps(tm,indent=2))
print('\nCONFUSION MATRIX [rows=true, cols=pred]:\n',confusion_matrix(ty,yp))
print('\nREPORT:\n',classification_report(ty,yp,target_names=['Healthy','Parkinson'],zero_division=0))
with open(MODEL_DIR/'test_metrics.json','w') as f: json.dump(tm,f,indent=2)
pd.DataFrame(history).to_csv(MODEL_DIR/'training_history.csv',index=False)

cm=confusion_matrix(ty,yp)
fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm); ax.set_xticks([0,1],['Healthy','Parkinson']); ax.set_yticks([0,1],['Healthy','Parkinson']); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Test Confusion Matrix')
for i in range(2):
 for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
fig.tight_layout(); fig.savefig(MODEL_DIR/'confusion_matrix.png',dpi=160); plt.close(fig)
print('Saved:',MODEL_DIR/'resnet18_spiral.pth')
