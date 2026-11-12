"""Small demo classifier trained on synthetic exposure variants of built-in images.
This is a demonstration of ML integration, not a validated exposure classifier.
"""
from functools import lru_cache
import numpy as np
from skimage import data
from sklearn.ensemble import RandomForestClassifier
from processing import features

@lru_cache(maxsize=1)
def get_model():
    base = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket()]
    X=[]; y=[]
    rng=np.random.default_rng(42)
    for img in base:
        if img.ndim != 3: continue
        img=img[:,:,:3].astype(np.float32)/255
        for label, factors in [('Low light',(.15,.25,.35,.45)),('Normal',(.75,.9,1.0,1.1)),('Overexposed',(1.6,2.0,2.5,3.0))]:
            for factor in factors:
                for _ in range(8):
                    v=np.clip(img*factor + rng.normal(0,.008,img.shape),0,1)
                    X.append(features((v*255).astype(np.uint8)))
                    y.append(label)
    model=RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42)
    model.fit(X,y)
    return model

def predict(image):
    model=get_model()
    label=model.predict([features(image)])[0]
    confidence=float(model.predict_proba([features(image)]).max())
    return label, confidence
