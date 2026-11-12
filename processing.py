import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def resize_rgb(image, max_side=1000):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return cv2.resize(image, (round(w*scale), round(h*scale))) if scale < 1 else image

def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    return np.r_[gray.mean()/255, gray.std()/255, np.percentile(gray,[10,50,90])/255, (gray<50).mean(), (gray>220).mean(), hist]

def gamma_enhance(img, gamma=1.8):
    # gamma > 1 brightens, using exponent 1/gamma
    lut = np.array([np.clip(255*((i/255)**(1/gamma)),0,255) for i in range(256)],dtype=np.uint8)
    return cv2.LUT(img,lut)

def clahe_enhance(img, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=float(clip),tileGridSize=(int(tiles),int(tiles))).apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)

def illumination_enhance(img, strength=0.7, sigma=15):
    # Simplified illumination-map-inspired enhancement; NOT full LIME implementation.
    x=img.astype(np.float32)/255
    illum=np.max(x,axis=2)
    smooth=cv2.GaussianBlur(illum,(0,0),sigmaX=float(sigma))
    adjusted=np.maximum(0.08, (1-strength)+strength*smooth)
    out=np.clip(x/adjusted[:,:,None],0,1)
    return (out*255).astype(np.uint8)

def denoise(img, method='None', kernel=3):
    if method=='Median': return cv2.medianBlur(img,int(kernel))
    if method=='Gaussian': return cv2.GaussianBlur(img,(int(kernel),int(kernel)),0)
    if method=='Mean': return cv2.blur(img,(int(kernel),int(kernel)))
    return img

def metrics(original, enhanced):
    a=original.astype(np.float32); b=enhanced.astype(np.float32)
    return {'Mean brightness (original)':round(float(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY).mean()),2),
            'Mean brightness (enhanced)':round(float(cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY).mean()),2),
            'Contrast std (original)':round(float(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY).std()),2),
            'Contrast std (enhanced)':round(float(cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY).std()),2),
            'PSNR vs input (dB)':round(float(peak_signal_noise_ratio(original,enhanced,data_range=255)),2),
            'SSIM vs input':round(float(structural_similarity(original,enhanced,channel_axis=2,data_range=255)),4)}
