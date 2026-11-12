import io
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from processing import resize_rgb, gamma_enhance, clahe_enhance, illumination_enhance, denoise, metrics
from ai_model import predict

st.set_page_config(page_title='AI RGB Enhancement Studio',page_icon='🌙',layout='wide')
st.title('🌙 AI-Based Low-Light RGB Image Enhancement Studio')
st.caption('SPA mini project | Adjustable parameters • Image enhancement • Demo ML classifier • Quality comparison')
with st.sidebar:
    st.header('Processing controls')
    method=st.selectbox('Enhancement algorithm',['Gamma correction','CLAHE','Illumination-inspired'])
    gamma=st.slider('Gamma (higher = brighter)',0.5,4.0,1.8,0.1)
    clip=st.slider('CLAHE clip limit',1.0,8.0,2.0,0.5)
    tiles=st.slider('CLAHE tile grid',2,16,8,2)
    strength=st.slider('Illumination correction strength',0.0,1.0,0.7,0.05)
    sigma=st.slider('Illumination smoothing sigma',1,50,15)
    noise_filter=st.selectbox('Noise reduction',['None','Median','Gaussian','Mean'])
    kernel=st.select_slider('Filter kernel size',options=[3,5,7,9],value=3)
    ai=st.checkbox('Show AI exposure prediction',value=True)
    auto=st.checkbox('Apply AI-recommended method',value=False)

upload=st.file_uploader('Upload an RGB image',type=['png','jpg','jpeg','webp'])
if upload is None:
    st.info('Upload a low-light image to start. All processing runs locally in your Python environment.')
    st.stop()
try:
    original=resize_rgb(np.array(Image.open(upload).convert('RGB')))
except Exception as e:
    st.error(f'Could not read image: {e}')
    st.stop()

if ai or auto:
    with st.spinner('Loading demonstration ML classifier...'):
        label,confidence=predict(original)
    st.write(f'**AI exposure category:** {label} (model confidence {confidence:.0%})')
    st.caption('Demo classifier trained on synthetic exposure variations of a few example images. Confidence is not clinical or production reliability.')
    if auto:
        method={'Low light':'Illumination-inspired','Normal':'CLAHE','Overexposed':'Gamma correction'}[label]
        if label=='Overexposed': gamma=0.7
        st.info(f'AI selected: {method}' + (' with gamma 0.7' if label=='Overexposed' else ''))

start=time.perf_counter()
if method=='Gamma correction': enhanced=gamma_enhance(original,gamma)
elif method=='CLAHE': enhanced=clahe_enhance(original,clip,tiles)
else: enhanced=illumination_enhance(original,strength,sigma)
enhanced=denoise(enhanced,noise_filter,kernel)
elapsed=time.perf_counter()-start

left,right=st.columns(2)
with left:
    st.subheader('Original RGB image');st.image(original,use_container_width=True)
with right:
    st.subheader('Enhanced RGB image');st.image(enhanced,use_container_width=True)
st.write(f'**Processing time:** {elapsed*1000:.1f} ms | **Algorithm:** {method} | **Filter:** {noise_filter}')

buf=io.BytesIO();Image.fromarray(enhanced).save(buf,format='PNG')
st.download_button('Download enhanced image',data=buf.getvalue(),file_name='enhanced_rgb.png',mime='image/png')

st.subheader('Image statistics and comparison')
st.dataframe([{'Metric':k,'Value':v} for k,v in metrics(original,enhanced).items()],hide_index=True,use_container_width=True)
st.warning('PSNR and SSIM here compare enhanced output to the dark input, NOT to a clean reference. Higher values do not necessarily mean better enhancement.')

fig,ax=plt.subplots(figsize=(9,3))
for img,name in [(original,'Original'),(enhanced,'Enhanced')]:
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ax.plot(np.arange(256),cv2.calcHist([gray],[0],None,[256],[0,256]).ravel(),label=name)
ax.set_xlabel('Pixel intensity');ax.set_ylabel('Count');ax.set_title('Grayscale intensity histogram');ax.legend();fig.tight_layout()
st.pyplot(fig);plt.close(fig)
with st.expander('RGB channel views'):
    for i,name in enumerate(['Red','Green','Blue']):
        a,b=st.columns(2)
        a.image(original[:,:,i],caption=f'Original {name}',clamp=True,use_container_width=True)
        b.image(enhanced[:,:,i],caption=f'Enhanced {name}',clamp=True,use_container_width=True)
with st.expander('Project methodology and limitations'):
    st.markdown('''**Research inspiration:** Guo et al., *LIME: Low-Light Image Enhancement via Illumination Map Estimation* (IEEE TIP, 2017). DOI: 10.1109/TIP.2016.2639450.\n\nThis app implements **a simplified illumination-map-inspired correction**, not the complete LIME algorithm. Gamma and CLAHE are baselines. The AI classifier uses a small synthetic training set and has not been validated on real exposure datasets. For a research report, evaluate with independent test images and, where possible, paired ground-truth reference images.''')
