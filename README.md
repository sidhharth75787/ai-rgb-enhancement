# AI-Based Low-Light RGB Image Enhancement Studio

An interactive, software-only SPA mini project for four students. Built with Streamlit, OpenCV and a small scikit-learn demonstration classifier.

## Run on Windows (VS Code terminal)

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `py` is unavailable, use `python` instead. On macOS/Linux activate with `source .venv/bin/activate`.

## Features
- Upload JPEG/PNG/WebP RGB images
- Gamma, CLAHE, simplified illumination-map-inspired enhancement
- Editable gamma, CLAHE clip/tile size, illumination strength/smoothing
- Median/Gaussian/mean smoothing with adjustable kernel size
- Side-by-side images, channel views, histograms, processing time, image statistics
- Download enhanced image
- AI demo exposure classifier and optional automatic method choice

## Research paper
X. Guo, Y. Li, H. Ling, “LIME: Low-Light Image Enhancement via Illumination Map Estimation,” IEEE Transactions on Image Processing, 2017. DOI: https://doi.org/10.1109/TIP.2016.2639450

**Academic honesty:** The project is *inspired by* LIME, not a full reproduction. The random-forest AI model is a proof of concept trained on synthetic exposure variants of four bundled sample images. Do not claim benchmark accuracy or robust generalization without a real independent dataset.

## Suggested team split
1. RGB loading, statistics and data preparation
2. Enhancement algorithms and parameter tuning
3. AI model and held-out dataset evaluation
4. Streamlit interface, testing and report integration

## Evaluation guidance
- Test on several real low-light images with varied lighting and colors.
- With paired clean reference images, compute PSNR/SSIM **against reference**.
- Without clean references, report brightness, contrast, visual comparisons and no-reference image quality metrics with appropriate caveats.
- Compare speed and visual quality of gamma, CLAHE and illumination-inspired correction.

## Dependencies
Python 3.10–3.13 recommended; install from requirements.txt. First launch can take longer because the AI model is trained locally. No dedicated GPU needed.
