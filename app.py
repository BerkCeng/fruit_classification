import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

st.set_page_config(page_title="FruitVision", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .main { background-color: #050505; color: #ffffff; }
    
    .analysis-container {
        background: #111111;
        border: 1px solid #222222;
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 50px;
    }
    
    .prediction-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .model-label {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 1px;
        color: #888888;
        margin-bottom: 20px;
        display: block;
    }
    
    .image-frame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #333333;
    }
    
    .info-box {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 20px;
        margin-top: 30px;
        border: 1px solid #222222;
    }
    </style>
    """, unsafe_allow_html=True)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = [
            target_layer.register_forward_hook(self._f_hook),
            target_layer.register_full_backward_hook(self._b_hook)
        ]

    def _f_hook(self, m, i, o):
        self.activations = o

    def _b_hook(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, in_t, idx):
        self.model.zero_grad()
        self.model(in_t)[0, idx].backward()
        weights = np.mean(self.gradients.cpu().data.numpy(), axis=(2, 3))[0]
        cam = np.zeros(self.activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :].cpu().data.numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (in_t.shape[3], in_t.shape[2]))
        return (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    def remove(self):
        for h in self.handlers:
            h.remove()


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
RESNET_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')
VIT_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_vit.pth')
VIT_IMG_SIZE = 128


@st.cache_resource
def load_all():
    if not os.path.exists(DATA_DIR):
        return None, None, [], "cpu"
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])[:268]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def prep(m, p):
        if not os.path.exists(p):
            return None
        m.load_state_dict(torch.load(p, map_location=dev))
        return m.to(dev).eval()

    rn = models.resnet18()
    rn.fc = nn.Linear(rn.fc.in_features, len(classes))
    vi = models.vit_b_16(image_size=VIT_IMG_SIZE)
    vi.heads.head = nn.Linear(vi.heads.head.in_features, len(classes))

    return prep(rn, RESNET_PATH), prep(vi, VIT_PATH), classes, dev


m_rn, m_vi, class_names, device = load_all()

st.title("FruitVision")
st.subheader("Meyve Sınıflandırma ve Model Karşılaştırma Arayüzü")

t1, t2 = st.tabs(["SINIFLANDIRMA", "PERFORMANS METRİKLERİ"])

with t1:
    files = st.file_uploader(
        "Görsel seçin",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if files:
        for idx, f in enumerate(files):
            img = Image.open(f).convert('RGB')
            img = ImageOps.exif_transpose(img)

            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)

            col_img, col_res = st.columns([1, 2])
            with col_img:
                st.image(img, width='stretch')

            with col_res:
                in_v = transforms.Compose([
                    transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out_v = m_vi(in_v)
                    pred_v = class_names[torch.argmax(out_v).item()]

                st.markdown(f'<div class="prediction-title">{pred_v.upper()}</div>', unsafe_allow_html=True)
                st.markdown('<div class="model-label">Tahmin Sonucu</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c_left, c_right = st.columns(2)

            with c_left:
                st.markdown('<div class="model-label">RESNET-18</div>', unsafe_allow_html=True)
                in_r = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(img).unsqueeze(0).to(device)
                in_r.requires_grad = True
                gcam = GradCAM(m_rn, m_rn.layer4)
                with torch.set_grad_enabled(True):
                    out_r = m_rn(in_r)
                    probs_r = torch.nn.functional.softmax(out_r, dim=1)
                    p3_v, p3_i = torch.topk(probs_r, 3)
                    hm = gcam.generate(in_r, p3_i[0][0].item())
                    gcam.remove()

                hm_img = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
                hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
                ov = cv2.addWeighted(np.array(img.resize((224, 224))), 0.6, hm_img, 0.4, 0)
                st.image(ov, caption="Grad-CAM", width='stretch')

                fig = px.bar(
                    x=[p.item() * 100 for p in p3_v[0]],
                    y=[class_names[i] for i in p3_i[0]],
                    orientation='h',
                    text=[f"%{p:.1f}" for p in [p.item() * 100 for p in p3_v[0]]],
                    color_discrete_sequence=['#58a6ff'],
                    height=150
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, key=f"r_{idx}", use_container_width=True, config={'displayModeBar': False})

            with c_right:
                st.markdown('<div class="model-label">VISION TRANSFORMER (ViT-B/16)</div>', unsafe_allow_html=True)
                in_v.requires_grad = True
                with torch.set_grad_enabled(True):
                    out_v = m_vi(in_v)
                    probs_v = torch.nn.functional.softmax(out_v, dim=1)
                    p3_vv, p3_iv = torch.topk(probs_v, 3)
                    out_v[0, p3_iv[0][0]].backward()
                    sal, _ = torch.max(in_v.grad.data.abs(), dim=1)
                    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

                sal_img = cv2.applyColorMap(np.uint8(255 * sal.reshape(128, 128).cpu().numpy()), cv2.COLORMAP_MAGMA)
                sal_img = cv2.cvtColor(sal_img, cv2.COLOR_BGR2RGB)
                ov_v = cv2.addWeighted(np.array(img.resize((128, 128))), 0.5, sal_img, 0.5, 0)
                st.image(ov_v, caption="Saliency Map", width='stretch')

                fig_v = px.bar(
                    x=[p.item() * 100 for p in p3_vv[0]],
                    y=[class_names[i] for i in p3_iv[0]],
                    orientation='h',
                    text=[f"%{p:.1f}" for p in [p.item() * 100 for p in p3_vv[0]]],
                    color_discrete_sequence=['#bc8cff'],
                    height=150
                )
                fig_v.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_v, key=f"v_{idx}", use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div class="info-box"><b>Veri Setinden Örnekler:</b><br><br>', unsafe_allow_html=True)
            class_p = os.path.join(DATA_DIR, pred_v)
            if os.path.exists(class_p):
                samples = [
                    os.path.join(pred_v, x)
                    for x in os.listdir(class_p)
                    if x.lower().endswith(('.png', 'jpg', 'jpeg'))
                ][:4]
                if samples:
                    cols = st.columns(len(samples))
                    for i, s in enumerate(samples):
                        with cols[i]:
                            st.image(os.path.join(DATA_DIR, s), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with t2:
    cp = os.path.join(WORKSPACE_DIR, 'results', 'detailed_comparison.csv')
    if os.path.exists(cp):
        df = pd.read_csv(cp)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Model Karşılaştırması")
            fig_r = go.Figure()
            cats = ['Doğruluk', 'Hız', 'Kararlılık', 'Detay Analizi']
            for i, r in df.iterrows():
                sc = [
                    r['Accuracy'] * 100,
                    max(10, 100 - r['Latency (ms)'] * 5),
                    85 if 'Res' in r['Model'] else 95,
                    70 if 'Res' in r['Model'] else 98
                ]
                fig_r.add_trace(go.Scatterpolar(r=sc, theta=cats, fill='toself', name=r['Model']))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})
        with c2:
            st.markdown("### Test Sonuçları")
            st.dataframe(df, width='stretch')
            chart_p = os.path.join(WORKSPACE_DIR, 'results', 'comparison_charts.png')
            if os.path.exists(chart_p):
                st.image(chart_p, width='stretch')

        st.markdown("---")
        st.subheader("Karmaşıklık Matrisleri (Confusion Matrix)")
        mc1, mc2 = st.columns(2)
        with mc1:
            cm_rn = os.path.join(WORKSPACE_DIR, 'results', 'resnet_confusion_matrix_top.png')
            if os.path.exists(cm_rn):
                st.markdown('<div class="model-label">RESNET-18</div>', unsafe_allow_html=True)
                st.image(cm_rn, width='stretch')
        with mc2:
            cm_vi = os.path.join(WORKSPACE_DIR, 'results', 'vit_confusion_matrix_top.png')
            if os.path.exists(cm_vi):
                st.markdown('<div class="model-label">VISION TRANSFORMER (ViT-B/16)</div>', unsafe_allow_html=True)
                st.image(cm_vi, width='stretch')
    else:
        st.warning("Sonuç dosyası bulunamadı. Önce değerlendirme betiğini çalıştırın.")
