import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "FaceTrace — Face Verification",
    page_icon  = "🔍",
    layout     = "wide"
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
CHECKPOINT  = BASE_DIR / "checkpoints" / "best_model.pth"
IMG_SIZE    = 112

# ─────────────────────────────────────────
# LOAD MODELS — cached so they load only once
# ─────────────────────────────────────────
@st.cache_resource
def load_detector():
    app = FaceAnalysis(
        name      = "buffalo_l",
        providers = ["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(320, 320))
    app.models['detection'].det_thresh = 0.3
    return app

@st.cache_resource
def load_finetuned_model():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    backbone   = tv_models.resnet50(weights=None)
    backbone.fc = nn.Sequential(
        nn.Linear(backbone.fc.in_features, 512),
        nn.BatchNorm1d(512)
    )
    backbone.load_state_dict(checkpoint["model_state"])
    backbone.eval()
    return backbone

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")),
                        cv2.COLOR_RGB2BGR)

def detect_and_align(bgr_img, detector):
    """Returns aligned face crop or None if no face detected"""
    faces = detector.get(bgr_img)
    if len(faces) == 0:
        return None, None
    face    = max(faces, key=lambda f: (
        (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
    ))
    aligned = face_align.norm_crop(bgr_img, landmark=face.kps)
    return aligned, face

def get_baseline_embedding(bgr_img, detector):
    """Get 512-d embedding using pretrained InsightFace"""
    faces = detector.get(bgr_img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: (
        (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
    ))
    return face.normed_embedding

def get_finetuned_embedding(aligned_bgr, model):
    """Get 512-d embedding using fine-tuned ResNet50"""
    rgb    = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    tensor = eval_transform(pil).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
        emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.squeeze(0).numpy()

def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2))

def get_verdict(score, threshold):
    if score >= threshold:
        return "✅ Same Person", "normal"
    else:
        return "❌ Different Person", "inverse"

# Thresholds from our evaluation
BASELINE_THRESHOLD  = 0.2905
FINETUNED_THRESHOLD = 0.5705

# ─────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────
st.title("FaceTrace — Face Verification System")
st.markdown(
    "Upload two face photos to check if they belong to the same person. "
    "Results are shown from both the **baseline** pretrained model and "
    "the **fine-tuned** model side by side."
)
st.divider()

# ─────────────────────────────────────────
# UI — FILE UPLOADERS
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Photo 1")
    file1 = st.file_uploader(
        "Upload first face image",
        type   = ["jpg", "jpeg", "png"],
        key    = "file1"
    )
    if file1:
        img1 = Image.open(file1)
        st.image(img1, caption="Photo 1", use_container_width=True)

with col2:
    st.subheader("Photo 2")
    file2 = st.file_uploader(
        "Upload second face image",
        type   = ["jpg", "jpeg", "png"],
        key    = "file2"
    )
    if file2:
        img2 = Image.open(file2)
        st.image(img2, caption="Photo 2", use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# UI — VERIFY BUTTON
# ─────────────────────────────────────────
if file1 and file2:
    if st.button("Verify Faces", type="primary", use_container_width=True):

        with st.spinner("Loading models..."):
            detector       = load_detector()
            finetuned_model = load_finetuned_model()

        with st.spinner("Detecting and analyzing faces..."):

            bgr1 = pil_to_bgr(Image.open(file1))
            bgr2 = pil_to_bgr(Image.open(file2))

            # ── Baseline embeddings ──
            base_emb1 = get_baseline_embedding(bgr1, detector)
            base_emb2 = get_baseline_embedding(bgr2, detector)

            # ── Fine-tuned embeddings ──
            aligned1, face1 = detect_and_align(bgr1, detector)
            aligned2, face2 = detect_and_align(bgr2, detector)

            ft_emb1 = get_finetuned_embedding(
                aligned1, finetuned_model
            ) if aligned1 is not None else None
            ft_emb2 = get_finetuned_embedding(
                aligned2, finetuned_model
            ) if aligned2 is not None else None

        st.divider()
        st.subheader("Results")

        # ── Show aligned face crops ──
        if aligned1 is not None and aligned2 is not None:
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                aligned1_rgb = cv2.cvtColor(aligned1,
                                            cv2.COLOR_BGR2RGB)
                st.image(aligned1_rgb,
                         caption="Detected face 1",
                         width=150)
            with c2:
                aligned2_rgb = cv2.cvtColor(aligned2,
                                            cv2.COLOR_BGR2RGB)
                st.image(aligned2_rgb,
                         caption="Detected face 2",
                         width=150)

        st.divider()

        # ── Side by side model comparison ──
        col_base, col_ft = st.columns(2)

        with col_base:
            st.markdown("### Baseline model")
            st.caption("Pretrained InsightFace — no fine-tuning")
            if base_emb1 is None or base_emb2 is None:
                st.error("Could not detect face in one or both images.")
            else:
                base_score   = cosine_similarity(base_emb1, base_emb2)
                base_verdict, _ = get_verdict(
                    base_score, BASELINE_THRESHOLD
                )
                st.metric("Similarity Score",
                          f"{base_score:.4f}",
                          f"Threshold: {BASELINE_THRESHOLD}")
                st.markdown(f"**Verdict: {base_verdict}**")
                # Score bar
                st.progress(min(max(base_score, 0.0), 1.0))

        with col_ft:
            st.markdown("### Fine-tuned model")
            st.caption("ResNet50 fine-tuned with ArcFace loss")
            if ft_emb1 is None or ft_emb2 is None:
                st.error("Could not detect face in one or both images.")
            else:
                ft_score     = cosine_similarity(ft_emb1, ft_emb2)
                ft_verdict, _ = get_verdict(
                    ft_score, FINETUNED_THRESHOLD
                )
                st.metric("Similarity Score",
                          f"{ft_score:.4f}",
                          f"Threshold: {FINETUNED_THRESHOLD}")
                st.markdown(f"**Verdict: {ft_verdict}**")
                st.progress(min(max(ft_score, 0.0), 1.0))

        st.divider()

        # ── Summary ──
        if (base_emb1 is not None and base_emb2 is not None and
                ft_emb1 is not None and ft_emb2 is not None):

            base_score = cosine_similarity(base_emb1, base_emb2)
            ft_score   = cosine_similarity(ft_emb1,   ft_emb2)

            st.subheader("Model comparison summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline score",    f"{base_score:.4f}")
            m2.metric("Fine-tuned score",  f"{ft_score:.4f}")
            m3.metric("Score improvement",
                      f"{ft_score - base_score:+.4f}")

else:
    st.info("Please upload both photos above to begin verification.")

# ─────────────────────────────────────────
# UI — SIDEBAR: Project info
# ─────────────────────────────────────────
with st.sidebar:
    st.header("About FaceTrace")
    st.markdown("""
    **FaceTrace** is a face verification system built as part of an 
    AI Developer Intern evaluation task for Jezt Technologies.
    
    **Model details:**
    - Baseline: InsightFace buffalo_l (ArcFace pretrained)
    - Fine-tuned: ResNet50 + ArcFace loss
    - Training: 20 epochs, AdamW, CosineAnnealingLR
    - Dataset: 8 identities, ~18,725 train images
    
    **Performance (on LQ eval set):**
    - Baseline ROC AUC: 0.7683
    - Fine-tuned ROC AUC: 0.9378
    - Improvement: +22.1%
    
    **Real-world application:**
    This system is designed toward solving the missing persons 
    identification problem in India — where 8.7 lakh people 
    go missing annually and 47% are never traced.
    """)

    st.divider()
    st.caption("Jezt Technologies — AI Developer Intern Task")
    st.caption("Edwin Viju | Amal Jyothi College of Engineering")