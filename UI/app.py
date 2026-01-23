import streamlit as st
import tempfile
from src.pipeline import DuplicateDetector
from PIL import Image

st.set_page_config(page_title="Mirror of Maya", layout="centered")

st.title("Mirror of Maya")
st.subheader("Near-Duplicate Image Detection")

mode = st.radio(
    "Choose Detection Mode",
    ["hybrid", "clip", "phash"]
)

st.markdown("---")

st.sidebar.title("Offline Evaluation")
st.sidebar.markdown("""
**Model:** Hybrid (pHash + CLIP)  
**Dataset:** 171 image pairs  

**pHash threshold:** 18  
**CLIP threshold:** 0.30  

**Precision:** 1.00  
**Recall:** 0.798  
**F1 Score:** 0.888  
""")

img1 = st.file_uploader("Upload First Image", type=["jpg", "png", "jpeg"])
img2 = st.file_uploader("Upload Second Image", type=["jpg", "png", "jpeg"])

if img1 and img2:
    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Image 1", use_column_width=True)

    with col2:
        st.image(img2, caption="Image 2", use_column_width=True)

    if st.button("Compare Images"):
        with tempfile.NamedTemporaryFile(delete=False) as f1, \
             tempfile.NamedTemporaryFile(delete=False) as f2:

            f1.write(img1.read())
            f2.write(img2.read())

            detector = DuplicateDetector(mode=mode)
            score, decision = detector.compare(f1.name, f2.name)

        st.markdown("---")
        st.subheader("Result")

        if mode == "phash":
            st.write(f"**pHash Distance:** {score:.2f}")
        else:
            st.write(f"**Similarity Score:** {score:.4f}")

        if decision:
            st.success("NEAR DUPLICATE")
        else:
            st.error("NOT DUPLICATE")
