import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import os

# Setup root path resolution
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from models.resnet18_transfer import ResNet18Transfer
from models.gradcam import GradCAMExplainer
from inference.change_detector import TemporalChangeDetector
from data.transforms import get_val_transforms
from configs.paths import OUTPUTS_DIR, CHECKPOINTS_DIR, CONF_MATRIX_DIR, HEATMAPS_DIR, ROC_DIR, EMBEDDINGS_DIR
from configs.constants import (
    EUROSAT_CLASSES, COLOR_FOREST_GREEN,
    SIMILARITY_THRESHOLD_HIGH_PRECISION,
    SIMILARITY_THRESHOLD_BALANCED,
    SIMILARITY_THRESHOLD_HIGH_RECALL
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EcoVision AI - Satellite Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT THEME ENFORCEMENT & STYLING ---
css_file = root_path / "app" / "styles" / "custom.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
        st.markdown('<script>document.documentElement.setAttribute("translate", "no"); document.documentElement.classList.add("notranslate");</script>', unsafe_allow_html=True)
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Styling sheet custom.css not found. Dashboard loaded with fallback themes.")

# Initialize History State
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(task: str, image1_name: str, image2_name: str, score: float, prediction: str):
    st.session_state.history.append({
        "Task": task,
        "Before Image": image1_name,
        "After Image": image2_name if image2_name else "N/A",
        "Metric/Score": f"{score:.4f}",
        "Prediction": prediction
    })

def load_uploaded_image(file):
    """Loads an uploaded image, handling TIFF/TIF files robustly with OpenCV if PIL fails."""
    try:
        file.seek(0)
        return Image.open(file).convert('RGB')
    except Exception as e:
        try:
            import cv2
            file.seek(0)
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb)
        except Exception:
            pass
        raise e


# --- CACHED MODEL LOADERS ---
@st.cache_resource
def load_resnet_model():
    """Loads and caches the fine-tuned ResNet18 classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18Transfer(num_classes=len(EUROSAT_CLASSES), pretrained=False)
    model_path = CHECKPOINTS_DIR / "resnet18_best.pth"
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, device, True
        except Exception as e:
            return model, device, False
    return model, device, False

@st.cache_resource
def load_change_detector():
    """Loads and caches the embedding-based Temporal Change Detector."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = CHECKPOINTS_DIR / "resnet18_best.pth"
    detector = TemporalChangeDetector(model_path=model_path, num_classes=len(EUROSAT_CLASSES), device=device)
    return detector

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🛰️ EcoVision AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Satellite Analysis Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Dataset Explorer", "Predict & Change Detection", "Training & Evaluation", "About"],
        icons=["house", "images", "search", "bar-chart-line", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": COLOR_FOREST_GREEN, "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "4px", "color": "#2C3E50"},
            "nav-link-selected": {"background-color": "#E8F5E9", "color": COLOR_FOREST_GREEN, "font-weight": "bold"},
        }
    )

# --- PAGE ROUTING ---
if selected_page == "Home":
    st.title("🏠 EcoVision AI Dashboard")
    st.write("Production-ready computer vision application executing high-accuracy satellite analysis.")
    
    st.markdown("""
    <div class="custom-card">
        <h3>Overview</h3>
        <p>EcoVision AI utilizes fine-tuned ResNet18 Deep Learning backbones to perform semantic category classification and embedding similarity-based land-cover change detection. It is designed to mitigate spatial leakage, identify complex shifts, and remain interpretable using GradCAM activation mapping.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_flow, col_stats = st.columns(2)
    with col_flow:
        st.subheader("🔁 Core Operations Pipeline")
        st.info("""
        1. **Ingest & Validate**: Download EuroSAT/UC Merced datasets and sweep image files for corruption.
        2. **Spatial Split**: Segregate image blocks to guarantee zero adjacent data leaks.
        3. **Train Models**: Optimize baseline CNNs and fine-tune ResNet18 via 2-phase learning schedulers.
        4. **Detect Temporal Changes**: Strip the classifier, extract 512-dim embeddings, and compare Cosine Similarity metrics.
        5. **Interpret Heatmaps**: Generate pixel-wise absolute difference maps, JET heatmaps, and GradCAM activations.
        """)
        
    with col_stats:
        st.subheader("📦 Model & Dataset Summary")
        st.success("""
        * **Fine-Tuned Model**: ResNet18 Transfer backbone (frozen backbone -> unfrozen blocks).
        * **Baseline Model**: 3-layer Convolutional Neural Network built from scratch.
        * **Primary Data source**: EuroSAT (27,000 image tiles, 10 categories).
        * **Holdout Validation**: UC Merced Land Use (2,100 images, 21 classes).
        """)

    # Prediction History
    st.subheader("📋 Session Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Download csv report
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Session Report (CSV)",
            data=csv_data,
            file_name="ecovision_session_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions executed in this session yet.")

elif selected_page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.write("Verify dataset configurations, class distributions, and sample grids.")
    
    dataset_choice = st.radio("Choose Dataset for Visual Ingestion", ["EuroSAT", "UC Merced"])
    
    if dataset_choice == "EuroSAT":
        st.subheader("EuroSAT Dataset Distribution")
        dist_img = OUTPUTS_DIR / "predictions" / "eurosat_class_distribution.png"
        grid_img = OUTPUTS_DIR / "predictions" / "eurosat_sample_grid.png"
        
        col_dist, col_grid = st.columns(2)
        with col_dist:
            if dist_img.exists():
                st.image(str(dist_img), caption="EuroSAT Class Distribution Counts", use_column_width=True)
            else:
                st.warning("Distribution chart not found. Run main.py first.")
        with col_grid:
            if grid_img.exists():
                st.image(str(grid_img), caption="EuroSAT Random Category Samples Grid", use_column_width=True)
            else:
                st.warning("Sample grid image not found. Run main.py first.")
                
    else:
        st.subheader("UC Merced Holdout Dataset Distribution")
        dist_img = OUTPUTS_DIR / "predictions" / "ucmerced_class_distribution.png"
        grid_img = OUTPUTS_DIR / "predictions" / "ucmerced_sample_grid.png"
        
        col_dist, col_grid = st.columns(2)
        with col_dist:
            if dist_img.exists():
                st.image(str(dist_img), caption="UC Merced Class Distribution Counts", use_column_width=True)
            else:
                st.warning("Distribution chart not found. Run main.py first.")
        with col_grid:
            if grid_img.exists():
                st.image(str(grid_img), caption="UC Merced Random Category Samples Grid", use_column_width=True)
            else:
                st.warning("Sample grid image not found. Run main.py first.")

elif selected_page == "Predict & Change Detection":
    st.title("🔍 Prediction & Temporal Change Detection")
    st.write("Upload satellite images to execute classification, change detection, and explainability.")
    
    model, device, loaded_successfully = load_resnet_model()
    detector = load_change_detector()
    
    if not loaded_successfully:
        st.error("⚠️ ResNet18 model checkpoints weights not found at `outputs/checkpoints/resnet18_best.pth`. Run main.py or complete Phase 4 training to generate weights.")
    else:
        # Config Threshold Toggle (Bonus Task B)
        st.subheader("⚙️ Operating Point Mode (Bonus Task B)")
        col_mode, col_slide = st.columns([1, 2])
        with col_mode:
            op_mode = st.radio(
                "Select Operating Point Mode",
                ["Custom Slider", "High Precision (Stricter)", "Balanced (Youden J)", "High Recall (More Sensitive)"]
            )
            
        if op_mode == "High Precision (Stricter)":
            threshold = SIMILARITY_THRESHOLD_HIGH_PRECISION
            with col_slide:
                st.info(f"🎯 **High Precision Mode (Threshold: {threshold})**: Minimizes false change alarms. Only flags changes if images are highly distinct.")
        elif op_mode == "Balanced (Youden J)":
            threshold = SIMILARITY_THRESHOLD_BALANCED
            with col_slide:
                st.info(f"⚖️ **Balanced Mode (Threshold: {threshold})**: Optimizes Youden's J statistic to balance False Positives and False Negatives.")
        elif op_mode == "High Recall (More Sensitive)":
            threshold = SIMILARITY_THRESHOLD_HIGH_RECALL
            with col_slide:
                st.info(f"🔍 **High Recall Mode (Threshold: {threshold})**: Catches all possible changes, even minor shifts in structure or cover.")
        else:
            with col_slide:
                threshold = st.slider("Custom Cosine Similarity Threshold", 0.0, 1.0, 0.75, 0.01)

        st.markdown("---")
        
        # File Uploaders
        st.markdown('<div class="notranslate">', unsafe_allow_html=True)
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            file1 = st.file_uploader("Upload Time T1 (Before) Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"], key="file1")
        with col_up2:
            file2 = st.file_uploader("Upload Time T2 (After) Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"], key="file2")
        st.markdown('</div>', unsafe_allow_html=True)

        if file1 and file2:
            img1 = load_uploaded_image(file1)
            img2 = load_uploaded_image(file2)
            
            # Display previews
            col_disp1, col_disp2 = st.columns(2)
            col_disp1.image(img1, caption=f"T1: {file1.name}", use_column_width=True)
            col_disp2.image(img2, caption=f"T2: {file2.name}", use_column_width=True)
            
            if st.button("🚀 Execute Change Analysis Pipeline", use_container_width=True):
                with st.spinner("Extracting embeddings, calculating similarity, and running hook GradCAM..."):
                    # 1. Classify individual tiles
                    transform = get_val_transforms()
                    t1_tensor = transform(img1).unsqueeze(0).to(device)
                    t2_tensor = transform(img2).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out1 = model(t1_tensor)
                        out2 = model(t2_tensor)
                        probs1 = F.softmax(out1, dim=1)[0]
                        probs2 = F.softmax(out2, dim=1)[0]
                        
                        val1, idx1 = torch.max(probs1, 0)
                        val2, idx2 = torch.max(probs2, 0)
                        
                        class1 = EUROSAT_CLASSES[idx1.item()]
                        class2 = EUROSAT_CLASSES[idx2.item()]
                        conf1 = val1.item() * 100
                        conf2 = val2.item() * 100
                    
                    # 2. Change Detection
                    sim = detector.calculate_similarity(img1, img2)
                    diff, heatmap = detector.generate_difference_map(img1, img2)
                    
                    is_changed = sim < threshold
                    status_text = "⚠️ CHANGE DETECTED" if is_changed else "✅ NO CHANGE"
                    status_class = "status-changed" if is_changed else "status-unchanged"
                    
                    # Add to session history
                    add_to_history("Change Detection", file1.name, file2.name, sim, "Changed" if is_changed else "Unchanged")
                    
                    # Output Results Card
                    st.markdown(f"""
                    <div class="custom-card {status_class}">
                        <h2 style='text-align: center;'>{status_text}</h2>
                        <p style='text-align: center; font-size: 18px;'>
                            Cosine Similarity: <b>{sim:.4f}</b> (Selected Threshold: {threshold})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Classification Columns
                    c_col1, c_col2 = st.columns(2)
                    c_col1.markdown(f"""
                    <div class="custom-card">
                        <h4>Before (T1) Category</h4>
                        <h3>{class1}</h3>
                        <p>Confidence: {conf1:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    c_col2.markdown(f"""
                    <div class="custom-card">
                        <h4>After (T2) Category</h4>
                        <h3>{class2}</h3>
                        <p>Confidence: {conf2:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 3. Interpretability and Maps
                    st.subheader("🗺️ Difference and GradCAM Maps")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.image(diff, caption="Absolute Difference Map", use_column_width=True)
                    m_col2.image(heatmap, caption="JET Change Heatmap", use_column_width=True)
                    
                    # Generate GradCAM overlay
                    explainer = GradCAMExplainer(model=model, target_layer=model.backbone.layer4[-1], device=device)
                    _, superimposed1, _ = explainer.generate_heatmap(img1, idx1.item())
                    _, superimposed2, _ = explainer.generate_heatmap(img2, idx2.item())
                    
                    st.subheader("🔥 GradCAM Focus Interpretability")
                    g_col1, g_col2 = st.columns(2)
                    g_col1.image(superimposed1, caption=f"T1 Focus - Class: {class1}", use_column_width=True)
                    g_col2.image(superimposed2, caption=f"T2 Focus - Class: {class2}", use_column_width=True)
                    
                    # Report generator markdown
                    report_content = f"""# EcoVision AI - Analysis Report
Generated for satellite images:
- **T1 (Before)**: {file1.name} (Predicted: {class1} | Confidence: {conf1:.2f}%)
- **T2 (After)**: {file2.name} (Predicted: {class2} | Confidence: {conf2:.2f}%)

## Change Detection Metrics
- **Cosine Similarity**: {sim:.6f}
- **Operating Threshold**: {threshold}
- **Status Decision**: {"Change Detected" if is_changed else "No Significant Change"}
"""
                    st.download_button(
                        label="📄 Download Analysis Markdown Report",
                        data=report_content,
                        file_name="ecovision_change_report.md",
                        mime="text/markdown"
                    )

elif selected_page == "Training & Evaluation":
    st.title("📈 Model Training & Evaluation")
    st.write("Visualizations detailing training loss curves, accuracy, confusion matrices, and projections.")
    
    tab_curves, tab_eval, tab_embeddings = st.tabs(["Learning Curves", "Metric Evaluation", "Embeddings Space Projections"])
    
    with tab_curves:
        st.subheader("Model Validation Loss & Accuracy Curves")
        c1, c2 = st.columns(2)
        
        baseline_curves = ROC_DIR / "baseline_learning_curves.png"
        resnet_curves = ROC_DIR / "resnet18_learning_curves.png"
        
        with c1:
            if baseline_curves.exists():
                st.image(str(baseline_curves), caption="Baseline CNN (Scratch)", use_column_width=True)
            else:
                st.warning("Baseline CNN curves not found. Complete training to populate.")
        with c2:
            if resnet_curves.exists():
                st.image(str(resnet_curves), caption="ResNet18 Fine-Tuning", use_column_width=True)
            else:
                st.warning("ResNet18 curves not found. Complete training to populate.")
                
    with tab_eval:
        st.subheader("Confusion Matrices & Threshold Selection")
        col_cm1, col_cm2 = st.columns(2)
        
        baseline_cm = CONF_MATRIX_DIR / "baseline_confusion_matrix.png"
        resnet_cm = CONF_MATRIX_DIR / "resnet18_confusion_matrix.png"
        roc_plot = ROC_DIR / "change_detection_roc.png"
        kde_plot = ROC_DIR / "similarity_distribution.png"
        
        with col_cm1:
            if baseline_cm.exists():
                st.image(str(baseline_cm), caption="Baseline CNN Confusion Matrix", use_column_width=True)
            else:
                st.warning("Confusion matrix not found.")
        with col_cm2:
            if resnet_cm.exists():
                st.image(str(resnet_cm), caption="ResNet18 Confusion Matrix", use_column_width=True)
            else:
                st.warning("Confusion matrix not found.")
                
        st.markdown("---")
        st.subheader("Temporal Change Threshold ROC & Distribution")
        col_roc1, col_roc2 = st.columns(2)
        with col_roc1:
            if roc_plot.exists():
                st.image(str(roc_plot), caption="Similarity Change Detection ROC", use_column_width=True)
            else:
                st.warning("ROC Curve plot not found.")
        with col_roc2:
            if kde_plot.exists():
                st.image(str(kde_plot), caption="Cosine Similarity KDE Density Distribution", use_column_width=True)
            else:
                st.warning("Distribution KDE plot not found.")
                
        st.markdown("---")
        st.subheader("Model Comparison Stats")
        # Quantitative comparison tables
        comparison_data = {
            "Model Name": ["Baseline CNN (Scratch)", "ResNet18 Transfer Learning"],
            "Validation Accuracy": ["~76.9%", "~96.5%"],
            "Holdout Accuracy (UC Merced)": ["~18.5% (Low)", "~48.2% (Moderate)"],
            "Optimizer Used": ["Lion", "Adam"],
            "Epochs Run": [10, "3 + 5 (Phase-wise)"],
            "Trainable Params": ["12.5M", "11.2M"]
        }
        st.table(pd.DataFrame(comparison_data))
        st.info("The ResNet18 model shows superior performance across all categories due to robust feature representations transferred from ImageNet pre-training.")

    with tab_embeddings:
        st.subheader("512-Dimensional Visual Projection Space")
        col_tsne, col_umap = st.columns(2)
        
        tsne_proj = EMBEDDINGS_DIR / "tsne_projection.png"
        umap_proj = EMBEDDINGS_DIR / "umap_projection.png"
        
        with col_tsne:
            if tsne_proj.exists():
                st.image(str(tsne_proj), caption="t-SNE 2D Projection Space", use_column_width=True)
            else:
                st.warning("t-SNE projection not found.")
        with col_umap:
            if umap_proj.exists():
                st.image(str(umap_proj), caption="UMAP 2D Projection Space", use_column_width=True)
            else:
                st.warning("UMAP projection not found.")

elif selected_page == "About":
    st.title("ℹ️ About the Project")
    
    st.markdown("""
    <div class="custom-card">
        <h3>Satellite Image Land-Use Classifier & Change Detector</h3>
        <p>This project is part of a complete computer vision submission for Land-Use analysis and temporal shift observation.</p>
        <hr>
        <h4>Tech Stack Details</h4>
        <ul>
            <li><b>Inference & Core:</b> Python, PyTorch, Torchvision, OpenCV</li>
            <li><b>Data Utilities:</b> KaggleHub, Pandas, NumPy, Scikit-Learn</li>
            <li><b>Visualizations:</b> Matplotlib, Seaborn, UMAP, t-SNE, GradCAM</li>
            <li><b>Web Framework:</b> Streamlit, streamlit-option-menu</li>
        </ul>
        <hr>
        <h4>Developer & License</h4>
        <p>Developed by: <b>Peeyush</b></p>
        <p>License: <b>MIT License</b></p>
    </div>
    """, unsafe_allow_html=True)
