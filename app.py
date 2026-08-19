import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Child Safety Monitor", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🛡️ Child Safety Monitoring System v2.0")
st.markdown("---")

# Sidebar info
st.sidebar.markdown("## 📊 System Info")
st.sidebar.markdown("**Status:** ✅ Running")
st.sidebar.markdown("**Model:** YOLOv8 Nano")
st.sidebar.markdown("**Mode:** Detection")

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model()

if model is None:
    st.error("Could not load YOLO model. Please check your installation.")
    st.stop()

# Main content
st.markdown("### 📸 Upload an Image for Detection")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image:**")
        st.image(image, use_column_width=True)
    
    # Run YOLO detection
    with st.spinner("🔍 Running detection..."):
        try:
            results = model(image, conf=0.5)
            
            # Get result image
            result_image = results[0].plot()
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("**Detection Results:**")
                st.image(result_image_rgb, use_column_width=True)
            
            # Show detection info
            st.markdown("---")
            st.markdown("### 📊 Detection Details")
            
            detections = results[0]
            num_detections = len(detections.boxes)
            
            if num_detections > 0:
                st.success(f"✅ Found {num_detections} person(s) in the image")
                
                # Show detection info
                for i, box in enumerate(detections.boxes):
                    conf = box.conf[0].item()
                    st.markdown(f"**Detection {i+1}:** Confidence: {conf:.2%}")
            else:
                st.info("ℹ️ No persons detected in this image")
        
        except Exception as e:
            st.error(f"❌ Error during detection: {e}")

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("**🔒 Secure System**  \nDeveloped for child safety monitoring")