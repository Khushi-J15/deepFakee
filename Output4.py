import streamlit as st
from PIL import Image
from api import process_image, process_video, process_audio
import time
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css = """
    <style>
        .main { padding: 2rem; }
        .stImage img {
            border-radius: 12px;
            max-width: 350px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .stImage img:hover { transform: scale(1.02); }
        .uploadedFile { display: none; }
        .st-emotion-cache-1hgxyac { border: none !important; }
        .stButton>button {
            background-color: #4a6bdf;
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #3a56c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .result-real {
            color: #6eb52f;
            font-weight: bold;
            animation: pulseGreen 1s;
        }
        .result-fake {
            color: #ff4b4b;
            font-weight: bold;
            animation: pulseRed 1s;
        }
        @keyframes pulseGreen {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        @keyframes pulseRed {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .result-card {
            background: black;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            margin: 1rem 0;
            border-left: 5px solid #4a6bdf;
        }
        .confidence-meter {
            height: 8px;
            background: linear-gradient(90deg, #ff4b4b 0%, #fdb913 50%, #6eb52f 100%);
            border-radius: 4px;
            margin: 1rem 0;
            position: relative;
        }
        .confidence-indicator {
            position: absolute;
            top: -6px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            border: 3px solid #4a6bdf;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transform: translateX(-50%);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeIn 0.6s ease-out forwards; }
        .audio-preview {
            background: #f9f9f9;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# Create uploads directory if not exists
os.makedirs("uploads", exist_ok=True)

def loading_animation():
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent_complete + 1)
    progress_bar.empty()

# Main app
st.title("🕵️ Deepfake Detection System")
st.markdown("Upload an image, video, or audio to analyze for potential deepfake content")

# Check for model files
if not (os.path.exists("svm_model.pkl") and os.path.exists("scaler.pkl")):
    st.markdown(
        """
        <div class="warning-box">
            <strong>Warning:</strong> Audio detection model files (svm_model.pkl, scaler.pkl) not found. 
            Please run <code>main.py</code> to train the model with audio files in <code>real_audio</code> and <code>deepfake_audio</code> directories.
        </div>
        """,
        unsafe_allow_html=True
    )

# Create columns
col1, col2 = st.columns([1, 1])

with col1:
    with st.container():
        st.header("📊 Settings")
        
        file_type = st.radio("Media Type", ("Image", "Video", "Audio"), horizontal=True)
        
        if file_type == "Image":
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        elif file_type == "Video":
            uploaded_file = st.file_uploader("Choose a video", type=["mp4"])
        else:  # Audio
            uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
        
        # Filter model options based on file type
        if file_type == "Audio":
            model = st.selectbox("Detection Model", ["SVM"])
        else:
            model = st.selectbox("Detection Model", ["EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST"])
        
        # Filter dataset options
        if file_type == "Audio":
            dataset = st.radio("Training Dataset", ["CustomAudio"])
        else:
            dataset = st.radio("Training Dataset", ["DFDC", "FFPP"])
        
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        
        if file_type == "Video":
            frames = st.slider("Frames to Analyze", 10, 100, 30)
        elif file_type == "Audio":
            duration = st.slider("Max Audio Duration (seconds)", 5, 60, 30)

with col2:
    with st.container():
        st.header("👁️ Preview & Results")
        
        if uploaded_file is not None:
            if file_type == "Image":
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=350)
                except:
                    st.error("Invalid image file")
            elif file_type == "Video":
                st.video(uploaded_file)
            else:  # Audio
                try:
                    st.audio(uploaded_file)
                    st.markdown('<div class="audio-preview">Audio Preview</div>', unsafe_allow_html=True)
                except:
                    st.error("Invalid audio file")
            
            if st.button("Check for Deepfake"):
                with st.spinner("Analyzing content..."):
                    loading_animation()
                    
                    try:
                        if file_type == "Image":
                            result, pred = process_image(
                                image=uploaded_file, model=model, dataset=dataset, threshold=threshold)
                            if pred == -1:
                                st.error(result)
                                st.stop()
                            
                            result_class = "result-real" if result == "real" else "result-fake"
                            icon = "✅" if result == "real" else "❌"
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>{icon} Detection Result</h3>
                                <p>The image is classified as: <span class="{result_class}">{result.upper()}</span></p>
                                <p>Confidence score: <span class="{result_class}">{pred:.2f}</span></p>
                                <div class="confidence-meter">
                                    <div class="confidence-indicator" style="left: {pred*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif file_type == "Video":
                            with open(f"uploads/{uploaded_file.name}", "wb") as f:
                                f.write(uploaded_file.read())
                            video_path = f"uploads/{uploaded_file.name}"
                            result, pred = process_video(video_path, model=model,
                                                        dataset=dataset, threshold=threshold, frames=frames)
                            if pred == -1:
                                st.error(result)
                                st.stop()
                            
                            result_class = "result-real" if result == "real" else "result-fake"
                            icon = "✅" if result == "real" else "❌"
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>{icon} Detection Result</h3>
                                <p>The video is classified as: <span class="{result_class}">{result.upper()}</span></p>
                                <p>Confidence score: <span class="{result_class}">{pred:.2f}</span></p>
                                <div class="confidence-meter">
                                    <div class="confidence-indicator" style="left: {pred*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:  # Audio
                            with open(f"uploads/{uploaded_file.name}", "wb") as f:
                                f.write(uploaded_file.read())
                            audio_path = f"uploads/{uploaded_file.name}"
                            result, pred = process_audio(
                                audio_path, model=model, dataset=dataset, threshold=threshold, duration=duration)
                            if pred == -1:
                                st.error(result)
                                st.stop()
                            
                            result_class = "result-real" if result == "real" else "result-fake"
                            icon = "✅" if result == "real" else "❌"
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>{icon} Detection Result</h3>
                                <p>The audio is classified as: <span class="{result_class}">{result.upper()}</span></p>
                                <p>Confidence score: <span class="{result_class}">{pred:.2f}</span></p>
                                <div class="confidence-meter">
                                    <div class="confidence-indicator" style="left: {pred*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        else:
            st.markdown("""
            <div style="border: 2px dashed #ccc; 
                     border-radius: 10px; 
                     padding: 40px; 
                     text-align: center;
                     color: #888;">
                <i class="fa fa-cloud-upload" style="font-size: 48px;"></i>
                <p>Please upload a file to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

        # Project Information

        This Streamlit app detects deepfakes in images, videos, and audio files.
        This app is created by [Khushi Jain](https://github.com/Khushi-J15/).

        The source code is available on [GitHub](https://github.com/Khushi-J15/deepFakee) <i class="fa fa-github"></i>
        ''', unsafe_allow_html=True
        )