import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from audio_recorder_streamlit import audio_recorder
import io
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="🎵 Voice Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: white;
    }
    
    .recording-section {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .emotion-emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Emotion mapping and emojis
emotion_mapping = {
    0: 'angry', 1: 'disgust', 2: 'fear', 
    3: 'happy', 4: 'neutral', 5: 'sad'
}

emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢'
}

emotion_colors = {
    'angry': '#FF6B6B', 'disgust': '#4ECDC4', 'fear': '#45B7D1',
    'happy': '#96CEB4', 'neutral': '#FFEAA7', 'sad': '#DDA0DD'
}

@st.cache_resource
def load_emotion_model():
    """Load the trained emotion recognition model"""
    try:
        model = load_model('model_cnn_bilstm.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the scaler if available"""
    try:
        # Try to load scaler if it exists
        import joblib
        scaler = joblib.load('scaler.pkl')
        st.success("✅ Scaler loaded successfully!")
        return scaler
    except:
        st.warning("⚠️ No scaler found. Using features without scaling.")
        return None

def preprocess_audio(audio_data, sr=22050, max_len=300):
    """Preprocess audio data for emotion recognition"""
    try:
        # Normalize audio (same as training)
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-7)
        
        # Extract features exactly as in training
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).T              # (T, 13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr).T                # (T, 12)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel).T                                        # (T, 128)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data).T                   # (T, 1)
        
        # Padding function (same as training)
        def pad(x):
            return np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') if x.shape[0] < max_len else x[:max_len, :]
        
        # Apply padding
        mfcc = pad(mfcc)
        chroma = pad(chroma)
        mel_db = pad(mel_db)
        zcr = pad(zcr)
        
        # Combine features in the same order as training
        combined = np.concatenate((mfcc, chroma, mel_db, zcr), axis=1)
        
        # Debug: Print shapes to verify
        print(f"Feature shapes - MFCC: {mfcc.shape}, Chroma: {chroma.shape}, Mel: {mel_db.shape}, ZCR: {zcr.shape}")
        print(f"Combined shape: {combined.shape}")
        
        return combined, mfcc, chroma, mel_db, zcr
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None, None, None

def predict_emotion(model, features, scaler=None):
    """Predict emotion from audio features"""
    try:
        # Reshape for model input
        features_reshaped = np.expand_dims(features, axis=0)
        
        # Apply scaling if scaler is provided (same as training)
        if scaler is not None:
            original_shape = features_reshaped.shape
            features_reshaped = scaler.transform(features_reshaped.reshape(-1, features_reshaped.shape[-1])).reshape(original_shape)
        
        # Make prediction
        prediction = model.predict(features_reshaped, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_emotion = emotion_mapping[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        # Debug: Print prediction details
        print(f"Raw predictions: {prediction[0]}")
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted emotion: {predicted_emotion}")
        print(f"Confidence: {confidence:.2f}%")
        
        return predicted_emotion, confidence, prediction[0]
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None, 0, None

def create_emotion_chart(predictions):
    """Create a beautiful emotion probability chart"""
    emotions = list(emotion_mapping.values())
    probabilities = predictions * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color=[emotion_colors[emotion] for emotion in emotions],
            text=[f'{prob:.1f}%' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🎭 Emotion Probability Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Emotions",
        yaxis_title="Probability (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def visualize_audio_features(mfcc, chroma, mel_db, zcr):
    """Create visualizations for audio features"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('black')
    
    # MFCC
    axes[0, 0].imshow(mfcc.T, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('MFCC Features', color='white')
    axes[0, 0].set_facecolor('black')
    
    # Chroma
    axes[0, 1].imshow(chroma.T, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Chroma Features', color='white')
    axes[0, 1].set_facecolor('black')
    
    # Mel Spectrogram
    axes[1, 0].imshow(mel_db.T, cmap='magma', aspect='auto')
    axes[1, 0].set_title('Mel Spectrogram', color='white')
    axes[1, 0].set_facecolor('black')
    
    # Zero Crossing Rate
    axes[1, 1].plot(zcr.flatten(), color='cyan', linewidth=2)
    axes[1, 1].set_title('Zero Crossing Rate', color='white')
    axes[1, 1].set_facecolor('black')
    axes[1, 1].tick_params(colors='white')
    
    # Style all axes
    for ax in axes.flat:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Voice Emotion Recognition System</h1>
        <p>AI-powered emotion detection from voice recordings using CNN + BiLSTM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: white; text-align: center;">🎭 Emotion Guide</h2>
        </div>
        """, unsafe_allow_html=True)
        
        for emotion, emoji in emotion_emojis.items():
            st.markdown(f"""
            <div style="background: {emotion_colors[emotion]}; 
                        padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; 
                        text-align: center; color: white;">
                {emoji} {emotion.title()}
            </div>
            """, unsafe_allow_html=True)
    
    # Load model and scaler
    model = load_emotion_model()
    scaler = load_scaler()
    
    if model is None:
        st.error("❌ Could not load the emotion recognition model. Please ensure 'model_cnn_bilstm.h5' is in the same directory.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Record & Analyze", "📊 Features Visualization", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        <div class="recording-section">
            <h2>🎤 Record Your Voice</h2>
            <p>Click the microphone button below to record your voice and detect emotions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Audio recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="6x",
            )
            
            # File uploader as alternative
            uploaded_file = st.file_uploader(
                "Or upload an audio file",
                type=['wav', 'mp3', 'flac'],
                help="Upload a WAV, MP3, or FLAC file"
            )
        
        with col2:
            if audio_bytes or uploaded_file:
                # Show processing status
                progress_container = st.empty()
                progress_container.markdown("""
                <div class="feature-card">
                    <h3>🔄 Processing Audio...</h3>
                    <p>Extracting features and analyzing emotions</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Process audio
                try:
                    if audio_bytes:
                        # Convert bytes to numpy array
                        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                    else:
                        # Process uploaded file
                        audio_data, sr = librosa.load(uploaded_file, sr=22050)
                    
                    # Show preprocessing step
                    progress_container.markdown("""
                    <div class="feature-card">
                        <h3>🔧 Preprocessing Audio...</h3>
                        <p>Extracting MFCC, Chroma, Mel Spectrogram, and ZCR features</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Preprocess audio
                    features, mfcc, chroma, mel_db, zcr = preprocess_audio(audio_data, sr)
                    
                    if features is not None:
                        # Show preprocessing complete
                        progress_container.markdown("""
                        <div class="feature-card">
                            <h3>✅ Preprocessing Complete!</h3>
                            <p>Features extracted successfully. Making prediction...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a small delay to show the message
                        import time
                        time.sleep(1)
                        
                        # Predict emotion
                        emotion, confidence, predictions = predict_emotion(model, features, scaler)
                        
                        if emotion:
                            # Clear progress and display results
                            progress_container.empty()
                            
                            # Display results
                            st.markdown(f"""
                            <div class="emotion-card">
                                <div class="emotion-emoji">{emotion_emojis[emotion]}</div>
                                <h2>Detected Emotion: {emotion.title()}</h2>
                                <h3>Confidence: {confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Emotion chart
                            fig = create_emotion_chart(predictions)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Audio player
                            st.audio(audio_bytes if audio_bytes else uploaded_file, format='audio/wav')
                            
                            # Display feature information
                            st.markdown(f"""
                            <div class="stats-container">
                                <h4>📊 Feature Information</h4>
                                <p>Audio Duration: {len(audio_data)/sr:.2f} seconds</p>
                                <p>Sample Rate: {sr} Hz</p>
                                <p>Feature Shape: {features.shape}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Store features for visualization
                            st.session_state.features = {
                                'mfcc': mfcc,
                                'chroma': chroma,
                                'mel_db': mel_db,
                                'zcr': zcr,
                                'audio_data': audio_data,
                                'sr': sr
                            }
                
                except Exception as e:
                    progress_container.empty()
                    st.error(f"Error processing audio: {e}")
                    st.error("Please check the console for detailed error information.")
    
    with tab2:
        st.markdown("""
        <div class="feature-card">
            <h2>📊 Audio Features Visualization</h2>
            <p>Explore the extracted features used for emotion recognition</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'features' in st.session_state:
            features = st.session_state.features
            
            # Create feature visualizations
            fig = visualize_audio_features(
                features['mfcc'], 
                features['chroma'], 
                features['mel_db'], 
                features['zcr']
            )
            st.pyplot(fig)
            
            # Feature statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="stats-container">
                    <h4>🎵 MFCC Stats</h4>
                    <p>Mean: {:.3f}</p>
                    <p>Std: {:.3f}</p>
                </div>
                """.format(
                    np.mean(features['mfcc']), 
                    np.std(features['mfcc'])
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stats-container">
                    <h4>🎶 Chroma Stats</h4>
                    <p>Mean: {:.3f}</p>
                    <p>Std: {:.3f}</p>
                </div>
                """.format(
                    np.mean(features['chroma']), 
                    np.std(features['chroma'])
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stats-container">
                    <h4>🔊 Mel Spectrogram</h4>
                    <p>Mean: {:.3f}</p>
                    <p>Std: {:.3f}</p>
                </div>
                """.format(
                    np.mean(features['mel_db']), 
                    np.std(features['mel_db'])
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="stats-container">
                    <h4>⚡ Zero Crossing Rate</h4>
                    <p>Mean: {:.3f}</p>
                    <p>Std: {:.3f}</p>
                </div>
                """.format(
                    np.mean(features['zcr']), 
                    np.std(features['zcr'])
                ), unsafe_allow_html=True)
            
        else:
            st.info("🎤 Record or upload an audio file first to see feature visualizations!")
    
    with tab3:
        st.markdown("""
        <div class="feature-card">
            <h2>ℹ️ About This System</h2>
            <p>This voice emotion recognition system uses advanced deep learning techniques to analyze and classify emotions from speech.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧠 Model Architecture
            - **CNN Layers**: Extract spatial features from audio spectrograms
            - **BiLSTM Layers**: Capture temporal dependencies in speech
            - **Dense Layers**: Final classification with softmax activation
            
            ### 🎵 Features Used
            - **MFCC**: Mel-frequency cepstral coefficients
            - **Chroma**: Pitch class profiles
            - **Mel Spectrogram**: Frequency representation
            - **Zero Crossing Rate**: Speech activity detection
            """)
        
        with col2:
            st.markdown("""
            ### 🎭 Supported Emotions
            - **Happy** 😊: Joy and positivity
            - **Sad** 😢: Sadness and melancholy
            - **Angry** 😠: Anger and frustration
            - **Fear** 😨: Fear and anxiety
            - **Disgust** 🤢: Disgust and aversion
            - **Neutral** 😐: Calm and neutral state
            
            ### 🔧 Technical Details
            - **Sample Rate**: 22,050 Hz
            - **Feature Length**: 300 frames
            - **Input Shape**: (300, 154)
            - **Model Format**: Keras H5
            """)
        
        st.markdown("""
        <div class="stats-container">
            <h3>🚀 How It Works</h3>
            <ol>
                <li><strong>Audio Recording</strong>: Capture voice input via microphone or file upload</li>
                <li><strong>Preprocessing</strong>: Extract audio features (MFCC, Chroma, Mel Spectrogram, ZCR)</li>
                <li><strong>Feature Engineering</strong>: Normalize and pad features to fixed length</li>
                <li><strong>Prediction</strong>: CNN+BiLSTM model predicts emotion probabilities</li>
                <li><strong>Visualization</strong>: Display results with confidence scores and feature analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
