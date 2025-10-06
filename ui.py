import streamlit as st
import pandas as pd
import base64
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from scipy.io import wavfile
import io

# This file contains functions that draw UI elements.

def apply_custom_styling():
    """Applies all the custom CSS for the app's theme."""
    st.markdown("""
    <style>
        /* --- General Layout & Background --- */
        [data-testid="stAppViewContainer"] > .main {
            background-color: #0F1116; /* A clean, dark background */
        }
        [data-testid="stHeader"] { background-color: rgba(0, 0, 0, 0); }
        .main .block-container { padding: 2rem 3rem; }

        /* --- Title Typography (Now Centered) --- */
        h1 {
            color: #FFFFFF;
            font-weight: 800;
            text-align: center; /* This centers the title */
            margin-bottom: 2rem; /* Adds some space below the title */
        }
        .title-main {
            font-size: 3.5rem;
            display: block;
            line-height: 1.2;
        }
        .title-sub {
            font-size: 1.1rem;
            font-weight: 400;
            color: #A0A0A0;
            display: block;
        }

        /* --- Section Headers --- */
        h2 {
            color: #EFEFEF; font-weight: 700;
            border-bottom: 2px solid rgba(255, 255, 255, 0.15);
            padding-bottom: 0.5rem; margin: 2.5rem 0 1.5rem 0;
        }
        h3 { color: #D3D3D3; font-weight: 600; }
        
        /* --- Widget & Container Styling --- */
        [data-testid="stVerticalBlock"] { gap: 2rem; }
        [data-testid="stExpander"] {
            background-color: rgba(40, 40, 40, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* --- Button Styling --- */
        [data-testid="stButton"] button {
            border-radius: 8px; font-weight: bold;
            transition: all 0.2s ease-in-out;
        }
        [data-testid="stButton"] button[kind="primary"] {
            background-color: #0078FF; color: white; border: none;
        }
        [data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #0056B3; transform: scale(1.02);
        }
        [data-testid="stButton"] button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.1); color: #EFEFEF;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        [data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
            transform: scale(1.02);
        }

        /* --- Metric Styling --- */
        [data-testid="stMetric"] {
            background-color: rgba(28, 28, 32, 0.85);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        [data-testid="stMetricLabel"] { color: #A0A0A0; }
    </style>
    """, unsafe_allow_html=True)

def render_signal_generation_options(sig_type, on_change_callback, default_samplerate):
    """Renders all the UI controls for generating a signal and returns the parameters."""
    params = {}
    st.write("### Options")
    
    if 'custom' not in sig_type.lower():
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            params['A_s'] = st.slider("Amplitude (A)", 0.1, 10.0, 1.0, 0.1, key='A_slider', on_change=on_change_callback)
        with gen_col2:
            params['f_s'] = st.slider("Frequency (f)", 0.1, 100.0, 2.0, 0.1, key='f_slider', on_change=on_change_callback)
        with gen_col3:
            params['phi_s'] = st.slider("Phase (ϕ)", 0.0, 360.0, 0.0, 5.0, key='phi_slider', on_change=on_change_callback)
        
        with st.expander("Set Precise Values"):
            prec_col1, prec_col2, prec_col3 = st.columns(3)
            with prec_col1:
                params['A'] = st.number_input("Amplitude", value=params['A_s'], min_value=0.1, max_value=10.0, step=0.01, key='A_num', on_change=on_change_callback)
            with prec_col2:
                params['f'] = st.number_input("Frequency", value=params['f_s'], min_value=0.1, max_value=100.0, step=0.01, key='f_num', on_change=on_change_callback)
            with prec_col3:
                params['phi'] = st.number_input("Phase", value=params['phi_s'], min_value=0.0, max_value=360.0, step=1.0, key='phi_num', on_change=on_change_callback)
        params['custom_data'] = None
    else:
        params['custom_data'] = st.text_area("Custom Data", "1,2,3,2,1", key='custom_data', on_change=on_change_callback)
        params['A'], params['f'], params['phi'] = 1.0, 1.0, 0.0

    if 'Sampled' in sig_type or 'Discrete' in sig_type:
        params['Fs'] = st.slider("Sampling Rate (Fs) in Hz", 8000, 48000, 16000, 100, key='sampling_rate', on_change=on_change_callback)
    else:
        params['Fs'] = default_samplerate
        
    return params

def render_scaling_options_and_buttons():
    """Renders the UI controls for scaling and the action buttons."""
    options = {}
    st.write("### Options")
    
    amp_s = st.slider("Amplitude Scaling Factor", 0.1, 5.0, 1.0, 0.1, key='amp_factor_slider')
    time_s = st.slider("Time Scaling Factor (a)", 0.2, 5.0, 1.0, 0.1, key='time_factor_slider')
    
    with st.expander("Set Precise Scaling Values"):
        prec_s_col1, prec_s_col2 = st.columns(2)
        with prec_s_col1:
            options['amp_scale_factor'] = st.number_input("Amplitude Factor", value=amp_s, min_value=0.1, max_value=5.0, step=0.01, key='amp_factor_num')
        with prec_s_col2:
            options['time_scale_factor'] = st.number_input("Time Factor", value=time_s, min_value=0.2, max_value=5.0, step=0.01, key='time_factor_num')
            
    b_col1, b_col2 = st.columns(2)
    options['apply_button'] = b_col1.button("Apply Scaling", use_container_width=True, type="primary")
    options['reset_button'] = b_col2.button("Reset", use_container_width=True, type="secondary")
    
    return options

def display_signal_properties(signal_obj):
    """Renders the signal properties using st.metric for a clean layout."""
    stats = signal_obj.calculate_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max", stats.get("Max", "N/A"))
        st.metric("Min", stats.get("Min", "N/A"))
        st.metric("Mean", stats.get("Mean", "N/A"))
    with col2:
        st.metric("RMS", stats.get("RMS", "N/A"))
        st.metric("Energy", stats.get("Energy", "N/A"))
        st.metric("Power", stats.get("Power", "N/A"))
    with col3:
        st.metric("Period", stats.get("Period", "N/A"))
        st.metric("Classification", stats.get("Classification", "N/A"))

def get_download_links(signal_obj, key_prefix):
    """Renders the download links for CSV and WAV."""
    if signal_obj is not None and signal_obj.t is not None and signal_obj.x is not None:
        col1, col2 = st.columns(2)
        df = pd.DataFrame({'time_s': signal_obj.t, 'amplitude': signal_obj.x})
        csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()
        col1.markdown(f'<a href="data:file/csv;base64,{b64_csv}" download="{key_prefix}_signal.csv" style="text-decoration:none;color:#0078FF;">Download CSV</a>', unsafe_allow_html=True)
        
        if signal_obj.is_discrete:
            try:
                audio_float = signal_obj.x.astype(np.float32)
                max_val = np.max(np.abs(audio_float))
                if max_val > 0: audio_float /= max_val
                
                source_fs = int(signal_obj.Fs)
                device_info = sd.query_devices(sd.default.device[1], 'output')
                target_fs = int(device_info['default_samplerate'])
                
                if source_fs != target_fs:
                    num_samples = int(len(audio_float) * target_fs / source_fs)
                    audio_float = resample(audio_float, num_samples)
                
                audio_int16 = (audio_float * 32767).astype(np.int16)
                buffer = io.BytesIO()
                wavfile.write(buffer, target_fs, audio_int16)
                b64_wav = base64.b64encode(buffer.getvalue()).decode()
                col2.markdown(f'<a href="data:audio/wav;base64,{b64_wav}" download="{key_prefix}_audio.wav" style="text-decoration:none;color:#0078FF;">Download WAV</a>', unsafe_allow_html=True)
            except Exception:
                pass

