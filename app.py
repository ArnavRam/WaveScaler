import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import resample

# Import from our modules
from signal_generation import generate_signal
from plotting import plot_signal
from signal_class import Signal
import ui # Our UI module

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Wave Scaler: Interactive Tool for Time and Amplitude Scaling of Signals",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.apply_custom_styling()

# --- Audio Hardware Query ---
try:
    DEFAULT_SAMPLERATE = int(sd.query_devices(sd.default.device[1], 'output')['default_samplerate'])
except Exception:
    DEFAULT_SAMPLERATE = 44100

# --- App State Management Callback ---
def signal_param_changed():
    """Flags that a signal generation parameter has changed."""
    st.session_state.params_changed = True
    sd.stop()
    st.session_state.playing_original = False
    st.session_state.playing_scaled = False

# --- Main UI Rendering ---
st.markdown("""
    <h1>
        <span class="title-main">WaveScaler</span>
        <span class="title-sub">Interactive Tool for Time and Amplitude Scaling of Signals</span>
    </h1>
""", unsafe_allow_html=True)

sig_type = st.selectbox(
    "Select which wave",
    ['Sine', 'Cosine', 'Exponential', 'Triangular', 'Sawtooth', 'Sampled Sine', 'Sampled Cosine', 'Sampled Exponential', 'Sampled Triangular', 'Sampled Sawtooth', 'Custom Continuous', 'Custom Discrete'],
    key='sig_type',
    on_change=signal_param_changed,
    label_visibility="collapsed"
)

# --- Original Signal Section ---
st.markdown("## Original Signal")
with st.container():
    gen_params = ui.render_signal_generation_options(sig_type, signal_param_changed, DEFAULT_SAMPLERATE)
    
    st.write("### Visualization & Properties")
    plot_col_orig, params_col_orig = st.columns([2, 1])
    original_signal = st.session_state.get('original_signal')
    
    with plot_col_orig:
        if original_signal:
            st.plotly_chart(plot_signal(original_signal, ""), use_container_width=True, key="original_chart")

    with params_col_orig:
        if original_signal:
            ui.display_signal_properties(original_signal)
            st.write("") 
            if original_signal.is_discrete:
                if st.session_state.get('playing_original', False):
                    if st.button("⏹️ Stop Audio", key="stop_orig", use_container_width=True, type="secondary"):
                        sd.stop(); st.session_state.playing_original = False; st.rerun()
                else:
                    if st.button("▶️ Play in Loop", key="play_orig", use_container_width=True, type="primary"):
                        try:
                            sd.stop(); st.session_state.playing_scaled = False
                            audio_float = original_signal.x.astype(np.float32)
                            max_val = np.max(np.abs(audio_float)) 
                            if max_val > 0: audio_float /= max_val
                            source_fs = int(original_signal.Fs)
                            if source_fs != DEFAULT_SAMPLERATE: audio_float = resample(audio_float, int(len(audio_float) * DEFAULT_SAMPLERATE / source_fs))
                            audio_int16 = (audio_float * 32767).astype(np.int16)
                            sd.play(audio_int16, samplerate=DEFAULT_SAMPLERATE, loop=True)
                            st.session_state.playing_original = True; st.rerun()
                        except Exception as e: st.error(f"Audio Error: {e}")
            ui.get_download_links(original_signal, "original")

# --- Scaling Operations Section ---
st.markdown("## Scaling Operations")
with st.container():
    scaling_ops = ui.render_scaling_options_and_buttons()
    
    st.write("### Visualization & Properties")
    plot_col_scaled, params_col_scaled = st.columns([2, 1])
    current_signal = st.session_state.get('current_signal')

    with plot_col_scaled:
        if current_signal:
            st.plotly_chart(plot_signal(current_signal, ""), use_container_width=True, key="scaled_chart")

    with params_col_scaled:
        if current_signal:
            ui.display_signal_properties(current_signal)
            st.write("") 
            if current_signal.is_discrete:
                if st.session_state.get('playing_scaled', False):
                    if st.button("⏹️ Stop Audio", key="stop_scaled", use_container_width=True, type="secondary"):
                        sd.stop(); st.session_state.playing_scaled = False; st.rerun()
                else:
                    if st.button("▶️ Play in Loop", key="play_scaled", use_container_width=True, type="primary"):
                        try:
                            sd.stop(); st.session_state.playing_original = False
                            audio_float = current_signal.x.astype(np.float32)
                            max_val = np.max(np.abs(audio_float))
                            if max_val > 0: audio_float /= max_val
                            source_fs = int(current_signal.Fs)
                            if source_fs != DEFAULT_SAMPLERATE: audio_float = resample(audio_float, int(len(audio_float) * DEFAULT_SAMPLERATE / source_fs))
                            audio_int16 = (audio_float * 32767).astype(np.int16)
                            sd.play(audio_int16, samplerate=DEFAULT_SAMPLERATE, loop=True)
                            st.session_state.playing_scaled = True; st.rerun()
                        except Exception as e: st.error(f"Audio Error: {e}")
            ui.get_download_links(current_signal, "scaled")

# --- App State Logic (runs invisibly) ---
if 'playing_original' not in st.session_state: st.session_state.playing_original = False
if 'playing_scaled' not in st.session_state: st.session_state.playing_scaled = False

if 'original_signal' not in st.session_state or st.session_state.get('params_changed', False):
    signal_args = {k: v for k, v in gen_params.items() if not k.endswith('_s')}
    t_orig, x_orig, is_discrete_orig = generate_signal(sig_type, **signal_args)
    
    if t_orig is not None:
        st.session_state.original_signal = Signal(t=t_orig, x=x_orig, is_discrete=is_discrete_orig, f=signal_args.get('f'), Fs=signal_args.get('Fs'))
        st.session_state.current_signal = st.session_state.original_signal.copy()
    elif 'original_signal' in st.session_state:
        st.session_state.original_signal, st.session_state.current_signal = None, None
    st.session_state.params_changed = False

if scaling_ops['reset_button']:
    sd.stop(); st.session_state.playing_original, st.session_state.playing_scaled = False, False
    if st.session_state.original_signal:
        st.session_state.current_signal = st.session_state.original_signal.copy()
    st.rerun()

if scaling_ops['apply_button']:
    sd.stop(); st.session_state.playing_original, st.session_state.playing_scaled = False, False
    if st.session_state.current_signal:
        # THE FIX IS HERE: Use 'time_scale_factor' to match the key from ui.py
        scaled_signal = st.session_state.current_signal.scale_amplitude(scaling_ops['amp_scale_factor']).scale_time(scaling_ops['time_scale_factor'])
        st.session_state.current_signal = scaled_signal
    st.rerun()

