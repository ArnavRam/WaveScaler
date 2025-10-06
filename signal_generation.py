import numpy as np

def generate_signal(sig_type, A, f, phi, Fs, duration=2.0, custom_data=None):
    """
    Generates a continuous or discrete signal based on user parameters.
    """
    is_discrete = sig_type in ['Sampled Sine', 'Sampled Cosine', 'Sampled Exponential', 
                               'Sampled Triangular', 'Sampled Sawtooth', 'Custom Discrete']

    if is_discrete:
        n_samples = int(duration * Fs)
        t = np.arange(n_samples) / Fs
    else:
        points_per_cycle = 50
        num_points = max(1000, int(duration * f * points_per_cycle))
        num_points = min(50000, num_points) 
        t = np.linspace(0, duration, num_points, endpoint=False)
    
    # Apply phase shift to time vector
    t_phased = t + np.deg2rad(phi) / (2 * np.pi * f) if f > 0 else t

    # --- THE FIX: Using a mathematically stable formula for Sawtooth wave ---
    sawtooth_wave = A * (2 * (t_phased * f - np.floor(0.5 + t_phased * f)))

    sig_map = {
        'Sine': A * np.sin(2 * np.pi * f * t + np.deg2rad(phi)),
        'Cosine': A * np.cos(2 * np.pi * f * t + np.deg2rad(phi)),
        'Exponential': A * np.exp(-f * t) * np.cos(2 * np.pi * 5 * t + np.deg2rad(phi)),
        'Triangular': A * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t + np.deg2rad(phi))),
        'Sawtooth': sawtooth_wave,
    }
    
    sampled_map = {
        'Sampled Sine': A * np.sin(2 * np.pi * f * t + np.deg2rad(phi)),
        'Sampled Cosine': A * np.cos(2 * np.pi * f * t + np.deg2rad(phi)),
        'Sampled Exponential': A * np.exp(-f * t) * np.cos(2 * np.pi * 5 * t + np.deg2rad(phi)),
        'Sampled Triangular': A * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t + np.deg2rad(phi))),
        'Sampled Sawtooth': sawtooth_wave,
    }

    if sig_type in sig_map:
        x = sig_map[sig_type]
    elif sig_type in sampled_map:
        x = sampled_map[sig_type]
    elif sig_type == 'Custom Continuous' or sig_type == 'Custom Discrete':
        try:
            if not custom_data or not any(char.isdigit() for char in custom_data): raise ValueError
            x = np.array([float(v.strip()) for v in custom_data.split(',')])
            t = np.linspace(0, duration, len(x)) if sig_type == 'Custom Continuous' else np.arange(len(x)) * (duration / len(x))
        except (ValueError, AttributeError):
            return None, None, False
    else:
        return None, None, False

    return t, x, is_discrete

