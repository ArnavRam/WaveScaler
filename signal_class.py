import numpy as np
from scipy.interpolate import interp1d
import copy

class Signal:
    """
    A class to represent a signal, containing its data and methods for manipulation and analysis.
    """
    def __init__(self, t, x, is_discrete, f=None, Fs=None):
        self.t = np.array(t, dtype=float) if t is not None else None
        self.x = np.array(x, dtype=float) if x is not None else None
        self.is_discrete = is_discrete
        self.f = f
        self.Fs = Fs

    def copy(self):
        """Returns a deep copy of the signal object to prevent unintended modifications."""
        return copy.deepcopy(self)

    def scale_amplitude(self, factor):
        """
        Performs amplitude scaling.
        Returns a new, scaled Signal object.
        """
        new_signal = self.copy()
        if new_signal.x is not None:
            new_signal.x *= factor
        return new_signal

    def scale_time(self, factor):
        """
        Performs time scaling.
        Returns a new, time-scaled Signal object.
        """
        new_signal = self.copy()
        if factor == 1.0 or new_signal.t is None or new_signal.x is None or len(new_signal.t) < 2:
            return new_signal

        t_original_duration = new_signal.t[-1] - new_signal.t[0]
        t_new_duration = t_original_duration / factor
        
        if self.is_discrete:
            MAX_SAMPLES = 250000
            num_original_samples = len(self.x)
            num_new_samples = int(num_original_samples / factor)

            # Safety cap to prevent freezing
            if num_new_samples > MAX_SAMPLES:
                # THE FIX: Removed the st.warning() call. The capping happens silently.
                num_new_samples = MAX_SAMPLES
            
            if num_new_samples < 2:
                new_signal.t = np.array([self.t[0]])
                new_signal.x = np.array([self.x[0]])
                return new_signal
            
            t_new = np.linspace(self.t[0], t_new_duration, num_new_samples)
            f_interp = interp1d(self.t, self.x, kind='linear', bounds_error=False, fill_value=0)
            x_new = f_interp(np.linspace(self.t[0], t_original_duration, num_new_samples))
            new_signal.t, new_signal.x = t_new, x_new
        else: # Continuous
            t_new = np.linspace(self.t[0], t_new_duration, len(self.t))
            f_interp = interp1d(self.t, self.x, kind='linear', bounds_error=False, fill_value=0)
            x_new = f_interp(t_new * factor)
            new_signal.t, new_signal.x = t_new, x_new
        
        # Update frequency if it exists
        if new_signal.f:
            new_signal.f *= factor

        return new_signal

    def calculate_stats(self):
        """Calculates key statistics for the signal and returns them as a dictionary."""
        stats = {}
        if self.x is None or len(self.x) == 0 or self.t is None or len(self.t) == 0:
            return { 'Max': 'N/A', 'Min': 'N/A', 'Mean': 'N/A', 'RMS': 'N/A', 'Power': 'N/A', 'Energy': 'N/A', 'Period': 'N/A', 'Classification': 'N/A'}

        stats['Max'] = f"{np.max(self.x):.3f}"
        stats['Min'] = f"{np.min(self.x):.3f}"
        stats['Mean'] = f"{np.mean(self.x):.3f}"
        stats['RMS'] = f"{np.sqrt(np.mean(self.x**2)):.3f}"
        
        duration = self.t[-1] - self.t[0] if len(self.t) > 1 else 1.0
        
        if self.is_discrete:
            energy = np.sum(np.abs(self.x)**2)
            stats['Classification'] = "Discrete"
        else:
            dt = self.t[1] - self.t[0] if len(self.t) > 1 else 1.0
            energy = np.sum(np.abs(self.x)**2) * dt
            stats['Classification'] = "Continuous"
            
        stats['Energy'] = f"{energy:.3f}"
        stats['Power'] = f"{energy / duration:.3f}" if duration > 0 else "N/A"
        
        if self.f and self.f > 0:
            stats['Period'] = f"{1/self.f:.3f} s"
        else:
            stats['Period'] = "Aperiodic"
            
        return stats

