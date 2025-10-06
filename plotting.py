import plotly.graph_objects as go
import numpy as np
from signal_class import Signal

def plot_signal(signal: Signal, title: str):
    """Plots the signal object using Plotly for interactivity."""
    fig = go.Figure()
    if signal is None or signal.x is None or signal.t is None or len(signal.x) == 0:
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            annotations=[{"text": "No Signal Data", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]
        )
        return fig

    # Unpack data from the signal object
    t, x, is_discrete = signal.t, signal.x, signal.is_discrete

    # Add an invisible trace to set the initial Y-axis range without locking it.
    if t is not None and len(t) > 1:
        fig.add_trace(go.Scatter(
            x=[t[0], t[-1]], 
            y=[-5.5, 5.5], 
            mode='markers', 
            marker={'opacity': 0}, # Invisible markers
            showlegend=False,
            hoverinfo='none'
        ))
        
    if is_discrete:
        # --- EFFICIENT STEM PLOTTING ---
        # Create coordinates for all stems at once.
        # Inserts None between each line segment to break the path.
        t_stems = np.repeat(t, 3)
        t_stems[2::3] = None # Gaps
        
        y_stems = np.zeros_like(t_stems)
        y_stems[1::3] = x # The tip of the stem

        # Plot all stems with one trace
        fig.add_trace(go.Scatter(
            x=t_stems, y=y_stems, 
            mode='lines', 
            line=dict(color='royalblue', width=2),
            hoverinfo='none' # Hide hover info for the stem lines
        ))

        # Add markers on top
        fig.add_trace(go.Scatter(
            x=t, y=x, 
            mode='markers', 
            marker=dict(color="royalblue", size=6),
            name='Sample' # Name for hover label
        ))

    else: # Continuous Signal
        fig.add_trace(go.Scatter(x=t, y=x, mode='lines', line=dict(color="royalblue")))
    
    xlabel = "Time (s)"
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=xlabel,
        yaxis_title="Amplitude",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=2, zerolinecolor='Black')

    return fig

