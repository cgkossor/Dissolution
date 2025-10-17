import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_color_gradient(n_colors, start=0.2, end=0.9):
    """Generate a color gradient with n_colors steps using Blues colormap."""
    blues = plt.cm.Blues(np.linspace(start, end, n_colors))
    return [mcolors.rgb2hex(color[:3]) for color in blues]

@st.cache_data
def setup_parameters(r0_values, si_values, si_max, m0, DL, rho, V, k, Cs0, k_cryst_ref, r_ref, alpha, C_cryst, t_max):
    """Define simulation parameters for multiple particle sizes."""
    params = {
        'm0': m0,
        'DL': DL,
        'rho': rho,
        'V': V,
        'k': k,
        'Cs0': Cs0,
        'k_cryst_ref': k_cryst_ref,
        'r_ref': r_ref,
        'alpha': alpha,
        'C_cryst': C_cryst,
        'si_values': si_values,
        'si_max': si_max,
        't_max': t_max,
        'r0_values': r0_values
    }
    params['m_drug0_base'] = params['DL'] * params['m0']
    params['C_b0_base'] = params['m_drug0_base'] / params['V']
    params['Cs_sink'] = params['si_max'] * params['C_b0_base']
    params['A0_values'] = [3 * params['m0'] / (params['rho'] * 1e6 * r0) * 1e4 for r0 in r0_values]
    params['k_cryst_values'] = [params['k_cryst_ref'] * (params['r_ref'] / r0) ** params['alpha'] for r0 in r0_values]
    return params

def Cs(t, Cs0, k_cryst, C_cryst):
    """Calculate time-dependent solubility for non-sink conditions."""
    return C_cryst + (Cs0 - C_cryst) * np.exp(-k_cryst * t)

def dissolution_model(Cb, t, k, A0, Cs_func, m_drug0, V):
    """Define dissolution model ODE."""
    m_t = max(m_drug0 - Cb * V, 0)
    A_t = A0 * (m_t / m_drug0)**(2/3)
    Cs_t = Cs_func(t)
    return k * A_t * (Cs_t - Cb) / V

@st.cache_data
def run_simulation(params, A0, r0_label, k_cryst, color_gradient, linestyle):
    """Run dissolution simulation for a given particle size."""
    t = np.logspace(-2, np.log10(params['t_max']), 100)
    colors = color_gradient
    
    Cs_sink_func = lambda t: params['Cs_sink']
    Cb_sink = odeint(dissolution_model, 0, t, args=(params['k'], A0, Cs_sink_func, params['m_drug0_base'], params['V'])).flatten()
    dissolved_pct_sink = (Cb_sink * params['V'] / params['m_drug0_base']) * 100
    m_sink = np.maximum(params['m_drug0_base'] - Cb_sink * params['V'], 0)
    A_sink = A0 * (m_sink / params['m_drug0_base'])**(2/3)
    
    Cb_nonsink_list, dissolved_pct_nonsink_list = [], []
    m_nonsink_list, A_nonsink_list, Cs_nonsink_list = [], [], []
    
    for si in params['si_values']:
        m_drug0 = params['C_cryst'] * params['V'] / si
        Cs_nonsink_func = lambda t: Cs(t, params['Cs0'], k_cryst, params['C_cryst'])
        Cb_nonsink = odeint(dissolution_model, 0, t, args=(params['k'], A0, Cs_nonsink_func, m_drug0, params['V'])).flatten()
        dissolved_pct_nonsink = (Cb_nonsink * params['V'] / m_drug0) * 100
        m_nonsink = np.maximum(m_drug0 - Cb_nonsink * params['V'], 0)
        A_nonsink = A0 * (m_nonsink / m_drug0)**(2/3)
        Cs_nonsink = Cs_nonsink_func(t)
        
        Cb_nonsink_list.append(Cb_nonsink)
        dissolved_pct_nonsink_list.append(dissolved_pct_nonsink)
        m_nonsink_list.append(m_nonsink)
        A_nonsink_list.append(A_nonsink)
        Cs_nonsink_list.append(Cs_nonsink)
    
    return {
        't': t,
        'Cb_sink': Cb_sink,
        'dissolved_pct_sink': dissolved_pct_sink,
        'A_sink': A_sink,
        'Cb_nonsink_list': Cb_nonsink_list,
        'dissolved_pct_nonsink_list': dissolved_pct_nonsink_list,
        'A_nonsink_list': A_nonsink_list,
        'Cs_nonsink_list': Cs_nonsink_list,
        'colors': colors,
        'linestyle': linestyle,
        'label': f'r0={r0_label*1e6:.1f}μm'
    }

def plot_comparison(results_list, params):
    """Plot comparison with dual legends for Sink Index and Particle Size."""
    si_all = sorted(params['si_values'] + [params['si_max']], reverse=False)  # Lowest to highest for color mapping
    n_colors = len(si_all)
    color_gradient = generate_color_gradient(n_colors, start=0.2, end=0.9)
    color_map = {si: color_gradient[i] for i, si in enumerate(si_all)}  # Highest SI -> lightest color
    
    # Dissolved Percentage Plot
    fig1 = go.Figure()
    
    # Add data traces (no legend entries)
    for si in si_all:
        for result in results_list:
            linestyle = result['linestyle']
            label = result['label']
            color = color_map[si]
            if si == params['si_max']:
                fig1.add_trace(go.Scatter(
                    x=result['t'], y=result['dissolved_pct_sink'],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f"SI={si:.2f}, {label}",
                    hovertemplate='Time: %{x:.2f} min<br>Dissolved: %{y:.2f}%<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ))
            else:
                idx = params['si_values'].index(si)
                fig1.add_trace(go.Scatter(
                    x=result['t'], y=result['dissolved_pct_nonsink_list'][idx],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f"SI={si:.2f}, {label}",
                    hovertemplate='Time: %{x:.2f} min<br>Dissolved: %{y:.2f}%<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ))

    # Add dummy traces for Sink Index legend (highest to lowest)
    si_legend = sorted(params['si_values'] + [params['si_max']], reverse=True)  # Highest to lowest for legend
    for i, si in enumerate(si_legend):
        legend_rank = 1000 + i  # Higher SI at top (lower rank)
        fig1.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible trace
            mode='lines',
            line=dict(color=color_map[si], width=2),
            name=f"SI={si:.2f}",
            legendgroup="sink_index",
            legendgrouptitle_text="Sink Index" if i == 0 else None,
            showlegend=True,
            legendrank=legend_rank
        ))

    # Add dummy traces for Particle Size legend
    for i, result in enumerate(results_list):
        linestyle = result['linestyle']
        label = result['label']
        legend_rank = 2000 + i  # Particle sizes below sink indices
        fig1.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible trace
            mode='lines',
            line=dict(color='white', dash=linestyle),
            name=label,
            legendgroup="particle_size",
            legendgrouptitle_text="Particle Size (μm)" if i == 0 else None,
            showlegend=True,
            legendrank=legend_rank,
            legend="legend2"
        ))

    fig1.update_layout(
        # title="Dissolution Profiles",
        margin=dict(l=20, r=20, t=0, b=20),
        xaxis_title="Time (min)",
        yaxis_title="Dissolved Percentage (%)",
        hovermode="closest",
        template="plotly_white",
        height=450,
        font=dict(size=16),
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,
            traceorder="normal",
            font=dict(size=16)
        ),
        legend2=dict(
            groupclick="toggleitem",
            yanchor="top",
            y=0.7,
            xanchor="left",
            x=1.05,
            traceorder="normal",
            font=dict(size=16)
        )
    )

    # Subplots for Surface Area, Solubility, and Concentration
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Surface Area vs Time", "Solubility vs Time", "Concentration vs Time"))

    # Surface Area
    for si in si_all:
        color = color_map[si]
        for result in results_list:
            linestyle = result['linestyle']
            label = result['label']
            if si == params['si_max']:
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=result['A_sink'],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f"SI={si:.2f}, {label}",
                    hovertemplate='Time: %{x:.2f} min<br>Surface Area: %{y:.2f} cm²<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=1)
            else:
                idx = params['si_values'].index(si)
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=result['A_nonsink_list'][idx],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f'SI={si:.2f}, {label}',
                    hovertemplate='Time: %{x:.2f} min<br>Surface Area: %{y:.2f} cm²<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=1)

    # Solubility
    for si in si_all:
        color = color_map[si]
        for result in results_list:
            linestyle = result['linestyle']
            label = result['label']
            if si == params['si_max']:
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=params['Cs_sink'] * np.ones_like(result['t']),
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f'SI={si:.2f}, {label}',
                    hovertemplate='Time: %{x:.2f} min<br>Solubility: %{y:.2f} mg/L<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=2)
            else:
                idx = params['si_values'].index(si)
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=result['Cs_nonsink_list'][idx],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f'SI={si:.2f}, {label}',
                    hovertemplate='Time: %{x:.2f} min<br>Solubility: %{y:.2f} mg/L<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=2)
        if si == params['si_max']:
            fig2.add_trace(go.Scatter(
                x=result['t'], y=params['C_cryst'] * np.ones_like(result['t']),
                mode='lines', line=dict(color='red', dash='dash'),
                name='C_crystal',
                hovertemplate='Time: %{x:.2f} min<br>Solubility: %{y:.2f} mg/L',
                showlegend=False
            ), row=1, col=2)

    # Concentration
    for si in si_all:
        color = color_map[si]
        for result in results_list:
            linestyle = result['linestyle']
            label = result['label']
            if si == params['si_max']:
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=result['Cb_sink'],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f"SI={si:.2f}, {label}",
                    hovertemplate='Time: %{x:.2f} min<br>Concentration: %{y:.2f} mg/L<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=3)
            else:
                idx = params['si_values'].index(si)
                fig2.add_trace(go.Scatter(
                    x=result['t'], y=result['Cb_nonsink_list'][idx],
                    mode='lines', line=dict(color=color, dash=linestyle),
                    name=f'SI={si:.2f}, {label}',
                    hovertemplate='Time: %{x:.2f} min<br>Concentration: %{y:.2f} mg/L<br>%{text}',
                    text=[label] * len(result['t']),
                    showlegend=False
                ), row=1, col=3)

    fig2.update_layout(
        height=500,
        template="plotly_white",
        showlegend=False,
        hovermode="closest"
    )
    fig2.update_yaxes(title_text="Surface Area (cm²)", row=1, col=1)
    fig2.update_yaxes(title_text="Solubility C_s (mg/L)", type="log", range=[1, 3], row=1, col=2)
    fig2.update_yaxes(title_text="Concentration C_b (mg/L)", row=1, col=3)
    fig2.update_xaxes(title_text="Time (min)", row=1, col=1)
    fig2.update_xaxes(title_text="Time (min)", row=1, col=2)
    fig2.update_xaxes(title_text="Time (min)", row=1, col=3)

    return fig1, fig2

@st.cache_data
def run_simulations(r0_values, si_values, si_max, m0, DL, rho, V, k, Cs0, k_cryst_ref, r_ref, alpha, C_cryst, t_max):
    """Run simulations with Blues color gradient."""
    params = setup_parameters(r0_values, si_values, si_max, m0, DL, rho, V, k, Cs0, k_cryst_ref, r_ref, alpha, C_cryst, t_max)
    linestyles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
    si_all = sorted(params['si_values'] + [params['si_max']], reverse=False)  # Lowest to highest for color mapping
    n_colors = len(si_all)
    color_gradient = generate_color_gradient(n_colors, start=0.2, end=0.9)
    results = []
    for i, (r0, A0, k_cryst) in enumerate(zip(r0_values, params['A0_values'], params['k_cryst_values'])):
        linestyle = linestyles[i % len(linestyles)]
        result = run_simulation(params, A0, r0, k_cryst, color_gradient, linestyle)
        results.append(result)
    return results, params

def main():
    st.set_page_config(layout="wide")
    
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-left: 1rem !important;
            padding-top: 0.1rem !important;
            padding-bottom: 0.1rem !important;
        }
        h2 {
            margin-bottom: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Interactive Dissolution Simulation")
    st.write("Adjust the sliders and inputs below to simulate dissolution profiles. Plots update automatically.")
    st.write("Chris Kossor (cgkossor@gmail.com)")

    st.sidebar.header("Simulation Parameters")
    DL = st.sidebar.slider("Drug Loading (DL)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="dl")
    k = st.sidebar.slider("Dissolution Rate Constant (k, m/min)", min_value=0.0001, max_value=0.005, value=1.80e-3, step=1e-4, format="%.4e", key="k")
    Cs0 = st.sidebar.slider("Initial Amorphous Solubility (Cs0, mg/L)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key="cs0")
    C_cryst = st.sidebar.slider("Crystalline Solubility (C_cryst, mg/L)", min_value=0.0, max_value=50.0, value=14.5, step=0.1, key="c_cryst")
    
    st.sidebar.header("Crystalization Constants")
    r_ref = st.sidebar.number_input("Reference Particle Radius (r_ref, μm)", min_value=0.0, value=22.5, step=0.1, key="r_ref") * 1e-6
    k_cryst_ref = st.sidebar.number_input("Reference Crystallization Rate (k_cryst_ref, /min)", min_value=0.0, value=1.36e-2, step=1e-3, format="%.3e", key="k_cryst_ref")
    alpha = st.sidebar.number_input("Power-law Exponent (alpha)", min_value=0.0, value=1.0, step=0.1, key="alpha")
   
    st.sidebar.header("Particle Sizes (r0, μm)")
    num_r0 = st.sidebar.number_input("Number of Particle Sizes", min_value=1, max_value=5, value=2, step=1, key="num_r0")
    r0_values = []
    for i in range(num_r0):
        default_value = 22.5 if i == 0 else 37.5 if i == 1 else 10.0 * (i + 1)
        r0 = st.sidebar.number_input(f"Particle Size {i+1} (μm)", min_value=0.0, value=default_value, step=0.1, key=f"r0_{i}") * 1e-6
        r0_values.append(r0)
    
    st.sidebar.header("Sink Index Values (SI)")
    si_max = st.sidebar.number_input("Maximum Sink Index (SI_max)", min_value=0.0, value=3.0, step=0.1, key="si_max")
    num_si = st.sidebar.number_input("Number of Non-Sink Indices", min_value=1, max_value=5, value=2, step=1, key="num_si")
    si_values = []
    for i in range(num_si):
        default_value = 0.145 if i == 0 else 0.25 if i == 1 else 0.1 * (i + 1)
        si = st.sidebar.number_input(f"Non-Sink Index {i+1}", min_value=0.0, value=default_value, step=0.01, key=f"si_{i}")
        si_values.append(si)

    st.sidebar.header("Experimental Setup")
    m0 = st.sidebar.number_input("Total Powder Mass (m0, mg)", min_value=0.0, value=180.0, step=1.0, key="m0")
    rho = st.sidebar.number_input("Density (rho, kg/m³)", min_value=0.0, value=1200.0, step=10.0, key="rho")
    V = st.sidebar.number_input("Volume (V, L)", min_value=0.0, value=0.9, step=0.01, key="v")
    t_max = st.sidebar.number_input("Maximum Time (t_max, min)", min_value=0.0, value=120.0, step=1.0, key="t_max")

    with st.container():
        results, params = run_simulations(r0_values, si_values, si_max, m0, DL, rho, V, k, Cs0, k_cryst_ref, r_ref, alpha, C_cryst, t_max)
        fig1, fig2 = plot_comparison(results, params)
        
        st.markdown("<h3>Dissolved Percentage</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True, config={'staticPlot': False, 'responsive': True})
        st.markdown("<h3>Surface Area, Solubility, and Concentration</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, config={'staticPlot': False, 'responsive': True})

if __name__ == "__main__":
    main()