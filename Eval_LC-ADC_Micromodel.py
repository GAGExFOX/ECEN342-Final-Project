"""
LC-ADC Micromodel for Event-Driven ECG Classification
Based on: "Evaluation of Level-Crossing ADCs for Event-Driven ECG Classification"

ECEN432 Final Project - TAMU
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate

# ─────────────────────────────────────────────
# ECG SIGNAL GENERATOR
# _____________________________________________
def generate_synthetic_ecg(duration_s=3.0, fs=360, heart_rate=72):
    """
    Generate a realistic synthetic ECG waveform.
    Output is in VOLTS scaled to match the paper's 10 mVpp full-scale range,
    centred around 0V (range approximately -2 mV to +8 mV).
    """
    t = np.arange(0, duration_s, 1/fs)
    ecg = np.zeros_like(t)
    rr_interval = 60.0 / heart_rate

    for beat_start in np.arange(0, duration_s, rr_interval):
        bt = t - beat_start
        # All amplitudes in Volts (paper uses 10 mVpp → scale to ~1e-3 V)
        # P wave  (~0.15 mV)
        p_mask = (bt > 0.05) & (bt < 0.25)
        ecg[p_mask] += 0.15e-3 * np.exp(-((bt[p_mask]-0.15)**2)/(2*0.03**2))
        # Q wave  (~-0.10 mV)
        q_mask = (bt > 0.25) & (bt < 0.30)
        ecg[q_mask] -= 0.10e-3 * np.exp(-((bt[q_mask]-0.27)**2)/(2*0.008**2))
        # R wave  (~1.2 mV  — dominant peak within 10 mVpp range)
        r_mask = (bt > 0.28) & (bt < 0.38)
        ecg[r_mask] += 1.20e-3 * np.exp(-((bt[r_mask]-0.33)**2)/(2*0.015**2))
        # S wave  (~-0.20 mV)
        s_mask = (bt > 0.35) & (bt < 0.42)
        ecg[s_mask] -= 0.20e-3 * np.exp(-((bt[s_mask]-0.39)**2)/(2*0.010**2))
        # T wave  (~0.35 mV)
        t_mask = (bt > 0.40) & (bt < 0.65)
        ecg[t_mask] += 0.35e-3 * np.exp(-((bt[t_mask]-0.52)**2)/(2*0.04**2))

    # Tiny noise (~5 µV)
    np.random.seed(42)
    ecg += np.random.normal(0, 5e-6, len(ecg))
    return t, ecg


# ─────────────────────────────────────────────
# 2.  LC-ADC MODEL  (core of the paper)
# ─────────────────────────────────────────────
def lcadc_model(t_in, ecg_in, M, Fc, N, delta_v_lsbs=1, A_FS=10e-3):
    """
    Simulate a Level-Crossing ADC.

    Parameters
    ----------
    t_in        : time axis of the input signal (s)
    ecg_in      : input ECG samples (V) — must be within [-A_FS/2, +A_FS/2]
    M           : ADC resolution (bits)
    Fc          : counter clock frequency (Hz)
    N           : counter clock resolution (bits)
    delta_v_lsbs: threshold gap k (eq. 2)
    A_FS        : full-scale voltage range (V) — paper uses 10 mVpp
    """
    q = A_FS / (2**M)            # LSB size (eq. 1)
    delta_v = delta_v_lsbs * q  # threshold gap (eq. 2)
    max_TI_count = 2**N - 1     # counter rollover value

    # Quantise first sample and set initial thresholds
    ecg_q0 = np.floor(ecg_in[0] / q) * q
    L_QL = ecg_q0
    U_QL = ecg_q0 + delta_v

    t_events, ecg_out, time_intervals = [], [], []
    last_event_t = t_in[0]
    last_amp = ecg_q0

    # Emit an initial sample so reconstruction starts correctly
    t_events.append(t_in[0])
    ecg_out.append(ecg_q0)
    time_intervals.append(0.0)

    for i in range(1, len(t_in)):
        sample = ecg_in[i]
        elapsed = t_in[i] - last_event_t
        TI_counts = elapsed * Fc

        # ── Clock rollover: repeat previous amplitude ──────────────
        if TI_counts >= max_TI_count:
            t_events.append(t_in[i])
            ecg_out.append(last_amp)
            time_intervals.append(max_TI_count / Fc)
            last_event_t = t_in[i]
            elapsed = 0.0

        # ── Upper threshold crossing ───────────────────────────────
        if sample >= U_QL:
            n_steps = int((sample - U_QL) / q) + 1
            for _ in range(n_steps):
                t_events.append(t_in[i])
                ecg_out.append(U_QL)
                time_intervals.append(elapsed)
                last_amp = U_QL
                last_event_t = t_in[i]
                elapsed = 0.0
                U_QL += q
                L_QL += q

        # ── Lower threshold crossing ───────────────────────────────
        elif sample <= L_QL:
            n_steps = int((L_QL - sample) / q) + 1
            for _ in range(n_steps):
                t_events.append(t_in[i])
                ecg_out.append(L_QL)
                time_intervals.append(elapsed)
                last_amp = L_QL
                last_event_t = t_in[i]
                elapsed = 0.0
                U_QL -= q
                L_QL -= q

    return (np.array(t_events), np.array(ecg_out), np.array(time_intervals))


# ─────────────────────────────────────────────
# 3.  SIGNAL RECONSTRUCTION (linear interp)
# ─────────────────────────────────────────────
def reconstruct_signal(t_events, ecg_out, t_uniform, Fc):
    """Reconstruct uniformly-sampled signal via linear interpolation."""
    if len(t_events) < 2:
        return np.zeros_like(t_uniform)
    f_interp = interpolate.interp1d(t_events, ecg_out,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=(ecg_out[0], ecg_out[-1]))
    return f_interp(t_uniform)


# ─────────────────────────────────────────────
# 4.  PERFORMANCE METRICS  (Section II-B)
# ─────────────────────────────────────────────
def compute_SDR(x_orig, x_hat):
    """Signal-to-Distortion Ratio (eq. 3)."""
    signal_power = np.mean((x_orig - np.mean(x_orig))**2)
    distortion_power = np.mean((x_orig - x_hat)**2)
    if distortion_power < 1e-20:
        return np.inf
    return 10*np.log10(signal_power / distortion_power)


def compute_CR(ecg_orig_len, fs_orig, t_events, M, N, Fc):
    """
    Compression Ratio = bits/sec (uniform) / bits/sec (LC-ADC).
    Uniform ADC assumed 11-bit at fs_orig Hz.
    """
    uniform_bps = fs_orig * 11
    lc_rate = len(t_events) / (t_events[-1] - t_events[0])
    lc_bps = lc_rate * (M + N)
    return uniform_bps / lc_bps


# ─────────────────────────────────────────────
# 5.  PAPER TABLE I  – LC-ADC MODEL PARAMETERS
# ─────────────────────────────────────────────
# (M, Fc_Hz, N)  extracted from Table I of the paper
TABLE_I = {
    2:  (74.13,    8),
    3:  (148.26,   8),
    4:  (296.51,   7),
    5:  (593.03,   7),
    6:  (1186.06,  7),
    7:  (2372.12,  4),
    8:  (4744.24,  3),
    9:  (9488.49,  3),
    10: (18976.97, 3),
    11: (37953.94, 3),
}


# ─────────────────────────────────────────────
# 6.  RUN SIMULATIONS
# ─────────────────────────────────────────────
def run_simulation():
    fs = 2385          # upsample to match paper's Fc for M=7
    t, ecg = generate_synthetic_ecg(duration_s=3.0, fs=fs)

    results = {}
    for M in range(2, 12):
        Fc, N = TABLE_I[M]
        t_ev, amp_ev, ti_ev = lcadc_model(t, ecg, M=M, Fc=Fc, N=N,
                                          delta_v_lsbs=1)
        if len(t_ev) < 3:
            continue
        ecg_hat = reconstruct_signal(t_ev, amp_ev, t, Fc)
        sdr = compute_SDR(ecg, ecg_hat)
        cr  = compute_CR(len(ecg), fs, t_ev, M, N, Fc)
        results[M] = dict(Fc=Fc, N=N, SDR=sdr, CR=cr,
                          t_ev=t_ev, amp_ev=amp_ev, ecg_hat=ecg_hat)
        print(f"  M={M:2d}  Fc={Fc:9.2f} Hz  N={N}  "
              f"CR={cr:6.2f}  SDR={sdr:6.2f} dB")

    return t, ecg, results


# ─────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────
def make_plots(t, ecg, results):
    M_vals = sorted(results.keys())
    SDRs   = [results[m]['SDR'] for m in M_vals]
    CRs    = [results[m]['CR']  for m in M_vals]

    # ── Figure 1: SDR & CR vs Resolution ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LC-ADC Performance vs Resolution\n"
                 "(Micromodel – Saeed et al. 2021)", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(M_vals, SDRs, 'bo-', linewidth=2, markersize=7)
    ax.axhline(21, color='red', linestyle='--', linewidth=1.5, label='SDR = 21 dB ("Good")')
    ax.set_xlabel("LC-ADC Resolution M (bits)")
    ax.set_ylabel("SDR (dB)")
    ax.set_title("Signal-to-Distortion Ratio")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(M_vals)

    ax = axes[1]
    ax.plot(M_vals, CRs, 'gs-', linewidth=2, markersize=7)
    ax.axhline(1.5, color='orange', linestyle='--', linewidth=1.5, label='CR = 1.5 (min acceptable)')
    ax.axhline(1.0, color='red',    linestyle=':',  linewidth=1.2, label='CR = 1 (no compression)')
    ax.set_xlabel("LC-ADC Resolution M (bits)")
    ax.set_ylabel("Compression Ratio (CR)")
    ax.set_title("Compression Ratio")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(M_vals)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/lcadc_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Waveform comparison for key resolutions ─────────
    show_M = [4, 7, 9]
    fig, axes = plt.subplots(len(show_M), 1, figsize=(13, 10), sharex=True)
    fig.suptitle("LC-ADC Event-Driven Sampling vs Original ECG\n"
                 "(Micromodel – Saeed et al. 2021)", fontsize=13, fontweight='bold')

    t_zoom = (t >= 0.2) & (t <= 2.5)

    for idx, M in enumerate(show_M):
        ax = axes[idx]
        r = results[M]
        ax.plot(t[t_zoom], ecg[t_zoom]*1e3, 'gray', linewidth=1.2,
                alpha=0.6, label='ECG_in (original)')
        ax.plot(t[t_zoom], r['ecg_hat'][t_zoom]*1e3, 'b-', linewidth=1.5,
                label='ECG reconstructed')
        ev_mask = (r['t_ev'] >= 0.2) & (r['t_ev'] <= 2.5)
        ax.scatter(r['t_ev'][ev_mask], r['amp_ev'][ev_mask]*1e3,
                   color='red', s=12, zorder=5, label='LC events')
        ax.set_ylabel("ECG (mV)")
        ax.set_title(f"M={M} bits | Fc={r['Fc']:.0f} Hz | N={r['N']} | "
                     f"CR={r['CR']:.2f} | SDR={r['SDR']:.1f} dB")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/lcadc_waveforms.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: SDR vs CR trade-off bubble chart ─────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(CRs, SDRs, c=M_vals, cmap='viridis',
                    s=120, zorder=5, edgecolors='k', linewidths=0.8)
    for M in M_vals:
        ax.annotate(f"M={M}", (results[M]['CR'], results[M]['SDR']),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(21,  color='red',    linestyle='--', label='SDR = 21 dB')
    ax.axvline(1.5, color='orange', linestyle='--', label='CR = 1.5')
    plt.colorbar(sc, label='M (bits)')
    ax.set_xlabel("Compression Ratio (CR)")
    ax.set_ylabel("SDR (dB)")
    ax.set_title("SDR vs CR Trade-off  (Saeed et al. 2021 – Micromodel)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/lcadc_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Event counts per beat ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    event_counts = [len(results[m]['t_ev']) for m in M_vals]
    bars = ax.bar(M_vals, event_counts, color='steelblue', edgecolor='k', alpha=0.85)
    ax.set_xlabel("LC-ADC Resolution M (bits)")
    ax.set_ylabel("Total Level-Crossing Events")
    ax.set_title("Event Count vs Resolution (3-second ECG segment)")
    ax.set_xticks(M_vals)
    for bar, cnt in zip(bars, event_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(cnt), ha='center', va='bottom', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/lcadc_event_counts.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nAll plots saved.")


# ─────────────────────────────────────────────
# 8.  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary(results):
    print("\n" + "="*65)
    print(f"{'M':>3} {'Fc (Hz)':>10} {'N':>3} {'CR':>7} {'SDR (dB)':>10}  {'Quality'}")
    print("-"*65)
    for M in sorted(results.keys()):
        r = results[M]
        quality = "GOOD ✓" if r['SDR'] >= 21 else ("OK" if r['SDR'] >= 15 else "POOR")
        print(f"{M:3d} {r['Fc']:10.2f} {r['N']:3d} {r['CR']:7.2f} {r['SDR']:10.2f}  {quality}")
    print("="*65)
    print("\nKey finding: M=7 offers best CR–SDR trade-off (paper conclusion)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("LC-ADC Micromodel Simulation")
    print("Saeed et al. IEEE TBCAS 2021  |  ECEN432 TAMU\n")
    print(f"{'M':>3} {'Fc (Hz)':>10} {'N':>3} {'CR':>7} {'SDR (dB)':>10}")
    print("-"*45)

    t, ecg, results = run_simulation()
    print_summary(results)
    make_plots(t, ecg, results)
