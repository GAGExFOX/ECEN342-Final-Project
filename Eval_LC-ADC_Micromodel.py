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
