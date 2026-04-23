"""
LC-ADC Micromodel for Event-Driven ECG Classification
Based on: "Evaluation of Level-Crossing ADCs for Event-Driven ECG Classification"

ECEN432 Final Project - TAMU
"""

import importlib

LC = importlib.import_module("LC-ADC")

# --- Private Variables ---
t =			list	# List of times for samples
ecg = 		list	# List of ECG Samples
results = 	list	# List of Results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("LC-ADC Micromodel Simulation")
    print("Saeed et al. IEEE TBCAS 2021  |  ECEN432 TAMU\n")
    print(f"{'M':>3} {'Fc (Hz)':>10} {'N':>3} {'CR':>7} {'SDR (dB)':>10}")
    print("-"*45)

    t, ecg, results = LC.run_simulation()
    
    LC.print_summary(results)
    
    LC.make_plots(t, ecg, results)
