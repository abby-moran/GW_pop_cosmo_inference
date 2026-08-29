"""
Check whether spin draws are independent of (m1, q, z) in the injection set.
If they are, p_draw(m1,q,z,spins) factorizes as p_draw(m1,q,z)*p_draw(spins),
and you can marginalize out spins analytically once you know p_draw(spins)'s
functional form (check the release .md doc for this).

Usage:
    python check_spin_mass_independence.py <path_to_hdf_file>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def main(path):
    df = pd.read_hdf(path, key="events")
    q = df['mass2_source'] / df['mass1_source']

    for i in [1, 2]:
        comps = [f"spin{i}x", f"spin{i}y", f"spin{i}z"]
        a = np.sqrt(sum(df[c].to_numpy(dtype=float) ** 2 for c in comps))
        costilt = df[f"spin{i}z"].to_numpy(dtype=float) / a

        print(f"--- spin{i} vs (m1, q, z) : Spearman correlation ---")
        for name, x in [("mass1_source", df['mass1_source']), ("q", q), ("redshift", df['redshift'])]:
            r_mag, _ = spearmanr(a, x)
            r_tilt, _ = spearmanr(costilt, x)
            print(f"  |spin{i}| vs {name}: rho={r_mag:+.4f}   cos(tilt{i}) vs {name}: rho={r_tilt:+.4f}")
        print()

    print("Interpretation: |rho| << 0.05 across the board -> spins effectively")
    print("independent of (m1,q,z) -> factorization assumption is safe.")
    print("Any |rho| that's clearly non-negligible -> spins correlate with")
    print("mass/redshift in this injection set -> analytic marginalization")
    print("over a fixed spin functional form is NOT valid; you'd need a")
    print("conditional p_draw(spins | m1,q,z) instead.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_spin_mass_independence.py <path_to_hdf_file>")
        sys.exit(1)
    main(sys.argv[1])