"""Post-processing of extended results: refined funnel stage, SymPy agreement on simulated
measurements, funnel figure. Run after shor_extended_experiments.py."""
import math
from fractions import Fraction
import numpy as np, pandas as pd, sympy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sympy.ntheory.continued_fraction import continued_fraction_convergents, continued_fraction_iterator
plt.rcParams.update({"font.family": "serif", "font.size": 10})
OUT = "out"
d = pd.read_csv(f"{OUT}/data_outcome_level.csv")
d = d[[c for c in d.columns if c not in ('useful_phase','r_convergent','period_ok_convergent')]]
f = pd.read_csv(f"{OUT}/table_E1_funnel.csv")
f = f[[c for c in f.columns if c not in ("obs_useful_pct","theory_useful_pct","sympy_convergent_agreement_pct","measurements")]]

def near_peak(y, t, r):
    """Useful phase estimate: |y/2^t - s/r| < 1/(2r^2) for some s = 1..r-1 (Legendre's
    criterion, which guarantees s/r appears among the continued-fraction convergents)."""
    x = y / 2 ** t
    return any(abs(x - s / r) < 1 / (2 * r * r) for s in range(1, r))

def conv_r(y, t, N):
    if y == 0: return 1
    dens = [c.q for c in continued_fraction_convergents(continued_fraction_iterator(sympy.Rational(y, 2 ** t))) if c.q <= N]
    return int(dens[-1]) if dens else 1

d["useful_phase"] = [near_peak(y, t, r) for y, t, r in zip(d.y, d.counting_qubits, d.true_r)]
d["r_convergent"] = [conv_r(y, t, N) for y, t, N in zip(d.y, d.counting_qubits, d.N)]
d["period_ok_convergent"] = [pow(int(a), int(rc), int(N)) == 1 for a, rc, N in zip(d.a, d.r_convergent, d.N)]
d.to_csv(f"{OUT}/data_outcome_level.csv", index=False)

rows = []
for (N, a), g in d.groupby(["N", "a"], sort=False):
    tot = g.observed_count.sum()
    useful_obs = 100 * g.observed_count[g.useful_phase].sum() / tot
    useful_th = 100 * g.theoretical_prob[g.useful_phase].sum()
    agree = 100 * g.observed_count[g.period_ok_convergent == g.period_recovered].sum() / tot
    rows.append(dict(N=N, a=a, obs_useful_pct=round(useful_obs, 2), theory_useful_pct=round(useful_th, 2),
                     sympy_convergent_agreement_pct=round(agree, 2), measurements=int(tot)))
extra = pd.DataFrame(rows)
f = f.merge(extra, on=["N", "a"])
f.to_csv(f"{OUT}/table_E1_funnel.csv", index=False)
print(f[["case", "r", "obs_useful_pct", "theory_useful_pct", "obs_period_pct", "theory_period_pct",
         "obs_factor_pct", "theory_factor_pct", "factor_ci_low", "factor_ci_high", "binom_p",
         "sympy_convergent_agreement_pct"]].to_string())
print("total measurements", d.observed_count.sum(),
      "overall agreement", round(100 * d.observed_count[d.period_ok_convergent == d.period_recovered].sum() / d.observed_count.sum(), 3))

sel = ["15/7", "21/2", "21/4", "21/5", "33/5", "35/2", "55/2"]
s = f.set_index("case").loc[sel]
x = np.arange(len(sel)); w = 0.27
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for i, (col, lab, c) in enumerate([("obs_useful_pct", "Useful phase estimate", "#BBBBBB"),
                                   ("obs_period_pct", "Period recovered", "#4C72B0"),
                                   ("obs_factor_pct", "Factor recovered", "#C44E52")]):
    ax.bar(x + (i - 1) * w, s[col], w, label=lab, color=c)
    th = col.replace("obs", "theory")
    ax.plot(x + (i - 1) * w, s[th], "k_", ms=9, mew=1.2)
ax.plot([], [], "k_", label="Theory")
ax.set_xticks(x, [f"{c}\n(r = {s.loc[c, 'r']})" for c in sel])
ax.set_ylabel("Share of runs (%)"); ax.set_ylim(0, 105)
ax.set_xlabel("Case (N / a)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_E1_funnel.png", dpi=200)
