"""
Extended experiments for the dissertation (run after shor_dissertation_sim.py).

E1  Success funnel: informative measurement -> period recovered -> factor recovered,
    observed and theoretical, with Clopper-Pearson 95% confidence intervals.
E2  Precision experiment: success versus number of counting qubits.
E3  Larger composites: N = 33, 35, 55 (18-qubit circuits).
E4  Resource analysis: qubits, depth and gate counts versus N.
E5  Independent verification with SymPy (multiplicative order, factorisation,
    continued-fraction convergents).

Model A = exact analytical distribution (NumPy); Model B = Qiskit circuit on Aer.
Run:  python shor_extended_experiments.py      (outputs to ./out)
"""
import math, time, json, sys
from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sympy
from sympy.ntheory import n_order, factorint
from sympy.ntheory.continued_fraction import continued_fraction_convergents, continued_fraction_iterator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, QFTGate
from qiskit_aer import AerSimulator

OUT, SEED, SHOTS = "out", 2026, 4096
plt.rcParams.update({"font.family": "serif", "font.size": 10})
log = lambda *a: print(*a, flush=True)


def order(a, N):
    r, v = 1, a % N
    while v != 1:
        v, r = (v * a) % N, r + 1
    return r


def modmul_gate(a, power, N, n):
    m = pow(a, power, N)
    U = np.zeros((2 ** n, 2 ** n))
    for y in range(2 ** n):
        U[(m * y) % N if y < N else y, y] = 1
    return UnitaryGate(U, label=f"{a}^{power} mod {N}")


def shor_circuit(a, N, t=None):
    n = math.ceil(math.log2(N))
    t = 2 * n if t is None else t
    qc = QuantumCircuit(t + n, t)
    qc.h(range(t))
    qc.x(t)
    for j in range(t):
        qc.append(modmul_gate(a, 2 ** j, N, n).control(1), [j] + list(range(t, t + n)))
    qc.append(QFTGate(t).inverse(), range(t))
    qc.measure(range(t), range(t))
    return qc, n, t


def theory(a, N, t):
    r, Q = order(a, N), 2 ** t
    probs, ys = np.zeros(Q), np.arange(Q)
    for k in range(r):
        xs = np.arange(k, Q, r)
        probs += np.abs(np.exp(2j * np.pi * np.outer(ys, xs) / Q).sum(axis=1) / Q) ** 2
    return probs / probs.sum()


def stages(y, a, N, t):
    """Return (informative, period_recovered, factor_recovered, candidate_r)."""
    r = order(a, N)
    if y == 0:
        return False, False, False, None
    r_c = Fraction(y, 2 ** t).limit_denominator(N).denominator
    informative = True
    period_ok = pow(a, r_c, N) == 1
    factor_ok = False
    if period_ok and r_c % 2 == 0:
        h = pow(a, r_c // 2, N)
        if h != N - 1 and 1 < math.gcd(h - 1, N) < N:
            factor_ok = True
    return informative, period_ok, factor_ok, r_c


def run_case(a, N, t, seed):
    qc, n, t = shor_circuit(a, N, t)
    sim = AerSimulator(seed_simulator=seed)
    counts = sim.run(transpile(qc, sim, seed_transpiler=SEED), shots=SHOTS).result().get_counts()
    obs = np.zeros(2 ** t, dtype=int)
    for b, c in counts.items():
        obs[int(b, 2)] += c
    return obs, theory(a, N, t), n, t


def ci(k, n):
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="exact")
    return round(100 * lo, 2), round(100 * hi, 2)


def funnel_row(obs, p, a, N, t, label):
    st = np.array([stages(y, a, N, t)[:3] for y in range(2 ** t)])
    row = dict(case=label, N=N, a=a, r=order(a, N), counting_qubits=t, shots=SHOTS)
    for i, name in enumerate(["informative", "period", "factor"]):
        k = int(obs[st[:, i]].sum())
        row[f"obs_{name}_pct"] = round(100 * k / SHOTS, 2)
        row[f"theory_{name}_pct"] = round(100 * float(p[st[:, i]].sum()), 2)
        if name == "factor":
            row["factor_ci_low"], row["factor_ci_high"] = ci(k, SHOTS)
            pt = float(p[st[:, i]].sum())
            row["binom_p"] = round(stats.binomtest(k, SHOTS, pt).pvalue, 4) if 0 < pt < 1 else None
    return row


# ---------------------------------------------------------------- E1 + E3
CASES = {15: [2, 4, 7, 8, 11, 13, 14], 21: [2, 4, 5, 8, 13, 20]}
NEW = {33: [2, 5, 10], 35: [2, 3, 6], 55: [2, 4, 21]}
rows, outcome_rows, case_no = [], [], 0
t0 = time.time()
for N, bases in list(CASES.items()) + list(NEW.items()):
    for a in bases:
        case_no += 1
        obs, p, n, t = run_case(a, N, None, SEED + case_no)
        rows.append(funnel_row(obs, p, a, N, t, f"{N}/{a}"))
        for y in np.nonzero((obs > 0) | (p > 1e-9))[0]:
            s = stages(int(y), a, N, t)
            outcome_rows.append(dict(N=N, a=a, counting_qubits=t, y=int(y),
                                     y_over_2t=round(y / 2 ** t, 6), observed_count=int(obs[y]),
                                     theoretical_prob=round(float(p[y]), 6), candidate_r=s[3],
                                     true_r=order(a, N), period_recovered=s[1],
                                     factor_recovered=s[2]))
        log(f"E1 {N}/{a} r={order(a,N)} factor obs={rows[-1]['obs_factor_pct']} "
            f"th={rows[-1]['theory_factor_pct']} ({time.time()-t0:.0f}s)")
funnel = pd.DataFrame(rows)
funnel.to_csv(f"{OUT}/table_E1_funnel.csv", index=False)
pd.DataFrame(outcome_rows).to_csv(f"{OUT}/data_outcome_level.csv", index=False)

# Combined H01 test over all informative cases
pv = funnel["binom_p"].dropna().values
fisher = stats.combine_pvalues(pv, method="fisher")
log("Fisher all cases", fisher, len(pv))

# ---------------------------------------------------------------- E2 precision
prec_rows = []
for (N, a, ts) in [(15, 7, range(2, 9)), (21, 2, range(3, 11)), (35, 2, range(4, 13))]:
    for t in ts:
        case_no += 1
        obs, p, n, _ = run_case(a, N, t, SEED + case_no)
        r = funnel_row(obs, p, a, N, t, f"{N}/{a}")
        prec_rows.append(r)
        log(f"E2 {N}/{a} t={t} period obs={r['obs_period_pct']} th={r['theory_period_pct']}")
prec = pd.DataFrame(prec_rows)
prec.to_csv(f"{OUT}/table_E2_precision.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 3.4))
for (lab, g), col in zip(prec.groupby("case", sort=False), ["#4C72B0", "#C44E52", "#55A868"]):
    ax.plot(g.counting_qubits, g.theory_period_pct, "-", color=col, alpha=0.6)
    ax.plot(g.counting_qubits, g.obs_period_pct, "o", color=col,
            label=f"N/a = {lab} (r = {g.r.iloc[0]})")
ax.set_xlabel("Counting qubits t (line = theory, dots = simulation)")
ax.set_ylabel("Period recovered per run (%)")
ax.set_ylim(0, 70)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_E2_precision.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- E5 SymPy verification
ver = []
for N, bases in list(CASES.items()) + list(NEW.items()):
    t = 2 * math.ceil(math.log2(N))
    for a in bases:
        r_sym = int(n_order(a, N))
        fac = factorint(N)
        agree = total = 0
        for y in range(1, 2 ** t):
            # candidate r by continued-fraction convergents: largest convergent denominator <= N
            dens = [c.q for c in continued_fraction_convergents(
                continued_fraction_iterator(sympy.Rational(y, 2 ** t))) if c.q <= N]
            r_conv = dens[-1] if dens else 1
            r_ld = Fraction(y, 2 ** t).limit_denominator(N).denominator
            ok_conv = pow(a, int(r_conv), N) == 1
            ok_ld = pow(a, r_ld, N) == 1
            total += 1
            agree += (ok_conv == ok_ld)
        ver.append(dict(N=N, a=a, order_simulation_code=order(a, N), order_sympy=r_sym,
                        order_match=order(a, N) == r_sym,
                        factorisation_sympy=" x ".join(f"{p_}" for p_ in sorted(fac)),
                        outcomes_checked=total, period_verdict_agreement_pct=round(100 * agree / total, 2)))
verify = pd.DataFrame(ver)
verify.to_csv(f"{OUT}/table_E5_sympy_verification.csv", index=False)
log(verify.to_string())

# ---------------------------------------------------------------- E4 resources
res = []
for N in [15, 21, 33, 35, 55]:
    qc, n, t = shor_circuit(2, N)
    t1 = time.time()
    tq = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                   seed_transpiler=SEED)
    ops = tq.count_ops()
    res.append(dict(N=N, bits=n, counting_qubits=t, total_qubits=t + n, depth=tq.depth(),
                    two_qubit_gates=int(ops.get("cx", 0)),
                    single_qubit_gates=int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x")))))
    log("E4", res[-1], f"{time.time()-t1:.0f}s")
resdf = pd.DataFrame(res)
resdf.to_csv(f"{OUT}/table_E4_resources.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.semilogy(resdf.N, resdf.two_qubit_gates, "o-", color="#C44E52", label="Two-qubit gates")
ax.semilogy(resdf.N, resdf.depth, "s-", color="#4C72B0", label="Circuit depth")
for _, r_ in resdf.iterrows():
    ax.annotate(f"{r_.total_qubits} qubits", (r_.N, r_.two_qubit_gates), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7)
ax.set_xlabel("Integer N (base a = 2)")
ax.set_ylabel("Count (log scale)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_E4_resources.png", dpi=200)
plt.close(fig)

json.dump(dict(fisher_all=dict(stat=round(fisher.statistic, 2), p=round(fisher.pvalue, 4),
                               k=len(pv))), open(f"{OUT}/summary_extended.json", "w"), indent=2)
log(funnel.to_string())
log(prec.to_string())
log(resdf.to_string())
log("DONE")
