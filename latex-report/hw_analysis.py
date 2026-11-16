"""Analysis and figures for the IBM Quantum hardware runs (dissertation section 4.8)."""
import json, math, numpy as np
from fractions import Fraction
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
plt.rcParams.update({"font.family": "serif", "font.size": 10})
SHOTS = 4096
# ---- N = 15 runs (3 counting qubits)
fez = {"000": 1032, "001": 33, "010": 950, "011": 47, "100": 1009, "101": 49, "110": 925, "111": 51}
mar = {"000": 1045, "001": 73, "010": 826, "011": 64, "100": 1081, "101": 85, "110": 839, "111": 83}
ideal15 = np.array([.25, 0, .25, 0, .25, 0, .25, 0])
def n15(c):
    p = np.array([c[format(y, "03b")] for y in range(8)]) / SHOTS
    k = c["010"] + c["110"]
    ci = stats.binomtest(k, SHOTS).proportion_ci(0.95, "exact")
    return dict(success=100 * k / SHOTS, ci=(100 * ci.low, 100 * ci.high), k=k,
                signal=100 * p[[0, 2, 4, 6]].sum(), tvd=0.5 * abs(p - ideal15).sum(),
                fid=np.sqrt(p * ideal15).sum() ** 2,
                p_random=stats.binomtest(k, SHOTS, 0.25, alternative="greater").pvalue,
                p_ideal=stats.binomtest(k, SHOTS, 0.5).pvalue, dist=p)
A, B = n15(fez), n15(mar)
z, pz = proportions_ztest([A["k"], B["k"]], [SHOTS, SHOTS])
print("N15 fez", {k: v for k, v in A.items() if k != "dist"}); print("N15 marrakesh", {k: v for k, v in B.items() if k != "dist"})
print("fez vs marrakesh z", z, "p", pz)
# ---- r = 6 runs (9 counting qubits)
T, Q, R = 9, 512, 6
def theory():
    ys = np.arange(Q); p = np.zeros(Q)
    for k in range(R):
        xs = np.arange(k, Q, R); p += np.abs(np.exp(2j*np.pi*np.outer(ys, xs)/Q).sum(1)/Q)**2
    return p / p.sum()
pth = theory()
def fmask(N, a):
    m = []
    for y in range(Q):
        ok = False
        if y:
            r = Fraction(y, Q).limit_denominator(N).denominator
            ok = pow(a, r, N) == 1 and r % 2 == 0 and pow(a, r//2, N) != N-1 and 1 < math.gcd(pow(a, r//2, N)-1, N) < N
        m.append(ok)
    return np.array(m)
near = sorted({(round(s * Q / R) + d) % Q for s in range(R) for d in (-1, 0, 1)})
v1 = json.load(open("/mnt/user-data/uploads/ibm_hardware_r6_counts.json"))["counts"]
ys1 = np.repeat([int(b, 2) for b in v1], list(v1.values()))
v2 = json.load(open("/mnt/user-data/uploads/ibm_r6_v2_bitstrings.json"))["bitstrings"]["plain"]
ys2 = np.array([int(b, 2) for b in v2["c"]]); keep2 = np.array([w != "11" for w in v2["w"]])
def r6(ys, keep=None):
    h = np.bincount(ys, minlength=Q) / len(ys); out = dict(near=100*h[near].sum(), fid=np.sqrt(h*pth).sum()**2,
                                                           tvd=0.5*abs(h-pth).sum())
    for N, a in [(21, 2), (77, 12)]:
        m = fmask(N, a); k = int(m[ys].sum()); ci = stats.binomtest(k, len(ys)).proportion_ci(0.95, "exact")
        out[N] = dict(s=100*k/len(ys), ci=(100*ci.low, 100*ci.high), ideal=100*pth[m].sum(), rnd=100*m.mean(),
                      p=stats.binomtest(k, len(ys), m.mean(), alternative="greater").pvalue)
        if keep is not None:
            kk = int(m[ys[keep]].sum()); out[N]["post"] = 100*kk/keep.sum()
    return out, h
C, h1 = r6(ys1); D, h2 = r6(ys2, keep2)
print("r6 v1", C); print("r6 v2", D, "kept", keep2.mean()); print("near bins", len(near), "uniform near", len(near)/Q)
print("ideal near", pth[near].sum(), "uniform fid", np.sqrt(pth/Q).sum()**2)
# ---- figures
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(8); w = 0.27
ax.bar(x - w, ideal15, w, color="#BBBBBB", label="Ideal (theory)")
ax.bar(x, A["dist"], w, color="#4C72B0", label="ibm_fez (14 Nov 2026)")
ax.bar(x + w, B["dist"], w, color="#C44E52", label="ibm_marrakesh (16 Nov 2026)")
ax.set_xticks(x, [f"{y:03b}\n(y={y})" for y in range(8)]); ax.set_ylabel("Probability")
ax.set_xlabel("Measured counting register"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("hw_n15.png", dpi=200); plt.close(fig)
fig, axs = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True); yy = np.arange(Q)
for ax, h, lab in [(axs[0], h1, "(a) First circuit: 302 two-qubit gates, depth 860"),
                   (axs[1], h2, "(b) Redesigned circuit: 73 two-qubit gates, depth 324")]:
    ax.vlines(yy, 0, pth, color="#C44E52", lw=1, label="Theory (ideal)")
    ax.plot(yy, h, ".", ms=3, color="#4C72B0", label="ibm_marrakesh")
    ax.set_ylim(0, 0.18); ax.set_ylabel("Probability"); ax.set_title(lab, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
axs[1].set_xlabel("Measured value y (9 counting qubits)")
fig.tight_layout(); fig.savefig("hw_r6.png", dpi=200); plt.close(fig)
print("figures written")
