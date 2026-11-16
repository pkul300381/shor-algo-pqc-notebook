"""Redraw resource figure against bit-length n (gate count of the generic construction depends on n)."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10})
r = pd.read_csv("out/table_E4_resources.csv")
g = r.groupby("bits").agg(N=("N", lambda s: ", ".join(map(str, s))), q=("total_qubits", "first"),
                          cx=("two_qubit_gates", "max"), d=("depth", "max")).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.semilogy(g.bits, g.cx, "o-", color="#C44E52", label="Two-qubit gates")
ax.semilogy(g.bits, g.d, "s-", color="#4C72B0", label="Circuit depth")
for _, x in g.iterrows():
    ax.annotate(f"N = {x.N}\n{x.q} qubits", (x.bits, x.cx), textcoords="offset points", xytext=(0, -26),
                ha="center", fontsize=7)
ax.set_xticks(g.bits); ax.set_xlim(3.6, 6.4); ax.set_ylim(150, 6e5)
ax.set_xlabel("Bit-length of N (work qubits n)"); ax.set_ylabel("Count (log scale)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("out/fig_E4_resources.png", dpi=200)
