"""Simulated error and power of the decision rule (P1: D >= k and lo > 0; P2: hi < k/N) under hit churn and talk
heterogeneity, planned before the decodes with an offline count of H0 = 10. Talk sizes = decision items per talk.
Needs numpy. Run: python3 -m stglossary.power [B] [R] > out.tsv  (defaults B = 9999, R = 400 runs per cell)"""
import numpy as np, sys
sizes = np.array([5,6,9,10,11,12,12,14,18,19]); N = sizes.sum(); talk = np.repeat(np.arange(10), sizes)
W6 = np.array([-np.sqrt(1.5),-1,-np.sqrt(.5),np.sqrt(.5),1,np.sqrt(1.5)])
R = int(sys.argv[2]) if len(sys.argv) > 2 else 400
B = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
H0 = 10
rng = np.random.default_rng(1)
k = max(0.10*N, 0)
def run(rise, keep, het, margins):
    out = {m: np.zeros(3) for m in margins}; Ds=[]
    for _ in range(R):
        tr = rng.gamma(1/het**2, het**2, 10) if het else np.ones(10)
        p = tr[talk]; p = p/p.sum()
        off = np.zeros(N, int); off[rng.choice(N, H0, replace=False, p=p)] = 1
        st = off * (rng.random(N) < keep)
        need = H0 + rise - st.sum()          # expected new hits so E[H_stream]=10+rise
        q = p*(1-off); q = q/q.sum()
        lam = np.clip(need*q, 0, 1)
        st = np.maximum(st, (rng.random(N) < lam).astype(int) * (1-off))
        d = st - off; D = d.sum(); Ds.append(D); mean = D/N
        S = np.bincount(talk, d - mean, 10)
        w = W6[rng.integers(0, 6, (B, 10))]
        bs = np.sort(mean + w @ S / N); lo, hi = bs[int(.025*B)], bs[int(.975*B)-1]
        for m in margins:
            if D >= k and lo > 0: out[m][0] += 1
            elif (m > 0 and lo >= -m and hi <= m) or (m < 0 and hi < -m): out[m][1] += 1
            else: out[m][2] += 1
    return {m: out[m]/R for m in margins}, np.mean(Ds)
margins = [-0.10, -0.05]
print("rise keep het | P1 P2 P3 (one-sided hi<.10) | P1 P2 P3 (one-sided hi<.05) | mean D")
for keep in (1.0, 0.6):
  for het in (0, 0.7):
    for rise in (0, 5, 10, 12, 14, 15, 20, 30):
        o, md = run(rise, keep, het, margins)
        print(rise, keep, het, "|", " ".join(f"{x:.3f}" for x in o[-0.10]), "|", " ".join(f"{x:.3f}" for x in o[-0.05]), "|", round(md,1)); sys.stdout.flush()
