# AI-Driven Continuous Network Slicing for Healthcare — Verified Issue List & Fix Methodology (v2)

**What changed from v1:** every item below was re-checked directly against the actual project files in
`intern.zip` (`config.py`, `environment.py`, `03_train_ddpg.ipynb`, `04_evaluate.ipynb`, `PROJECT_WORKFLOW.md`,
`dataset.csv`) rather than taken from the prior conversation's memory. Status: **all 11 original items are
still unresolved in the current code.** One new, more severe bug was found that was not on the previous list.
It is placed first because it invalidates every DDPG result currently produced by `04_evaluate.ipynb`.

**Ground rules (unchanged):**
1. Fix one item at a time. Do not batch unrelated fixes into a single run.
2. Items marked **DECISION REQUIRED** need an explicit choice before any code is written.
3. After any change to `config.py`, `environment.py`, or the actor/eval logic, re-run the **full evaluation
   table** (Equal Sharing / Priority-DataSize Heuristic / Static Priority / Round Robin / Random / DDPG /
   DDPG-Ablated) before trusting any comparison.
4. Report the full per-slice table every time — never a filtered "SUCCESS" summary.

---

## Sequencing overview

```
0 (new, blocks all DDPG numbers) → 1 → 2 → 3(decision) → 4 → 5 → 6 → 7(decision)
  → [re-run full evaluation] → 9 → 8 → 10 → 11
```

---

## 0. NEW — CRITICAL: The evaluated actor is architecturally different from the trained actor

**Problem:** confirmed directly by diffing the `Actor` class in both notebooks.

`03_train_ddpg.ipynb` (this is what `ddpg_actor.pth` was actually trained as):
```python
def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x)) * 15.0   # bounded [0, 15] output
```
Training then converts this into an executable action via proportional rescaling:
```python
pred_actions = pred_raw_actions * (batch_total_bw / sum_pred)
```

`04_evaluate.ipynb` (this is what actually gets run at evaluation time):
```python
def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)                          # raw, unbounded, can be negative
```
and then computes the executed action with:
```python
raw_logits = actor(state_t)
probs = torch.softmax(raw_logits, dim=0).cpu().numpy().flatten()
actions = probs * TOTAL_BW_test
```

Because both `Actor` definitions have identical layer shapes (`10 → 64 → 32 → 1`), `load_state_dict()` succeeds
silently — there is no error to signal the mismatch. But the weights were optimized under a saturating sigmoid
output feeding a proportional-rescale action rule; evaluation instead feeds the same weights' raw linear output
into a batch-wide softmax, a completely different function. The two notebooks also disagree with
`PROJECT_WORKFLOW.md` Section 6.5, which specifies `softmax_normalize(raw_actions) * TOTAL_BW` as the intended
rule — meaning the *training* notebook's sigmoid+proportional-rescale approach was itself a deliberate deviation
from spec (made earlier to fix actor collapse), and evaluation was never updated to match it.

**Net effect:** every "DDPG" number currently produced by `04_evaluate.ipynb` — Cumulative Utility, SLA
violation rates, everything — reflects a policy that was never trained, not the model in `ddpg_actor.pth`.
This is more severe than the earlier eval-metric-formula bug because it doesn't just miscompute a score, it
runs an entirely different action-selection function than the one the weights were optimized for.

**Fix:** make the `Actor` class and action-selection rule in `04_evaluate.ipynb` byte-for-byte identical to
`03_train_ddpg.ipynb` — i.e., import the same `Actor` class (or copy it exactly) with the `sigmoid(...) * 15.0`
output, and compute the executed action the same way training does:
```python
raw_actions = actor(state_t)                      # sigmoid-bounded, shape (N, 1)
actions = raw_actions * (TOTAL_BW_test / raw_actions.sum())
```
Apply the same fix to the "DDPG (Ablated)" branch, which shares the same broken action logic.

**Verification before trusting any result downstream:** after the fix, run a single forward pass on one batch
of test states and confirm the executed actions sum to `TOTAL_BW_test` and fall within a sane range (compare
distribution against the training-time action distribution logged during Item 8's diagnostics, if available).
Then re-run the full evaluation table and treat every prior DDPG number as void until this passes.

---

## 1. The reward function the agent actually optimizes is not in the paper

**Status: confirmed still open.** `environment.py` still applies the SLA penalty on top of the formal utility:
```python
rewards[violating] = rewards[violating] - penalties[violating] * (1.0 + violation_ratio[violating])
```
while the paper's Eq. 21/23 only shows `U = w1·Th + w2·R − w3·L − w4·Ptotal + w5·REU`, with no penalty term.

**Fix:** add to Section 3.4:
```
R(t) = U(t) − Σᵢ 1[Lᵢ(t) > Lᵢ^req] · κᵢ · (1 + (Lᵢ(t) − Lᵢ^req)/Lᵢ^req)
```
This must land before anything else, since it's the actual objective everything is evaluated against.

---

## 2. Penalty constants are inconsistent across three sources

**Status: confirmed still open.** `config.py` right now:
```python
SLA_PENALTIES = {1: 350.0, 2: 300.0, 3: 150.0, 4: 100.0, 5: 50.0}
```
`PROJECT_WORKFLOW.md` line 228 specifies:
```
{ SliceID 1: 350, SliceID 2 (ICU): 300, SliceID 3: 20, SliceID 4: 10, SliceID 5: 5 }
```
Slices 3–5 are still penalized 7.5–10× heavier than the project's own spec. The paper still doesn't state
penalty values at all (pending Item 1).

**Fix:** change `config.py` to `{1: 350.0, 2: 300.0, 3: 20.0, 4: 10.0, 5: 5.0}` to match spec, unless there's
a specific reason to keep the heavier weighting — in which case update `PROJECT_WORKFLOW.md` and the paper to
match `config.py` instead. Either way, the three sources need to agree.

**Verification:** re-run the full evaluation table after the change — "Equal Sharing beats Static Priority"
was measured under 150/100/50 and may flip under 20/10/5.

---

## 3. Priority/criticality formulation — DECISION REQUIRED

**Status: confirmed still open, and confirmed on the actual dataset.** `dataset.csv`'s `Emergency` column is
binary (`{0,1}`), and `SliceID==2` (ICU) is `Emergency=1` for 100% of rows (verified: mean Emergency by slice
is exactly `1.0` for slice 2 vs. ~0.04 for all others). No health-severity index, no per-sensor reference
ranges, no criticality weighting `xs` exist anywhere in the dataset or code. Paper Eqs. 9–14 describe a
continuous `ρᶜ_{p,t} ∈ [0,1]` score that was never built, and the unedited note ("Change the variable
accordingly...") is still sitting in the paper draft directly before Eq. 9.

**Two options (unchanged from before):**
- **(a)** Rewrite paper Eqs. 9–14 to describe the actual binary `Emergency` flag. Fast, honest, thinner
  contribution.
- **(b)** Implement the real per-sensor severity index (`θl,s`, `θu,s`, `xs` for HeartRate/BP/SpO2) in
  `config.py`, compute it during dataset generation, and replace `Emergency` everywhere it's used (state
  vector, `Priority`, dataset generation). Changes what the agent observes as state.

Must be resolved before Item 4, since `Priority` depends on it.

---

## 4. Priority score equation (Eq. 15) is missing normalization terms the code already has

**Status: code is correct, paper is not.** `config.py`/`environment.py`/`PROJECT_WORKFLOW.md` line 105 all
agree on:
```
Priority_i(t) = clip( w1·Emergency_i + w2·(LatencyReq_min/LatencyReq_i) + w3·min(Traffic_i(t)/Traffic_max,1), 0, 1 )
```
with `w1=0.5, w2=0.3, w3=0.2` (matches `config.py` exactly). The paper's Eq. 15,
`Pi(t) = w1·Ci + w2·(1/D^req_i) + w3·Ti(t)`, is unbounded as written.

**Fix:** update Eq. 15 to match the code/spec above, and swap `Ci` for whichever Item 3 outcome is chosen.

---

## 5. Bandwidth allocation formula (Eq. 3) doesn't match the implemented baseline

**Status: confirmed still open.** `PROJECT_WORKFLOW.md` lines 116–118 specify:
```
weight_i(t) = Priority_i(t) × DataSize_i(t)
Bandwidth_i(t) = ( weight_i(t) / Σⱼ weight_j(t) ) × TOTAL_BW
```
This is what's actually baked into `dataset.csv`'s `Bandwidth` column and used as the "Priority-DataSize
Heuristic" baseline in `04_evaluate.ipynb`. Paper Eq. 3 only has `Bi(t) = Pi(t)/ΣPj(t) × Btotal` — no
`DataSize` term.

**Fix:** update Eq. 3 to include the `DataSize` weighting, with the rationale already in
`PROJECT_WORKFLOW.md`: it keeps the transmission-delay ratio `DataSize/Bandwidth` stable regardless of
payload size, so `Priority` alone determines who's served faster.

---

## 6. Formal constraints in the paper are never enforced or checked in code

**Status: confirmed still open.** No `RMIN`, `THMIN`, `PMAX`, `REUMIN` constants exist in `config.py`.
`environment.py`'s `step()` does not clip `powers` to any maximum, and nothing tracks violation rates for
`Ri(t) ≥ Rmin`, `Thtotal(t) ≥ Thmin`, or `REU ≥ REUmin` in `04_evaluate.ipynb`.

**Fix:** either (a) add the constants, hard-clip power at `Pmax` in `environment.py`, and add violation-rate
tracking for the rest to `04_evaluate.ipynb`; or (b) reframe Eqs. 26–30 in the paper as observed target ranges
rather than enforced constraints.

---

## 7. Queue-dynamics mechanism — DECISION REQUIRED

**Status: confirmed still open, unchanged.** `environment.py` still has:
```python
self.queues = np.minimum(np.maximum(self.queues + t_data['Traffic']*0.0600 - throughputs*0.78, 0.0), 50.0)
self.lambdas = np.minimum((self.queues/(1.0+self.queues))*mu, mu - 0.1)
```
Neither `PROJECT_WORKFLOW.md` nor the paper defines this — both describe `Queue`/`Lambda` as static columns.
`dataset.csv`'s own static `Queue` column tops out at 15, but the live environment's dynamic queue is capped
at 50 — over 3× higher than what the dataset (and the `/15.0` normalizer in `_get_state()`) were built around.

**Two options (unchanged):** (a) revert to spec, read `Queue`/`Lambda` statically per time step; or (b) keep
the dynamic system but formally document it in the paper (updating Eq. 1's `λ`/`µ` to be state-dependent) and
justify the `0.06`/`0.78` constants. Don't spend more time calibrating them until this is settled.

---

## [CHECKPOINT] Re-run full evaluation

After Items 0–7, re-run the complete evaluation and record fresh numbers. Nothing produced before this point —
including every DDPG comparison so far — should be treated as final.

---

## 8. Batch-coupling in the actor isn't fully eliminated — it moved into the gradient

**Status: confirmed still open**, verified directly in `03_train_ddpg.ipynb`'s training loop:
```python
pred_raw_actions = actor(S)
sum_pred = torch.sum(pred_raw_actions, dim=0, keepdim=True)
pred_actions = pred_raw_actions * (batch_total_bw / sum_pred)
loss_actor_accum = loss_actor_accum + -critic(S, pred_actions).mean()
```
`sum_pred` is not detached from the autograd graph, so patient *i*'s gradient still has an ~O(1/N) dependency
on every other patient's output (N=1000 in training).

**Fix:** measure `d(loss_i)/d(other patient's raw output)` to quantify the cross-term. If non-negligible,
detach `sum_pred` (`sum_pred.detach()`) so it's used only to compute the executed action, not inside the loss.

---

## 9. Local Critic proposal is unverified and contradicts an earlier result

**Status: confirmed still unresolved.** `test_local_critic.py` is not present anywhere in `intern.zip`. The
`Critic` in `03_train_ddpg.ipynb` still uses global mean/max pooling over the whole batch:
```python
mean_X = torch.mean(X, dim=0, keepdim=True).expand(...)
max_X = torch.max(X, dim=0, keepdim=True)[0].expand(...)
```
unchanged from the design that scored both +0.55 and -0.17 Q-calibration correlation in different rounds.

**Fix:** re-run Q-calibration with this centralized critic *after* Items 0–2 are fixed (both the eval-actor
bug and the penalty schedule change what the critic is learning to predict) before concluding the pooling
architecture itself is the problem.

---

## 10. State normalization does not bound to [0,1] — worse than previously measured

**Status: confirmed open, and confirmed worse than previously described**, verified against `dataset.csv`:

| Feature | Code | Observed range | Bound violated |
|---|---|---|---|
| `norm_queue` | `queue / 15.0` | live queue capped at 50 → up to **3.33** | yes |
| `norm_lambda` | `lambda / 10.0` | dataset `Lambda` max 10.66 → up to **1.07**, higher once dynamic queue feeds back | yes |
| `norm_traffic` | `Predicted_Traffic / 100.0` | **6.6% of all rows** exceed 100 (max ~100.6); also doesn't use `config.TRAFFIC_MAX` (150), so it's inconsistent with the `Priority` formula's own traffic normalizer | yes |
| `norm_prev_bw` | `prev_bandwidth / (TOTAL_BW/10000.0)` = `/3.0` | baseline `Bandwidth` column ranges up to 63.5 → up to **21.17** | yes, by far the worst offender |

**Fix:** recompute each divisor from the actual bound of its quantity: 50 for queue (matches the live cap),
an empirical or spec-defined max for lambda, `config.TRAFFIC_MAX` (not a hardcoded 100) for traffic, and the
actual maximum observed/possible bandwidth allocation for `prev_bw` — not `TOTAL_BW/10000`, which has no
documented derivation and is off by roughly 7× from the real data range.

**Verification:** compute the true max of each normalized feature across the full dataset (and across a live
rollout, for queue/lambda) post-fix, confirm all fall within [0,1].

---

## 11. Exploration noise added before batch-sum normalization (low priority)

**Status: confirmed open**, verified in `03_train_ddpg.ipynb`:
```python
noise = np.random.normal(0, noise_std * 15.0, size=raw_actions.shape)
noisy_actions = np.clip(raw_actions + noise, 0.05, 15.0)
actions = noisy_actions * (TOTAL_BW_train / sum_noisy)
```
Noise is added before the batch-sum rescale, so its effective magnitude varies with `sum_noisy` episode to
episode. Low priority — revisit only if exploration looks inconsistent after Items 0–10 land.

---

## Discipline reminders

- **Q-calibration gate:** correlation between predicted Q and empirical returns must clear 0.5+ before
  trusting any policy comparison built on that critic (baseline observed: -0.0128).
- **Isolate before combining fixes.** Where combining is unavoidable, add a diagnostic that gives some
  post-hoc attribution evidence (e.g. gradient-norm logging, as already present in the training loop).
- **Always report the full per-slice results table.**
- **PER (Prioritized Experience Replay)** stays deferred until Items 0–9 are resolved.
