# Prompt for Antigravity — Full Bug/Mismatch List, Fixes, and Execution Methodology

You are working on the DDPG+LSTM continuous network slicing project (`environment.py`, `config.py`,
`03_train_ddpg.ipynb`, `04_evaluate.ipynb`, `PROJECT_WORKFLOW.md`, `dataset.csv`). Below is the complete,
verified list of open issues — confirmed directly against the current code, not from memory — in the exact
order to fix them. Follow the **execution rules** below strictly; do not skip ahead or batch unrelated fixes.

---

## Execution rules

1. **Work one item at a time**, in the numbered order given. Do not start item *N+1* until item *N*'s
   verification step has passed.
2. **Items marked `DECISION REQUIRED` are not yours to resolve unilaterally.** Stop and report both options
   back with your recommendation; wait for an explicit choice before writing any code for that item.
3. **After any change to `config.py`, `environment.py`, the Actor/Critic classes, or the reward computation**,
   re-run the full evaluation (Equal Sharing / Priority-DataSize Heuristic / Static Priority / Round Robin /
   Random / DDPG / DDPG-Ablated) before reporting results. Do not reuse numbers computed before the fix.
4. **Always report the complete per-slice results table** (all 5 slices, all policies). Never report a
   filtered "SUCCESS/FAIL" summary in place of the actual numbers.
5. **Do not introduce any new tunable constant that isn't already defined in `config.py`.** If a fix needs a
   new constant, add it to `config.py` with a comment explaining where it came from (spec, calibration, etc.)
   — don't hardcode magic numbers inline.
6. **Before any full retrain**, run the specific standalone verification check listed under that item.
   Only proceed to a full training run after that check passes.

---

## 0. [CRITICAL — fix first] Evaluated actor is architecturally different from the trained actor

**Problem:** `03_train_ddpg.ipynb`'s `Actor.forward()` is:
```python
def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x)) * 15.0
```
and the executed action during training is a proportional rescale:
```python
pred_actions = pred_raw_actions * (batch_total_bw / sum_pred)
```

`04_evaluate.ipynb`'s `Actor.forward()` is instead:
```python
def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)   # no sigmoid, no scaling
```
and the executed action is computed with a batch-wide softmax instead:
```python
probs = torch.softmax(raw_logits, dim=0).cpu().numpy().flatten()
actions = probs * TOTAL_BW_test
```

Both classes have identical layer shapes (`10 → 64 → 32 → 1`), so `actor.load_state_dict(torch.load('ddpg_actor.pth'))`
loads without any error. But the weights were optimized under a saturating sigmoid output feeding a
proportional-rescale action rule; evaluation instead runs the same weights through an unbounded linear output
feeding a softmax — a function the network was never trained to be evaluated through. **Every DDPG number
currently produced by `04_evaluate.ipynb` reflects a policy that does not correspond to `ddpg_actor.pth`.**

**Fix:** Make `04_evaluate.ipynb`'s `Actor` class and action-selection logic byte-for-byte identical to
`03_train_ddpg.ipynb`. Concretely:
```python
class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 15.0

# action selection, matching training exactly:
raw_actions = actor(state_t)                              # (N, 1), bounded [0, 15]
actions = (raw_actions * (TOTAL_BW_test / raw_actions.sum())).cpu().numpy().flatten()
```
Apply the identical fix to the "DDPG (Ablated)" branch, which shares the same broken logic.

**Verification (must pass before trusting anything downstream):**
1. Run one forward pass on a batch of test states; confirm `actions.sum()` equals `TOTAL_BW_test` (within
   floating-point tolerance) and that individual actions fall in a sane range (compare against the
   distribution of actions logged during training, if available).
2. Only after this passes, re-run the full evaluation table and discard every DDPG number produced before
   this fix.

---

## 1. Reward function the agent actually optimizes is missing from the paper

**Problem:** `environment.py` applies a large SLA-violation penalty on top of the formal utility:
```python
rewards[violating] = rewards[violating] - penalties[violating] * (1.0 + violation_ratio[violating])
```
This penalty is the dominant driver of agent behavior (it's why Static Priority "starves" lower slices), but
the paper's Eq. 21/23 only shows `U = w1·Th + w2·R − w3·L − w4·Ptotal + w5·REU` — no penalty term.

**Fix:** Add a formal equation to Section 3.4 of the paper:
```
R(t) = U(t) − Σᵢ 1[Lᵢ(t) > Lᵢ^req] · κᵢ · (1 + (Lᵢ(t) − Lᵢ^req)/Lᵢ^req)
```
where `κᵢ` is the per-slice penalty constant (see Item 2). This must be documented before anything else,
since it's the actual objective everything downstream is evaluated against.

**Verification:** Numerically reproduce `R(t)` from the new equation on 10 sample patient-steps and diff
against `environment.py`'s actual `rewards` array; confirm match to floating-point precision.

---

## 2. Penalty constants disagree across three sources

**Problem:**
- `config.py` (what runs): `{1: 350.0, 2: 300.0, 3: 150.0, 4: 100.0, 5: 50.0}`
- `PROJECT_WORKFLOW.md` (spec, line 228): `{1: 350, 2: 300, 3: 20, 4: 10, 5: 5}`
- Paper: unspecified

**Fix:** Change `config.py`'s `SLA_PENALTIES` to `{1: 350.0, 2: 300.0, 3: 20.0, 4: 10.0, 5: 5.0}` to match the
project spec, unless there's a documented reason to keep the heavier weighting — in which case update
`PROJECT_WORKFLOW.md` and the paper instead. All three sources must agree.

**Verification:** Confirm `04_evaluate.ipynb` sources penalties dynamically from `config.py` (it already
does — don't hardcode a second copy). Re-run the full evaluation table; the "Equal Sharing beats Static
Priority" result was measured under 150/100/50 and may not hold under 20/10/5.

---

## 3. DECISION REQUIRED — Priority/criticality formulation

**Problem:** Paper Eqs. 9–14 define a continuous, per-sensor health severity index (deviation from
`[θl,s, θu,s]`, weighted by medical-criticality constant `xs`, averaged into `ρᶜ_{p,t} ∈ [0,1]`). None of this
exists in the code. `dataset.csv`'s `Emergency` column is binary
(`HeartRate>100 OR BP>140 OR SpO2<92`), and `SliceID==2` (ICU) is hardcoded `Emergency=1` for 100% of rows.
The paper draft still contains an unedited note ("Change the variable accordingly...") directly before Eq. 9.

**Report these two options back and wait for a choice:**
- **(a) Simplify the paper** to describe the binary `Emergency` flag actually used. Fast, honest, thinner
  contribution.
- **(b) Build the real thing:** implement per-sensor reference ranges and criticality weights in `config.py`,
  compute the severity index during dataset generation, replace `Emergency` everywhere downstream (state
  vector, `Priority`, dataset generation). Non-trivial — new constants, new dataset columns, changes agent state.

**Do not proceed to Item 4 until this is resolved**, since `Priority` depends on it.

---

## 4. Priority score equation (Eq. 15) missing normalization terms present in the code

**Problem:** Code/spec (matching, `PROJECT_WORKFLOW.md` line 105):
```
Priority_i(t) = clip( w1·Emergency_i + w2·(LatencyReq_min/LatencyReq_i) + w3·min(Traffic_i(t)/Traffic_max, 1), 0, 1 )
```
with `w1=0.5, w2=0.3, w3=0.2`. Paper Eq. 15, `Pi(t) = w1·Ci + w2·(1/D^req_i) + w3·Ti(t)`, is unbounded as
written — the code is correct, the paper needs to catch up.

**Fix:** Update Eq. 15 to match the code exactly, and swap `Ci` for whichever Item 3 outcome is chosen.

**Verification:** Reproduce `Priority` from the updated equation on a patient sample; diff against
`config.py`'s actual computation.

---

## 5. Bandwidth allocation formula (Eq. 3) doesn't match the implemented baseline

**Problem:** `PROJECT_WORKFLOW.md` (lines 116–118) specifies:
```
weight_i(t) = Priority_i(t) × DataSize_i(t)
Bandwidth_i(t) = ( weight_i(t) / Σⱼ weight_j(t) ) × TOTAL_BW
```
This is what's baked into `dataset.csv`'s `Bandwidth` column (the "Priority-DataSize Heuristic" baseline).
Paper Eq. 3 only has `Bi(t) = Pi(t)/ΣPj(t) × Btotal` — missing the `DataSize` term.

**Fix:** Update Eq. 3 to include `DataSize` weighting, with the documented rationale: it keeps the
transmission-delay ratio `DataSize/Bandwidth` stable regardless of payload size, so `Priority` alone
determines who is served faster.

**Verification:** Confirm Σᵢ Bi(t) ≤ Btotal holds under the updated formula on the full dataset.

---

## 6. Formal constraints in the paper are never enforced or checked in code

**Problem:** Eqs. 26–30 state `Ri(t) ≥ Rmin`, `Thtotal(t) ≥ Thmin`, `Ptotal(t) ≤ Pmax`, `REU ≥ REUmin`. None
of `RMIN`, `THMIN`, `PMAX`, `REUMIN` exist in `config.py`; `environment.py` doesn't clip power to any maximum;
`04_evaluate.ipynb` reports averages but never checks against thresholds.

**Fix — report both options, pick one:**
- **(a)** Add the constants to `config.py`, hard-clip `powers` at `Pmax` in `environment.py`'s `step()`, add
  violation-rate tracking for the other three to `04_evaluate.ipynb`.
- **(b)** Reframe Eqs. 26–30 in the paper as observed target ranges, not enforced constraints.

**Verification:** If (a) — confirm 100% compliance with `Pmax` post-fix; report violation rates for the rest.

---

## 7. DECISION REQUIRED — Queue-dynamics mechanism not in spec or paper

**Problem:** `environment.py` has an invented live-update system:
```python
self.queues = np.minimum(np.maximum(self.queues + t_data['Traffic']*0.06 - throughputs*0.78, 0.0), 50.0)
self.lambdas = np.minimum((self.queues/(1.0+self.queues))*mu, mu - 0.1)
```
Neither `PROJECT_WORKFLOW.md` nor the paper defines this — both describe `Queue`/`Lambda` as static,
pre-generated columns (`Poisson(Lambda/Mu)`). The dataset's static `Queue` column tops out at 15, but the
live dynamic queue is capped at 50 — over 3× the range the `/15.0` state normalizer was built around.

**Report these two options back and wait for a choice:**
- **(a) Revert to spec:** read `Queue`/`Lambda` directly from `dataset.csv` per time step, action-independent.
  Simpler, matches spec exactly, but the agent's bandwidth choices stop affecting future congestion.
- **(b) Keep the dynamic version**, but document it in the paper as an explicit extension (update Eq. 1 to
  show `λ`/`µ` as state-dependent) and justify the `0.06`/`0.78` constants with a calibration procedure.

**Do not spend further time calibrating `0.06`/`0.78` until this is settled.**

---

## [CHECKPOINT] Re-run full evaluation

After Items 0–7 land, re-run the complete evaluation table and record fresh numbers. Every conclusion drawn
before this checkpoint (including any prior "DDPG beats X" claim) is provisional until reconfirmed here.

---

## 8. Batch-coupling in the actor isn't fully eliminated — it moved into the gradient

**Problem:** The actor's forward pass is per-patient independent, but the training loop still does:
```python
pred_raw_actions = actor(S)
sum_pred = torch.sum(pred_raw_actions, dim=0, keepdim=True)
pred_actions = pred_raw_actions * (batch_total_bw / sum_pred)
loss_actor_accum = loss_actor_accum + -critic(S, pred_actions).mean()
```
`sum_pred` is not detached from the autograd graph, so patient *i*'s gradient carries an ~O(1/N) dependency
on every other patient's output (N=1000 in training).

**Fix:** Quantify `d(loss_i)/d(other patient's raw output)` to measure the cross-term's magnitude. If
non-negligible, detach `sum_pred` (`sum_pred.detach()`) so it's used only to compute the environment-executed
action, not inside the actor's loss.

**Verification:** Re-run the action-diversity check (bandwidth std across patients, single-patient batch vs.
padded-into-999) specifically through the loss/gradient path, not just the forward pass.

---

## 9. Local Critic proposal is unverified and contradicts an earlier result

**Problem:** Q-calibration correlation went from +0.55 to -0.17 across two rounds using the *same*
global mean/max-pooling critic design — inconsistent with "global pooling is fundamentally broken" as a
blanket explanation. `test_local_critic.py`, the basis for a proposed Local Critic redesign, is not present
in the project files.

**Fix:** Obtain and review `test_local_critic.py` if it exists; confirm it used the same reward scaling and
constraints as the real run. Re-run Q-calibration with the current centralized critic **after** Items 0–2 are
fixed (both change what the critic is learning to predict) before concluding the pooling architecture itself
is the problem.

**Verification:** Report Q-calibration correlation with the centralized critic post-fix. Only pursue a Local
Critic redesign if correlation is still poor after this re-check.

---

## 10. State normalization does not bound to [0,1]

**Problem (verified against `dataset.csv`):**

| Feature | Code | Observed range | Bound violated |
|---|---|---|---|
| `norm_queue` | `queue / 15.0` | live queue capped at 50 → up to 3.33 | yes |
| `norm_lambda` | `lambda / 10.0` | dataset max 10.66 → up to 1.07+ | yes |
| `norm_traffic` | `Predicted_Traffic / 100.0` | 6.6% of rows exceed 100; also ignores `config.TRAFFIC_MAX` (150) | yes |
| `norm_prev_bw` | `prev_bandwidth / (TOTAL_BW/10000.0)` = `/3.0` | baseline `Bandwidth` up to 63.5 → up to 21.17 | yes, worst offender |

**Fix:** Recompute each divisor from the actual bound of its quantity: 50 for queue (matches the live cap),
an empirical/spec-defined max for lambda, `config.TRAFFIC_MAX` (not a hardcoded 100) for traffic, and the
real maximum possible bandwidth allocation for `prev_bw` — not `TOTAL_BW/10000`, which has no documented
derivation.

**Verification:** Compute the true max of each normalized feature across the full dataset (and a live
rollout, for queue/lambda) post-fix; confirm all fall within [0,1].

---

## 11. Exploration noise added before batch-sum normalization (low priority)

**Problem:**
```python
noise = np.random.normal(0, noise_std * 15.0, size=raw_actions.shape)
noisy_actions = np.clip(raw_actions + noise, 0.05, 15.0)
actions = noisy_actions * (TOTAL_BW_train / sum_noisy)
```
Noise is added before the batch-sum rescale, so effective exploration magnitude varies with `sum_noisy`
episode to episode.

**Fix:** Low priority — revisit only if exploration looks inconsistent after Items 0–10 land.

---

## Discipline reminders

- **Q-calibration gate:** correlation between predicted Q and empirical returns must clear 0.5+ before
  trusting any policy comparison built on that critic (baseline observed: -0.0128).
- **Isolate before combining fixes.** Where unavoidable, add a diagnostic (e.g. gradient-norm logging) that
  gives post-hoc attribution evidence.
- **Report the full per-slice results table every time.**
- **PER (Prioritized Experience Replay)** stays deferred until Items 0–9 are resolved.
