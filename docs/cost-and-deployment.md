# GPU, Hosting and Cost Analysis

**Status:** proposal — for commercial planning
**Date:** 2026-08-27
**Companion:** [`model-selection.md`](./model-selection.md) · [`architecture.md`](./architecture.md)

Prices below were checked on 2026-08-27 and **change frequently** — verify against the
provider's calculator before committing. Throughput figures marked **[measured]** come
from real runs on the dev laptop; figures marked **[estimated]** are extrapolations and
must be validated in Phase 11 before any pricing is promised to a client.

---

## 1. The one measurement everything else derives from

Base Quran Whisper, CTranslate2 int8, 5.5 s audio segment, on the dev laptop
(i7-9750H, 2019, AVX2, **no** AVX-512/VNNI), averaged over 3 runs: **[measured]**

| CPU threads | Wall time | Core-seconds | RTF |
|---|---|---|---|
| 1 | 8.32 s | 8.32 | 1.51 |
| 2 | 5.20 s | 10.40 | 0.95 |
| 4 | 3.94 s | 15.77 | 0.72 |
| 6 | 3.65 s | 21.92 | 0.66 |

Two things to read out of this table:

1. **Thread scaling is poor.** 6× the cores buys 2.3× the speed. Adding cores to one
   request is wasteful; the efficient move is **more concurrent 2-thread workers**, not
   fatter ones. The cheapest setting in core-seconds is 1 thread; 2 threads is the
   sensible latency/cost balance.
2. **~10 core-seconds per 5.5 s segment** is the unit of cost. Everything below is that
   number multiplied by a price.

A modern server CPU (Sapphire Rapids / Genoa) has **AVX-512 with VNNI**, which
CTranslate2 uses for int8 matrix multiply. The dev laptop has neither. A conservative
**2.5×** speedup is assumed for server CPUs below — *this is the single biggest
assumption in this document and must be benchmarked in Phase 11.* **[estimated]**

---

## 2. "I have a VPS KVM 4 — will it work?"

Hostinger KVM 4 = **4 vCPU, 16 GB RAM, 200 GB NVMe, ~$13–15/month.**

**Yes — for development, demos and a small pilot. Not for scale.**

| Use | Verdict |
|---|---|
| Phases 1–5 (data, offline ASR, alignment, REST API) | ✅ Comfortable. 16 GB RAM is plenty; base model is 145 MB. |
| Phases 6–7 (WebSocket streaming, live recitation) | ✅ Works, for roughly **2–4 concurrent reciters** [estimated] |
| Client demo / pilot with a handful of testers | ✅ Ideal — and it costs you almost nothing |
| Tier B (large-v3-turbo) | ❌ Measured at 15.8 s for a 2.9 s clip even on 6 laptop cores |
| Tier C (Muaalem, 0.6 B, phoneme/Tajweed) | ❌ Needs ~3 GB VRAM; CPU inference would be minutes per clip |
| Production at 50+ concurrent users | ❌ |

Sizing: with 2 threads per worker you get 2 workers on 4 vCPU. At an assumed 2.5×
server-CPU speedup, each worker handles ~2 concurrent reciters → **~4 concurrent**,
minus headroom for FastAPI, VAD and the OS → **plan for 2–3.**

**Recommendation: start here.** It carries you through Phase 7 at ~$15/month, which is
the correct amount to spend before you have paying users. Do not rent a GPU yet.

---

## 3. "Can I develop on Colab's free GPU?"

**For development and benchmarking: yes, and you should.**
**For running the product: no.**

| | Colab Free | Colab Pro (~$10/mo) |
|---|---|---|
| GPU | T4 16 GB, when available | T4/L4, priority |
| Session limit | ~12 h, disconnects after ~90 min idle | longer |
| Weekly budget | ~15–30 GPU-hours, **varies with demand** | higher, still capped |
| Inbound connections | ❌ none — cannot host a WebSocket API | ❌ |
| Uptime guarantee | none | none |

Where Colab genuinely earns its place in this project:

* **Converting the Whisper checkpoints to CTranslate2.** Conversion needs `torch`, which
  cannot be installed on your Intel Mac at all (see model-selection §1.1). Colab is the
  fastest way to produce the CT2 artifact — run the converter, download the ~145 MB
  result, commit it to `models/`. One-off, free, no Docker needed.
* **Benchmarking Tier B and Tier C on a real GPU** to get the numbers this document
  currently has to estimate. That directly de-risks the pricing below.
* **Running the Muaalem phoneme model** (Phase 8) for evaluation, since it can't run on
  your laptop.

What it can't do: serve the API. There is no inbound networking, no uptime guarantee,
and tunnelling a production service through it violates the spirit of the terms. Use it
as a **GPU workbench**, not a server.

---

## 4. GPU pricing landscape (checked 2026-08-27)

| Provider | Instance / GPU | $/hour | $/month (24×7) | Notes |
|---|---|---|---|---|
| **AWS** | g4dn.xlarge (T4 16 GB) | ~$0.526 | ~$384 | The standard cheap GPU box |
| AWS | g6.xlarge (L4 24 GB) | ~$0.805 | ~$588 | Newer, faster than T4 |
| AWS | g5.xlarge (A10G 24 GB) | ~$1.86 | ~$1,358 | Overkill here |
| AWS | g4dn.xlarge **spot** | ~$0.16–0.21 | ~$115–155 | Up to ~70% off, **interruptible** |
| **Azure** | NC4as_T4_v3 (T4) | ~$0.526 | ~$384 | Matches AWS |
| Azure | NC4as_T4_v3 **spot** | ~$0.153 | ~$111 | Cheapest big-cloud GPU |
| **RunPod** | L4 (pod) | ~$0.39 | ~$285 | |
| RunPod | RTX 4090 (pod) | ~$0.69 | ~$504 | |
| RunPod | **serverless** RTX 4090 | ~$1.10 | **active seconds only** | Best fit for spiky load |
| **Vast.ai** | RTX 4090 | ~$0.29–0.59 | ~$210–430 | Marketplace — ⚠️ see §7 |

Reserved / Savings Plans: up to ~72% off on-demand for a 1-year commitment. **Do not
sign one until you have a stable load curve** — a year of unused GPU is the most common
way this kind of project loses money.

---

## 5. What it actually costs per user

### 5.1 Throughput assumptions

| Symbol | Value | Basis |
|---|---|---|
| Segment cost, server CPU, 2 threads | ~2.1 s per 5.5 s segment | measured 5.20 s ÷ 2.5× **[estimated]** |
| Concurrent reciters per 2-thread CPU worker | ~2.5 | **[estimated]** |
| Segment cost, T4 GPU, fp16 | ~0.2 s per 5.5 s segment | **[estimated]** — validate on Colab |
| Concurrent reciters per T4 | ~60–80 | **[estimated]** |
| VAD saving | ~30% of session audio is silence, never transcribed | architecture §7 |
| Recitation per active user | 20 min/day | assumption — change it for your market |
| Peak-to-average concurrency factor | 5× (strong Fajr / post-Maghrib peaks) | assumption |

Daily-active-users per concurrent slot ≈ `(24×60) / (20 × 5)` ≈ **14 DAU per concurrent reciter.**

### 5.2 The three realistic configurations

| | **A. VPS (your KVM 4)** | **B. Cloud CPU** | **C. Cloud GPU** |
|---|---|---|---|
| Hardware | 4 vCPU | c7i.2xlarge, 8 vCPU | g4dn.xlarge, T4 |
| Cost/month | **~$15** | ~$257 | ~$384 (~$115 spot) |
| Concurrent reciters | ~2–3 | ~10 | ~70 |
| Supported DAU | ~30–40 | ~140 | ~1,000 |
| **Infra cost / DAU / month** | **~$0.40** | **~$1.83** | **~$0.38** (~$0.12 spot) |
| **Infra cost / recitation-minute** | ~$0.0007 | ~$0.0031 | ~$0.00064 |

**The counter-intuitive result: the small VPS and the GPU box have almost the same cost
per user — and cloud CPU is the worst of the three.** Cloud CPU is expensive per unit of
compute and can't amortise; the VPS wins because it's absurdly cheap, and the GPU wins
because it's absurdly fast. The dead zone is in between.

**So the growth path is: VPS → (skip cloud CPU) → single GPU.**
The crossover is around **35–40 daily active users**, when the VPS saturates.

### 5.3 The cost trap that isn't compute

Mode B (serving verified recitation audio) can cost **more than the AI** if done naively.
A 2 MB ayah × 100k requests/month = 200 GB egress. On AWS S3 that's ~$18/month in egress
alone and it scales linearly with users, forever.

**Fix: serve Mode B audio from Cloudflare R2 (zero egress fees) or any CDN with free
egress.** This is a ~$20/month decision at pilot scale and a four-figure decision later.
Compute is elastic; bandwidth is a tax.

---

## 6. How to keep cost far below revenue

Ordered by impact:

1. **Stay on the VPS until users force you off it.** ~$15/month through Phase 7. Most
   projects at this stage burn $300+/month on an idle GPU.
2. **Keep Tier A on CPU; put only Tier B/C on GPU — serverless.** This is the biggest
   structural saving in the design. Tier B only runs when Tier A *suspects* a mistake —
   call it 5% of segments. Serverless GPU billed per active second (RunPod ~$1.10/hr
   active) means a suspected-mistake confirmation costs a fraction of a cent, and you
   pay **nothing** when nobody is reciting. Architecture §5.2 already isolates Tier B
   behind a config flag, so this is a deployment choice, not a rewrite.
3. **Scale to zero overnight.** Quran recitation has extreme diurnal peaks (Fajr, after
   Maghrib) and a seasonal one (Ramadan). A 24×7 instance sized for peak is idle most of
   the day. Serverless or autoscaling turns a 5× peak factor from a cost multiplier into
   a non-issue — potentially **60–70% off** a fixed-instance bill.
4. **VAD gating.** Already in the architecture, and it's a direct cost lever: ~30% of a
   session is silence you never pay to transcribe.
5. **int8 quantization.** Already chosen; roughly 2–4× cheaper than fp32 on CPU.
6. **2-thread workers, not 6.** From §1: 6 threads costs 2.1× the core-seconds of 2
   threads for 1.4× the speed. Right-sizing worker threads is free money.
7. **Cloudflare R2 for audio** (§5.3).
8. **Spot/interruptible for anything not user-facing** — batch evaluation, model
   conversion, benchmarking. ~70% off.
9. **Defer Reserved Instances** until the load curve is stable — then take the ~72%.

---

## 7. Risks to price in

* **Vast.ai and similar marketplaces run on other people's hardware.** You would be
  streaming users' voice recordings — reciting Quran — through machines you don't
  control. Given the privacy stance in the brief (§36, `STORE_AUDIO=false`), I would not
  use marketplace GPUs for live user audio. They're fine for benchmarking with public
  reciter audio.
* **Spot instances get reclaimed**, typically on ~2 minutes' notice. Never put live
  WebSocket sessions on spot without a fallback.
* **The 2.5× server-CPU speedup is an assumption** (§1). If it's only 1.5×, configuration
  B/C concurrency drops ~40% and cost per user rises accordingly. **Benchmark before quoting a client.**
* **GPU throughput figures are estimated, not measured.** Validate on Colab first — free,
  and it's the same T4 as g4dn.xlarge.
* **Whisper cold-start on serverless** is real: the model load was measured at 1.62 s
  (base) and 7.65 s (turbo). Serverless needs a warm pool or you pay that per request.

---

## 8. Suggested commercial model

Infra cost lands at roughly **$0.0006–0.003 per minute of recitation**, or
**$0.40–1.80 per daily-active user per month.**

| Model | Suggested price | Gross margin |
|---|---|---|
| B2B API, per minute | $0.01–0.02 / min | ~85–95% |
| B2B seat licence | $3–5 / active user / month | ~70–90% |
| White-label flat tier | $299–999 / month for a capacity band | high, if you cap concurrency |

Per-minute is the honest fit — your cost is genuinely per-minute-of-audio, so the pricing
tracks the cost and you can never be bankrupted by a heavy user.

**The margin is comfortable at every scale.** The risk to profitability in this project
is not the unit economics — it's paying for idle capacity before you have users. Hence
recommendation #1.

### Break-even sketch

At $4/user/month against ~$0.40 infra: **one paying user covers the VPS.** A single
g4dn.xlarge ($384/mo) breaks even at ~**96 paying users** and serves ~1,000 — so the
step up to GPU is safe once you're past ~150 users, and profitable well before it saturates.

---

## 9. Recommended sequence

1. **Now → Phase 7:** your KVM 4 VPS. ~$15/month. Use Colab (free) for CT2 conversion and
   any GPU benchmarking.
2. **Phase 8–9 (phoneme/Tajweed dev):** Colab Pro (~$10/mo) or RunPod on-demand by the
   hour. Do not leave it running.
3. **Pilot / first client:** stay on the VPS, add RunPod **serverless** for Tier B/C.
   Expect ~$15–50/month total.
4. **Past ~40 concurrent users:** one g4dn.xlarge or RunPod L4. Skip cloud CPU entirely.
5. **Stable load:** 1-year Reserved/Savings Plan for the ~72% discount.

**Before quoting any client a price, run `scripts/benchmark_asr.py` on the actual target
hardware.** Section 32 of the brief says not to claim latency before measuring it; the
same discipline applies to cost.

---

## Sources

- https://aws.amazon.com/ec2/pricing/on-demand/ · https://instances.vantage.sh/aws/ec2/g4dn.xlarge
- https://www.runpod.io/gpu-models/rtx-4090 · https://computeprices.com/providers/runpod
- https://cloudprice.net/vm/Standard_NC4as_T4_v3 · https://www.thundercompute.com/blog/azure-gpu-instances
- https://www.synpixcloud.com/blog/vast-ai-vs-runpod-rtx-4090-pricing
- https://smarthostfinder.com/hostinger-vps-pricing/
- https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/

---

## 10. Decision: agreed deployment plan (2026-08-27)

**Chosen:** develop on the Mac + Colab, deploy to the already-purchased Hostinger KVM 4
(yearly), scale by moving to bigger CPU hosts rather than GPU. Client is a school
(educational use).

**I agree with this plan.** For a single-school deployment the CPU path is genuinely
cheaper than GPU, not just cheaper-feeling — GPU only wins past roughly 50 concurrent
reciters, which one school is unlikely to reach. Three things need to be right, though.

### 10.1 Correction to an earlier statement

I said PyTorch "cannot be installed" — that is true **only of your Intel Mac**.
Your **VPS runs Linux x86_64, where PyTorch installs normally.** So the VPS is in one
respect *more* capable than the dev laptop: it can run the Muaalem phoneme model (Tier C)
that the Mac cannot. Whether it can run it *fast enough* is unmeasured — 0.6 B params on
4 vCPU is likely tens of seconds per clip, so treat it as an offline/async feature, not
live. Benchmark it before promising it.

### 10.2 The classroom-burst risk — the main thing to plan for

Consumer apps have a peak factor of ~5×. **A school does not.** Thirty students in one
computer lab at 10:00 means **30 simultaneous streams**, not thirty spread across a day.
That is the load pattern that breaks this deployment, and my earlier "30–40 daily active
users" figure assumed consumer spread — it does not apply to a classroom.

| Scenario | Concurrent | KVM 4 (4 vCPU) |
|---|---|---|
| Individual homework / practice | 1–3 | ✅ fine |
| Small group / teacher demo | 5–8 | ⚠️ marginal |
| One full class in a lab | 25–40 | ❌ needs ~6–10× the CPU |

Ask the client directly: **do students use it individually at home, or as a whole class
at once?** The answer changes the hardware bill by 10×. If it's whole-class, size for
the class, not the roll.

### 10.3 Better scaling steps than "a bigger Hostinger plan"

| Step | Spec | ~Cost/mo | Est. concurrent |
|---|---|---|---|
| Now — Hostinger KVM 4 | 4 vCPU / 16 GB | ~$13 (paid) | 2–3 |
| Hostinger KVM 8 | 8 vCPU / 32 GB | ~$26 | 5–6 |
| **Hetzner AX42 (dedicated)** | **8 real Ryzen 7 PRO 8700GE cores / 64 GB** | **~€46** | **~12–15** |
| Hetzner AX52 | Ryzen 7 7700, 8 cores | ~€64 | ~15–18 |
| 2–3 × AX42 behind a load balancer | | ~€92–138 | ~25–45 |

Two reasons the Hetzner dedicated boxes beat stacking VPS plans here:

1. **Real cores, not vCPU.** A "vCPU" is usually one hyperthread; §1 already showed this
   workload scales badly across threads, so hyperthreads help less than the count suggests.
2. **Zen 4 has AVX-512 with VNNI**, which is precisely the instruction set CTranslate2
   uses for int8 inference. Older EPYC (Zen 2/3) does not have it. This could be worth
   **2–3× per core** — and it means *which* CPU you rent matters more than how many vCPU
   you get. Verify `lscpu | grep avx512_vnni` on any host before renting it.

Avoid **Hetzner Cloud CCX** for this: CCX33 (8 vCPU) is ~€138/mo after the April 2026
price rise, three times the AX42 dedicated box for less CPU.

**Practical rule to give the client:** each additional ~10 simultaneous reciters costs
roughly **€45–50/month** in CPU hosting. That is a clean, honest line item to quote.

### 10.4 Consequence: Tier B confirmation must change

The architecture used Tier B (large-v3-turbo on GPU) to confirm a suspected mistake
before reporting it — the main defence against false corrections. **With no GPU, Tier B
as specified is unavailable** (measured: 15.8 s for a 2.9 s clip). It must be replaced,
not dropped — the hallucination problem in model-selection §3.2 is real and unsolved
without it.

CPU-affordable replacements, in preference order:

1. **Cross-window agreement (free).** Only confirm a word if two overlapping inference
   windows produce it identically. The overlapping windows already exist in the streaming
   design, so this costs nothing extra and directly attacks the hallucination pattern —
   trailing hallucinations differ between runs (measured: `قُطْ قُطْ` vs `مُعْلَمُونَ`
   vs `مَا يَغْفَى` on the same clip), whereas genuine words are stable.
2. **Second decode at a different temperature, on suspect segments only (~+5% compute).**
   Same logic, applied on demand.
3. **A `whisper-small` Quran fine-tune as CPU Tier B** (~3.3× base cost, run on ~5% of
   segments → ~+16% total). Viable, but no well-adopted small Quran model was found in
   research — would need validation.

**Recommendation: (1) + (2).** Both are CPU-only, cost almost nothing, and exploit a
property I actually measured — hallucinations are unstable across decodes, real words
are not. This becomes the Phase 4 false-correction defence.

### 10.5 Ask the client one product question first

Does the school need **live word-by-word feedback while the student recites**, or is
**per-ayah feedback after they finish** enough?

If per-ayah is enough, you can ship Phases 1–5 only — record, submit, analyse, respond.
No WebSockets, no streaming buffer, no VAD, no real-time latency budget. On the KVM 4
that supports **far more students** (requests queue instead of needing simultaneous
capacity), and it removes the hardest and riskiest third of the build. For a school
memorisation workflow that is often exactly what's wanted.

Streaming can be added later without rework — the architecture already separates the
analysis pipeline from the transport.

### 10.6 Flag: children's voice data

The users are schoolchildren, and voice recordings of minors are personal data under
GDPR (and equivalents), with children's data given special protection. Before launch,
confirm with the client: who is the data controller, what consent the school holds from
parents, and where the data is stored. `STORE_AUDIO=false` (already the default in the
architecture) is the strongest position — if audio is never persisted, most of this
problem disappears. Choose an EU region if the school is in the EU.

This is a contractual matter for the client, not a coding task, but it should be settled
before the pilot rather than after.

---

## 11. Decision: streaming demo, capped at 3 concurrent sessions (2026-08-27)

**Chosen:** build streaming (Mode A) as originally specced, hard-capped at 3 concurrent
sessions on the KVM 4, to demo to the school. Scale decision deferred until after the demo.

**Agreed — with one correction to how the scaling conversation should be framed.**

### 11.1 Measured: 3 concurrent sessions on one shared model

Dev laptop (6 cores, AVX2), base int8, one shared `WhisperModel`, 5.5 s segments,
3 worker threads submitting concurrently: **[measured]**

| Sessions | cpu_threads | Median latency | Aggregate RTF | Keeps up? |
|---|---|---|---|---|
| 1 | 1 | 8.21 s | 1.49 | ❌ |
| 3 | 1 | 10.26 s | 0.63 | ✅ |
| 3 | 2 | 7.99 s | 0.49 | ✅ |

**Throughput was never the problem.** At 3 sessions the server keeps up comfortably
(RTF 0.49 — roughly 2× headroom). **Latency is the problem:** ~8 s between a student
finishing a phrase and the correction appearing, on this laptop.

### 11.2 The thing that must be understood before the demo

**More CPU buys concurrency, not speed.** Section 1 measured it: 6× the threads gives
2.3× the speed. So:

| Client's reaction to the demo | Correct upsell |
|---|---|
| "We need more students at once" | ✅ Bigger VPS (KVM 8 / 16, or Hetzner AX42) |
| "It feels laggy" | ❌ Bigger VPS will **not** fix this — needs a GPU or a smaller model |

Selling a KVM 16 to fix perceived lag would take the client's money and not fix their
complaint. Keep the two axes separate in that conversation.

### 11.3 The fix for latency: tiny as the provisional tier

`whisper-tiny-ar-quran` (CT2 int8, 39 M params) measured on the same laptop: **[measured]**

| Model | 1 session | 3 sessions | Aggregate RTF @3 |
|---|---|---|---|
| **tiny** | **1.75 s** | **3.03 s** | **0.19** |
| base | 5.34 s | 7.99 s | 0.49 |

**Tiny is ~3× faster and, on short segments, actually *cleaner* than base.** On the two
short clips base hallucinated trailing words (`قُطْ قُطْ`, `الْحَمْدُ`) while tiny
produced exactly the right text with no tail. Streaming segments are short by
construction (waqf-delimited, 3–8 s), which is tiny's best case.

But tiny degrades badly on long audio, and — critically — on Ayat al-Kursi it produced
**`خَلْفَهُمُونَ` at probability 0.90**, a wrong word above the proposed confidence
threshold. That is exactly the false-correction landmine the whole design exists to avoid.

**Therefore:**

* **tiny → PROVISIONAL tier.** Drives live cursor, word highlighting, "we hear you".
  Feedback in ~1.8–3 s, which feels responsive. **It is never allowed to report a mistake.**
* **base → COMMITTED tier.** The only tier permitted to emit `mistake_detected`.
* **base runs on demand, not on every segment** — only when tiny's output disagrees with
  the expected ayah text (~20–30 % of segments). Combined load ≈ 0.19 + ~0.15 ≈ **0.34
  aggregate RTF**, well inside budget.

This maps exactly onto the provisional/committed split already in architecture §7.1 — no
redesign, just a second (cheaper) engine bound to the provisional role.

### 11.4 Deployment config for the demo

```
MAX_CONCURRENT_SESSIONS=3      # hard cap; 4th connection rejected with a clear error
ASR_PROVISIONAL_MODEL=hafizku/faster-whisper-tiny-ar-quran
ASR_COMMITTED_MODEL=OdyAsh/faster-whisper-base-ar-quran
COMPUTE_TYPE=int8
CPU_THREADS=2                  # per worker — NOT 4; thread scaling is poor
NUM_WORKERS=3                  # one shared model instance, 3 parallel slots
CONFIRM_STRATEGY=agreement
```

One shared `WhisperModel` per tier with `num_workers=3` — **never one model per session.**
Memory is fine either way (~145 MB + ~75 MB), but per-session models would waste CPU.

### 11.5 The one unknown: which CPU the VPS actually has

Everything above scales with per-core speed, and Hostinger's EPYC generation is not
visible from here. **Zen 4+ has AVX-512 VNNI, which CTranslate2 uses for int8 — worth
roughly 2–3× on this exact workload. Zen 2/3 does not have it.**

Run on the VPS before building anything:

```bash
lscpu | grep -E 'Model name|avx512_vnni'
```

| Result | Expected outcome at 3 sessions |
|---|---|
| `avx512_vnni` present | Provisional ~1.2–1.5 s, committed ~3–4 s. **Demo will feel good.** |
| absent | Provisional ~3–4 s, committed ~8–10 s. **Cap at 2 sessions**, or demo on one. |

If VNNI is absent, a Hetzner AX42 (Ryzen 7 PRO 8700GE, Zen 4, ~€46/mo) becomes the
sensible demo host rather than a bigger Hostinger plan — CPU *generation* matters more
here than vCPU *count*.

### 11.6 Making 3 s feel like 0 s

Since Scenario A tells us the expected ayah in advance, the UI can advance a **reading
cursor from VAD alone** — no ASR, effectively free. The student sees the app tracking
them word-by-word in real time; corrections settle in behind. This hides most of the
perceived lag and costs nothing. Worth building into the demo client.

### 11.7 VPS CPU confirmed: AMD EPYC 9354P (Zen 4 / Genoa) — green light

`lscpu` on the production VPS returned:

```
Model name: AMD EPYC 9354P 32-Core Processor
Flags: ... avx2 avx512f avx512dq avx512cd avx512bw avx512vl avx512_bf16
       avx512vbmi avx512_vbmi2 avx512_vnni avx512_bitalg vaes gfni sha_ni ...
```

**`avx512_vnni` is present.** This is the good branch of the table in §11.5 — CTranslate2's
int8 VNNI kernels are available. The EPYC 9354P is also a high-clock Zen 4 server part
with a large L3, versus the dev laptop's 2019 Coffee Lake (AVX2, no VNNI).

Estimated **2.5–3× per-core** over the dev laptop **[estimated]**. Projected at 3
concurrent sessions:

| Tier | Dev laptop (measured) | VPS (projected) |
|---|---|---|
| tiny — provisional / live view | 3.03 s | **~1.4–1.7 s** |
| base — committed / corrections | 7.99 s | **~3.7–4.4 s** |

That lands in the "demo will feel good" band. **Proceed with 3 sessions**; 4–5 is
probably achievable and should be tested once Phase 7 exists.

Worth testing on the box itself: `cpu_threads=1, num_workers=3` may beat
`cpu_threads=2` there, since 3×2 threads oversubscribes 4 vCPU while 3×1 does not.
Benchmark both in Phase 11 — it's a two-line config change.

### 11.8 Correction to §10.3: stay with Hostinger, skip Hetzner

§10.3 recommended a Hetzner AX42 over a larger Hostinger plan, on the assumption that
Hostinger was running older EPYC without AVX-512. **That assumption was wrong** — the
9354P is Zen 4 with full VNNI, matching or beating the AX42's Ryzen 7 PRO 8700GE on
cache and memory bandwidth.

**Revised scaling ladder** (same silicon throughout, so concurrency scales roughly
linearly with vCPU — a much cleaner story for the client):

| Plan | vCPU | ~Cost/mo | Est. concurrent sessions |
|---|---|---|---|
| KVM 4 (owned, yearly) | 4 | ~$13 | **3–5** |
| KVM 8 | 8 | ~$26 | ~8–10 |
| KVM 16 | 16 | ~$50 | ~16–20 |

No provider migration needed at any step. Hetzner remains a fallback only if Hostinger
ever moves the account to older silicon — re-run `lscpu` after any plan change.
