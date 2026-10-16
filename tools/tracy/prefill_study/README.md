# Llama 3.1 8B prefill profiling on Blackhole: study guide

How the prefill profiling study of October 2026 was done, what it found, and how to redo it. The study covers one
decoder layer of Llama-3.1-8B-Instruct on one P150 (Blackhole, 1350 MHz, 110 worker cores), prompts 128–32,768 tokens.

- Published report: https://claude.ai/code/artifact/23303fd9-dda8-4b5a-8a87-3412e8a9c175 (source: `report/blackhole-prefill.html`)
- Original sheet: "Llama 3.1 8B profiling", tab "Blackhole prefill",
  https://docs.google.com/spreadsheets/d/1TxVEBprRxPeSXiWnYWH0FBAYObo5u_Kx2aRJVM8dZis (superseded by the report's corrected table)
- Raw data (git-ignored, local only): `generated/profiler/`

## Contents of this folder

| File | What it does |
|---|---|
| `env.sh` | Environment for device runs (source from the repo root) |
| `capture.sh` | Profile one length: `counters` mode (perf counters) or `zones` mode (SDPA stage zones) |
| `test_model_prefill_min_padding.patch` | Test change: 512/1k/2k test ids and the `TT_PREFILL_MIN_PADDING` switch |
| `sdpa_stage_zones.patch` | Temporary profiling zones around the 8 SDPA stages. Apply only for zone runs, then revert |
| `layer_table.py` | Main table: layer time, MatMul %, SDPA %, FPU/SFPU utilization per length |
| `layer_units.py` | Whole-layer FPU and SFPU utilization (all ops, incl. SDPA's internal matmuls and the SiLU × mul) |
| `plot_sweep.py` | Static PNG of the sweep from `layer_table.py` output |
| `layer_dag.py` | Longest dependency chain of the layer's 23 ops |
| `sdpa_stages.py` | Per-stage SDPA breakdown from zone runs |
| `sdpa_schedule.py` | Measured unpack / math / pack schedule of SDPA chunk steps |
| `sdpa_unit_split.py` | FPU / SFPU / idle cycles per SDPA stage |
| `quasar_model.py` | Hypothetical SDPA schedule with separate FPU and SFPU threads |
| `fast_exp_model.py` | What-if: exp at one cycle per SFPU instruction (128 cycles/tile); layer time and FPU/SFPU utilization |
| `sfpu_offload_model.py` | What-if: SDPA keeps only QKᵀ/PV on the FPU, all else on the SFPU; 1 vs 2 threads, measured vs fast exp |
| `fpu_clock_slack.py` | What-if: how far the FPU clock could drop in the Quasar two-thread model before SDPA slows down |
| `report/blackhole-prefill.html` | Source of the published report |
| `results/prefill_sweep_1dev.png` | Static sweep figure (`plot_sweep.py`) |
| `results/early_whole_run/` | First-day per-run timelines and summaries (Oct 12–13): whole-log scope (`select=all`, includes warmup), and the 512 / 3k ones are padded runs (512 ran at 1024, 3072 at 4096). Kept for reference only; superseded by the layer-scope numbers |

`tools/tracy/perf_util_report.py` (one level up) is the per-run utilization report the scripts build on.
`perf_util_report.py`, this folder and the test change are all uncommitted.

## Redo the study

From the repo root. Device runs go one at a time; budget about 3 min per length and 7 min for 32k (mostly the CPU
reference model). Run long sweeps detached (`setsid nohup ... &`) so a terminal or tool timeout does not kill them.

```bash
git apply tools/tracy/prefill_study/test_model_prefill_min_padding.patch   # skip if already applied
P=tools/tracy/prefill_study

# 1. Counter runs (main table, utilization)
for L in 128 512 1k 2k 3k 4k 8k 16k 32k; do $P/capture.sh counters $L; done

# 2. Main table, PNG, layer dependency chain
source $P/env.sh
python $P/layer_table.py out/layer.json 128:generated/profiler/prefill_128_1dev 512:generated/profiler/prefill_512_1dev \
  1024:generated/profiler/prefill_1k_1dev 2048:generated/profiler/prefill_2k_1dev 3072:generated/profiler/prefill_3k_1dev \
  4096:generated/profiler/prefill_4k_1dev 8192:generated/profiler/prefill_8k_1dev 16384:generated/profiler/prefill_16k_1dev \
  32768:generated/profiler/prefill_32k_1dev
python $P/layer_units.py out/layer.json out/layer_units.json
python $P/plot_sweep.py out/layer.json prefill_sweep_1dev.png
python $P/layer_dag.py generated/profiler/prefill_4k_1dev 1113      # 1113 = row of the layer's Embeddings op

# 3. SDPA zone runs (temporarily instruments the kernel; kernels are JIT-compiled, no rebuild needed)
git apply $P/sdpa_stage_zones.patch
for L in 128 512 1k 2k 3k 4k 8k 16k 32k; do $P/capture.sh zones $L; done
git checkout -- ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp   # revert!

# 4. SDPA analysis
Z=generated/profiler/sdpa_zones
python $P/sdpa_stages.py out/stages.json 128:$Z/prefill_128 512:$Z/prefill_512 1024:$Z/prefill_1k 2048:$Z/prefill_2k \
  3072:$Z/prefill_3k 4096:$Z/prefill_4k 8192:$Z/prefill_8k 16384:$Z/prefill_16k 32768:$Z/prefill_32k
python $P/sdpa_schedule.py out/schedule.json $Z/prefill_4k
python $P/sdpa_unit_split.py out/split.json out/stages.json \
  --counters 32768:generated/profiler/prefill_32k_1dev 1024:generated/profiler/prefill_1k_1dev
python $P/quasar_model.py out/quasar.json out/split.json out/layer.json
python $P/fast_exp_model.py out/fast_exp.json out/split.json out/layer.json out/layer_units.json   # needs layer_units.py output
python $P/sfpu_offload_model.py out/sfpu_offload.json out/split.json out/layer.json out/layer_units.json
python $P/fpu_clock_slack.py out/split.json out/fpu_clock_slack.json
```

Existing data from the October 13–14 runs: the counter runs are `generated/profiler/prefill_{128,1k,2k,4k,8k,16k,32k}_1dev`
plus `prefill_512_1dev_nopad` and `prefill_3k_1dev_nopad` for 512 and 3072 (the plain `prefill_512_1dev` and
`prefill_3k_1dev` folders hold padded runs: 512 ran at 1024, 3072 at 4096). Zone runs are in `generated/profiler/sdpa_zones/`.

To update the report, edit `report/blackhole-prefill.html` (data is in the `DATA`, `SDPA`, `UNITS` and Quasar arrays in
its script) and republish it to the same URL.

## Definitions (keep these fixed between runs)

- **Op scope "layer"**: the Embeddings op plus every op from the first LayerNorm to the last BinaryNg (the MLP residual
  add). 21 ops (23 at 3072, where attention adds a Pad and a Slice). Same set whether prefill was traced or not. Leaves
  out the final norm, the LM head (8 × `32×4096×16032` matmuls) and sampling. Scope "full" = whole last forward pass.
- **Why the scope must be pinned**: prefill is traced up to 1k and untraced from 2k, so `perf_util_report.py --select auto`
  picks the trace replay (one layer) for 128–1k but the whole forward pass for 2k+. That inconsistency was in the
  original sheet (MatMul % up to 7 pt too high, SDPA % up to 5 pt too low at 2k+).
- **Matmul / SDPA**: op codes matching `matmul` (Matmul, MinimalMatmul) and `sdpa`; shares are of device kernel time in scope.
- **Active-core utilization** (the headline number): per op, median over the cores that ran it of counter ÷ that core's
  counter window; then averaged over ops weighted by kernel time. The sheet mixed this with the "working core" median at
  2k+, which counts only the 8 of 80 LM-head cores that do work.
- **Full-grid utilization**: per op, counter ÷ op kernel duration averaged over all 110 worker cores (idle = 0). Chip level.
- **Counters**: FPU / SFPU = cycles an instruction is valid on that unit; MATH = either. In SDPA, FPU + SFPU = MATH exactly.

## Findings (October 2026)

Layer scope, active-core utilization:

| Prompt | Layer time | MatMul % | SDPA % | FPU in matmul | SFPU in SDPA |
|---|---|---|---|---|---|
| 128 | 1.33 ms | 77.48 | 2.46 | 45.76 | 20.32 |
| 512 | 2.25 ms | 70.81 | 7.65 | 66.17 | 45.92 |
| 1k | 3.83 ms | 62.86 | 16.55 | 79.35 | 47.09 |
| 2k | 8.12 ms | 57.91 | 24.75 | 78.53 | 58.89 |
| 3k | 15.3 ms | 54.24 | 28.33 | 73.27 | 59.10 |
| 4k | 20.0 ms | 47.99 | 37.84 | 75.66 | 59.21 |
| 8k | 53.9 ms | 35.42 | 54.32 | 75.75 | 59.33 |
| 16k | 164 ms | 23.18 | 70.16 | 75.77 | 59.37 |
| 32k | 556 ms | 13.70 | 82.38 | 75.71 | 59.38 |

- SDPA overtakes matmul between 4k and 8k. Layer time grows 3.4× from 16k to 32k.
- **Whole-layer FPU / SFPU** (all ops, mean over active cores, weighted by kernel time; `layer_units.py`): FPU 36.8 /
  48.7 / 53.6 / 51.7 / 46.7 / 45.6 / 40.0 / 34.4 / 30.0% and SFPU 2.3 / 7.1 / 11.9 / 18.4 / 19.8 / 25.5 / 34.5 / 43.2 /
  49.8% for 128 … 32k; they cross between 8k and 16k. Each op's counters already include everything it runs (SDPA's
  FPU counter includes QKᵀ/PV). The MLP's SiLU × multiply (`BinaryNg`, 58–69% SFPU) is half of all SFPU at 512.
  Data-movement ops (create/concat heads, KV-cache writes, pad/slice) launch no compute kernel and count as 0.
  Inter-op idle is excluded; the `DEVICE FW START/END CYCLE` columns are unusable for gaps (overlapping, and garbage in
  the `_nopad` runs), so gaps come from `OP TO OP LATENCY` in the regular runs only.
- **SDPA stages** (share of the math thread, 256-token chunks, 2k–32k): sub + exp 66%, QKᵀ 12%, PV 11%, row max 6%,
  correction 4–5%. At 64-token chunks (below 2k) row max + correction grow to 29–32%.
- **FPU vs SFPU in SDPA** (256-token chunk step = 149,918 cycles): the matmuls keep the FPU 93–98% busy (64 cycles per
  tile product at HiFi4) and are 92% of FPU cycles. Sub + exp is 88% SFPU-busy, about 1,356 SFPU cycles per tile, and is
  97% of all SFPU cycles. Row max and correction are 73–76% idle (waiting on unpack/pack).
- **Why exp is costly here**: prefill SDPA uses `exp_approx_mode=False` (`models/tt_transformers/tt/model_config.py:1723`)
  and HiFi4 with fp32 dest accumulation, which also keeps SDPA on the legacy compute loop (`compute_common.hpp`), not the
  streaming one. In that path every tile gets a separate SFPU multiply by 1/√d and re-runs the multiply and exp setup
  (`compute_common.hpp:404–407`).
- **Schedule** (4k, all 64 cores): unpack, math and pack go through the stages in lockstep; math sets the pace (unpack
  finishes sub + exp 7,604 cycles earlier); next QKᵀ starts a median 55 cycles after the correction, so K/V reads keep up.
- **Dependencies**: the next chunk's QKᵀ needs nothing from the current step. It waits only because one math thread runs
  the stages in order and there is one scores buffer (`cb_qk_im`, `Sq_chunk_t × Sk_chunk_t` tiles,
  `sdpa_program_factory.cpp:509`).
- **Layer dependency chain** at 4k: 17.59 of 19.97 ms (88%). Off the chain: W3 2.18 ms (beside W1), K/V branch 0.14 ms,
  KV-cache writes 0.05 ms, RoPE slices 0.01 ms. SDPA reads the type-cast Q/K/V, not the KV cache.
- **Quasar model** (separate FPU and SFPU threads, Blackhole stage costs, two scores buffers): SDPA step 149,918 →
  101,318 cycles (1.48×), SFPU thread 100% busy, FPU 48%. Layer −27% at 32k, −12% at 4k, −8% at 2k.
- **Single-cycle exp what-if** (exp = 1 SFPU instruction, 4 single-cycle instructions per 32-lane slot = 128
  cycles/tile vs ~1,356 measured; stage waits kept): SDPA step 149,911 → 71,326 cycles (2.10×); layer −0.8% (128),
  −6.4% (1k), −19.8% (4k), −43.2% (32k); whole-layer FPU 30.0 → 52.8%, SFPU 49.8 → 11.7% at 32k. With Quasar threads
  as well: step 48,588 cycles, FPU-thread bound; 32k layer 556 → 246 ms.
- **Only QKᵀ/PV on the FPU what-if** (row-max reduce, subtract, correction = ~192 element-wise tiles/step, 2,812
  FPU cycles, moved to the SFPU; "same cost" vs "SFPU rate" = 128 cycles/tile vs ~15 on the FPU): one thread — no
  gain (same cost) or 15% slower (SFPU rate). Two threads + measured exp — worse (101,323 → 115,645 cycles; SFPU is
  already the bottleneck). Two threads + single-cycle exp — threads balance: 48,588 → 37,060 cycles (4.05× today),
  FPU 88% busy, 32k layer 211 ms; but at SFPU rate 58,818 cycles (worse than not moving). In the two-thread model the
  SFPU thread must run MAX_{j+1}, EX_{j+1} before CORR_j (EX needs only the new row max), else the chain
  EX → PV → CORR serializes the steps.
- **FPU clock headroom on Quasar** (two-thread model, FPU tasks stretched until the step grows >0.1%): today's exp
  and split — FPU could run at 40–48% of its clock during SDPA (SFPU sets the step); single-cycle exp — none (FPU sets
  the step); matmuls-only split + single-cycle exp — 92%. Matmul ops need the full FPU clock (SFPU idle there), so a
  policy would switch per phase. SDPA holds 65% of the layer's FPU-busy cycles at 32k, 20% at 4k, 11% at 2k. Power:
  at fixed voltage, energy per FPU op (∝ C·V²) is unchanged and SDPA time is unchanged, so frequency alone saves only
  clock-distribution and ungated idle toggling; real savings need DVFS on an FPU power domain. Codebase shows Quasar
  data-movement and Neo compute clock domains (1.4 GHz, `profiler_analysis.cpp`), no separate FPU/SFPU domain.
- **Clock (AICLK)**: one chip-wide clock for FPU, SFPU, unpack/pack and the RISC-Vs; UMD exposes only BUSY (max) /
  IDLE (min) via firmware (`set_clock_state`, 200 ms timeout). From 512 tokens up, 84–97% of layer time is in ops whose
  math thread is busy 56–87%, and SDPA's non-busy time is waiting on unpack/pack (same clock), so a lower clock slows the
  layer roughly in proportion. Low *average* utilization comes from mixing phases, not from slack in the bottleneck.
- **Data quality**: every run passed PCC (0.998); no "Profiler DRAM buffers were full" warnings; counter-derived
  utilization matches the ops CSV columns to 1e-14 pt.

## Carrying over to a decode study

Reusable as is:
- `env.sh`, the capture pattern in `capture.sh` (needs a decode test and `-k` filter), `perf_util_report.py`
  (`--select auto` picks the last trace replay, and decode is traced), and every trap listed below.
- Definitions: active-core vs full-grid utilization, time weighting, data-movement ops counted as 0, the counter
  semantics (FPU + SFPU = MATH when the units never overlap), and `layer_units.py`'s method.
- The "DRAM full" check, the profiler limits (~125 zones per RISC per op; sum profiling has 2 slots) and the zone
  sampling + exact-count scaling used by `sdpa_stages.py`.
- The two-thread (Quasar) and what-if models are generic in-order schedulers; only the task list and costs are SDPA
  prefill specific.

Different in decode, re-derive before reusing numbers:
- **SDPA kernel**: decode uses `paged_scaled_dot_product_attention_decode` (flash-decode,
  `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`), not the prefill
  `compute_common.hpp` loop. It has no profiling zones; `sdpa_stage_zones.patch` does not apply.
- **Exp**: decode also sets `exp_approx_mode=False` (`get_attn_sdpa_decode_program_config`, `model_config.py` ~1749,
  1756). Fidelity for `SDPA_DECODE` is HiFi4 in most optimization modes, HiFi2 in one (`model_config.py` ~288/318/411),
  so whether fp32 dest accumulation forces the accurate fp32 exp has to be checked per mode.
- **Op scope**: the layer-scope rule in `layer_table.py` (Embeddings + first LayerNorm .. last BinaryNg) is prefill
  specific; decode has different ops (QKV heads decode, paged cache update, SDPA decode). Pin a decode scope the same
  way and keep it fixed across batch sizes and context lengths.
- **Bound**: decode matmuls run on M = batch rows and are likely limited by reading weights from DRAM, not by the FPU.
  Prefill at 128 tokens already showed matmuls at 33–60% FPU on 32 cores. That changes the clock question: DRAM has its
  own clock, so a lower AICLK may cost little in decode, unlike prefill. Measure it rather than reuse the prefill answer.
- **Stage costs**: per-tile costs fitted here (exp ~1,356 SFPU cycles per tile, ~15 FPU cycles per element-wise tile,
  64 cycles per HiFi4 tile product) are kernel- and fidelity-specific; refit for flash-decode.

## Method notes and traps

- **Prompt padding**: `get_padded_prefill_len` (`models/tt_transformers/tt/common.py:729`) rounds to 128, 1024, then powers
  of 2 (512 → 1024, 3072 → 4096). `TT_PREFILL_MIN_PADDING=1` (test patch) rounds only to what the model needs: multiples of
  128, then 512 (Blackhole MLP block), then 1024. At 3072 the QKV matmul still pads to 4096 rows (`attention.py:1050`).
- **Environment**: the default shell python lacks `tracy`; use `env.sh`. `HF_MODEL` and `MESH_DEVICE` are required.
  `MAX_PREFILL_CHUNK_SIZE=32` matches the original runs (default would be 4096).
- **pytest filter**: use `-k '1layer and not accuracy and max128k-<len>-'`. `performance` alone also matches the
  `models_performance_bare_metal` marker and selects the accuracy variant too.
- **Dropped markers ("DRAM full")**: check `grep -c 'markers were dropped' run.log`; in the ops CSV, rows without device
  data that sit together and are "captured but never replayed" trace ops are expected, not drops.
- **SDPA zones**: each TRISC holds ~125 zones per op, so zone runs are sampled. `sdpa_stages.py` scales each stage's mean
  by its exact execution count; coverage of 96–100% of measured math time confirms it. Zones did not change SDPA time
  (7553.2 vs 7553.5 µs at 4k). Tracy's `--enable-sum-profiling` would give exact totals but has only 2 slots.
- **Short stages**: at 64-token chunks the math thread's timestamps run ahead of the hardware (PV shows 53 cycles per tile
  product, below the 64-cycle rate). Per-stage FPU/SFPU splits and the Quasar model are only given for 2k and up.
- **Tracking processes**: `pgrep -f '<pattern>'` also matches the shell that runs it; use PIDs. Don't kill a device run
  mid-way; let it finish.
- **Scratch files**: anything kept outside the repo (session scratchpads) can disappear between days. Keep scripts here.
