# HMT 개발 TODO

> 본 문서는 [README.md](README.md)의 12단계 개발 로드맵을 실행 가능한 단위 작업으로 분해한 것이다.
> 각 항목은 **what / verify** 형식이며, verify 조건이 만족되어야 다음으로 넘어간다 (CLAUDE.md "Goal-Driven Execution").

## 환경 / 도구

- 로컬: macOS (Apple Silicon), Python 3.11, **uv** 패키지 관리
- 원격(필요 시): Linux + CUDA GPU (Stage 1+의 실제 학습, Stage 4 CPU pinned cache, Stage 5 Triton)
- macOS 한계: `bitsandbytes` / `triton` 사용 불가 → Stage 0 QLoRA baseline과 Stage 5 kernel은 원격에서만 검증

## 진행 표시 범례

- `[ ]` 미완료
- `[~]` 진행 중
- `[x]` 완료 (verify 조건 통과)
- `[!]` 블로커 / 디펜던시 대기

## 현재 상태 (2026-04-27)

- Stage 0 코드 작성 + 0.1 ~ 0.4 verify 통과. `hmt/data.py`의 causal-LM label off-by-one 버그 수정 완료 (HF 모델이 내부 shift를 수행하므로 dataset에서 사전 shift하지 않음).
- 0.5는 CUDA + bitsandbytes 필요 → 로컬 macOS에서는 실행 불가, 원격 GPU 환경 확보 후 진행.
- 다음 작업: **Stage 1** (1.1 GaLore-style projector부터). 사전에 C.3(로깅 추상화) 도입 여부 결정 필요.

---

## Stage 0 — 기준선 마무리

- [x] **0.1 환경 부트스트랩**
  - what: `uv venv --python 3.11 && uv pip install -e '.[logging,dev]'`
  - verify: `uv run python -c "import torch, transformers, datasets, omegaconf"`가 에러 없음

- [x] **0.2 AdamW baseline smoke test**
  - what: `uv run python train_baseline.py --config configs/baseline_adamw.yaml training.max_steps=10 logging.log_interval=1`
  - verify: `outputs/baseline_adamw/metrics.jsonl`에 10줄, loss 단조 감소 경향, NaN 없음

- [x] **0.3 평가 루프 추가**
  - what: `hmt/eval.py` — wikitext-2 valid split에서 perplexity 계산. `train_baseline.py`에 `eval_interval` 도입. `hmt/data.py`의 label off-by-one 버그 동시 수정
  - verify: smoke run에서 step 4 → 8 동안 ppl 69.0 → 65.5로 감소 ✅

- [x] **0.4 메트릭 시각화 스크립트**
  - what: `scripts/plot_metrics.py` — JSONL의 train/eval 이벤트를 분리해 loss / ppl / tokens-per-sec / peak-mem 4-패널 PNG 생성
  - verify: `outputs/smoke_0_3/plots.png` 정상 생성 ✅

- [!] **0.5 (선택) 원격 CUDA에서 QLoRA baseline 검증** — 블록: 로컬 macOS는 CUDA 없음
  - what: 원격 GPU에서 `[qlora]` extras 설치 후 동일 데이터로 200 step
  - verify: trainable params 비율 ~1% 미만, peak mem이 AdamW 대비 명백히 감소

---

## Stage 1 — GaLore-style Low-Rank Optimizer

- [ ] **1.1 `hmt/optim/projector.py`**
  - what: `LayerProjector(P, Q, rank)` + `project(grad)` / `reconstruct(low_update)`
  - verify: shape 일치, BF16 round-trip 오차 < 토이 데이터에서 1%

- [ ] **1.2 SVD 기반 basis 초기화**
  - what: `update_projection_basis(grad, rank)` — `torch.linalg.svd` (정확도 검증용)
  - verify: pytest로 reconstruction error가 rank↑일수록 단조 감소

- [ ] **1.3 `hmt/optim/lowrank_adamw.py`**
  - what: `LowRankAdamW` — m, v를 `r×r` 또는 `r×min(out,in)` shape로 보관. weight decay decoupled
  - verify: 단일 linear layer GaLore 방식으로 학습하여 toy convex 문제에서 vanilla AdamW와 같이 수렴

- [ ] **1.4 `hmt/trainer.py`**
  - what: 직접 작성하는 트레이너. backward 후 layer hook에서 grad → project → low-rank state update → full grad 즉시 해제. HF Trainer 사용 X
  - verify: `torch.cuda.max_memory_allocated`로 peak GPU mem이 baseline AdamW 대비 감소

- [ ] **1.5 `configs/hmt_stage1.yaml`**
  - what: rank 고정값(e.g. 128), 대상 layer 패턴(`q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj`)
  - verify: 동일 시드에서 baseline AdamW와 step별 loss 차이 < 5%

- [ ] **1.6 회귀 테스트 `tests/test_lowrank_optim.py`**
  - what: 작은 모델(125M 또는 dummy 2-layer transformer)에서 baseline 대비 loss/메모리 비교
  - verify: CI에서 30초 이내 완료, peak mem 감소량 보고

---

## Stage 2 — Dynamic Rank Selection

- [ ] **2.1 `hmt/optim/spectrum.py`** — randomized SVD + power iteration
  - verify: rank 64 추정에서 full SVD 대비 ≥3× 빠름, top-k singular value 상대오차 < 5%

- [ ] **2.2 `hmt/optim/rank_scheduler.py`** — `EnergyRankScheduler(τ=0.95, candidates=[32,64,128,256])`
  - verify: 합성 gradient(known spectrum)에서 의도한 rank 선택

- [ ] **2.3 update interval K 도입**
  - what: 매 step이 아니라 K step마다만 basis 갱신. gradient norm trigger도 실험
  - verify: K=50/100/200에서 loss 영향 < 1% 차이, tokens/sec 향상

- [ ] **2.4 layer-wise rank 로깅**
  - what: `metrics.jsonl`에 layer별 rank/energy 기록
  - verify: attention layer가 평균적으로 더 작은 rank로 수렴, MLP는 큰 rank 유지하는 경향 관찰

- [ ] **2.5 `configs/hmt_stage2.yaml`** + Stage 1 대비 평균 rank 감소량 보고

---

## Stage 3 — Hybrid Activation Compression

- [ ] **3.1 `hmt/memory/activation_compress.py`** — `BlockwiseInt8Compressor(block_size=256)`
  - verify: BF16 입력에서 평균 상대오차 < 0.5%, INT8 storage가 BF16 대비 정확히 1/2

- [ ] **3.2 `hmt/autograd/compressed_linear.py`** — `CompressedLinearFunction`
  - verify: gradcheck가 fp32 reference 대비 1e-2 이내 통과

- [ ] **3.3 `hmt/memory/policy.py`** — `ActivationPolicy` (per-layer-type: keep / recompute / compress_int8 / compress_fp8)
  - verify: YAML 정책이 정확히 mapping됨 (단위테스트)

- [ ] **3.4 모델 패칭 유틸**
  - what: HF 모델의 MLP/Attention output linear을 `CompressedLinear`로 교체하는 `patch_model(model, policy)`
  - verify: forward 출력이 패칭 전후 동일 (numerical tolerance)

- [ ] **3.5 baseline 비교 — gradient checkpointing vs HMT activation compression**
  - verify: peak VRAM 감소, step time이 grad ckpt 대비 동등 이상

---

## Stage 4 — CPU Basis Cache (Linux+CUDA)

- [ ] **4.1 `hmt/memory/cpu_basis_cache.py`** — pinned memory에 P, Q 보관, async H2D/D2H
  - verify: cache put/get 단위테스트, pinned memory 사용 확인

- [ ] **4.2 cache lifecycle policy**
  - what: 현재 step에서 필요한 layer의 basis만 GPU resident, 나머지는 CPU
  - verify: GPU 상에 동시 resident인 basis 수가 N(=동시 활성 layer)으로 제한됨

- [ ] **4.3 prefetch** — 다음 layer basis를 별도 CUDA stream으로 미리 H2D
  - verify: profiler에서 H2D와 compute가 overlap

- [ ] **4.4 macOS placeholder**
  - what: MPS는 pinned memory 미지원 → no-op fallback. Linux+CUDA에서만 활성
  - verify: mac에서 import 에러 없음, "cache disabled on this platform" 경고만 출력

---

## Stage 5 — Triton Kernels (Linux+CUDA only)

- [ ] **5.1 `hmt/kernels/quantize.py` / `dequantize.py`** — blockwise INT8
  - verify: PyTorch reference 대비 결과 동일, 2× 이상 throughput

- [ ] **5.2 `hmt/kernels/lowrank_update.py`** — fused `W -= η · (P @ Update(Ĝ) @ Q.T)`
  - verify: PyTorch eager 대비 step time 단축

- [ ] **5.3 fused optimizer update**
  - what: m, v 갱신 + reconstruction을 한 kernel로
  - verify: kernel launch 횟수 감소, GPU util 상승

- [ ] **5.4 platform 가드**
  - what: `import_kernels()` 함수가 mac에서는 Python fallback 반환
  - verify: mac CI에서 import 에러 없음

---

## Stage 6 — APOLLO-style Optimizer

- [ ] **6.1 `hmt/optim/scaling.py`** — tensor-wise / channel-wise scaling
  - verify: 같은 합성 문제에서 GaLore-style과 수렴 비교

- [ ] **6.2 `hmt/optim/apollo.py`** — APOLLO + APOLLO-Mini (rank-1)
  - verify: optimizer state mem이 GaLore Mode A 대비 추가 감소

- [ ] **6.3 GaLore vs APOLLO ablation**
  - what: 동일 dataset/모델/예산
  - verify: 학습 안정성 + loss 동등 또는 더 나음, mem ↓

---

## Stage 7 — 통합 HMT Trainer

- [ ] **7.1 모든 정책을 단일 YAML로 제어** (`configs/hmt_1b.yaml`, `hmt_3b.yaml`, `hmt_7b.yaml`)
  - verify: 한 yaml에서 baseline → HMT-full로 전환되며 동일 코드 경로

- [ ] **7.2 NVMe checkpoint-only policy** — `hmt/memory/checkpoint.py`
  - what: low-rank state + basis snapshot + tokenizer state 저장/복원
  - verify: 학습 중간 kill → resume 시 step/loss 연속

- [ ] **7.3 실험 매트릭스 자동화**
  - what: baseline×HMT 변형의 메트릭을 한 표로. `scripts/run_matrix.py`
  - verify: N config 차례로 실행, summary.json 생성

- [ ] **7.4 최종 리포트 가능 그래프**
  - what: peak mem, tokens/sec, val ppl, rank distribution, compression error
  - verify: README에서 직접 인용 가능한 PNG/SVG 5종

---

## 횡단 (Cross-cutting)

- [ ] **C.1 GitHub Actions** — lint(ruff) + Stage 1/2 단위테스트만 (모델 다운로드 X, dummy transformer 사용)
- [ ] **C.2 결정성** — `seed_everything()`, `torch.use_deterministic_algorithms(True)` 옵션
- [ ] **C.3 로깅 추상화** — JSONL / TensorBoard / W&B를 동일 인터페이스로. Stage 1 시작 전에 도입
- [ ] **C.4 원격 학습 가이드** — RunPod / Lambda Labs 같은 원격 CUDA 머신에서 실행 절차 (mac에서는 큰 모델 학습 불가)

---

## 권장 진행 순서

1. **지금**: 0.1 → 0.2 (Stage 0 sanity 통과)
2. **다음 세션**: 0.3 (eval) + C.3 (로깅 추상화) — Stage 1+에서 필요
3. **그다음**: Stage 1 (1.1 ~ 1.6) — HMT의 첫 진짜 기여. 여기까지 끝나면 GaLore 논문 재현 수준
4. Stage 2 ~ 3은 독립적이라 병행 가능. 4 ~ 5는 Linux/CUDA가 필요하니 원격 환경 확보 후
5. Stage 6 ~ 7은 6번이 먼저, 7번은 모든 부품이 모인 뒤 통합
