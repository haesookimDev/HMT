# HMT Backlog (Deferred Work)

> 본 문서는 **현재 macOS 로컬 환경에서 실행/검증 불가능한 항목**과,
> Stage 0–7 진행 중 발견된 follow-up 이슈를 한 곳에 정리한다.
> 전제 환경은 **원격 Linux + CUDA GPU**이며, 일부 항목은 long-run 학습이 필요하다.
>
> 진행 상황은 [TODO.md](TODO.md) 참조. 본 BACKLOG는 "deferred" 항목에 집중.

## 환경 전제

- Linux x86_64 또는 ARM64 (CUDA 지원)
- CUDA 12.x + 호환 NVIDIA driver
- GPU VRAM 권장: 24GB 이상 (Phase 2~3 1B/3B), 80GB+ (Phase 4 7B)
- 의존성 추가: `uv pip install -e '.[qlora,kernels,viz,dev,logging]'`
- 권장 클라우드: RunPod, Lambda Labs, AWS p4d/p5, Colab Pro+ A100

---

## Stage 0.5 — QLoRA baseline (CUDA-only)

기존 코드는 이미 작성됨([hmt/model_loader.py](hmt/model_loader.py)의 `load_baseline_qlora`,
[configs/baseline_qlora.yaml](configs/baseline_qlora.yaml)). 실제 실행만 deferred.

- [ ] **0.5.1 의존성 설치**
  - what: `uv pip install -e '.[qlora]'` (linux-only marker로 macOS에서는 no-op)
  - verify: `uv run python -c "import bitsandbytes as bnb; print(bnb.__version__)"`
- [ ] **0.5.2 200-step run + 메트릭 수집**
  - what: `uv run python train_baseline.py --config configs/baseline_qlora.yaml`
  - 모델 액세스: Llama-3.2-1B (HF token 필요) — 또는 `pythia-160m`로 교체 가능
  - verify: `outputs/baseline_qlora/metrics.jsonl` 생성, eval ppl < AdamW baseline 동일 step
- [ ] **0.5.3 trainable params 비율 확인**
  - verify: 출력의 `trainable / total` 로그 — LoRA r=16에서 약 0.5–1% 예상
- [ ] **0.5.4 peak GPU mem 측정 + plot**
  - verify: `peak_mem_mb`가 AdamW full BF16 대비 명백히 감소 (≥30%)

**예상 소요**: 30분 (셋업) + 1–2시간 (200 step run, A100 기준)

---

## Stage 4 — CPU Basis Cache

**전제**: pinned memory + async CUDA stream → CUDA-only

### 4.1 `hmt/memory/cpu_basis_cache.py`

- [ ] **CPUBasisCache 클래스**
  - what: README pseudocode 기반:
    ```python
    class CPUBasisCache:
        def put(self, key, P, Q):    # CPU pinned memory에 저장
        def get(self, key, device):   # async H2D
    ```
  - LRU 또는 layer-id 직접 mapping
  - `pin_memory=True`로 할당 → `tensor.to(device, non_blocking=True)`로 async copy
- [ ] **단위 테스트** (`tests/test_cpu_basis_cache.py`)
  - put/get round-trip equality
  - pinned flag 확인 (`tensor.is_pinned()`)
  - non-CUDA platform에서 graceful fallback (no-op 또는 동기 copy)
  - **verify (CUDA)**: cache 크기 N entries × 평균 (P+Q) 바이트 == 호스트 메모리 사용량

### 4.2 Lifecycle Policy

- [ ] **`refresh_projectors_from_grads`에 cache 통합**
  - 현재 step에 사용 중인 projector만 GPU resident
  - 나머지는 즉시 CPU로 이동 (`P.to('cpu', non_blocking=True)`)
  - 다음 step에서 필요해지면 prefetch
- [ ] **GPU 동시 resident projector 수 카운팅**
  - hook으로 측정, jsonl logging
  - **verify**: 동시 resident 개수 ≤ N (config로 지정한 상한)

### 4.3 Async Prefetch

- [ ] **별도 CUDA stream으로 다음 layer basis prefetch**
  - 현재 layer의 forward/backward와 overlap
  - `torch.cuda.Stream` + `torch.cuda.current_stream().wait_stream(prefetch_stream)`
- [ ] **profiler 검증**
  - what: `torch.profiler`로 H2D copy와 compute가 overlap되는지 확인
  - **verify**: timeline에서 memcpy와 GEMM 동시 실행, prefetch 추가 시 step time ≤ no-prefetch

### 4.4 macOS placeholder

- [ ] **MPS / CPU 환경에서 graceful no-op**
  - `pin_memory()` 호출이 MPS에서 에러 시 catch → 일반 CPU tensor로 fallback
  - `.to(device, non_blocking=True)`도 MPS에서는 동기로 동작 — OK
  - **verify**: macOS에서 import + 동작 + "cache disabled" 경고만 출력

**예상 소요**: 1–2일 (구현 + 디버깅 + profiler 분석)

---

## Stage 5 — Triton Kernels (Linux+CUDA only)

`triton`은 [pyproject.toml](pyproject.toml)에서 `kernels` extra + linux-only marker로
이미 분리되어 있음. macOS에서는 no-op 설치, CUDA 환경에서만 실제 설치.

### 5.1 Blockwise INT8 quantize / dequantize kernels

- [ ] **`hmt/kernels/quantize.py`** — Triton kernel
  - 현재 [hmt/memory/activation_compress.py](hmt/memory/activation_compress.py)의 PyTorch
    구현은 several individual ops로 split됨. Triton fused kernel로:
    1. `amax` per block
    2. scale 계산
    3. round + clamp
    4. int8 packing
  - **verify**: PyTorch eager 결과와 정확 일치 (bit-perfect quantization is OK)
- [ ] **`hmt/kernels/dequantize.py`** — INT8 → fp32 매트릭스 multiply 직전 fused
- [ ] **벤치마크**: 1024×4096 활성화, block_size=256
  - **verify**: ≥2× throughput vs PyTorch eager

### 5.2 Fused low-rank reconstruction + weight update

- [ ] **`hmt/kernels/lowrank_update.py`** — 핵심 hot path
  - 현재 [hmt/optim/lowrank_adamw.py](hmt/optim/lowrank_adamw.py)의 `_step_lowrank`:
    ```python
    full_update = projector.reconstruct(low_update.to(p.dtype))
    p.data.add_(full_update, alpha=-lr)
    ```
  - 이는 (a) `P @ low_update @ Q^T` 매트릭스 곱 2회 + (b) tensor 할당 + (c) in-place add
  - Fused kernel: in-place `W -= η · P @ U @ Q.T` 단일 launch
  - **verify**: PyTorch eager와 결과 동일 (atol 1e-5), step time 단축
- [ ] **5.3 Fused optimizer state update** — m, v 갱신 + 위 reconstruction까지 한 kernel로
  - kernel launch 횟수 큰 폭 감소
  - **verify**: `torch.profiler` `cuda_kernel_count` 로그가 layer당 ~3회 → ~1회

### 5.4 Platform 가드

- [ ] **`hmt/kernels/__init__.py`**
  - `try: import triton` → 실패 시 PyTorch fallback 함수 노출
  - 동일 API: `quantize_int8(x, block_size)`, `lowrank_update_(W, P, U, Q, lr)` 등
  - **verify**: macOS / non-CUDA Linux에서 import 에러 없이 PyTorch path 사용

**예상 소요**: 3–5일 (Triton 학습 곡선 + 정확도 검증 + 벤치마크)

---

## Stage 6.2 — APOLLO-Mini (rank-1 auxiliary state)

현재 [hmt/optim/apollo.py](hmt/optim/apollo.py)는 tensor / channel 모드만 구현.
6.2는 `m`, `v` 모두 rank-1 (벡터 2개)로 표현하는 APOLLO 논문의 "Mini" 변형.

- [ ] **이론 설계 문서화**
  - what: `m ≈ u_m ⊗ w_m`, `v ≈ u_v ⊗ w_v` (outer product 근사)
  - 갱신 규칙: u, w 벡터를 어떻게 EMA로 유지하는가
    - 후보 1: SVD-rank-1 of running gradient (느림)
    - 후보 2: streaming power iteration (1회 iter per step)
    - 후보 3: row-mean & column-mean 곱 (간단, 정확도 낮음)
  - 결정 후 짧은 design note 작성 (`docs/apollo_mini_design.md`)
- [ ] **구현 + 테스트**
  - `ApolloAdamW(scaling="rank1")` 추가
  - state size: u_m [out] + w_m [in] + u_v [out] + w_v [in] = 2(out + in)
  - **verify**: state 원소 수 ≤ GaLore Mode A (r×r) at r=64
- [ ] **수렴 검증**
  - toy regression에서 tensor / channel과 비교
  - **verify**: loss 단조 감소, AdamW 대비 ≤ 5% 차이

**예상 소요**: 3–5일 (research-grade 알고리즘)

---

## Stage 6.3 — GaLore vs APOLLO long-run ablation

코드는 모두 준비됨 (`configs/hmt_stage1.yaml`, `hmt_stage2.yaml`,
`hmt_stage6_apollo.yaml`). 실제 long-run 비교만 deferred.

- [ ] **실험 매트릭스 정의**
  - 모델: pythia-160m → 1B → 3B (단계적)
  - 데이터: wikitext-2 (small) + C4 / FineWeb-Edu subset (large)
  - 옵티마이저: AdamW / GaLore-fixed / GaLore-scheduler / APOLLO-tensor / APOLLO-channel
  - Steps: 1k smoke / 10k full
- [ ] **`scripts/run_matrix.py`로 일괄 실행**
  - 동일 시드, 동일 데이터, 동일 LR schedule
- [ ] **메트릭 비교 표 + plot 생성**
  - peak GPU mem, optimizer state mem, train loss, val ppl, tokens/sec
  - **verify**: README §7 비교 표 채울 수 있는 5 PNG (기존 4-panel + rank distribution + 추가 1개)
- [ ] **분석 노트**
  - 어떤 layer 패턴에서 어떤 optimizer가 우세한가
  - 메모리 vs 정확도 pareto frontier

**예상 소요**: 1주 (실험 + 분석 + plot)
**비용 추정**: A100 80GB × 24h × 5 configs ≈ $50–100

---

## Cross-cutting C.3 — 로깅 추상화

현재 [train_baseline.py](train_baseline.py)는 JSONL 직접 쓰기. 외부 로거 도입 시 시점에:

- [ ] **`hmt/utils/logger.py`** — 단일 인터페이스
  ```python
  class MetricLogger(Protocol):
      def log(self, step: int, metrics: dict, *, event: str = "train") -> None: ...
      def close(self) -> None: ...
  ```
- [ ] **3가지 백엔드**
  - `JsonlLogger` (기존 동작 유지)
  - `TensorBoardLogger` (옵션 `[logging]` extras에 이미 있음)
  - `WandBLogger`
- [ ] **train_baseline.py에서 dispatch**
  - `cfg.logging.backend: jsonl | tensorboard | wandb` (또는 list)
  - 동시 다중 백엔드 지원 (broadcast)
- [ ] **verify**: 동일 학습 run에서 3개 백엔드 모두 같은 step별 loss 기록

**예상 소요**: 반나절. 도입 시점 권장: long-run ablation 시작 직전 (수치 비교를 W&B 대시보드에서)

---

## Cross-cutting C.4 — 원격 학습 가이드

- [ ] **`docs/remote_training.md` 작성** (또는 README 부록)
  - **RunPod**: instance 선택 가이드 (A100 40/80GB, H100), Pod template, persistent volume, SSH key
  - **Lambda Labs**: 1-Click GPU, ephemeral instance, S3 sync 패턴
  - **공통 셋업 스크립트** (`scripts/setup_remote.sh`):
    1. `uv` 설치
    2. `uv venv && uv pip install -e '.[qlora,kernels,viz,logging]'`
    3. HF token export (Llama 사용 시)
    4. `nvidia-smi` 확인 + `torch.cuda.is_available()` 검증
    5. wandb login (있다면)
- [ ] **데이터 캐시 가이드**
  - HF datasets cache → 영속 볼륨 마운트
  - WikiText / C4는 cold start 1회만 다운로드
- [ ] **결과 회수 패턴**
  - `outputs/` rsync to local, 또는 W&B artifact upload

**예상 소요**: 반나절

---

## Stage 0–7 진행 중 발견된 follow-up 이슈

다음 항목들은 코드 path는 정확히 동작하나, 학습 품질을 더 끌어올리려면 추후 다룰 가치 있음.

### F.1 LowRankAdamW basis 갱신 시 m / v 좌표계 미정렬

- **발견**: Stage 1.4 smoke. K=4 같은 짧은 refresh interval에서 PPL 진동.
- **원인**: `LayerProjector.refresh_`로 P, Q를 갈아끼울 때 m, v는 옛 basis 좌표계에 머무름. 새 basis에서는 의미가 다름.
- **해결 방향**:
  - GaLore 논문의 m/v projection 보정 적용 (오래된 m을 새 basis로 transform)
  - 또는 K를 충분히 크게 (≥50) 유지하여 영향 최소화 (현재 default)
- **verify**: K=4 ~ 200 sweep, K 작아도 PPL 단조 감소 유지

### F.2 INT8 absmax round-trip error 한계

- **발견**: Stage 3.1 smoke. block_size=256에서 Frob rel err ~0.7% (README의 0.5% 미달)
- **해결 방향**:
  - block_size=32 또는 64로 줄이기 (storage overhead 증가)
  - NF4-style nonlinear quantization 도입
  - INT8을 FP8 (E4M3 또는 E5M2)로 교체 (Hopper 이상)
- **verify**: 새 변형으로 round-trip < 0.5% 유지하면서 storage ≤ 0.6× BF16

### F.3 Checkpoint resume 시 dataset 위치 미저장

- **발견**: Stage 7.2 구현 시 명시적으로 deferred.
- **현재 동작**: resume 시 데이터 iterator를 스트리밍 처음부터 재시작.
  - 짧은 fine-tuning에서는 문제 없음
  - 장기 pretraining에서는 같은 batch를 재사용 → loss spike 가능
- **해결 방향**:
  - `Datasets.IterableDataset.skip(n)` 활용
  - 또는 token count 기반 checkpoint (`global_token_count`)로 resume
- **verify**: 100 step 학습 → 50 step에서 checkpoint → resume 후 51부터 진행 시 50까지의 loss 정확 일치 + 51부터의 loss가 연속

### F.4 활성화 압축 패칭 후 model state_dict 호환성

- **발견**: Stage 7.2 구현 시 동작 확인됨. CompressedLinear는 `.weight`, `.bias`만 가지므로 state_dict는 nn.Linear와 동일.
- **잠재 이슈**: HF `model.save_pretrained()` 사용 시 CompressedLinear가 nn.Linear가 아니라서 일부 헬퍼가 깨질 수 있음
- **해결 방향**: `patch_model_int8_linear`의 역연산 `unpatch_model_int8_linear` 추가
- **verify**: patch → save_pretrained → load_pretrained 라운드트립 (CUDA 환경)

### F.5 `LowRankAdamW`의 sparse 가정 미검증

- **현재**: `if p.grad.is_sparse: raise RuntimeError(...)` — 즉시 에러
- **HF Embedding의 `sparse=True`** 사용 시 학습 진입 불가
- **해결 방향**: sparse grad는 lowrank 적용 안 하고 dense AdamW path로 fallback
- **verify**: embedding이 sparse=True인 모델에서 lowrank_adamw 정상 동작

### F.6 multi-GPU (FSDP / DDP) 호환성 미검증

- **현재**: 단일 GPU 가정. `torch.cuda.set_rng_state_all`은 여러 device 다루지만, FSDP wrapper와는 미테스트
- **해결 방향**: Stage 4 작업 시점에 FSDP wrapper 안에서 LowRankAdamW가 정상 작동하는지 검증
- **verify**: 2-GPU FSDP에서 `train_baseline.py --config configs/hmt_full.yaml` 진행 + 단일 GPU와 동등 loss 수렴 (allow some divergence due to all-reduce)

---

## 권장 진행 순서 (원격 GPU 확보 후)

1. **C.4 (반나절)** — 원격 환경 셋업 가이드 + setup script
2. **C.3 (반나절)** — W&B 로거 (이후 비교 실험에서 필수)
3. **0.5 (1–2시간)** — QLoRA baseline 데이터 1세트 확보
4. **6.3 short-run (1일)** — 1k step × 5 optimizer 변형, smoke 결과 확인
5. **F.1 (1일)** — m/v projection 보정 적용 후 K-sweep
6. **Stage 4 (1–2일)** — CPU basis cache, peak mem 실측
7. **Stage 5 (3–5일)** — Triton kernels (시간 여유 있을 때)
8. **6.3 full (1주)** — 10k step ablation matrix, 최종 비교 표
9. **F.3 (1일)** — checkpoint resume 정확성 (long-run에서 중요)
10. **6.2 (3–5일)** — APOLLO-Mini rank-1 (research)

총 예상: **3–4주**의 GPU 사용 + 약 $100–300 클라우드 비용 (작은 모델), Phase 2~3로 가면 추가 비용
