# HMT: Hierarchical Memory Training with Dynamic Low-Rank Optimization and Hybrid Activation Compression
---

작은 VRAM 환경에서 거대 언어모델 학습을 위한 계층형 메모리 학습 알고리즘

초록

대규모 언어모델 학습은 모델 파라미터, gradient, optimizer state, activation으로 인해 막대한 GPU VRAM을 요구한다. 기존 접근은 LoRA/QLoRA, ZeRO offload, gradient checkpointing, CPU/NVMe offload를 통해 메모리 부족을 완화하지만, 많은 경우 학습 자유도 제한, PCIe/NVMe 병목, 재계산 비용 증가라는 한계를 가진다. 본 논문은 작은 VRAM 환경에서 거대 LLM을 더 효율적으로 학습하기 위한 계층형 메모리 학습 알고리즘 HMT를 제안한다. HMT는 단순히 데이터를 CPU나 디스크로 내리는 방식이 아니라, 학습 중 생성되는 gradient, optimizer state, activation의 표현 자체를 작게 만든다. 구체적으로는 동적 저랭크 gradient projection, APOLLO/GaLore 계열 optimizer state 압축, layer-wise hybrid activation compression, CPU basis cache, NVMe checkpoint-only policy를 결합한다. 목표는 offload 중심 학습보다 높은 처리량을 유지하면서 full-parameter fine-tuning에 가까운 학습 자유도를 확보하는 것이다.


---

1. 서론

거대 언어모델은 파라미터 수가 증가할수록 추론과 학습 모두에서 메모리 요구량이 급격히 증가한다. 특히 학습 단계에서는 단순히 모델 weight만 저장하는 것이 아니라 gradient, optimizer state, activation까지 저장해야 하므로 메모리 사용량이 추론보다 훨씬 크다. AdamW optimizer를 사용하는 일반적인 mixed precision 학습에서는 optimizer state가 모델 weight보다 더 큰 메모리를 차지할 수 있다.

작은 VRAM 환경에서 사용되는 대표적인 방법은 다음과 같다.

첫째, LoRA 또는 QLoRA는 원본 모델 weight를 고정하고 작은 adapter만 학습함으로써 trainable parameter와 optimizer state를 줄인다. 그러나 weight update 공간이 제한되기 때문에 full fine-tuning과 다른 학습 동역학을 갖는다.

둘째, DeepSpeed ZeRO, FSDP, CPU/NVMe offload는 optimizer state, gradient, parameter를 CPU RAM 또는 NVMe로 이동시켜 GPU VRAM 사용량을 줄인다. 그러나 CPU↔GPU 또는 NVMe↔GPU 전송이 빈번해질수록 PCIe와 디스크 I/O가 병목이 된다.

셋째, gradient checkpointing은 activation 저장량을 줄이는 대신 backward 과정에서 forward 계산을 다시 수행한다. 이 방식은 메모리 절감 효과는 크지만 학습 시간이 증가한다.

최근에는 기존 방식과 다른 접근이 등장하고 있다. GaLore는 weight space가 아니라 gradient space에서 low-rank projection을 수행하여 full-parameter learning에 가까운 학습을 유지하면서 optimizer state 메모리를 줄인다. 해당 연구는 optimizer state 메모리를 최대 65.5% 줄이고, 8-bit GaLore에서는 optimizer memory를 최대 82.5%, 전체 training memory를 63.3% 줄였다고 보고한다. 또한 24GB RTX 4090에서 7B 모델 pretraining 가능성을 보였다고 설명한다. 

APOLLO는 AdamW의 learning-rate adaptation에 중복성이 있다는 관찰에서 출발하여, auxiliary low-rank optimizer state로 AdamW 수준의 성능을 SGD 수준 메모리 비용에 가깝게 달성하려는 방법이다. 특히 APOLLO-Mini는 rank-1 변형에서도 강한 메모리 효율을 목표로 한다. 

CompAct는 optimizer가 아니라 backward에 필요한 activation compute graph를 대상으로 하며, compressed activation을 저장해 pretraining에서 GPU peak memory를 25~30%, fine-tuning에서 50% 줄였다고 보고한다. 

본 논문은 이러한 흐름을 결합하여, 작은 VRAM 환경에서 거대 LLM을 학습하기 위한 새로운 통합 알고리즘을 제안한다.


---

2. 문제 정의

2.1 목표

본 연구의 목표는 다음과 같다.

작은 VRAM 환경에서 거대 LLM을 학습하되, 단순 CPU/NVMe offload에 의존하지 않고 학습 중 생성되는 주요 메모리 항목을 수학적으로 압축한다.

대상 메모리 항목은 다음과 같다.

1. Model parameters
2. Gradients
3. Optimizer states
4. Activations
5. Temporary buffers

우리가 최적화하려는 목적 함수는 다음과 같이 정의할 수 있다.

minimize   peak_gpu_memory
maximize   tokens_per_second
maintain   validation_loss_quality

즉, 단순히 메모리를 줄이는 것이 아니라 메모리, 속도, 학습 품질의 균형을 최적화한다.


---

2.2 기존 방식의 한계

기존 작은 VRAM 학습 방식은 크게 두 부류로 나뉜다.

첫 번째는 학습 범위를 줄이는 방식이다. LoRA/QLoRA가 대표적이다. 이 방식은 매우 실용적이지만, 원본 weight 전체를 직접 업데이트하지 않는다.

두 번째는 학습 상태를 외부 메모리로 이동하는 방식이다. ZeRO-Offload, ZeRO-Infinity, CPU/NVMe offload가 여기에 해당한다. 이 방식은 모델을 실행 가능하게 만들지만, GPU가 계산을 기다리는 시간이 증가할 수 있다.

본 논문에서는 다음 관점을 취한다.

> 작은 VRAM 문제를 해결하기 위해서는 메모리를 외부로 밀어내는 것보다, 학습 상태 자체를 더 작은 표현으로 바꾸는 것이 우선되어야 한다.




---

3. 제안 방법: HMT

HMT는 다음 네 가지 핵심 구성요소로 이루어진다.

HMT = Dynamic Low-Rank Gradient Projection
    + Memory-Efficient Optimizer State
    + Hybrid Activation Compression
    + Hierarchical Basis and Checkpoint Storage


---

3.1 Dynamic Low-Rank Gradient Projection

Transformer의 linear layer weight를 다음과 같이 둔다.

W_l ∈ R^{out × in}

일반적인 학습에서는 backward 과정에서 full gradient가 생성된다.

G_l = ∂L / ∂W_l

HMT는 이 gradient를 그대로 optimizer에 전달하지 않는다. 대신 gradient를 저차원 부분공간으로 project한다.

G_l \approx P_l \, \hat{G}_l \, Q_l^\top

여기서:

G_l      : layer l의 원래 gradient
P_l      : output 방향 projection basis
Q_l      : input 방향 projection basis
Ĝ_l      : low-rank gradient
rank r   : r << min(out, in)

low-rank gradient는 다음과 같이 계산한다.

Ĝ_l = P_l^T G_l Q_l

optimizer는 full gradient G_l이 아니라 Ĝ_l에 대한 state만 유지한다. 따라서 AdamW의 momentum, variance도 full matrix 크기가 아니라 low-rank matrix 크기로 저장된다.


---

3.2 동적 rank 선택

기존 low-rank 방식은 보통 layer마다 고정 rank를 사용한다. 그러나 모든 layer의 gradient spectrum이 동일하지 않다. 일부 layer는 rank 32로도 충분할 수 있고, 일부 MLP layer는 rank 128 이상이 필요할 수 있다.

HMT는 gradient energy ratio를 기반으로 layer별 rank를 동적으로 선택한다.

E(r) = sum_{i=1}^{r} σ_i^2 / sum_{i=1}^{n} σ_i^2

여기서 σ_i는 gradient의 singular value이다.

rank 선택 규칙은 다음과 같다.

if E(64) >= τ:
    rank = 64
elif E(128) >= τ:
    rank = 128
elif E(256) >= τ:
    rank = 256
else:
    rank = full 또는 high-rank fallback

보통 τ = 0.90 ~ 0.98 사이에서 실험한다.

실제 구현에서는 매 step마다 SVD를 수행하면 너무 느리므로 다음을 사용한다.

1. randomized SVD
2. power iteration
3. update interval
4. gradient norm 기반 rank 예측
5. layer type별 rank prior


---

3.3 Memory-Efficient Optimizer State

AdamW는 parameter마다 보통 다음 state를 저장한다.

m_t : first moment
v_t : second moment

HMT에서는 full weight shape에 대해 m_t, v_t를 저장하지 않는다. 대신 low-rank gradient 공간에서 optimizer state를 저장한다.

m̂_t, v̂_t ∈ R^{r × r}

또는 APOLLO-style로 channel-wise 또는 tensor-wise scaling을 근사한다.

APOLLO는 AdamW의 adaptive scaling을 full parameter 단위가 아니라 구조화된 learning-rate scaling으로 근사하는 방식이며, auxiliary low-rank state를 사용한다. 

HMT의 optimizer는 다음 두 모드를 지원한다.

Mode A: GaLore-style low-rank AdamW
Mode B: APOLLO-style approximate gradient scaling

초기 구현에서는 Mode A가 더 직관적이다. 이후 성능이 안정되면 APOLLO-style scaling을 추가한다.


---

3.4 Hybrid Activation Compression

훈련 중 activation은 backward를 위해 저장된다. 긴 sequence나 큰 batch에서는 activation이 optimizer state보다 큰 병목이 될 수 있다. CompAct는 compressed activation을 통해 peak memory를 줄일 수 있음을 보였다. 

HMT는 모든 activation에 동일한 정책을 적용하지 않는다. Transformer 내부 구성요소별로 다른 정책을 사용한다.

Embedding activation      : keep BF16
Attention Q/K/V           : recompute
Attention output          : compress FP8 또는 INT8
MLP intermediate          : compress INT8
Residual stream           : keep BF16 또는 FP8
LayerNorm input/stat      : keep BF16
LM head                   : keep BF16

이 정책의 이유는 다음과 같다.

Attention activation은 재계산 비용이 상대적으로 허용 가능한 경우가 많고, MLP intermediate는 크기가 커서 compression 이득이 크다. LayerNorm과 residual stream은 수치 안정성에 민감하므로 과도하게 압축하지 않는다.


---

3.5 CPU Basis Cache

기존 offload는 weight, optimizer state, gradient를 CPU나 NVMe로 이동한다. HMT는 이와 다르게 CPU RAM을 projection basis cache로 사용한다.

즉, CPU에는 다음만 저장한다.

P_l, Q_l의 오래된 버전
layer별 rank history
gradient spectrum statistics
optimizer low-rank history

GPU에는 현재 step에서 필요한 basis만 유지한다.

이 방식의 장점은 CPU↔GPU 전송량이 weight offload보다 훨씬 작다는 점이다.


---

3.6 NVMe Checkpoint-Only Policy

NVMe는 학습 중 frequent offload 대상으로 사용하지 않는다. HMT에서 NVMe는 다음 용도로 제한한다.

1. adapter 또는 low-rank optimizer checkpoint
2. projection basis snapshot
3. training state snapshot
4. dataset cache

즉, NVMe를 “느린 VRAM”으로 쓰지 않고, “저빈도 영속 저장소”로만 사용한다.


---

4. HMT 전체 알고리즘

Algorithm 1. HMT Training Loop

Input:
    model θ
    dataset D
    memory policy π
    optimizer Ω
    rank threshold τ
    projection update interval K

Initialize:
    Load model in BF16 or 8-bit
    Initialize projection bases P_l, Q_l for selected layers
    Initialize low-rank optimizer states
    Initialize CPU basis cache
    Initialize activation compression policy

For each training step t:
    1. Load batch x_t

    2. Forward pass:
        For each layer l:
            Apply layer-specific activation policy:
                - keep
                - recompute
                - compress_int8
                - compress_fp8

    3. Compute loss L_t

    4. Backward pass:
        For each trainable linear layer l:
            Compute gradient G_l

            If t mod K == 0:
                Estimate gradient spectrum
                Select rank r_l dynamically
                Update projection bases P_l, Q_l
                Store old bases in CPU cache

            Project gradient:
                Ĝ_l = P_l^T G_l Q_l

            Release full gradient G_l from GPU memory

    5. Optimizer step:
        Update low-rank optimizer states m̂_l, v̂_l
        Reconstruct update:
            ΔW_l = P_l Update(Ĝ_l) Q_l^T

        Apply weight update:
            W_l ← W_l - η ΔW_l

    6. Periodic checkpoint:
        Save model delta, low-rank states, basis snapshots to NVMe

Output:
    trained model θ
    low-rank optimizer states
    final projection bases


---

5. 핵심 알고리즘 상세 설명

5.1 Projection Basis Update Algorithm

가장 중요한 부분은 projection basis를 어떻게 갱신하느냐다.

매 step마다 full SVD를 하면 느리다. 따라서 HMT는 K step마다만 basis를 갱신한다.

Input:
    gradient G_l
    previous basis P_l, Q_l
    target energy threshold τ
    rank candidates R = {32, 64, 128, 256}

Process:
    1. Randomized range finder로 approximate singular vectors 계산
    2. singular value energy ratio 계산
    3. 최소 rank r_l 선택
    4. P_l, Q_l 갱신
    5. 이전 basis는 CPU cache로 이동

Pseudo-code:

@torch.no_grad()
def update_projection_basis(grad, rank_candidates, energy_threshold):
    # grad: [out_dim, in_dim]
    # 실제 구현에서는 torch.linalg.svd 대신 randomized SVD를 사용해야 함
    U, S, Vh = torch.linalg.svd(grad.float(), full_matrices=False)

    energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)

    selected_rank = rank_candidates[-1]
    for r in rank_candidates:
        if energy[r - 1] >= energy_threshold:
            selected_rank = r
            break

    P = U[:, :selected_rank].to(grad.dtype)
    Q = Vh[:selected_rank, :].T.to(grad.dtype)

    return P, Q, selected_rank

개발 초기에는 위처럼 SVD로 검증하고, 이후 randomized SVD 또는 Triton kernel로 최적화한다.


---

5.2 Low-Rank Optimizer Algorithm

Low-rank gradient를 계산한 뒤 optimizer state는 Ĝ에 대해서만 유지한다.

class HMTLowRankAdamW:
    def __init__(
        self,
        params,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        state_dtype=torch.float16,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state_dtype = state_dtype
        self.state = {}

    @torch.no_grad()
    def step_layer(self, weight, grad, projector):
        # grad: full gradient, temporary
        P, Q = projector.P, projector.Q

        # project full gradient to low-rank space
        low_grad = P.T @ grad @ Q

        sid = id(weight)
        if sid not in self.state:
            self.state[sid] = {
                "step": 0,
                "m": torch.zeros_like(low_grad, dtype=self.state_dtype),
                "v": torch.zeros_like(low_grad, dtype=self.state_dtype),
            }

        st = self.state[sid]
        st["step"] += 1

        m = st["m"]
        v = st["v"]

        m.mul_(self.beta1).add_(low_grad, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(low_grad, low_grad, value=1 - self.beta2)

        m_hat = m / (1 - self.beta1 ** st["step"])
        v_hat = v / (1 - self.beta2 ** st["step"])

        low_update = m_hat / (torch.sqrt(v_hat.float()) + self.eps)
        low_update = low_update.to(weight.dtype)

        # reconstruct update to full weight shape
        update = P @ low_update @ Q.T

        if self.weight_decay > 0:
            weight.mul_(1 - self.lr * self.weight_decay)

        weight.add_(update, alpha=-self.lr)

        # full grad는 즉시 제거 가능
        grad = None

실제 구현에서는 P @ low_update @ Q.T가 병목이 될 수 있으므로 Triton kernel로 fused reconstruction-update를 구현하는 것이 좋다.


---

5.3 Activation Compression Algorithm

Activation compression은 custom autograd로 구현한다.

초기 버전은 INT8 block-wise quantization을 사용한다.

class BlockwiseInt8Compressor:
    def __init__(self, block_size=256):
        self.block_size = block_size

    def compress(self, x):
        orig_shape = x.shape
        flat = x.reshape(-1, self.block_size)

        scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        q = torch.round(flat / scale).clamp(-128, 127).to(torch.int8)

        meta = {
            "orig_shape": orig_shape,
            "scale": scale.to(torch.float16),
        }
        return q, meta

    def decompress(self, q, meta):
        x = q.float() * meta["scale"].float()
        return x.reshape(meta["orig_shape"])

custom linear:

class CompressedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, compressor):
        y = torch.nn.functional.linear(x, weight, bias)

        packed_x, meta = compressor.compress(x)

        ctx.save_for_backward(weight)
        ctx.packed_x = packed_x
        ctx.meta = meta
        ctx.compressor = compressor
        ctx.has_bias = bias is not None

        return y

    @staticmethod
    def backward(ctx, grad_y):
        (weight,) = ctx.saved_tensors

        x = ctx.compressor.decompress(ctx.packed_x, ctx.meta)

        grad_x = grad_y @ weight
        grad_w = grad_y.reshape(-1, grad_y.shape[-1]).T @ x.reshape(-1, x.shape[-1])

        grad_b = None
        if ctx.has_bias:
            reduce_dims = tuple(range(grad_y.ndim - 1))
            grad_b = grad_y.sum(dim=reduce_dims)

        return grad_x, grad_w, grad_b, None

주의할 점은 위 코드는 연구 prototype이다. 실제 LLM의 linear layer shape와 tensor layout에 맞게 수정해야 하며, 성능을 위해 Triton kernel화가 필요하다.


---

6. 스테이지별 개발 계획

아래는 직접 개발 가능한 단계별 계획이다.


---

Stage 0. 실험 기준선 구축

목표

기존 방법과 비교할 기준선을 만든다.

구현 언어

Python
PyTorch
Hugging Face Transformers
YAML

구현 내용

1. Llama 계열 1B~3B 모델 로딩
2. BF16 학습 루프 작성
3. AdamW baseline
4. QLoRA baseline
5. VRAM, tokens/sec, loss curve 기록

핵심 알고리즘

Standard causal language modeling training loop

산출물

train_baseline.py
configs/baseline_adamw.yaml
configs/baseline_qlora.yaml
memory_profiler.py

예시 구조

hmt_train/
  train_baseline.py
  configs/
    baseline_adamw.yaml
    baseline_qlora.yaml
  hmt/
    data.py
    model_loader.py
    profiler.py


---

Stage 1. GaLore-style low-rank optimizer 구현

목표

full gradient를 생성하되 optimizer state는 low-rank로 저장한다.

구현 언어

Python
PyTorch

구현 알고리즘

Dynamic Low-Rank Gradient Projection
Low-Rank AdamW

개발 순서

1. nn.Linear layer만 대상으로 선정
2. backward 후 gradient hook에서 G_l 확보
3. SVD 기반 P_l, Q_l 계산
4. Ĝ_l = P_l^T G_l Q_l 계산
5. low-rank AdamW state 저장
6. ΔW_l = P_l Update(Ĝ_l) Q_l^T로 weight update
7. full gradient 즉시 삭제

핵심 파일

hmt/optim/projector.py
hmt/optim/lowrank_adamw.py
hmt/trainer.py

성공 기준

AdamW 대비 GPU memory 감소
loss가 발산하지 않음
작은 모델에서 baseline과 유사한 validation loss


---

Stage 2. Dynamic Rank Selection 추가

목표

layer별 gradient spectrum에 따라 rank를 자동 조절한다.

구현 언어

Python
PyTorch

구현 알고리즘

Gradient spectrum estimation
Energy-based rank selection
Rank scheduling

개발 순서

1. rank 후보군 정의: [32, 64, 128, 256]
2. projection update interval K 설정
3. K step마다 spectrum 추정
4. energy threshold τ 기준으로 rank 선택
5. layer별 rank log 저장

핵심 파일

hmt/optim/rank_scheduler.py
hmt/optim/spectrum.py

rank scheduler 예시

class EnergyRankScheduler:
    def __init__(self, candidates=(32, 64, 128, 256), threshold=0.95):
        self.candidates = candidates
        self.threshold = threshold

    def select_rank(self, singular_values):
        energy = torch.cumsum(singular_values ** 2, dim=0)
        energy = energy / energy[-1]

        for r in self.candidates:
            if r <= len(energy) and energy[r - 1] >= self.threshold:
                return r

        return self.candidates[-1]

성공 기준

고정 rank 대비 비슷한 loss
평균 rank 감소
tokens/sec 또는 memory 효율 향상


---

Stage 3. Activation Compression prototype

목표

gradient checkpointing보다 빠르거나, 비슷한 속도에서 더 낮은 activation memory를 달성한다.

구현 언어

Python
PyTorch custom autograd

구현 알고리즘

Blockwise INT8 activation compression
Hybrid activation policy

개발 순서

1. MLP activation만 INT8 압축
2. attention activation은 recompute 유지
3. residual과 layernorm은 BF16 유지
4. custom autograd Function 작성
5. memory/속도 비교

핵심 파일

hmt/memory/activation_compress.py
hmt/autograd/compressed_linear.py
hmt/memory/policy.py

정책 예시

activation:
  embedding: keep_bf16
  attention_qkv: recompute
  attention_output: compress_int8
  mlp_intermediate: compress_int8
  residual: keep_bf16
  layernorm: keep_bf16

성공 기준

no compression 대비 peak VRAM 감소
gradient checkpointing 대비 step time 개선 또는 유사
loss degradation 허용 범위 내 유지


---

Stage 4. CPU Basis Cache 구현

목표

CPU RAM을 weight offload가 아니라 projection basis cache로 사용한다.

구현 언어

Python
PyTorch

구현 알고리즘

Basis lifecycle management
Asynchronous CPU-GPU transfer
Pinned memory staging

개발 순서

1. 오래된 P_l, Q_l을 CPU pinned memory로 이동
2. 현재 step에 필요한 basis만 GPU에 유지
3. layer별 basis cache hit/miss 기록
4. basis prefetch 구현

핵심 파일

hmt/memory/cpu_basis_cache.py
hmt/memory/staging.py

basis cache 예시

class CPUBasisCache:
    def __init__(self, max_entries=1024):
        self.cache = {}
        self.max_entries = max_entries

    def put(self, key, P, Q):
        self.cache[key] = {
            "P": P.detach().to("cpu", non_blocking=True).pin_memory(),
            "Q": Q.detach().to("cpu", non_blocking=True).pin_memory(),
        }

    def get(self, key, device):
        item = self.cache[key]
        P = item["P"].to(device, non_blocking=True)
        Q = item["Q"].to(device, non_blocking=True)
        return P, Q

성공 기준

GPU에 유지되는 basis memory 감소
CPU transfer overhead가 전체 step time의 작은 비율로 유지


---

Stage 5. Triton kernel 최적화

목표

Python/PyTorch prototype에서 병목이 되는 연산을 GPU kernel로 최적화한다.

구현 언어

Python
Triton

최적화 대상

1. blockwise activation quantization
2. blockwise activation dequantization
3. low-rank projection matmul
4. fused reconstruction + weight update
5. optimizer state update

핵심 파일

hmt/kernels/quantize.py
hmt/kernels/dequantize.py
hmt/kernels/lowrank_update.py

우선순위

1. activation compress/decompress
2. reconstructed update 적용
3. low-rank gradient projection
4. fused optimizer update

성공 기준

Python implementation 대비 tokens/sec 향상
GPU utilization 상승
torch.profiler에서 kernel launch overhead 감소


---

Stage 6. APOLLO-style optimizer 추가

목표

GaLore-style optimizer 외에 APOLLO-style approximate gradient scaling을 구현한다.

구현 언어

Python
PyTorch
Triton optional

구현 알고리즘

Approximated Gradient Scaling
Channel-wise learning-rate scaling
Low-rank auxiliary optimizer state

개발 순서

1. tensor-wise scaling 버전 구현
2. channel-wise scaling 버전 구현
3. rank-1 APOLLO-Mini 스타일 구현
4. GaLore-style optimizer와 비교

핵심 파일

hmt/optim/apollo.py
hmt/optim/scaling.py

성공 기준

GaLore보다 optimizer state memory 감소
AdamW 또는 GaLore와 유사한 loss curve
학습 안정성 확보


---

Stage 7. 통합 HMT Trainer 개발

목표

모든 정책을 YAML config로 제어할 수 있는 학습 프레임워크를 만든다.

구현 언어

Python
PyTorch
YAML
OmegaConf

구현 내용

1. model loading
2. dataset packing
3. dynamic rank optimizer
4. activation compression
5. CPU basis cache
6. NVMe checkpoint-only save
7. profiling
8. experiment logging

프로젝트 구조

hmt_train/
  train.py
  configs/
    hmt_1b.yaml
    hmt_3b.yaml
    hmt_7b.yaml

  hmt/
    model_loader.py
    data.py
    trainer.py

    optim/
      projector.py
      lowrank_adamw.py
      rank_scheduler.py
      apollo.py

    memory/
      activation_compress.py
      cpu_basis_cache.py
      policy.py
      checkpoint.py

    autograd/
      compressed_linear.py

    kernels/
      quantize.py
      dequantize.py
      lowrank_update.py

    utils/
      profiler.py
      logging.py


---

7. 실험 설계

7.1 비교 대상

Baseline A: BF16 AdamW
Baseline B: QLoRA
Baseline C: GaLore fixed-rank
Baseline D: APOLLO
Proposed: HMT dynamic-rank + hybrid activation compression


---

7.2 측정 지표

1. Peak GPU memory
2. Average GPU memory
3. tokens/sec
4. step time
5. validation loss
6. CPU RAM usage
7. PCIe traffic
8. checkpoint size
9. rank distribution
10. activation compression error


---

7.3 실험 모델

초기 실험은 작은 모델부터 시작해야 한다.

Phase 1: 125M ~ 350M
Phase 2: 1B
Phase 3: 3B
Phase 4: 7B
Phase 5: 13B 이상

처음부터 7B 이상으로 가면 디버깅 비용이 너무 크다.


---

8. 개발 스택 요약

8.1 언어 선택

영역	언어/도구	이유

학습 루프	Python	실험 속도
모델 실행	PyTorch	LLM 생태계 호환성
config	YAML + OmegaConf	실험 재현성
optimizer prototype	Python + PyTorch	디버깅 용이
activation compression	PyTorch custom autograd	backward 제어
고성능 kernel	Triton	CUDA보다 빠른 개발
최종 극한 최적화	CUDA C++	필요할 때만
데이터 전처리	Python 또는 Rust	대용량이면 Rust 고려
로깅	TensorBoard / W&B	실험 추적
profiling	torch.profiler, Nsight Systems	병목 분석



---

8.2 권장 구현 순서

1. Python/PyTorch로 correctness 검증
2. 작은 모델에서 loss curve 확인
3. memory profiler로 실제 절감 확인
4. 병목 kernel만 Triton으로 이동
5. APOLLO-style scaling 추가
6. CPU basis cache 추가
7. 3B/7B 모델로 확장
8. 필요할 때만 CUDA extension 작성


---

9. 예상 기여점

본 방법의 예상 기여점은 다음과 같다.

첫째, CPU/NVMe offload 중심이 아니라 학습 상태 표현 자체를 압축한다.

둘째, LoRA처럼 weight update 공간을 고정 adapter로 제한하지 않고, full-parameter update에 가까운 gradient-space update를 수행한다.

셋째, 모든 layer에 같은 메모리 정책을 적용하지 않고, attention, MLP, residual, layernorm, lm_head에 서로 다른 정책을 적용한다.

넷째, CPU RAM을 weight offload 저장소가 아니라 projection basis cache로 사용한다.

다섯째, NVMe는 학습 중 빈번한 swap 공간이 아니라 checkpoint-only 저장소로 제한한다.


---

10. 한계와 위험 요소

10.1 수치 안정성

activation compression과 low-rank gradient projection은 모두 학습 품질에 영향을 줄 수 있다. 특히 LayerNorm, residual stream, lm_head는 압축에 민감할 수 있으므로 BF16 유지가 안전하다.

10.2 Projection overhead

basis 갱신에 SVD를 사용하면 매우 느릴 수 있다. 따라서 실제 구현에서는 randomized SVD, power iteration, update interval이 필수다.

10.3 Reconstruction cost

low-rank update를 다시 full weight shape으로 복원하는 과정이 병목이 될 수 있다. 이 부분은 Triton fused kernel로 최적화해야 한다.

10.4 구현 난이도

Hugging Face Trainer를 그대로 사용하기 어렵다. 직접 training loop, optimizer step, gradient hook, activation hook을 제어해야 한다.


---

11. 결론

본 논문은 작은 VRAM 환경에서 거대 언어모델을 학습하기 위한 계층형 메모리 학습 알고리즘 HMT를 제안하였다. 기존 방식은 주로 모델 상태를 CPU나 NVMe로 이동하거나, 학습 가능한 파라미터 수를 줄이는 방식에 의존한다. 반면 HMT는 gradient, optimizer state, activation의 표현 자체를 줄여 GPU VRAM 요구량을 낮춘다.

핵심 아이디어는 다음과 같다.

1. gradient는 full matrix로 오래 보관하지 않고 low-rank space로 project한다.
2. optimizer state는 full parameter shape이 아니라 low-rank shape으로 유지한다.
3. activation은 layer type별로 keep, recompute, compress를 다르게 적용한다.
4. CPU RAM은 weight offload가 아니라 projection basis cache로 사용한다.
5. NVMe는 checkpoint-only 저장소로 제한한다.
6. 성능 병목은 Triton kernel로 단계적으로 최적화한다.

직접 개발 관점에서는 Python + PyTorch로 prototype을 만들고, 병목이 확인된 compression/projection/update 연산만 Triton으로 옮기는 방식이 가장 현실적이다. CUDA C++는 최종 단계에서만 필요하다.


---

12. 최종 개발 로드맵

Stage 0:
    Baseline 구축
    언어: Python, PyTorch
    알고리즘: AdamW, QLoRA

Stage 1:
    Low-rank optimizer 구현
    언어: Python, PyTorch
    알고리즘: GaLore-style gradient projection

Stage 2:
    Dynamic rank 추가
    언어: Python, PyTorch
    알고리즘: energy-based rank scheduling

Stage 3:
    Activation compression 추가
    언어: Python, PyTorch custom autograd
    알고리즘: blockwise INT8/FP8 activation compression

Stage 4:
    CPU basis cache 추가
    언어: Python, PyTorch
    알고리즘: pinned memory basis staging

Stage 5:
    Triton 최적화
    언어: Python, Triton
    알고리즘: fused quantize/dequantize/projection/update

Stage 6:
    APOLLO-style optimizer 추가
    언어: Python, PyTorch, Triton optional
    알고리즘: approximate gradient scaling

Stage 7:
    HMT 통합 trainer 완성
    언어: Python, PyTorch, YAML
    알고리즘: dynamic low-rank optimizer + hybrid activation policy
