# %% [markdown]
# # `attn.py` — 오늘의 어텐션 **정답지**
#
# ⚠️ **먼저 `01_self_attention.py` 의 `## 4`·`## 5` 를 스스로 채운 뒤에 여는 파일이다.**
# 오늘의 핵심은 그 네 줄을 직접 짜 보는 것이고, 여기엔 그 답이 이미 적혀 있다.
#
# 다 짜고 나서 열면 확인할 것이 하나 있다 — **내가 짠 것과 여기 있는 것이 같은가.**
# 같다면 오늘은 끝난 것이다.
#
# 논문의 식은 딱 한 줄이다:
#
# $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
#
# 이 파일에는 그 한 줄과, 그것을 감싼 셀프 어텐션 한 덩어리,
# 그리고 손계산에 쓸 작은 예제가 들어 있다.
#
# | | 어제 (Day 1) | 오늘 (Day 2) |
# |---|---|---|
# | 점수 내는 법 | 작은 신경망 `v·tanh(Wq·q + Wk·k)` (덧셈형) | 그냥 **내적** `q·k` (논문의 것) |

# %%
import math

import torch
import torch.nn as nn

# %% [markdown]
# ## 1. 논문의 식 한 줄
#
# **오늘 배우는 것의 전부가 이 함수 하나다.** 논문 식과 한 줄씩 대응한다.
#
# ```
# softmax( Q Kᵀ / √d_k ) V
#    ③       ①    ②     ④
# ```


# %%
def scaled_dot_product_attention(Q, K, V):
    """논문 §3.2.1 Scaled Dot-Product Attention.

    Args:
        Q: (L, d_k)  질의 — "나는 무엇을 찾고 있나"
        K: (L, d_k)  이름표 — "나는 무엇인가"
        V: (L, d_v)  내용 — "내가 가진 값"

    Returns:
        out     : (L, d_v)  각 자리의 새 표현 (V 들의 가중합)
        weights : (L, L)    어텐션 가중치. 행마다 합이 1이다
    """
    d_k = Q.size(-1)

    # ① 모든 짝의 궁합 점수를 한 번에.  (L,d_k) @ (d_k,L) → (L,L)
    scores = Q @ K.transpose(-2, -1)

    # ② √d_k 로 나눈다 (논문 각주 4).
    #    d_k 가 커지면 내적의 분산이 d_k 만큼 커져 softmax 가 포화된다.
    #    √d_k 로 나누면 흩어짐이 다시 1 근처로 돌아온다.
    scores = scores / math.sqrt(d_k)

    # ③ 점수 → 가중치.  dim=-1 은 "행 방향" — 한 자리가 여러 자리를 볼 때
    #    그 배분의 합이 1이 되어야 하므로 행마다 정규화한다.
    weights = torch.softmax(scores, dim=-1)

    # ④ 가중합.  (L,L) @ (L,d_v) → (L,d_v)
    out = weights @ V
    return out, weights


# %% [markdown]
# ## 2. 셀프 어텐션 — Q·K·V 를 "같은 입력"에서 만든다
#
# 위 함수는 Q·K·V 를 **받아서** 계산만 한다. 그 Q·K·V 를 **만드는** 것이 여기다.
# 어제와 오늘의 차이가 `forward` 의 **인자 개수 하나**에 들어 있다.
#
# | | 어제 (cross) | 오늘 (self) |
# |---|---|---|
# | `forward` | `forward(self, h, enc_out)` — 둘 | `forward(self, x)` — **하나** |


# %%
class SelfAttention(nn.Module):
    """한 문장이 자기 자신을 보는 어텐션.

    어제는 Q가 디코더에서, K·V가 인코더에서 왔다 (두 문장 사이 = cross).
    오늘은 셋 다 같은 x 에서 나온다 (한 문장 안 = self).

    투영이 왜 세 개나 필요한가?
      x 끼리 그냥 내적하면 두 가지가 망가진다.
        (1) 자기 고착 — x_i·x_i ≈ d 인데 x_i·x_j ≈ ±√d 라
            대각만 압도적으로 커져 각 단어가 '자기 자신만' 보게 된다 (= 항등층)
        (2) 학습할 것이 없음 — 파라미터가 0개라 데이터에 맞춰 달라질 수가 없다
      투영 세 개가 이 둘을 동시에 푼다. 01_self_attention.py 에서 직접 확인한다.
    """

    def __init__(self, d):
        super().__init__()
        # bias=False — 논문의 식에 편향이 없고, 나중에 파이토치 것과 맞출 때도 편하다
        self.Wq = nn.Linear(d, d, bias=False)   # 찾는 것
        self.Wk = nn.Linear(d, d, bias=False)   # 이름표
        self.Wv = nn.Linear(d, d, bias=False)   # 내용
        self.Wo = nn.Linear(d, d, bias=False)   # 마지막에 한 번 섞기

    def forward(self, x):
        """x: (L, d) → out: (L, d), weights: (L, L)

        Q·K·V 가 **셋 다 같은 x** 에서 나오는 것 — 이 세 줄이 'self' 다.
        마지막 Wo 는 멀티헤드에서 헤드들을 이어 붙인 뒤 한 번 섞는 자리인데,
        단일 헤드인 여기서도 파이토치와 구조를 맞추려고 그대로 둔다
        (손계산 예제에서는 Wo 를 항등행렬로 둬서 없는 것과 같게 만들었다).
        """
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)      # ← 셋 다 같은 x 에서. 이게 'self'
        out, weights = scaled_dot_product_attention(Q, K, V)
        return self.Wo(out), weights


# %% [markdown]
# ## 3. 파이토치 것과 맞춰 보기
#
# 우리가 짠 것과 `nn.MultiheadAttention` 이 **같은 계산**인지 확인하려면,
# 같은 가중치를 심어 놓고 출력을 비교하면 된다.


# %%
def plant_into_torch(mine, d):
    """내 가중치를 nn.MultiheadAttention 에 그대로 심는다.

    nn.MultiheadAttention 은 Wq·Wk·Wv 를 따로 두지 않고
    (3d, d) 짜리 in_proj_weight 하나에 세로로 쌓아 둔다.

        in_proj_weight  (3d, d)      우리 것
        ┌──────────┐
        │   Wq     │  0   ~  d       mine.Wq.weight  (d, d)
        ├──────────┤
        │   Wk     │  d   ~ 2d       mine.Wk.weight  (d, d)
        ├──────────┤
        │   Wv     │ 2d   ~ 3d       mine.Wv.weight  (d, d)
        └──────────┘

    우리도 nn.Linear 로 만들었으므로 저장 방식이 같아 — 전치 없이 — 그냥 이어 붙이면 된다.
    (직접 torch.matmul(x, W) 로 짰다면 여기에 .T 를 네 군데 붙여야 했다.)

    ⚠️ num_heads=1 이 전제다.
       헤드가 둘 이상이면 파이토치가 이 (3d, d) 를 헤드별로 잘라 쓰기 때문에
       같은 값을 심어도 계산이 달라진다 (실측: 손계산 예제 0.1478 · 무작위 d=8 0.0734).
       부동소수점 오차 1e-8 과 자릿수가 다르다. 멀티헤드는 shape 만 따로 본다.
    """
    mha = nn.MultiheadAttention(embed_dim=d, num_heads=1, bias=False, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(
            torch.cat([mine.Wq.weight, mine.Wk.weight, mine.Wv.weight], dim=0)
        )
        mha.out_proj.weight.copy_(mine.Wo.weight)
    return mha


# %%
def run_torch(mha, x):
    """nn.MultiheadAttention 은 배치를 요구한다. (L,d) → (1,L,d) 로 넣고 다시 벗긴다.

    x 를 세 번 넣는 것에 주목 — 파이토치 함수는 원래 (query, key, value) 를 따로 받는다.
    어제의 cross attention 이 그 형태였다. **셋에 같은 것을 넣으면 셀프**가 된다.
    """
    out, weights = mha(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
    return out.squeeze(0), weights.squeeze(0)


# %% [markdown]
# ## 4. 손계산용 작은 예제
#
# 손으로 계산할 수 있게 일부러 이렇게 골랐다:
#
# - 가중치가 전부 **0 아니면 1** → `Q·K·V` 가 전부 **정수**로 나온다
# - `d = 4` 라서 **`√d_k` 가 정확히 2** → 나눗셈이 암산으로 된다
# - `Wo` 는 **항등행렬** → 가중합 결과가 그대로 최종 출력이다
#
# ⚠️ 이 가중치들은 학습된 것이 아니라 **손계산이 되도록 고른 것**이다.
# 그래서 이 어텐션 격자에는 아무 의미가 없다. 오늘은 계산이 맞는지만 본다.

# %%
TOKENS = ["나는", "배를", "먹었다", "."]

# nn.Linear 는 y = x @ W.T 로 계산한다. 그래서 아래 W? 의 **각 행**이
# 출력 한 성분을 만든다 — 예를 들어 Q 의 첫 성분은 x 와 WQ 의 0번 행의 내적이다.
#   Q[나는] = X[나는] @ WQ.T = [1,1,0,0] @ WQ.T = [1, 1, 1, 1]
# (WQ 의 네 행 [1,0,0,1] [1,0,1,0] [1,0,0,0] [0,1,1,0] 과 [1,1,0,0] 을 각각 내적)
#
# 각 토큰을 나타내는 벡터. 진짜 임베딩은 학습된 실수지만,
# 오늘은 손으로 곱할 수 있도록 0과 1만 썼다.
X = torch.tensor(
    [
        [1, 1, 0, 0],   # 나는
        [0, 1, 1, 0],   # 배를
        [1, 0, 0, 1],   # 먹었다
        [0, 0, 1, 1],   # .
    ],
    dtype=torch.float,
)

WQ = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0]], dtype=torch.float)
WK = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float)
WV = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]], dtype=torch.float)
WO = torch.eye(4)


# %%
def toy():
    """손계산 예제를 만들어 돌려준다. (모델, x, 토큰이름)"""
    m = SelfAttention(4)
    with torch.no_grad():
        m.Wq.weight.copy_(WQ)
        m.Wk.weight.copy_(WK)
        m.Wv.weight.copy_(WV)
        m.Wo.weight.copy_(WO)
    return m, X, TOKENS


# %% [markdown]
# ## 5. 보기 좋게 찍는 도우미
#
# 여기부터는 **출력을 예쁘게 만드는 코드**다. 어텐션과는 상관없으니 그냥 지나가도 된다.


# %%
def _pad(s, width):
    """한글은 터미널에서 두 칸을 차지한다. 그걸 감안해 오른쪽 정렬한다.

    파이썬 기본 f-string 정렬은 글자 '개수'만 세기 때문에
    한글이 섞이면 표가 어긋난다. 폭을 직접 계산해서 맞춘다.
    """
    폭 = sum(2 if ord(c) > 0x1100 else 1 for c in s)
    return " " * max(0, width - 폭) + s


# %%
def show_matrix(name, M, rows=None):
    """행렬을 이름표와 함께 찍는다. 정수면 정수로 보여 준다."""
    is_int = torch.equal(M, M.round())
    print(f"{name}  {tuple(M.shape)}")
    for i, row in enumerate(M):
        label = _pad(rows[i] if rows else str(i), 8)
        cells = " ".join(f"{v:>4.0f}" if is_int else f"{v:>8.4f}" for v in row)
        print(f"  {label} | {cells}")


# %%
def show_grid(weights, tokens):
    """어텐션 격자를 찍는다. 행 = 보는 쪽, 열 = 보이는 쪽.

    행 합만 찍는다 — 열 합은 1이 될 이유가 없어서 일부러 안 찍는다.
    궁금하면 weights.sum(0) 을 직접 찍어 보라 (제각각으로 나온다).
    """
    head = " ".join(_pad(t, 8) for t in tokens)
    print(f"{_pad('', 8)} | {head}   (행 합)")
    for i, row in enumerate(weights):
        cells = " ".join(f"{v:>8.4f}" for v in row)
        print(f"{_pad(tokens[i], 8)} | {cells}    {row.sum():.4f}")


# %% [markdown]
# ## 6. 자가 점검 도우미 — `01` 노트북이 쓴다
#
# `01_self_attention.py` 에서 **네 단계를 직접 채우는 칸**이 있다.
# 거기서 쓰는 채점기가 여기 있다.
#
# 점검은 **값을 외워 두고 비교하는 방식이 아니다.** 기계마다·실행마다 무작위 값이 달라지므로,
# 그때그때 같은 `Q·K·V` 로 정답을 새로 계산해서 맞춰 본다.
#
# ⚠️ **스스로 채워 보기 전에 아래 `단계별_정답` 의 본문을 열어 보지 말 것.**


# %%
def 단계별_정답(Q, K, V):
    """네 단계의 중간값을 한 번에 돌려준다. 노트북 자가 점검용."""
    d_k = Q.size(-1)
    점수 = Q @ K.transpose(-2, -1)
    스케일 = 점수 / math.sqrt(d_k)
    가중치 = torch.softmax(스케일, dim=-1)
    출력 = 가중치 @ V
    return {"점수": 점수, "스케일": 스케일, "가중치": 가중치, "출력": 출력}


# %%
def 확인(이름, 학생값, 정답, 힌트=""):
    """학생이 채운 값을 점검하고 **다음 칸에서 쓸 값을 돌려준다.**

    핵심은 마지막 줄이다 — 틀렸으면 정답을 돌려준다.
    그래야 한 칸에서 막혀도 노트북이 끝까지 돌아간다.
    다시 채워 넣고 이 칸만 돌리면 그때 ✓ 로 바뀐다.
    """
    if 학생값 is None:
        print(f"✗ {이름} — 아직 안 채웠다.  {힌트}")
    elif not torch.is_tensor(학생값):
        print(f"✗ {이름} — 텐서가 아니다 ({type(학생값).__name__}).  {힌트}")
    elif 학생값.shape != 정답.shape:
        print(f"✗ {이름} — 모양이 {tuple(학생값.shape)} 인데 {tuple(정답.shape)} 여야 한다.  {힌트}")
    elif not torch.allclose(학생값, 정답, atol=1e-5):
        print(f"✗ {이름} — 모양은 맞는데 값이 다르다.  {힌트}")
    else:
        print(f"✓ {이름}  {tuple(학생값.shape)}")
        return 학생값
    print(f"   → 지금은 정답으로 이어서 간다. 위를 고치고 이 칸을 다시 돌리면 ✓ 가 뜬다.")
    return 정답


# %%
def 함수확인(fn, Q, K, V):
    """학생이 정의한 어텐션 함수가 정답과 같은 값을 내는지 본다. True/False 를 돌려준다."""
    try:
        결과 = fn(Q, K, V)
    except Exception as e:
        print(f"✗ 함수가 돌다가 멈췄다 — {type(e).__name__}: {e}")
        return False

    try:
        출력, 가중치 = 결과
    except (TypeError, ValueError):
        print(f"✗ 반환값이 (출력, 가중치) 두 개가 아니다 — 받은 것: {type(결과).__name__}")
        return False

    if not torch.is_tensor(출력) or not torch.is_tensor(가중치):
        print("✗ 아직 안 채웠다 — 네 줄이 전부 None 이다. TODO ①~④ 를 채운다.")
        return False

    정답 = 단계별_정답(Q, K, V)
    if 출력.shape != 정답["출력"].shape:
        print(f"✗ 출력 모양이 {tuple(출력.shape)} 인데 {tuple(정답['출력'].shape)} 여야 한다.")
        return False
    if not torch.allclose(출력, 정답["출력"], atol=1e-5):
        print("✗ 출력 값이 다르다. ÷√d_k 를 빠뜨렸거나 softmax 방향(dim=-1)이 다를 수 있다.")
        return False
    if not torch.allclose(가중치, 정답["가중치"], atol=1e-5):
        print("✗ 출력은 맞는데 가중치가 다르다. softmax 를 통과한 값을 돌려주고 있는지 보라.")
        return False

    행합 = 가중치.sum(-1)
    print(f"✓ 출력·가중치 둘 다 정답과 같다  {tuple(출력.shape)}")
    print(f"✓ 가중치 행 합이 전부 1  (최소 {행합.min():.6f} · 최대 {행합.max():.6f})")
    return True


# %% [markdown]
# ## 7. 이 파일만 돌려 보기 (자가 확인)
#
# 위에서 정의한 것들이 실제로 도는지 여기서 한 번 확인한다.
# **`01_self_attention.py` 가 불러다 쓸 때는 이 칸이 실행되지 않는다** (`__name__` 이 다르다).

# %%
if __name__ == "__main__":
    m, x, tokens = toy()
    Q, K, V = m.Wq(x), m.Wk(x), m.Wv(x)
    out, weights = scaled_dot_product_attention(Q, K, V)

    print("문장:", " ".join(tokens))
    print(f"x {tuple(x.shape)} → Q·K·V {tuple(Q.shape)} → 점수 {tuple((Q @ K.T).shape)}"
          f" → 출력 {tuple(out.shape)}")
    print("\n어텐션 격자 — 행마다 합이 1인 것만 본다 (값에는 의미가 없다):")
    show_grid(weights, tokens)
    print("\n1행 출력:", [round(v, 6) for v in out[0].tolist()],
          "  ← 종이로 푼 1행과 같은 값")
