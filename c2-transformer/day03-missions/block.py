# %% [markdown]
# # `block.py` — 오늘의 **정답지**이자 도우미
#
# ⚠️ **먼저 `01_block.py` 의 `## 3`·`## 4`·`## 5` 를 스스로 채운 뒤에 여는 파일이다.**
# 오늘의 핵심은 어제 만든 어텐션에 **잔차 · 층 정규화 · FFN** 을 붙여 블록을 완성하는 것이고,
# 여기엔 그 답이 이미 적혀 있다.
#
# 논문 §3.1 이 말하는 인코더 블록은 이 한 줄이다:
#
# $$\text{LayerNorm}\bigl(x + \text{Sublayer}(x)\bigr)$$
#
# 여기서 `Sublayer` 는 **어텐션**이거나 **FFN** 둘 중 하나다. 그게 전부다.

# %%
import math

import torch
import torch.nn as nn

BERT_ID = "klue/bert-base"


# %% [markdown]
# ## 1. 어제 만든 셀프 어텐션 (그대로)
#
# 오늘 새로 배우는 것이 아니다. 어제 짠 것을 **붙일 대상**으로 다시 가져왔다.

# %%
def scaled_dot_product_attention(Q, K, V, mask=None):
    """논문 §3.2.1. 어제 것에 `mask` 인자 하나만 늘었다 (`## 7` 에서 쓴다)."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 가리는 자리를 -inf 로 만들면 softmax 를 지난 뒤 정확히 0이 된다.
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


# %%
class SelfAttention(nn.Module):
    """어제 만든 것. Q·K·V 가 셋 다 같은 x 에서 나온다."""

    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x, mask=None):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.Wo(out), weights


# %% [markdown]
# ## 2. 위치 인코딩 — 논문 §3.5 의 사인·코사인
#
# 논문 식 그대로다:
#
# ```
# PE(pos, 2i  ) = sin( pos / 10000^(2i/d) )     ← 짝수 차원
# PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )     ← 홀수 차원
# ```
#
# 차원마다 **파장이 다른 파도**를 하나씩 그린다고 보면 된다. 앞쪽 차원은 빠르게 출렁이고
# 뒤쪽 차원은 아주 느리게 움직인다. 그래서 위치마다 **고유한 무늬**가 생긴다.


# %%
def sinusoidal_pe(max_len, d):
    """(max_len, d) 짜리 사인·코사인 위치 인코딩. 학습 파라미터가 하나도 없다."""
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
    # 10000^(2i/d) 를 로그로 계산한다 — 지수가 커서 그냥 거듭제곱하면 값이 넘친다
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# %%
def robust_limit(M):
    """히트맵 색 범위를 2~98 백분위로 잡는다.

    학습된 위치 임베딩은 극소수 값이 나머지보다 10배 넘게 크다.
    최댓값으로 색을 맞추면 나머지가 전부 흰색이 되어 **빈 화면처럼** 보인다.
    """
    v = M.flatten().float()
    return max(torch.quantile(v, 0.02).abs().item(), torch.quantile(v, 0.98).abs().item())


# %%
def neighbor_similarity(pe, offsets=(1, 2, 5, 10, 20, 50, 100), n=200):
    """'가까운 위치끼리 비슷한가' 를 재는 도구.

    위치 벡터를 길이 1로 맞춘 뒤, `k` 칸 떨어진 두 위치의 코사인 유사도를 평균낸다.
    1에 가까우면 비슷하고, 0이면 무관하고, 음수면 반대 방향이다.

    `n=200` — 앞 200자리만 쓴다. 실습 문장이 그 안에 다 들어가기 때문이다.
    구간을 바꿔도 경향은 같다 (0~199 / 200~399 / 312~511 모두 +1 에서 0.71~0.75,
    +100 에서 0 근처). 다만 **+100 값은 구간에 따라 −0.04 ~ +0.01 로 흔들리므로
    "멀면 반대 방향" 이 아니라 "거의 무관해진다" 로 읽어야 한다.**

    Returns:
        dict[int, float]: {떨어진 칸 수: 평균 유사도}
    """
    x = pe[:n]
    x = x / x.norm(dim=1, keepdim=True)
    out = {}
    for k in offsets:
        if k >= n:
            continue
        pairs = (x[:-k] * x[k:]).sum(dim=1)
        out[k] = pairs.mean().item()
    return out


# %% [markdown]
# ## 3. ★ 오늘 만드는 것 — Transformer 인코더 블록
#
# 어제 어텐션 한 덩어리에 **세 가지를 붙였을 뿐**이다.
#
# | 붙인 것 | 무엇을 해결하나 |
# |---|---|
# | 잔차 연결 `x + …` | 깊게 쌓아도 입력이 끝까지 살아남는다 |
# | 층 정규화 `LayerNorm` | 더하다 보면 커지는 값의 크기를 매번 되돌린다 |
# | FFN | 어텐션이 **섞기만** 하므로, 각 자리를 따로 가공할 곳이 필요하다 |
#
# ⚠️ 어텐션은 어제 짠 것 대신 `nn.MultiheadAttention` 을 쓴다.
# 어제 그 둘이 **같은 값을 낸다는 걸 이미 확인했으므로**, 오늘은 붙이는 데만 집중한다.


# %%
class 내블록(nn.Module):
    """논문 Fig 1 의 인코더 블록 하나.

    forward 가 딱 두 줄이다. 두 줄 모두 LayerNorm(x + Sublayer(x)) 꼴이고,
    Sublayer 자리에 각각 어텐션과 FFN 이 들어간다.
    """

    def __init__(self, d, d_ff, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        # 넓혔다가(d → d_ff) 다시 접는다(d_ff → d). 논문은 4배를 썼다.
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),          # 논문은 ReLU, BERT 는 GELU (config 의 hidden_act 로 확인된다)
            nn.Linear(d_ff, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        """x: (B, L, d) → (B, L, d). 들어간 모양 그대로 나온다 = 몇 층이든 쌓인다."""
        a, _ = self.attn(x, x, x)          # Sublayer ①
        x = self.norm1(x + a)              # 잔차 → LN
        f = self.ffn(x)                    # Sublayer ②
        x = self.norm2(x + f)              # 잔차 → LN
        return x


# %% [markdown]
# ## 4. 파라미터 세기
#
# 오늘의 정점은 **세 가지의 파라미터 개수가 같다**는 것이다. 그걸 세는 도구.


# %%
def n_params(module):
    """모듈 안의 학습 파라미터 총 개수."""
    return sum(p.numel() for p in module.parameters())


# %%
def torch_layer(d, d_ff, h):
    """파이토치가 기본 제공하는 인코더 블록.

    ⚠️ dim_feedforward 를 **반드시 넘긴다.** 기본값이 2048 이라
       그냥 두면 우리 것(3072)과 개수가 안 맞는다 (5,513,984 가 나온다).
    """
    return nn.TransformerEncoderLayer(
        d_model=d, nhead=h, dim_feedforward=d_ff, batch_first=True
    )


# %%
def 파라미터표(모듈, 항목들):
    """모듈을 부분별로 세어 표로 찍는다.

    항목들: [(이름, 서브모듈 또는 서브모듈 리스트), ...]
    """
    총 = n_params(모듈)
    print(f"{'부분':<28s} {'개수':>12s}   비율")
    print("-" * 52)
    for 이름, m in 항목들:
        c = sum(n_params(x) for x in m) if isinstance(m, (list, tuple)) else n_params(m)
        print(f"{이름:<28s} {c:>12,d}   {c / 총 * 100:5.1f}%")
    print("-" * 52)
    print(f"{'합계':<28s} {총:>12,d}   100.0%")
    return 총


# %% [markdown]
# ## 5. 실제 BERT 불러오기
#
# 한 번 받아 두면 캐시에 남아서 두 번째부터는 즉시 뜬다.
#
# ⚠️ 처음 부를 때 `LOAD REPORT` 라는 표와 함께 `UNEXPECTED` 라고 적힌 줄이 몇 개 나올 수 있다.
# **오류가 아니다.** 저장된 파일 안에 지금 안 쓰는 부분(다른 용도의 머리)이 들어 있다는 뜻이다.

# %%
_BERT = None


def load_bert(model_id=BERT_ID):
    """klue/bert-base 를 불러온다. 같은 세션에서 여러 번 불러도 한 번만 받는다."""
    global _BERT
    if _BERT is None:
        from transformers import AutoModel
        import transformers.utils.logging as hf_logging

        hf_logging.set_verbosity_error()      # LOAD REPORT 소음을 줄인다
        _BERT = AutoModel.from_pretrained(model_id)     # BERT_ID = "klue/bert-base"
        _BERT.eval()                          # 학습이 아니라 구경이 목적이다
    return _BERT


# %%
def bert_layer0(model=None):
    """BERT 의 첫 번째 블록 하나. 12개가 전부 같은 모양이라 아무거나 하나면 된다."""
    model = model or load_bert()
    return model.encoder.layer[0]


# %% [markdown]
# ## 6. causal mask — 뒤를 못 보게 막는다
#
# 어제 `## 6` 에서 *"못 보게 막을 자리가 여기다"* 라고만 하고 넘어갔던 그 자리다.


# %%
def causal_mask(L):
    """(L, L) bool. True 인 칸이 '가려질 자리' 다 = 자기보다 뒤에 있는 자리."""
    return torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)


# %% [markdown]
# ## 7. 자가 점검 도우미 — `01_block.py` 가 쓴다
#
# ⚠️ **스스로 채워 보기 전에 아래 정답 함수들의 본문을 열어 보지 말 것.**


# %%
def 정답_잔차(x, sublayer_out):
    return x + sublayer_out


def 정답_잔차LN(x, sublayer_out, ln):
    return ln(x + sublayer_out)


def 정답_ffn(d, d_ff):
    return nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))


# %%
def 확인(이름, 내값, 정답, 힌트=""):
    """내가 채운 값을 점검하고 **다음 칸에서 쓸 값을 돌려준다.**

    틀렸으면 정답을 돌려준다 — 한 칸에서 막혀도 노트북이 끝까지 돌아가게.
    고쳐 넣고 이 칸만 다시 돌리면 ✓ 로 바뀐다.
    """
    if 내값 is None:
        print(f"✗ {이름} — 아직 안 채웠다.  {힌트}")
    elif not torch.is_tensor(내값):
        print(f"✗ {이름} — 텐서가 아니다 ({type(내값).__name__}).  {힌트}")
    elif 내값.shape != 정답.shape:
        print(f"✗ {이름} — 모양이 {tuple(내값.shape)} 인데 {tuple(정답.shape)} 여야 한다.  {힌트}")
    elif not torch.allclose(내값, 정답, atol=1e-5):
        print(f"✗ {이름} — 모양은 맞는데 값이 다르다.  {힌트}")
    else:
        print(f"✓ {이름}  {tuple(내값.shape)}")
        return 내값
    print("   → 지금은 정답으로 이어서 간다. 위를 고치고 이 칸을 다시 돌리면 ✓ 가 뜬다.")
    return 정답


# %%
def 모듈확인(이름, 내모듈, d, d_ff, 힌트=""):
    """내가 만든 FFN 이 제대로 생겼는지 본다. 모듈을 돌려준다."""
    정답 = 정답_ffn(d, d_ff)
    if 내모듈 is None:
        print(f"✗ {이름} — 아직 안 채웠다.  {힌트}")
        return 정답
    if not isinstance(내모듈, nn.Module):
        print(f"✗ {이름} — nn.Module 이 아니다 ({type(내모듈).__name__}).  {힌트}")
        return 정답
    try:
        out = 내모듈(torch.zeros(2, d))
    except Exception as e:
        print(f"✗ {이름} — 돌다가 멈췄다: {type(e).__name__}: {e}  {힌트}")
        return 정답
    if out.shape != (2, d):
        print(f"✗ {이름} — 나온 모양이 {tuple(out.shape)} 인데 (2, {d}) 여야 한다.  {힌트}")
        return 정답
    if n_params(내모듈) != n_params(정답):
        print(f"✗ {이름} — 파라미터가 {n_params(내모듈):,}개인데 {n_params(정답):,}개여야 한다.")
        print(f"   → 가운데를 {d_ff} 로 넓혔는지 확인한다.  {힌트}")
        return 정답
    print(f"✓ {이름}  파라미터 {n_params(내모듈):,}개")
    return 내모듈


# %% [markdown]
# ## 8. 이 파일만 돌려 보기 (자가 확인)
#
# **`01_block.py` 가 불러다 쓸 때는 이 칸이 실행되지 않는다** (`__name__` 이 다르다).

# %%
if __name__ == "__main__":
    d, d_ff, h, L = 768, 3072, 12, 6

    내 = 내블록(d, d_ff, h)
    토치 = torch_layer(d, d_ff, h)
    print("내 블록                    :", f"{n_params(내):>12,d}")
    print("nn.TransformerEncoderLayer :", f"{n_params(토치):>12,d}")

    x = torch.randn(1, L, d)
    print("\n모양 유지:", tuple(x.shape), "→", tuple(내(x).shape))

    pe = sinusoidal_pe(512, d)
    print("\n사인·코사인 이웃 유사도:",
          {k: round(v, 3) for k, v in neighbor_similarity(pe).items()})

    print("\ncausal mask (L=5) — True 가 가려질 자리:")
    print(causal_mask(5).int())
