"""model.py — 어텐션이 달린 seq2seq(작은 번역기)와 학습·추론 함수.

[구조 한눈에]
  인코더(Encoder)   입력 날짜를 읽어 글자마다 hidden 벡터를 남긴다.
  어텐션(Attention) 디코더가 한 글자를 낼 때, 인코더 hidden들 중 '지금 필요한 것'에
                    가중치를 크게 준다. (가중치 합 = 1)  ← 오늘의 핵심
  디코더(Decoder)   그 가중합(context)을 받아 다음 글자를 예측한다.

[정직성] 여기 어텐션은 Day 2에서 배울 Q·K·V의 '초기형'(Bahdanau, 2015)이다.
오늘은 "디코더가 매 시점 입력의 어디를 보는지"만 눈으로 본다. 정식화는 Day 2.
"""

# %%
import time
import torch
import torch.nn as nn

H = 64   # hidden 크기 — 작게 잡아 CPU로 몇 초면 학습된다


# %% [markdown]
# ## 1. 인코더 — 입력을 읽어 글자마다 흔적을 남긴다
#
# 입력 `27 July 2026` 의 글자 하나하나를 벡터로 바꾸고(`Embedding`),
# 양방향 GRU로 훑는다. 결과는 **글자 수만큼의 hidden 벡터**다.
#
# 여기가 중요하다 — 옛 seq2seq 는 이걸 **하나로 압축**해서 디코더에 넘겼다.
# 우리는 압축하지 않고 **전부 남겨 둔다**. 그래야 디코더가 골라 볼 것이 있다.

# %%
class Encoder(nn.Module):
    """입력을 읽어 글자마다 hidden을 남긴다. 양방향 GRU(앞/뒤 둘 다 읽음)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.gru = nn.GRU(32, H, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(self.emb(x))[0]          # (B, S, 2H) — 글자마다 하나씩


# %% [markdown]
# ## 2. 어텐션 — 오늘의 핵심 ★
#
# `forward` 안의 네 줄이 오늘 배운 네 단계 그대로다.
#
# | 코드 | 오전에 손으로 한 것 |
# |---|---|
# | `score = self.v(torch.tanh(q + k))` | 채점 — 입력 글자마다 점수 하나 |
# | `torch.softmax(score, dim=1)` | 점수 → 합이 1인 가중치 |
# | `(weights * enc_out).sum(1)` | 가중합 = context |
#
# `Wq`·`Wk`·`v` 는 **채점 방식을 학습으로 정하는** 부분이다.
# 내일(Day 2)은 이 채점기가 통째로 **내적** 한 번으로 바뀐다.

# %%
class Attention(nn.Module):
    """디코더의 지금 상태(query)와 인코더 hidden(key)들의 '궁합' 점수를 매겨,
    softmax로 0~1 가중치를 만들고 가중합(context)을 돌려준다."""

    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(H, H)                # 디코더 상태 변환
        self.Wk = nn.Linear(2 * H, H)            # 인코더 hidden 변환
        self.v = nn.Linear(H, 1)                 # 점수 하나로

    def forward(self, dec_h, enc_out):
        q = self.Wq(dec_h).unsqueeze(1)          # (B, 1, H)
        k = self.Wk(enc_out)                     # (B, S, H)
        score = self.v(torch.tanh(q + k)).squeeze(-1)   # (B, S) — 입력 글자마다 점수
        weights = torch.softmax(score, dim=1)    # (B, S) — 합이 1인 가중치 ★
        context = (weights.unsqueeze(-1) * enc_out).sum(1)   # (B, 2H) — 가중합
        return context, weights


# %% [markdown]
# ## 3. 디코더 — 한 글자 낼 때마다 어텐션을 다시 부른다
#
# `forward` 가 **출력 글자 하나**를 담당한다. 그래서 어텐션도 글자마다 새로 계산된다 —
# 히트맵의 행이 여러 개인 이유가 이것이다.
#
# `uniform_len` 은 실험용 스위치다. 이걸 주면 어텐션을 끄고 **전부 1/L 로 똑같이** 본다.
# 미션 2에서 이 스위치를 켜서 "고르기를 뺏으면 무너진다"를 확인한다.

# %%
class Decoder(nn.Module):
    """context를 받아 다음 글자를 예측한다. 한 글자씩(GRUCell)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.attn = Attention()
        self.cell = nn.GRUCell(32 + 2 * H, H)
        self.out = nn.Linear(H, vocab_size)

    def forward(self, y_prev, enc_out, h, uniform_len=None):
        """uniform_len 을 주면 어텐션을 끄고 **균등 가중치(=단순 평균)**를 쓴다.
        §4-2 "평균으로는 안 된다"를 직접 확인하는 실험용. (실제 길이만큼 1/L)"""
        if uniform_len is None:
            context, weights = self.attn(h, enc_out)
        else:
            weights = torch.zeros(enc_out.size(0), enc_out.size(1))
            weights[:, :uniform_len] = 1.0 / uniform_len       # 다 똑같이 본다
            context = (weights.unsqueeze(-1) * enc_out).sum(1)
        h = self.cell(torch.cat([self.emb(y_prev), context], dim=-1), h)
        return self.out(h), h, weights           # 예측 · 새 상태 · 어텐션 가중치


# %% [markdown]
# ## 4. 학습 — 정답을 한 글자씩 넣어 주며(teacher forcing)
#
# 안쪽 `for t in range(...)` 가 출력 글자를 하나씩 도는 부분이다.
# 다음 글자를 예측할 때 **모델의 예측이 아니라 정답**을 넣어 준다 —
# 초반에 한 글자 틀리면 그 뒤가 전부 무너지는 걸 막는 흔한 방법이다.
#
# 어텐션에는 "여기를 봐라"라는 정답이 **없다.** 번역만 맞히라고 시켰는데
# 정렬은 그 부산물로 **저절로** 생긴다 — 오늘 히트맵이 놀라운 이유다.

# %%
def train(data, epochs=6, batch=256, lr=2e-3, seed=0, log=print):
    """작은 번역기를 학습한다. CPU로 10초 안팎. (enc, dec)를 돌려준다."""
    torch.manual_seed(seed)
    enc, dec = Encoder(len(data.src_vocab)), Decoder(len(data.tgt_vocab))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=0)  # PAD 자리는 손실에서 뺀다

    X, Y = data.X, data.Y
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        total, nb = 0.0, 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            enc_out = enc(xb)
            h = torch.zeros(xb.size(0), H)
            loss = 0.0
            for t in range(yb.size(1) - 1):      # 정답을 한 글자씩 넣어주며(teacher forcing)
                logit, h, _ = dec(yb[:, t], enc_out, h)
                loss = loss + lossf(logit, yb[:, t + 1])
            loss = loss / (yb.size(1) - 1)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if log:
            log(f"  epoch {ep}  loss {total / nb:.3f}  ({time.time() - t0:.1f}s)")
    return enc, dec


# %% [markdown]
# ## 5. 번역 — 그리고 어텐션 행렬을 모아 온다
#
# 글자를 하나 낼 때마다 그 시점의 가중치 `w` 를 `atts` 에 쌓는다.
# 다 쌓으면 **행 = 출력 글자 · 열 = 입력 글자** 인 표가 되고, 그게 히트맵이다.

# %%
@torch.no_grad()
def translate(enc, dec, data, src, uniform=False):
    """입력 날짜 문자열을 번역하고, (예측문자열, 어텐션행렬)을 돌려준다.
    어텐션행렬 A: 행=출력 글자, 열=입력 글자, 값=얼마나 봤나(0~1).

    uniform=True 로 주면 어텐션을 끄고 **균등 가중치(평균)**로 번역한다 —
    "고르기"를 빼면 어떻게 되는지 보는 실험용."""
    enc.eval(); dec.eval()
    x = torch.tensor([data.encode_src(src)])
    enc_out = enc(x)
    h = torch.zeros(1, H)
    y = torch.tensor([data.tgt_stoi["\1"]])      # SOS로 시작
    chars, atts = [], []
    for _ in range(data.TGT_LEN):
        logit, h, w = dec(y, enc_out, h, uniform_len=len(src) if uniform else None)
        atts.append(w[0, :len(src)].tolist())    # 실제 글자 길이만큼만(PAD 제외)
        y = logit.argmax(-1)
        c = data.tgt_vocab[y.item()]
        if c == "\2":                            # EOS면 끝
            break
        chars.append(c)
    import numpy as np
    return "".join(chars), np.array(atts)
