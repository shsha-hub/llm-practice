"""
🟢 A-1. 관찰 — 세 모델을 내 손으로 비교하고 기록한다  (가벼움 · 목표 40분)

강사와 함께 본 비교를 **여러분 컴퓨터에서 다시** 한다. 그리고 여기서는
숫자를 하나만 적는 게 아니라, **05번에서 배운 절차대로** 기록한다.

    ① 같은 설정을 여러 번 돌려 → 흔들림(잡음)을 잰다
    ② 설정을 바꿔 → 차이(신호)를 잰다
    ③ 신호가 잡음보다 뚜렷할 때만 "낫다"고 말한다

⚠️ **이 파일은 그대로 실행하면 돌아간다.** 고칠 것은 없다.
   대신 **표를 채우고 판단하는 것**이 여러분 몫이다. 결과를 보고 생각하는 미션이다.
"""

# %%
import statistics
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                      # 노트북 셀에는 __file__ 이 없다
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))  # day04/ 의 모듈을 쓰기 위해

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor
from model import ALL_MODELS, count_params

VOCAB_SIZE, MAX_LEN, EMBED, HIDDEN = 2000, 100, 32, 32
BATCH, LR, EPOCHS = 64, 1e-3, 8

# 여기를 바꿔 가며 실험한다
MODELS = ALL_MODELS          # [("RNN", RnnClassifier), ("LSTM", ...), ("GRU", ...)]
SEEDS = [42, 43, 44]


# %% [markdown]
# ## 준비 — 데이터 (세 모델에 똑같은 조건)

# %%
train, val = load_imdb(n_train=5000, n_val=2000)
ttok = [tokenize_en(t) for t in train["text"]]
vtok = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(ttok, max_size=VOCAB_SIZE)
X_train = pad_and_tensor(ttok, word2idx, MAX_LEN)
y_train = torch.tensor(train["label"], dtype=torch.float32)
X_val = pad_and_tensor(vtok, word2idx, MAX_LEN)
y_val = torch.tensor(val["label"], dtype=torch.float32)
print(f"사전 {len(word2idx)} · 훈련 {len(X_train)} · 검증 {len(X_val)}")


# %%
def train_once(ModelClass, seed, epochs=EPOCHS):
    """한 번 훈련하고 (최고 정확도, 파라미터 수, 걸린 시간)을 돌려준다."""
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
    m = ModelClass(len(word2idx), EMBED, HIDDEN)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.BCELoss()
    best, t0 = 0.0, time.time()
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            best = max(best, ((m(X_val) > 0.5).float() == y_val).float().mean().item())
    return best, count_params(m), time.time() - t0


# %% [markdown]
# ## 실험 — 세 모델 × 세 시드
#
# 시간이 좀 걸린다(CPU 기준 대략 2~4분). 돌아가는 동안 아래 「기록표」를 미리 그려 두자.

# %%
table = {}
for name, ModelClass in MODELS:
    scores, params, secs = [], None, []
    for seed in SEEDS:
        acc, params, t = train_once(ModelClass, seed)
        scores.append(acc); secs.append(t)
        print(f"  {name:4s} seed {seed}: {acc:.3f}  ({t:.0f}초)")
    table[name] = {"scores": scores, "params": params, "time": statistics.mean(secs)}

# %% [markdown]
# ## 결과 정리

# %%
print(f"\n  {'모델':<6} {'세 번의 결과':<24} {'평균':>7} {'흔들림':>8} {'파라미터':>10} {'평균시간':>9}")
print("  " + "-" * 70)
for name, _ in MODELS:
    r = table[name]
    spread = max(r["scores"]) - min(r["scores"])
    print(f"  {name:<6} {str([f'{s:.3f}' for s in r['scores']]):<24} "
          f"{statistics.mean(r['scores']):>7.3f} {spread:>8.3f} {r['params']:>10,} {r['time']:>8.0f}초")

# %% [markdown]
# ## 판단 — 무엇을 말해도 되나
#
# 아래 출력이 **여러분이 정당하게 주장할 수 있는 것**의 목록이다.

# %%
noise = max(max(r["scores"]) - min(r["scores"]) for r in table.values())
print(f"\n  이 실험의 잡음(같은 설정 재실행 최대 흔들림) = {noise:.3f}\n")

names = [name for name, _ in MODELS]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ma, mb = statistics.mean(table[a]["scores"]), statistics.mean(table[b]["scores"])
        diff = abs(ma - mb)
        hi = a if ma > mb else b
        if diff > noise * 2:
            print(f"  {a} vs {b}: 차이 {diff:.3f} > 잡음×2 → ✅ '{hi} 가 낫다' 고 말해도 된다")
        else:
            print(f"  {a} vs {b}: 차이 {diff:.3f} ≤ 잡음×2 → ❌ 차이가 있다고 말할 수 없다")

print("""
  ⚠️ 여러분 숫자는 위 예시와 다를 수 있다. **판정 규칙만 같으면 된다.**
     "차이가 있다"와 "차이를 확인하지 못했다"는 다른 말이다. 후자를 정직하게 쓰는 연습을 하자.
""")


# %% [markdown]
# ## 기록표 — 직접 채우자
#
# | 모델 | 게이트 수 | 평균 정확도 | 흔들림 | 파라미터 | 평균 시간 |
# |------|-----------|-------------|--------|----------|-----------|
# | RNN  | 0         |             |        |          |           |
# | LSTM | 3         |             |        |          |           |
# | GRU  | 2         |             |        |          |           |
#
# **내가 내린 결론** (한 문장으로):
#
# > 이 데이터·이 설정에서 __________ 는 __________ 보다 낫다고 말할 수 있다.
# > 반면 __________ 와 __________ 사이는 차이를 확인하지 못했다.


# %% [markdown]
# ## 더 해 보기 (시간이 남으면)
#
# 위 `MODELS` · `SEEDS` 는 그대로 두고, 아래 설정을 하나씩 바꿔 다시 돌려 보자.
# **한 번에 하나만** 바꿔야 원인을 알 수 있다.
#
# 1. `HIDDEN = 64` — 은닉 차원을 키우면 세 모델의 순위가 바뀌는가?
# 2. `MAX_LEN = 200` — 문장을 길게 살리면 게이트의 이점이 커지는가?
#    (RNN은 길게 보여줘도 앞부분을 잊어서 별 도움이 안 됐다. LSTM은 다를까?)
# 3. `EPOCHS = 15` — 더 오래 훈련하면? 과적합은 언제 시작되는가?
#
# **회고 때 나눌 것**
# - 2번 실험이 오늘의 핵심과 가장 가깝다. 결과가 어땠는지 꼭 공유하자.
# - 시간(초)은 모델별로 얼마나 달랐나? 파라미터 수와 비례하던가?
