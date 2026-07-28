"""
🟡 B-1. 응용 — GRU 모델과 훈련 루프를 직접 채운다  (보통 · 목표 50분)

지금까지는 `model.py` 에 미리 만들어 둔 클래스를 가져다 썼다. 편했지만,
**직접 써 보지 않으면 내 것이 되지 않는다.** 여기서는 처음부터 손으로 만든다.

    빈칸(TODO)은 6개. 위에서부터 하나씩 채우고 실행하면 된다.
    막히면 04_gru.py 와 model.py 를 열어 보자. 정답은 solutions/ 에 있다.

목표: 이 파일을 끝까지 돌려 **GRU 모델의 검증 정확도를 출력**하는 것.
"""

# %%
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor

VOCAB_SIZE, MAX_LEN, EMBED, HIDDEN = 2000, 100, 32, 32
BATCH, LR, EPOCHS = 64, 1e-3, 8
torch.manual_seed(42)

# %% [markdown]
# ## 준비 — 데이터 (여기는 다 되어 있다)

# %%
train, val = load_imdb(n_train=5000, n_val=2000)
ttok = [tokenize_en(t) for t in train["text"]]
vtok = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(ttok, max_size=VOCAB_SIZE)
X_train = pad_and_tensor(ttok, word2idx, MAX_LEN)
y_train = torch.tensor(train["label"], dtype=torch.float32)
X_val = pad_and_tensor(vtok, word2idx, MAX_LEN)
y_val = torch.tensor(val["label"], dtype=torch.float32)
print(f"준비 완료 — 사전 {len(word2idx)} · 훈련 {len(X_train)} · 검증 {len(X_val)}")


# %% [markdown]
# ## TODO 1~4 — GRU 분류기 만들기
#
# 부품 세 개를 순서대로 잇는다.
#
#     정수 문장 → [임베딩] → [GRU] → [Linear] → sigmoid → 확률
#
# 힌트
# - 임베딩: `nn.Embedding(사전크기, 임베딩차원, padding_idx=0)`
# - GRU:   `nn.GRU(입력차원, 은닉차원, batch_first=True)`
#          ⚠️ 입력차원은 **임베딩 차원**이다 (사전 크기가 아니다!)
# - 분류기: `nn.Linear(은닉차원, 1)` — 확률 하나를 낼 거니까 출력이 1

# %%
class MyGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        # TODO 1: 임베딩 층을 만든다. <pad>(0)는 학습하지 않도록 padding_idx 를 준다.
        self.embedding = None

        # TODO 2: GRU 층을 만든다. batch_first=True 를 잊지 말자.
        self.gru = None

        # TODO 3: 은닉 상태 하나를 받아 확률 하나로 줄이는 Linear 를 만든다.
        self.fc = None

    def forward(self, x):
        # x: (배치, 길이) 정수 인덱스
        x = self.embedding(x)              # → (배치, 길이, 임베딩차원)

        # TODO 4: GRU 에 통과시킨다.
        #   GRU 의 반환은 (output, h_n) — LSTM 과 달리 짝이 아니다.
        #     h_n 의 모양은 (층수, 배치, 은닉차원) 이고, 우리는 맨 위 층의 마지막이 필요하다.
        output, hidden = None, None

        last = hidden[-1]                  # (배치, 은닉차원) — 문장 전체의 요약
        return torch.sigmoid(self.fc(last)).squeeze(-1)


# %%
model = MyGRU(len(word2idx), EMBED, HIDDEN)
if model.embedding is None or model.gru is None or model.fc is None:
    raise SystemExit("→ TODO 1~3 을 먼저 채우자. (nn.Embedding · nn.GRU · nn.Linear)")

n_params = sum(p.numel() for p in model.parameters())
print(f"모델 생성 완료 — 파라미터 {n_params:,}개")
print("  (04_gru.py 에서 본 GRU 파라미터 수와 비슷하게 나오면 잘 만든 것이다)")

# 만들자마자 한 번 통과시켜 본다. 여기서 터지면 위 TODO 를 다시 보자.
with torch.no_grad():
    test_out = model(X_val[:4])
print(f"시험 통과 — 출력 shape {tuple(test_out.shape)}  (배치 4개니까 (4,) 여야 한다)")


# %% [markdown]
# ## TODO 5~6 — 훈련 루프 채우기
#
# 앞에서 배운 학습 5단계를 그대로 쓴다.
#
#     ① 예측한다        pred = model(xb)
#     ② 손실을 잰다      loss = criterion(pred, yb)
#     ③ 기울기를 지운다   optimizer.zero_grad()
#     ④ 기울기를 구한다   loss.backward()
#     ⑤ 가중치를 고친다   optimizer.step()
#
# ⚠️ ③을 빼먹으면 기울기가 **누적**돼 훈련이 이상해진다. 순서에 주의하자.

# %%
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        # TODO 5: 위 ①~⑤ 를 순서대로 쓴다. (5줄이면 된다)

        total += loss.item()

    model.eval()
    with torch.no_grad():
        # TODO 6: 검증 정확도를 구한다.
        #   힌트: model(X_val) 은 0~1 확률이다. 0.5 보다 크면 긍정으로 친다.
        #         맞은 비율 = ((확률 > 0.5) 과 y_val 이 같은 개수) / 전체
        acc = None

    best_acc = max(best_acc, acc)
    print(f"에포크 {epoch+1:2d}: 훈련 손실 {total/len(train_loader):.4f} · 검증 정확도 {acc:.4f}")

print(f"\n최고 검증 정확도: {best_acc:.4f}")


# %% [markdown]
# ## 확인 — 잘 됐나?
#
# - 손실이 **에포크마다 내려가고** 있는가? (오르기만 하면 TODO 5의 순서를 다시 보자)
# - 정확도가 0.5(찍기)보다 **뚜렷하게 높은가**? 0.5 근처에 머문다면 모델이 아무것도 못 배운 것이다.
# - A-1 에서 잰 GRU 결과와 **비슷한 범위**인가? 크게 다르면 어딘가 다르게 만든 것이다.
#
# ⚠️ 정확한 숫자를 맞히려 하지 말자. A-1 에서 봤듯 같은 설정도 실행마다 흔들린다.
#
# ## 더 해 보기
#
# 1. `nn.GRU` 를 `nn.LSTM` 으로 바꿔 보자. **어디가 터지는가?** 어떻게 고쳐야 하나?
#    (02_lstm_swap.py 에서 본 '받는 모양' 이야기다 — 이번엔 스스로 고쳐 보자)
# 2. `hidden[-1]` 대신 `output[:, -1, :]` 를 써 보자. 결과가 달라지는가? 왜?
# 3. `padding_idx=0` 을 빼면 어떻게 되나? 정확도가 눈에 띄게 달라지는가?
