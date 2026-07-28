"""
🟡 B-1 정답 — GRU 모델 + 훈련 루프

TODO 6개를 채운 결과다. **먼저 스스로 해 보고** 막힐 때만 열어 보자.
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
sys.path.insert(0, str(_HERE.parent.parent))       # day04/ 의 모듈

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor

VOCAB_SIZE, MAX_LEN, EMBED, HIDDEN = 2000, 100, 32, 32
BATCH, LR, EPOCHS = 64, 1e-3, 8
torch.manual_seed(42)

# %%
train, val = load_imdb(n_train=5000, n_val=2000)
ttok = [tokenize_en(t) for t in train["text"]]
vtok = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(ttok, max_size=VOCAB_SIZE)
X_train = pad_and_tensor(ttok, word2idx, MAX_LEN)
y_train = torch.tensor(train["label"], dtype=torch.float32)
X_val = pad_and_tensor(vtok, word2idx, MAX_LEN)
y_val = torch.tensor(val["label"], dtype=torch.float32)
print(f"준비 완료 — 사전 {len(word2idx)}")


# %%
class MyGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        # TODO 1 — 임베딩. padding_idx=0 으로 <pad> 자리는 학습하지 않는다
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # TODO 2 — GRU. 입력 차원은 **임베딩 차원**이다 (사전 크기가 아님!)
        #          batch_first=True 라야 (배치, 길이, 차원) 순서가 된다
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

        # TODO 3 — 은닉 상태 → 확률 하나
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)                    # (배치, 길이, 임베딩차원)

        # TODO 4 — GRU 는 (output, h_n) 을 돌려준다. LSTM 처럼 튜플이 아니다.
        output, hidden = self.gru(x)
        #   output (배치, 길이, 은닉)   : 맨 위 층이 매 시간마다 낸 은닉 상태
        #   hidden (층수, 배치, 은닉)   : 각 층이 마지막 시간에 낸 은닉 상태

        last = hidden[-1]                        # 맨 위 층의 마지막 = 문장 요약
        return torch.sigmoid(self.fc(last)).squeeze(-1)


# %%
model = MyGRU(len(word2idx), EMBED, HIDDEN)
print(f"파라미터 {sum(p.numel() for p in model.parameters()):,}개")
with torch.no_grad():
    print("시험 통과 — 출력 shape", tuple(model(X_val[:4]).shape))


# %%
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        # TODO 5 — 학습 5단계
        pred = model(xb)                 # ① 예측
        loss = criterion(pred, yb)       # ② 손실
        optimizer.zero_grad()            # ③ 지난 기울기 지우기 (빼먹으면 누적된다!)
        loss.backward()                  # ④ 기울기 계산
        optimizer.step()                 # ⑤ 가중치 수정
        total += loss.item()

    model.eval()
    with torch.no_grad():
        # TODO 6 — 검증 정확도
        probs = model(X_val)                             # 0~1 확률
        pred_label = (probs > 0.5).float()               # 0.5 기준으로 긍정/부정
        acc = (pred_label == y_val).float().mean().item()  # 맞은 비율

    best_acc = max(best_acc, acc)
    print(f"에포크 {epoch+1:2d}: 훈련 손실 {total/len(train_loader):.4f} · 검증 정확도 {acc:.4f}")

print(f"\n최고 검증 정확도: {best_acc:.4f}")
print("""
⚠️ 이 숫자는 실행할 때마다 조금씩 달라진다. A-1 에서 잰 GRU 범위 안에 들어오면 잘 만든 것이다.

[자주 하는 실수]
  · nn.GRU(vocab_size, hidden) — 사전 크기를 넣었다. 입력은 **임베딩 차원**이다.
  · batch_first=True 를 빼먹음 — 그러면 (길이, 배치, 차원) 순서라 결과가 엉킨다.
  · optimizer.zero_grad() 를 loss.backward() **뒤에** 씀 — 방금 구한 기울기를 지워 버린다.
  · squeeze(-1) 을 빼먹음 — (배치, 1) 과 (배치,) 가 안 맞아 BCELoss 에서 경고나 오류가 난다.
""")
