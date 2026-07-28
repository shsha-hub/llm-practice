"""
🟡 B-1. 모델과 훈련 루프를 직접 채우기 (응용 · 목표 45분)

[이 파일은 지금 그대로는 실행되지 않는다.] TODO 4곳을 채워야 돈다.
   에러 메시지가 힌트다. 특히 shape(모양) 에러는 "차원이 안 맞는다"는 신호다.

    python missions/B1_build_TODO.py

[밑천]
- 모델 조립: 04_rnn_model.py · model.py
- 훈련 루프: 05_train.py · (Day 1 학습 루프)
- 데이터: 03_imdb_pipeline.py 는 그대로 가져다 쓴다(이미 배운 것)

막히면 위 파일들을 열어 놓고 비교하며 채우자. 정답은 solutions/B1_build_solution.py.
"""

import sys
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor

torch.manual_seed(42)
VOCAB_SIZE, MAX_LEN, EMBED, HIDDEN = 2000, 100, 32, 32

# ----- 데이터 (이미 배운 것 — 그대로) -----
train, val = load_imdb(n_train=5000, n_val=2000)
tr_toks = [tokenize_en(t) for t in train["text"]]
va_toks = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(tr_toks, max_size=VOCAB_SIZE)
Xtr = pad_and_tensor(tr_toks, word2idx, MAX_LEN)
ytr = torch.tensor(train["label"], dtype=torch.float32)
Xva = pad_and_tensor(va_toks, word2idx, MAX_LEN)
yva = torch.tensor(val["label"], dtype=torch.float32)
tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
vl = DataLoader(TensorDataset(Xva, yva), batch_size=64)


# ===== TODO 1 & 2: 모델 =====
class MyRnn(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # TODO 1: 세 레이어를 선언하라. 차원을 맞추는 게 핵심이다.
        #   - self.embedding : 정수 vocab_size개를 EMBED 차원 벡터로 (padding_idx=0 잊지 말 것)
        #   - self.rnn       : EMBED 차원 입력을 HIDDEN 차원 은닉으로 (batch_first=True)
        #   - self.fc        : HIDDEN 차원을 1개 점수로
        raise NotImplementedError("TODO 1: __init__ 의 세 레이어를 채우세요")

    def forward(self, x):
        # TODO 2: 순전파. x(배치,길이) → 임베딩 → RNN → 마지막 은닉 → 점수 → 확률
        #   힌트: output, hidden = self.rnn(...) 에서 hidden[-1] 이 "마지막 은닉"
        #        마지막에 torch.sigmoid(...).squeeze(-1) 로 (배치,) 확률을 만든다
        raise NotImplementedError("TODO 2: forward 를 채우세요")


model = MyRnn(len(word2idx))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----- 훈련 -----
for epoch in range(8):
    model.train()
    for xb, yb in tl:
        # ===== TODO 3: 한 배치 학습 5단계 (Day 1에서 배운 그대로) =====
        #   zero_grad → forward(pred) → loss 계산 → backward → step
        raise NotImplementedError("TODO 3: 학습 5단계를 채우세요")

    # ----- 평가 -----
    model.eval(); correct = n = 0
    with torch.no_grad():
        for xb, yb in vl:
            # ===== TODO 4: 검증 정확도 =====
            #   pred = model(xb); 0.5보다 크면 긍정으로 보고 정답과 비교해 correct 누적
            raise NotImplementedError("TODO 4: 검증 정확도 계산을 채우세요")
    print(f"에포크 {epoch+1}: 검증정확도 {correct/n:.4f}")

print("완성했다면 정확도가 0.5(찍기)보다 확실히 높게 나온다.")
