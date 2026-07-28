"""
🔴 C. 도전 (목표 35분 · 3개 중 1개 이상)

[이 파일은 지금 그대로는 실행되지 않는다.] 고른 도전의 TODO를 채운다.
   먼저 끝난 사람은 여러 개 해도 좋다. 정답은 solutions/C_challenge_solution.py.

    python missions/C_challenge_TODO.py

세 도전 모두 오늘 배운 것에서 한 걸음씩 더 나간다.
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

torch.manual_seed(0)


# ============================================================
# (a) 손 구현 = 라이브러리  — 은닉 상태 재귀를 진짜 이해했는가
# ============================================================
# 02_rnn_cell_byhand.py 처럼, nn.RNN 안의 가중치를 꺼내 손으로 굴려
# nn.RNN 과 같은 마지막 은닉이 나오는지 확인하라.
def challenge_a():
    I, H, L = 4, 6, 7
    x = torch.randn(1, L, I)
    rnn = nn.RNN(I, H, batch_first=True)

    Wih, Whh = rnn.weight_ih_l0, rnn.weight_hh_l0
    bih, bhh = rnn.bias_ih_l0, rnn.bias_hh_l0

    h = torch.zeros(H)
    for t in range(L):
        # TODO (a): 은닉 상태 갱신식을 채워라.
        #   h = tanh( Wih·x_t + Whh·h + bih + bhh )
        #   힌트: 행렬곱은 @, x의 t번째 단어는 x[0, t]
        raise NotImplementedError("TODO (a): h 갱신식을 채우세요")

    _, lib_hidden = rnn(x)
    print("(a) 손 구현 == nn.RNN ?", torch.allclose(h, lib_hidden[-1, 0], atol=1e-6))


# ============================================================
# (b) 학습된 임베딩이 도움이 되나 — GloVe 로 초기화
# ============================================================
# 어제 만든 glove_50d_top30k.pt 로 임베딩을 초기화한 모델과, 랜덤 초기화 모델의
# 검증 정확도·수렴 속도를 비교하라. (작은 데이터에서 사전학습이 유리한가?)
# GloVe 는 50차원이므로 EMBED_DIM=50 으로 맞춘다.
def challenge_b():
    from imdb_data import load_imdb
    from textutils import tokenize_en, build_vocab, pad_and_tensor
    from model import IMDBRnn
    from torch.utils.data import TensorDataset, DataLoader

    # GloVe 파일은 어제(day02) 폴더에 있다.
    glove_path = _HERE.parents[2] / "day02_shared"  # ← TODO (b)에서 올바른 경로로
    # TODO (b):
    #   1) 어제 glove_50d_top30k.pt 를 torch.load 로 읽는다 (경로는 아래 힌트 참고)
    #   2) 랜덤 임베딩 모델 하나, GloVe로 embedding.weight 를 채운 모델 하나를 만든다
    #      (사전의 단어가 GloVe에 있으면 그 벡터를, 없으면 그대로 둔다)
    #   3) 둘을 같은 조건으로 몇 에포크 훈련해 검증 정확도를 비교한다
    #
    #   힌트(경로): day02 의 glove 는 가 있는 경로에
    #     즉  ../glove/glove_50d_top30k.pt
    raise NotImplementedError("TODO (b): GloVe 초기화 vs 랜덤 비교를 구현하세요")


# ============================================================
# (c) 마지막만 볼까, 다 볼까 — mean pooling
# ============================================================
# 분류에 hidden[-1](마지막 은닉) 대신, 모든 시간 위치 output 의 '평균'을 써서
# 정확도를 비교하라. 무엇을 '문장의 요약'으로 쓰는 게 나은가?
def challenge_c():
    from imdb_data import load_imdb
    from textutils import tokenize_en, build_vocab, pad_and_tensor
    from torch.utils.data import TensorDataset, DataLoader

    class MeanRnn(nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.embedding = nn.Embedding(vocab, 32, padding_idx=0)
            self.rnn = nn.RNN(32, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            e = self.embedding(x)
            output, hidden = self.rnn(e)     # output: (배치, 길이, 은닉)
            # TODO (c): hidden[-1] 대신 output 을 '시간 축(길이)'으로 평균 내서 쓰라.
            #   힌트: output.mean(dim=1) 이 (배치, 은닉) 이 된다
            raise NotImplementedError("TODO (c): mean pooling 으로 요약을 만드세요")

    print("(c) MeanRnn 을 05_train 방식으로 훈련해 hidden[-1] 방식과 정확도를 비교하라.")


if __name__ == "__main__":
    # 도전 하나를 골라 주석을 풀고 실행하라.
    challenge_a()
    # challenge_b()
    # challenge_c()
