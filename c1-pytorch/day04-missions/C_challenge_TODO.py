"""
🔴 C. 도전 — 게이트 안을 들여다본다  (어려움 · 목표 60분+)

아래 세 갈래 중 **하나 이상**을 골라 한다. 셋 다 하면 훌륭하다.
어렵게 느껴지는 게 정상이다. 답을 못 내도 **어디까지 갔는지**가 남는다.

  C-1. LSTM 한 스텝을 손으로 구현해 nn.LSTM 과 같은 값이 나오는지 검증   ← 가장 확실한 이해
  C-2. 훈련된 모델의 게이트 값을 꺼내 본다 — 모델이 언제 문을 여닫는가   ← 가장 재미있음
  C-3. MAX_LEN 을 300으로 늘려, 게이트의 이점이 커지는지 재본다          ← A-2 (d)의 숙제

정답은 solutions/ 에 있다. 먼저 30분은 스스로 붙잡아 보자.
"""

# %%
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))

torch.manual_seed(0)


# %% [markdown]
# ---
# # C-1. LSTM 한 스텝을 손으로
#
# `nn.LSTM` 이 내부에서 하는 계산을 **우리 손으로 똑같이** 해 본다.
# RNN 셀을 손으로 만들어 본 것의 LSTM 판이다. 수식은 이렇다.
#
# ```
# 게이트 3개 + 후보값 1개 (모두 같은 방식으로 한꺼번에 계산된다)
#     i_t = σ(W_ii·x_t + b_ii + W_hi·h_{t-1} + b_hi)      input  게이트
#     f_t = σ(W_if·x_t + b_if + W_hf·h_{t-1} + b_hf)      forget 게이트
#     g_t = tanh(W_ig·x_t + b_ig + W_hg·h_{t-1} + b_hg)   후보 셀 상태 (c̃)
#     o_t = σ(W_io·x_t + b_io + W_ho·h_{t-1} + b_ho)      output 게이트
#
# 상태 갱신 — 여기가 오늘의 핵심
#     c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t       ← 곱셈이 아니라 **덧셈**으로 이어진다
#     h_t = o_t ⊙ tanh(c_t)
# ```
#
# ⚠️ 파이토치는 `weight_ih_l0` 하나에 **네 벌(i, f, g, o)을 세로로 붙여** 저장한다.
#   그래서 shape 이 `(4×은닉, 입력)` 이다. 순서는 **i · f · g · o** 로 정해져 있다.

# %%
H, E = 4, 3                                   # 은닉 4 · 입력 3 (작게 잡아야 눈으로 본다)
cell = nn.LSTM(E, H, batch_first=True)

W_ih = cell.weight_ih_l0        # (4H, E)
W_hh = cell.weight_hh_l0        # (4H, H)
b_ih = cell.bias_ih_l0          # (4H,)
b_hh = cell.bias_hh_l0          # (4H,)
print(f"  W_ih {tuple(W_ih.shape)} · W_hh {tuple(W_hh.shape)} · 은닉 H={H}")
print(f"  → 4H = {4*H} 이 i, f, g, o 네 벌이 붙어 있다는 뜻이다\n")

x_t = torch.randn(1, 1, E)                    # (배치 1, 길이 1, 입력 E)
h_prev = torch.zeros(1, 1, H)
c_prev = torch.zeros(1, 1, H)

# TODO C-1-a: 위 수식대로 i, f, g, o 를 직접 계산한다.
#   힌트 ① 한꺼번에 계산한 뒤 4등분하는 것이 편하다:
#            gates = x_t.view(-1) @ W_ih.T + b_ih + h_prev.view(-1) @ W_hh.T + b_hh
#            i, f, g, o = gates.chunk(4)        # 순서는 i, f, g, o
#   힌트 ② torch.sigmoid / torch.tanh 를 쓴다.
i_t = f_t = g_t = o_t = None

# TODO C-1-b: c_t 와 h_t 를 구한다. (⊙ 는 원소별 곱 = 그냥 * 이다)
c_t = None
h_t = None

# 검증 — nn.LSTM 이 낸 값과 같은가?
with torch.no_grad():
    out, (h_ref, c_ref) = cell(x_t, (h_prev, c_prev))

if h_t is not None:
    print("  내 손계산 h_t :", [round(v, 4) for v in h_t.flatten().tolist()])
    print("  nn.LSTM  h_t :", [round(v, 4) for v in h_ref.flatten().tolist()])
    print("  일치? ", torch.allclose(h_t.flatten(), h_ref.flatten(), atol=1e-6))
else:
    print("  (TODO C-1-a, C-1-b 를 채우면 검증이 돌아간다)")

# %% [markdown]
# ### C-1 확인 질문
# 1. `f_t` 를 강제로 1로, `i_t` 를 0으로 두면 `c_t` 는 어떻게 되는가? 직접 넣어 보자.
#    → 이것이 "덧셈 고속도로"가 열린 상태다.
# 2. 반대로 `f_t = 0` 이면? 모델은 무엇을 하고 있는 셈인가?
# 3. `g_t` 만 tanh 이고 나머지는 sigmoid 인 이유는? (힌트: 게이트는 '얼마나', 후보는 '무엇을')


# %% [markdown]
# ---
# # C-2. 훈련된 모델의 게이트 값 들여다보기
#
# 게이트는 **매 단어마다 다시 계산된다.** 그러니 모델이 어떤 단어에서 문을 열고 닫는지
# 볼 수 있다. 훈련된 LSTM에서 forget 게이트 값을 단어별로 꺼내 보자.
#
# 방법: `nn.LSTM` 은 게이트를 밖으로 내주지 않는다. **C-1에서 만든 손계산을 그대로 써서**
# 훈련된 가중치로 한 스텝씩 직접 굴리면 된다.

# %%
# TODO C-2-a: A-2 처럼 LSTM 을 하나 훈련시킨다 (또는 B-1 에서 만든 것을 재사용).
#
# TODO C-2-b: 문장 하나를 골라 한 단어씩 손계산으로 굴리면서,
#             각 단어에서의 forget 게이트 평균값 f_t.mean() 을 기록한다.
#
# TODO C-2-c: 단어와 f 값을 나란히 출력한다. 이런 모양이면 된다.
#
#     단어          forget 평균
#     the           0.52
#     movie         0.55
#     terrible      0.31      ← 감정 단어에서 값이 튀는가?
#
# 볼 것:
#   · <pad> 자리에서 f 값이 어떤가? (모델이 빈칸을 '무시'하기로 배웠을까?)
#   · 감정이 실린 단어(great, terrible)에서 다른 단어와 다르게 움직이는가?
#   · ⚠️ 사람이 보기에 그럴듯한 패턴을 **찾고 싶어지는** 것을 조심하자.
#     아무 단어나 몇 개 뽑아 비교해 보고, 정말 차이가 있는지 따져 보자.


# %% [markdown]
# ---
# # C-3. MAX_LEN 을 늘리면 게이트의 이점이 커지는가
#
# A-2 (d)에서 이런 결과가 나왔다 — 긴 리뷰에서 LSTM의 우위가 **더 크지 않았다.**
# 그리고 그 이유를 `MAX_LEN=100` 때문이라고 짐작했다. 짐작을 **검증**하자.

# %%
# TODO C-3-a: A-1 의 비교를 MAX_LEN = 300 으로 다시 돌린다.
#             (시간이 3배쯤 걸린다. EPOCHS 를 6으로 줄여도 된다)
#
# TODO C-3-b: MAX_LEN 100 일 때와 300 일 때의 **RNN↔LSTM 격차**를 비교한다.
#             격차가 커졌다면 짐작이 맞은 것이다.
#
# TODO C-3-c: 여기서도 규칙은 같다. 시드를 2개 이상 써서 **잡음을 먼저 재고** 판단하자.
#
# 예상해 볼 것 (먼저 적고 시작하자):
#   · MAX_LEN 300 이면 RNN 은 더 나아질까, 더 나빠질까? 왜?
#   · LSTM 은?
#   · 둘의 격차는?


# %% [markdown]
# ---
# ## 회고 때 나눌 것
#
# - C-1을 끝냈다면: `f_t = 1, i_t = 0` 일 때 무슨 일이 일어나는지 **말로** 설명해 보자.
#   이걸 설명할 수 있으면 오늘 이론은 다 이해한 것이다.
# - C-2를 했다면: 가장 인상적인 게이트 패턴을 공유하자. (없었다면 "없었다"도 결과다)
# - C-3을 했다면: 짐작이 맞았는가? 틀렸다면 다음 짐작은 무엇인가?
