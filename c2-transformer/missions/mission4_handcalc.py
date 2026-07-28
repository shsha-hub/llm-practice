# %% [markdown]
# # 🔴 미션 4 (선택 · ~25분) — 손으로 계산하고 코드로 대조
#
# 오전에 점수 `2.0 / 1.0 / 0.5` 로 손계산을 했습니다.
# 이번엔 **내가 고른 점수**로 손계산한 뒤, 코드로 맞는지 확인합니다.
#
# 제출: 손계산 과정(종이·메모) + 코드 출력이 **일치**함을 확인한 캡처.

# %%
import math
import torch

# %% [markdown]
# ## 1. 내 점수를 정하고, 손으로 계산한다
# 먼저 **계산기로** 아래를 채운 뒤 코드를 돌립니다. (순서 지킬 것)
#
# ```
# 내 점수:  s₁ = ____   s₂ = ____   s₃ = ____
#
# 1단계  e^s₁ = ____    e^s₂ = ____    e^s₃ = ____      합 = ____
# 2단계  α₁ = ____      α₂ = ____      α₃ = ____        합 = ____  (1이어야 한다)
# 3단계  context = α₁·h₁ + α₂·h₂ + α₃·h₃ = [ ____ , ____ ]
# ```

# %%
MY_SCORES = [1.5, 0.5, -1.0]      # TODO: 내 점수로 (음수도 넣어 보자)

H = [[1, 0], [0, 1], [1, 1]]      # h₁, h₂, h₃ (오전과 같음)

# %% [markdown]
# ## 2. 코드로 확인

# %%
exps = [math.exp(s) for s in MY_SCORES]
total = sum(exps)
alphas = [e / total for e in exps]

print("1단계  지수:", [f"{e:.3f}" for e in exps], " 합", f"{total:.3f}")
print("2단계  가중치:", [f"{a:.3f}" for a in alphas], " 합", f"{sum(alphas):.3f}")

ctx = [sum(alphas[i] * H[i][d] for i in range(3)) for d in range(2)]
print("3단계  context:", [f"{c:.3f}" for c in ctx])

# %% [markdown]
# ## 3. PyTorch의 softmax와도 같은지

# %%
t = torch.softmax(torch.tensor(MY_SCORES), dim=0)
print("torch.softmax:", [f"{v:.3f}" for v in t.tolist()])
# 위 2단계 값과 같아야 한다.

# %% [markdown]
# ## 생각해 볼 것
#
# - 점수에 **음수**를 넣었는데도 가중치가 전부 0 이상인가? 왜 그런가?
# - 점수를 전부 **똑같게**(예: 1.0, 1.0, 1.0) 주면 가중치는 어떻게 되나?
#   그건 미션 2의 **균등(평균)**과 같은 상황이다.
