# %% [markdown]
# # 🟡 미션 2 (~30분) — 가중치를 **균등**하게 바꾸면?
#
# 오전에 이렇게 유도했습니다: *"평균은 안 된다 — 다 똑같이 중요해지니까."*
# 정말 그런지 **직접 확인**합니다.
#
# 학습된 모델은 그대로 두고, **번역할 때만** 어텐션을 끄고
# 모든 입력을 똑같이(1/L) 보게 만듭니다.
#
# 제출: 비교 표 캡처 + "무엇이 어떻게 망가졌나" **한 줄**.

# %%
import sys
# 노트북(cwd=이 폴더)에서도, .py 를 상위에서 돌려도 attn/koplot 을 찾게 한다.
# ⚠️ __file__ 은 주피터 커널에 정의되지 않는다 — 노트북에서 NameError 로 죽는다.
sys.path[:0] = ["..", "."]

from dates import DateData
from model import train, translate

data = DateData(n=8000, seed=0)
enc, dec = train(data, epochs=6)

# %% [markdown]
# ## 어텐션 사용 vs 균등(평균)

# %%
tests = ["27 July 2026", "July 27, 2026", "27/07/2026", "2026년 7월 27일"]

print(f"{'입력':18s} {'어텐션 사용':14s} 균등(평균)")
print("-" * 52)
for s in tests:
    normal, _ = translate(enc, dec, data, s)
    uniform, _ = translate(enc, dec, data, s, uniform=True)   # ← 어텐션 끔
    print(f"{s:18s} {normal:14s} {uniform}")

# %% [markdown]
# ## 내 날짜로도

# %%
MY_DATE = "3 March 2001"        # TODO: 내 날짜로

n, _ = translate(enc, dec, data, MY_DATE)
u, _ = translate(enc, dec, data, MY_DATE, uniform=True)
print(f"어텐션: {n}   /   균등: {u}")
# TODO(쓰기): 균등으로 바꾸니 무엇이 어떻게 망가졌나? 한 줄.

# %% [markdown]
# ## ⚠️ 결과를 해석할 때 조심할 것
#
# 이 모델은 **어텐션이 있는 채로 학습**했습니다.
# 그러니 이 실험이 말해 주는 것은
#
# > "평균만 쓰는 모델은 아예 못 배운다"  ❌ (그건 여기서 확인 못 함)
# > **"이 모델이 실제로 '고르기'에 의존하고 있다"**  ✅
#
# 입니다. 고르는 능력을 빼앗으니 번역이 무너진다 — 그게 오늘 유도한 것의 증거입니다.
