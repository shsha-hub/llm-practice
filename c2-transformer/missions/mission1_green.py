# %% [markdown]
# # 🟢 미션 1 (필수) — 히트맵 뽑고 · 가중치 읽고 · 왜인지 쓰기
#
# 세 가지를 한다:
# 1. **뽑기** — 내가 고른 날짜를 **재정렬 포맷 하나** + **OOD(처음 보는 입력) 하나**로 히트맵
# 2. **읽기** — 재정렬 히트맵에서 출력 연도 글자의 **가중치를 숫자로** 확인
# 3. **쓰기** — 어느 출력이 어느 입력을 봤는지 + OOD는 왜 흔들리는지 **두 줄**
#
# 제출: 히트맵 1장(`my_alignment.png`) + 두 줄.

# %%
import sys
# 노트북(cwd=이 폴더)에서도, .py 를 상위에서 돌려도 attn/koplot 을 찾게 한다.
# ⚠️ __file__ 은 주피터 커널에 정의되지 않는다 — 노트북에서 NameError 로 죽는다.
sys.path[:0] = ["..", "."]

import matplotlib
# 노트북이면 그림이 셀 아래에 바로 뜬다. 터미널이면 화면 없이 파일로만 저장한다.
NOTEBOOK = "ipykernel" in sys.modules
if not NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koplot; koplot.use_korean()

from dates import DateData
from model import train, translate
from viz import plot_alignment, print_weight_row

data = DateData(n=8000, seed=0)
enc, dec = train(data, epochs=6)

# %% [markdown]
# ## 1. 재정렬 포맷으로 뽑기 — 여기 두 줄만 바꾼다
# 재정렬 포맷: `27 July 2026` · `July 27, 2026` · `27/07/2026` 꼴

# %%
MY_DATE = "3 March 2001"       # TODO: 내 날짜(재정렬 포맷)로

pred, A = translate(enc, dec, data, MY_DATE)
plot_alignment(MY_DATE, pred, A)
plt.savefig("my_alignment.png", dpi=120)
print(f"{MY_DATE} -> {pred}")
if not NOTEBOOK:
    plt.close()                # 노트북에서는 닫지 않는다 — 닫으면 셀에 안 보인다

# %% [markdown]
# ## 2. 가중치를 숫자로 — 출력 연도 첫 글자(0번째)

# %%
print_weight_row(MY_DATE, pred, A, out_idx=0)
# TODO(쓰기): 값이 가장 큰 입력은 어디였나? 그게 왜 그 위치인가? 한 줄.

# %% [markdown]
# ## 3. OOD — 처음 보는 입력 하나
# 예: `Mar 3 2001`(약자) · `2001.03.03`(점) · `31 March 2001`(3월 31일이 있나?) 중 하나

# %%
MY_OOD = "Mar 3 2001"          # TODO: OOD 하나 골라 바꾸기
p, a = translate(enc, dec, data, MY_OOD)
print(f"{MY_OOD} -> {p}")
# TODO(쓰기): 제대로 나왔나 흔들렸나? 모델이 이 입력을 왜 어려워하나? 한 줄.
