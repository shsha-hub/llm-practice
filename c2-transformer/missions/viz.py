"""viz.py — 어텐션 가중치를 보는 두 가지 방법.

  plot_alignment()    정렬(alignment) 히트맵 — 전체를 한눈에
  print_weight_row()  출력 한 글자의 가중치를 숫자로 — 합이 1인지 확인

히트맵 읽는 법:
  행(y) = 출력 글자(YYYY-MM-DD)   ·   열(x) = 입력 글자
  칸이 밝을수록 그 출력 글자를 낼 때 그 입력 글자를 '많이 봤다'.
"""

# %%
import matplotlib.pyplot as plt


# %% [markdown]
# ## plot_alignment — 어텐션 행렬을 그림 한 장으로
#
# `A` 는 그냥 숫자 표다. `imshow` 로 칸마다 색을 칠하면 그것이 히트맵이 된다.
# 눈금 라벨을 **글자로** 바꿔 주는 것이 핵심 — 그래야 "출력 `2` 가 입력 `2026` 을 봤다"가 읽힌다.

# %%
def plot_alignment(src, pred, A, ax=None):
    """입력 src, 예측 pred, 어텐션행렬 A(행=출력·열=입력)를 히트맵으로."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(A, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(list(src), fontsize=9)
    ax.set_yticks(range(len(pred)))
    ax.set_yticklabels(list(pred), fontsize=9)
    ax.set_xlabel("입력")
    ax.set_ylabel("출력")
    ax.set_title(f"{src}  →  {pred}")
    if own:
        plt.tight_layout()
    return ax


# %% [markdown]
# ## print_weight_row — 히트맵의 한 줄을 숫자로 펼치기
#
# 히트맵은 한눈에 보이지만 **값이 안 보인다**. 이 함수는 행 하나를 골라
# 입력 글자마다 값을 찍고, 합이 정말 1인지 보여 준다.
# 그림과 숫자가 같은 것이라는 걸 확인하는 용도다.

# %%
def print_weight_row(src, pred, A, out_idx):
    """한 출력 글자의 어텐션 가중치를 '숫자로' 찍는다. 히트맵의 한 행을 펼친 것.
    가중치는 입력 글자마다 하나씩이고 다 더하면 1이다."""
    row = A[out_idx]
    total = float(row.sum())
    print(f"출력 '{pred[out_idx]}' (출력 {out_idx}번째)의 어텐션 가중치 — 합 = {total:.2f}")
    for j, c in enumerate(src):
        bar = "█" * int(round(row[j] * 30))          # 값 크기를 막대로
        show = "'　'" if c == " " else f"'{c}'"
        print(f"  입력[{j:2d}] {show:5s} {row[j]:.2f}  {bar}")


# %% [markdown]
# ## 자가 진단 — 히트맵은 결국 '숫자 표'다
#
# 학습된 모델 없이, **손으로 지어낸 표**를 그대로 넣어 본다.
# 모델이 없어도 그림이 그려진다는 것 자체가 요점이다 —
# 히트맵은 어텐션이 아니라, **어텐션이 남긴 숫자를 색으로 칠한 것**뿐이다.
#
# 아래 `A` 는 "출력 앞은 입력 꼬리를, 출력 뒤는 입력 머리를 본다"를 손으로 적어 넣은 것이다.
# 실제 히트맵의 **계단 무늬**가 이 모양이다.

# %%
if __name__ == "__main__":
    import sys
    import numpy as np

    # 이 파일은 백엔드를 고르지 않는다 — 그건 부르는 쪽(01_… · missions/…)의 몫이다.
    # 여기서는 plt.show() 를 부르지 않으므로 창이 뜰 일이 없고, 저장은 어느 백엔드에서나 된다.
    NOTEBOOK = "ipykernel" in sys.modules

    import koplot; koplot.use_korean()

    src, pred = "27 Jul 2026", "2026-07-27"
    A = np.zeros((len(pred), len(src)))
    for i in range(len(pred)):
        j = (len(src) - 1 - i) if i < 4 else (i - 4)   # 앞 4글자는 꼬리를, 나머지는 머리를
        A[i, j] = 0.7
        A[i] += 0.3 / len(src)                          # 나머지는 옅게 — 합이 1이 되게

    print_weight_row(src, pred, A, out_idx=0)
    plot_alignment(src, pred, A)
    plt.savefig("viz_demo.png", dpi=110)
    print("\n저장: viz_demo.png  (지어낸 숫자다 — 모델 결과가 아니다)")
    if not NOTEBOOK:
        plt.close()
