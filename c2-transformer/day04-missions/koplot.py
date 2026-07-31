"""
koplot.py — matplotlib 그래프에서 한글이 깨지지 않게 해주는 도우미

[왜 필요한가]
matplotlib의 기본 폰트(DejaVu Sans)에는 한글 글자가 없다.
그래서 그래프 제목이나 축 이름에 한글을 쓰면 네모(□□□)로 나오고
"Glyph ... missing from font(s) DejaVu Sans" 경고가 쏟아진다.

해결은 두 단계다.
  1) 시스템에 한글 폰트를 설치한다   ← 터미널에서 한 번만
  2) matplotlib에게 그 폰트를 쓰라고 알려준다  ← 이 파일이 하는 일

[1단계 — 터미널에서 한 번만 실행]
    sudo apt update
    sudo apt install -y fonts-nanum
    rm -rf ~/.cache/matplotlib      # 폰트 목록 캐시를 지워야 새 폰트를 인식한다

[2단계 — 파이썬 코드에서]
    import koplot
    koplot.use_korean()             # 이 한 줄이면 끝

    import matplotlib.pyplot as plt
    plt.title("손실 곡선")           # 이제 한글이 제대로 나온다

[한글 폰트가 없어도]
    경고 대신 친절한 안내를 출력하고, 그래프는 영어 라벨로 정상 동작한다.
    (설치가 안 돼도 실습이 멈추지 않게 하기 위함)
"""

# %%
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 우선순위 순서. 위에 있는 것이 발견되면 그것을 쓴다.
#   - NanumGothic  : 리눅스에서 가장 흔함 (fonts-nanum 패키지)
#   - Noto Sans CJK: 최근 배포판 기본
#   - Malgun Gothic: 윈도우 기본 (윈도우에서 직접 돌릴 때)
#   - AppleGothic  : macOS
_CANDIDATES = [
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Malgun Gothic",
    "AppleGothic",
]

_INSTALL_HINT = """
[안내] 한글 폰트를 찾지 못했습니다. 그래프의 한글이 네모(□)로 보일 수 있습니다.
       터미널에서 아래를 한 번만 실행하면 해결됩니다:

           sudo apt update && sudo apt install -y fonts-nanum
           rm -rf ~/.cache/matplotlib

       그다음 파이썬을 다시 실행하세요.
       (설치하지 않아도 실습은 그대로 진행할 수 있습니다 — 라벨만 영어로 쓰면 됩니다)
"""


# %% [markdown]
# ## 1. 설치된 폰트 중에 한글 되는 것 찾기

# %%
def find_korean_font():
    """설치된 폰트 중 한글을 지원하는 것을 찾아 이름을 돌려준다. 없으면 None."""
    # fontManager.ttflist 에 시스템이 인식한 폰트들이 들어 있다
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in installed:
            return name
    return None


# %% [markdown]
# ## 2. 찾은 폰트를 matplotlib 기본값으로
#
# 폰트가 없어도 **멈추지 않는다** — 안내만 찍고 영어 라벨로 계속 간다.
# 실습이 폰트 문제로 중단되지 않게 하려는 것이다.

# %%
def use_korean(verbose=True):
    """matplotlib이 한글 폰트를 쓰도록 설정한다.

    Returns:
        str | None: 적용된 폰트 이름. 못 찾았으면 None.
    """
    font = find_korean_font()

    if font is None:
        if verbose:
            print(_INSTALL_HINT)
        return None

    # 기본 폰트를 한글 폰트로 지정
    plt.rcParams["font.family"] = font

    # [중요] 한글 폰트를 쓰면 마이너스 기호(−)가 깨지는 문제가 있다.
    #        False로 두면 일반 하이픈(-)을 써서 깨지지 않는다.
    plt.rcParams["axes.unicode_minus"] = False

    if verbose:
        print(f"[koplot] 한글 폰트 적용: {font}")
    return font


# %% [markdown]
# ## 3. 로그축 눈금을 ASCII 로
#
# [왜 필요한가]
# 로그축 눈금은 matplotlib 이 **mathtext**(수식 렌더러)로 그린다. 거기 쓰이는 마이너스는
# 일반 하이픈(-)이 아니라 유니코드 U+2212 인데, **NanumGothic 에 그 글자가 없다.**
# 그래서 한글 폰트를 적용한 채 로그축을 쓰면 지수의 마이너스가 □ 로 깨진다.
#
# `axes.unicode_minus = False` 는 **일반 축에만** 듣고 mathtext 에는 안 듣는다.
# 그래서 눈금 라벨 자체를 ASCII 문자열로 직접 만들어 준다.
#
#     ax.set_yscale("log")
#     koplot.ascii_log_axis(ax, "y")

# %%
def _ascii_pow10(v, _pos=None):
    """10의 거듭제곱을 ASCII 로. 1e-4 → '10^-4', 100 → '100'."""
    import math as _m

    if v <= 0:
        return ""
    e = int(round(_m.log10(v)))
    if abs(v - 10 ** e) > 1e-9 * max(1.0, v):
        return ""              # 10의 거듭제곱이 아닌 보조 눈금은 비운다
    if -2 <= e <= 3:
        return f"{10 ** e:g}"  # 0.01 ~ 1000 은 그냥 숫자로 읽는 게 낫다
    return f"10^{e}"           # 그 밖은 ASCII 지수 표기


def ascii_log_axis(ax, which="y"):
    """로그축 눈금 라벨을 ASCII 로 바꾼다. which: 'x' | 'y' | 'both'."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    for w in ("x", "y") if which == "both" else (which,):
        axis = getattr(ax, f"{w}axis")
        axis.set_major_formatter(FuncFormatter(_ascii_pow10))
        axis.set_minor_formatter(NullFormatter())
    return ax


# %% [markdown]
# ## 4. 자가 진단 — 한글과 음수 기호가 제대로 나오나
#
# `python koplot.py` 로 직접 실행하거나, 노트북에서 아래 셀을 돌리면 된다.
# 제목·축 이름의 한글이 네모(□)로 보이면 위 안내대로 폰트를 설치하자.

# %%
if __name__ == "__main__":
    import sys
    import matplotlib

    # 노트북이면 그림이 셀 아래에 바로 뜬다. 터미널이면 화면 없이 파일로만 저장한다.
    NOTEBOOK = "ipykernel" in sys.modules
    if not NOTEBOOK:
        matplotlib.use("Agg")   # WSL·서버에서 안전

    applied = use_korean()

    plt.figure(figsize=(4, 3))
    plt.plot([1, 2, 3, 4], [-1, 0.5, -0.3, 1.2], marker="o")
    plt.title("한글 제목 테스트")
    plt.xlabel("에포크")
    plt.ylabel("손실 (음수 포함)")
    plt.tight_layout()
    plt.savefig("koplot_test.png", dpi=100)
    if not NOTEBOOK:
        plt.close()

    if applied:
        print("koplot_test.png 를 열어보세요. 한글과 음수 기호가 정상이면 성공입니다.")
    else:
        print("koplot_test.png 를 만들었지만 한글은 깨져 보일 것입니다. 위 안내대로 폰트를 설치하세요.")
