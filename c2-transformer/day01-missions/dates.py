"""dates.py — 합성 날짜 데이터 만들기 + 문자 어휘.

[왜 합성(synthetic) 날짜인가]
어텐션이 "입력의 어디를 보는지"를 눈으로 보려면 **정렬이 선명한** 문제가 필요하다.
실제 번역(영-한)은 데이터·학습이 무거워 정렬이 흐리다. 날짜 정규화는
  "27 July 2026"  ->  "2026-07-27"
처럼 답이 규칙적이라, 몇 초 학습만으로도 정렬이 깨끗하게 나온다.

[핵심 — 입력 포맷을 일부러 섞는다]
출력은 언제나 YYYY-MM-DD. 그런데 입력은 연/월/일 순서가 제각각이다.
  27 July 2026   -> 2026-07-27   (연도가 입력의 '꼬리'에 있다)
  2026년 7월 27일 -> 2026-07-27   (연도가 입력의 '머리'에 있다)
이렇게 순서가 뒤바뀌는 포맷이 있어야, 어텐션이 대각선이 아니라
**필요한 곳으로 건너뛰며** 보는 게 히트맵에 드러난다.
"""

# %%
import random
import torch

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# 특수 토큰: 자리 채움 · 시작 · 끝
PAD, SOS, EOS = "\0", "\1", "\2"


# %% [markdown]
# ## 1. 날짜 하나 만들기 — 포맷 4종을 섞는다
#
# 네 포맷 중 셋(`fmt 0·1·2`)은 **연도가 뒤에** 오고, 하나(`fmt 3`)만 순서가 그대로다.
# 이 섞임이 오늘 실습의 전부다 — 섞이지 않으면 히트맵이 그냥 대각선이라
# "어텐션이 고른다"가 눈에 안 보인다.

# %%
def make_one(rng):
    """날짜 하나를 (입력문자열, 정답 YYYY-MM-DD)로 만든다. 포맷은 4종 중 무작위."""
    y = rng.randint(2000, 2030)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)                       # 28까지만 — 달마다 말일이 달라 생기는 예외 회피
    target = f"{y:04d}-{m:02d}-{d:02d}"
    fmt = rng.randint(0, 3)
    if fmt == 0:
        src = f"{d} {MONTHS[m-1]} {y}"           # 27 July 2026     (재정렬)
    elif fmt == 1:
        src = f"{MONTHS[m-1]} {d}, {y}"          # July 27, 2026    (재정렬)
    elif fmt == 2:
        src = f"{d:02d}/{m:02d}/{y}"             # 27/07/2026       (재정렬)
    else:
        src = f"{y}년 {m}월 {d}일"                 # 2026년 7월 27일   (단조 — 순서 그대로)
    return src, target


# %% [markdown]
# ## 2. 글자를 번호로 — 그리고 길이를 맞춘다
#
# 신경망은 글자를 모른다. 등장하는 문자를 모아 번호를 매기고(`src_stoi`),
# 짧은 입력의 뒤를 `PAD`(0)로 채워 길이를 통일한다.
#
# `encode_src` 의 `.get(c, 0)` 한 줄이 조용히 중요하다 — **배운 적 없는 글자를
# PAD로 넘긴다.** 그래서 §7의 OOD 입력(`2026.07.27` 의 점)이 터지지 않고
# "흔들리는" 결과를 낸다.

# %%
class DateData:
    """데이터 한 벌 + 문자↔번호 사전을 함께 들고 있는 그릇.

    사용:
        data = DateData(n=8000, seed=0)
        data.X, data.Y            # 학습용 정수 텐서
        data.encode_src("27/07/2026")
    """

    def __init__(self, n=8000, seed=0):
        rng = random.Random(seed)
        self.pairs = [make_one(rng) for _ in range(n)]

        # 입력·출력에 등장하는 문자를 모아 사전을 만든다
        src_chars = sorted(set("".join(s for s, _ in self.pairs)))
        tgt_chars = sorted(set("".join(t for _, t in self.pairs)))
        self.src_vocab = [PAD] + src_chars
        self.tgt_vocab = [PAD, SOS, EOS] + tgt_chars
        self.src_stoi = {c: i for i, c in enumerate(self.src_vocab)}
        self.tgt_stoi = {c: i for i, c in enumerate(self.tgt_vocab)}

        self.SRC_LEN = max(len(s) for s, _ in self.pairs)   # 가장 긴 입력에 맞춰 자리 채움
        self.TGT_LEN = len("YYYY-MM-DD") + 1                # 정답 10글자 + EOS

        self.X = torch.tensor([self.encode_src(s) for s, _ in self.pairs])
        self.Y = torch.tensor([self.encode_tgt(t) for _, t in self.pairs])

    def encode_src(self, s):
        # 학습에 없던 문자는 PAD(0)로 처리한다 — OOD(처음 보는 포맷) 입력도 터지지 않게.
        ids = [self.src_stoi.get(c, 0) for c in s]
        return ids + [0] * (self.SRC_LEN - len(ids))        # 뒤를 PAD(0)로 채운다

    def encode_tgt(self, t):
        return [self.tgt_stoi[SOS]] + [self.tgt_stoi[c] for c in t] + [self.tgt_stoi[EOS]]


# %% [markdown]
# ## 3. 눈으로 확인 — 실제로 어떤 짝이 나오나
#
# 노트북에서는 아래 셀이 그대로 돈다. 포맷 네 종류가 섞여 나오는지 보자.

# %%
if __name__ == "__main__":
    data = DateData(n=8, seed=0)
    print("입력 최대 길이:", data.SRC_LEN, "· 입력 어휘:", len(data.src_vocab),
          "· 출력 어휘:", len(data.tgt_vocab))
    for s, t in data.pairs:
        print(f"  {s:18s} -> {t}")
