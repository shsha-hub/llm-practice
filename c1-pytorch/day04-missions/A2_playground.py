"""
🟢 A-2. 놀이터 — RNN이 틀린 문장을 LSTM은 맞히나?  🎮 (가벼움 · 목표 40분)

RNN은 긴 리뷰의 앞부분을 잊는다. 그래서 틀리는 문장이 생긴다.

    "이 한계를 고치려면 무엇이 필요할까? → 내일 LSTM"

오늘이 그 내일이다. **같은 문장을 RNN과 LSTM에게 동시에 먹여** 답이 갈리는지 본다.

  (a) 같은 문장, 두 모델의 답을 나란히
  (b) RNN이 확신에 차서 틀린 리뷰 — LSTM은 어떤가
  (c) 반전 위치 실험 — 게이트가 앞쪽 기억을 지켜 주는가
  (d) 문장을 점점 길게 늘이면 어디서부터 무너지나

⚠️ **LSTM이 다 맞히지는 않는다.** 그게 정상이다.
   "어디까지 나아졌고 어디는 그대로인가"를 찾는 게 이 미션의 재미다.
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
sys.path.insert(0, str(_HERE.parent))

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor
from model import RnnClassifier, LstmClassifier

VOCAB_SIZE, MAX_LEN, EMBED, HIDDEN = 2000, 100, 32, 32

# %% [markdown]
# ## 준비 — 두 모델을 나란히 훈련시킨다 (2~3분)
#
# 놀려면 먼저 선수가 둘 있어야 한다. **완전히 같은 조건**으로 훈련한다.

# %%
train, val = load_imdb(n_train=5000, n_val=2000)
ttok = [tokenize_en(t) for t in train["text"]]
vtok = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(ttok, max_size=VOCAB_SIZE)
X_train = pad_and_tensor(ttok, word2idx, MAX_LEN)
y_train = torch.tensor(train["label"], dtype=torch.float32)
X_val = pad_and_tensor(vtok, word2idx, MAX_LEN)
y_val = torch.tensor(val["label"], dtype=torch.float32)


def train_model(ModelClass, epochs=8, seed=42):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    m = ModelClass(len(word2idx), EMBED, HIDDEN)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = ((m(X_val) > 0.5).float() == y_val).float().mean().item()
        if acc > best_acc:                       # 가장 좋았던 순간을 보관
            best_acc, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m, best_acc


rnn, rnn_acc = train_model(RnnClassifier)
lstm, lstm_acc = train_model(LstmClassifier)
print(f"준비 완료 — RNN {rnn_acc:.3f} · LSTM {lstm_acc:.3f}")


# %%
def predict(model, text):
    """영어 문장 하나 → 긍정일 확률. 훈련 때와 같은 파이프라인을 통과시킨다."""
    x = pad_and_tensor([tokenize_en(text)], word2idx, MAX_LEN)
    with torch.no_grad():
        return model(x).item()


def duel(text, label=""):
    """같은 문장을 두 모델에 먹이고 나란히 보여 준다."""
    pr, pl = predict(rnn, text), predict(lstm, text)
    mark_r = "긍정" if pr > 0.5 else "부정"
    mark_l = "긍정" if pl > 0.5 else "부정"
    flag = "  ← 답이 갈렸다!" if (pr > 0.5) != (pl > 0.5) else ""
    print(f"  RNN [{mark_r} {pr:.2f}] │ LSTM [{mark_l} {pl:.2f}]{flag}")
    print(f"     {label}{text[:80]}{'…' if len(text) > 80 else ''}")
    return pr, pl


# %% [markdown]
# ## (a) 같은 문장, 두 답
#
# 아래 문장을 **마음대로 바꿔 가며** 실행하자. 어떤 문장에서 둘의 답이 갈리는가?
#
# ⚠️ **먼저 알아둘 것** — 이 모델은 **긴 영화 리뷰(보통 100~300단어)로 훈련**됐다.
# 우리가 지어낸 한 줄짜리 문장은 모델이 평소 보던 것과 딴판이다.
# (앞쪽 패딩까지 감안하면, 10단어 문장은 90칸이 빈칸이고 10칸만 내용이다.)
# 그래서 **엉뚱한 답이 자주 나온다.** 여기서 나온 답으로 "이 모델은 못 쓴다"고 결론 내지 말자.
# 이건 어디까지나 **감을 잡는 놀이**이고, 제대로 된 증거는 (b)·(d)의 진짜 데이터에서 나온다.

# %%
print("\n(a) 같은 문장을 둘에게\n")
duel("this movie was absolutely wonderful and i loved every minute of it")
duel("terrible film boring acting and a complete waste of time")
duel("this movie was good")
duel("this movie was not good")                # ← not 을 이해할까?
duel("i expected a masterpiece but got a mess")


# %% [markdown]
# ## (b) RNN이 확신에 차서 틀린 리뷰 — LSTM은?
#
# 검증 데이터에서 **RNN이 크게 틀린** 리뷰를 뽑아, 같은 리뷰를 LSTM에게 준다.

# %%
with torch.no_grad():
    p_rnn = rnn(X_val)
    p_lstm = lstm(X_val)

rnn_wrong = ((p_rnn > 0.5).float() != y_val)
lstm_right = ((p_lstm > 0.5).float() == y_val)

n_rnn_wrong = int(rnn_wrong.sum())
n_rescued = int((rnn_wrong & lstm_right).sum())
n_lstm_wrong = int((~lstm_right).sum())
n_broken = int((~rnn_wrong & ~lstm_right).sum())

print(f"\n(b) 검증 {len(y_val)}편 중")
print(f"    RNN 이 틀림              : {n_rnn_wrong}")
print(f"    LSTM 이 틀림             : {n_lstm_wrong}")
print(f"    RNN 틀림 → LSTM 맞힘 ✅  : {n_rescued}   (게이트가 구해 낸 것)")
print(f"    RNN 맞힘 → LSTM 틀림 ❌  : {n_broken}   (오히려 놓친 것)")
print(f"    → 순이득 {n_rescued - n_broken}편")
print("\n    📌 구해 낸 것만 세면 안 된다. **놓친 것도 있다.** 둘을 같이 봐야 정직한 비교다.")

# 구해 낸 리뷰 중 RNN이 가장 확신했던 것 3편
rescued = (rnn_wrong & lstm_right).nonzero(as_tuple=True)[0]
if len(rescued) > 0:
    conf = (p_rnn[rescued] - 0.5).abs()
    top = rescued[conf.argsort(descending=True)][:2]
    for i in top:
        i = i.item()
        truth = "긍정" if y_val[i] == 1 else "부정"
        print(f"\n    [정답 {truth}] RNN {p_rnn[i]:.2f}(틀림) → LSTM {p_lstm[i]:.2f}(맞힘)")
        print(f"    원문: {val['text'][i][:180].strip()}…")


# %% [markdown]
# ## (c) 반전 위치 실험 — 반전이 앞에 있으면 어떻게 되나
#
# RNN은 **뒤쪽 반전**에 훨씬 민감하다. 마지막 은닉 상태만 보고 판단하기 때문이다.
# 게이트가 앞쪽 기억을 지켜 준다면, LSTM은 **앞뒤 차이가 줄어야** 한다.

# %%
GOOD = "the acting was great and the story was beautiful and i enjoyed it so much"
BAD = "but honestly it was terrible boring and a complete waste of time"

print("\n(c) 반전을 앞에 둘 때 vs 뒤에 둘 때\n")
r_end, l_end = duel(f"{GOOD} {BAD}", label="반전이 뒤 → ")
print()
r_front, l_front = duel(f"{BAD} {GOOD}", label="반전이 앞 → ")

gap_rnn = abs(r_end - r_front)
gap_lstm = abs(l_end - l_front)
print(f"\n    앞·뒤에 따라 답이 흔들린 폭")
print(f"      RNN  : {gap_rnn:.2f}")
print(f"      LSTM : {gap_lstm:.2f}")
if gap_lstm < gap_rnn:
    print("    → LSTM 쪽이 위치에 덜 흔들렸다. 앞쪽 정보가 더 살아남았다는 신호다.")
else:
    print("    → 이번엔 LSTM도 만만찮게 흔들렸다. 왜 그럴지 (d)를 보고 다시 생각해 보자.")
print("""    ⚠️ **문장 하나로 낸 결과다. 이걸로 결론 내면 안 된다.**
       (a)의 경고대로 짧은 합성 문장은 모델이 평소 보던 것이 아니다.
       GOOD/BAD 를 바꿔 대여섯 쌍쯤 해 보면 답이 이리저리 뒤집히는 걸 볼 수 있다.
       → **그래서 (d)에서 진짜 데이터 2,000편으로 다시 잰다.** 이게 오늘의 방식이다.""")


# %% [markdown]
# ## (d) 긴 리뷰에서 게이트가 더 값을 하는가 — 진짜 데이터로
#
# (a)~(c)는 **우리가 지어낸 짧은 문장**이었다. 재미있지만 증거로는 약하다.
# 여기서는 **검증 데이터 2,000편을 길이로 나눠** 정확도를 따로 잰다.
#
# 세운 가설: *게이트는 긴 문장에서 값을 하니, **긴 리뷰에서 격차가 더 벌어질 것**이다.*
# 재 보자. 가설이 맞을 수도, 틀릴 수도 있다.

# %%
lengths = torch.tensor([len(t) for t in vtok])       # 리뷰마다 토큰 개수(자르기 전 원래 길이)
short_mask = lengths <= 100                   # 자르지 않아도 통째로 들어가는 리뷰
long_mask = lengths > 250                     # 앞부분이 잘려 나가는 긴 리뷰

print(f"\n(d) 리뷰 길이로 나눠 보기")
print(f"    짧은 리뷰(≤100단어) {int(short_mask.sum())}편 · 긴 리뷰(>250단어) {int(long_mask.sum())}편\n")

print(f"    {'모델':<6} {'전체':>8} {'짧은 리뷰':>10} {'긴 리뷰':>10}")
print("    " + "-" * 38)
accs = {}
for name, probs in [("RNN", p_rnn), ("LSTM", p_lstm)]:
    ok = ((probs > 0.5).float() == y_val).float()
    accs[name] = (ok.mean().item(), ok[short_mask].mean().item(), ok[long_mask].mean().item())
    print(f"    {name:<6} {accs[name][0]:>8.3f} {accs[name][1]:>10.3f} {accs[name][2]:>10.3f}")

gain_short = accs["LSTM"][1] - accs["RNN"][1]
gain_long = accs["LSTM"][2] - accs["RNN"][2]
print(f"\n    LSTM 이 RNN 보다 나은 폭 — 짧은 리뷰 {gain_short:+.3f} · 긴 리뷰 {gain_long:+.3f}")

if gain_long > gain_short:
    print("    → 가설대로다. 긴 리뷰에서 격차가 더 벌어졌다.")
else:
    print("    → ❗ 가설이 빗나갔다. 오히려 짧은 리뷰에서 격차가 더 크다.")
    print("""
    왜 그럴까? **MAX_LEN=100** 을 떠올리자.
    300단어짜리 리뷰도 우리는 **뒤 100단어만 잘라서** 넣는다.
    그러니 '긴 리뷰'라고 이름 붙였지만 모델이 실제로 본 것은 **잘려 나간 조각**이다.
    앞의 200단어는 게이트가 지킬 기회조차 없었다 — 애초에 모델에 들어가지 않았으니까.

    📌 기억해 둘 것.
      "게이트는 긴 문장에 강하다"는 말은 맞다. 하지만 **우리 실험은 그걸 확인할 수 없는 설계**였다.
      결과가 예상과 다를 때, 모델을 의심하기 전에 **내 실험 설계를 먼저 의심하자.**""")

print("""
    ↑ 이 결과를 뒤집어 보려면? MAX_LEN 을 300으로 올리고 A-1 을 다시 돌려 보자.
      (시간이 3배쯤 걸린다. 도전해 볼 사람은 미션 C 에서.)
""")


# %% [markdown]
# ## 기록하고 이야기하자
#
# | 실험 | 내가 넣은 문장 | RNN | LSTM | 갈렸나? | 왜 그럴까 |
# |---|---|---|---|---|---|
# | (a) | | | | | |
# | (c) | | | | | |
# | (d) | | | | | |
#
# **회고 때 나눌 것**
# 1. 두 모델의 답이 갈린 문장 중 **가장 재미있는 것**을 하나 공유하자.
# 2. (b)에서 "LSTM이 오히려 놓친" 리뷰가 있었다. 그걸 읽어 보면 무슨 특징이 있나?
# 3. (d)에서 LSTM도 결국 무너졌다면, 게이트로도 해결되지 않는 것은 무엇일까?
#    → 이 질문이 **Day 6 어텐션**으로 이어진다.
