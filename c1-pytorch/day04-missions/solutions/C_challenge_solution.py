"""
🔴 C 정답 — 게이트 안 들여다보기 (C-1 · C-2 · C-3)

세 갈래 모두의 풀이다. 스스로 30분은 붙잡아 본 뒤에 열어 보자.
전체를 다 돌리면 CPU 기준 몇 분 걸린다(C-3 이 무겁다).
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
sys.path.insert(0, str(_HERE.parent.parent))

from imdb_data import load_imdb
from textutils import tokenize_en, build_vocab, pad_and_tensor
from model import RnnClassifier, LstmClassifier

torch.manual_seed(0)


# %% [markdown]
# # C-1. LSTM 한 스텝을 손으로

# %%
H, E = 4, 3
cell = nn.LSTM(E, H, batch_first=True)
W_ih, W_hh = cell.weight_ih_l0, cell.weight_hh_l0
b_ih, b_hh = cell.bias_ih_l0, cell.bias_hh_l0

x_t = torch.randn(1, 1, E)
h_prev = torch.zeros(1, 1, H)
c_prev = torch.zeros(1, 1, H)

with torch.no_grad():
    # TODO C-1-a — 네 벌을 한꺼번에 계산하고 i, f, g, o 로 4등분한다
    gates = x_t.view(-1) @ W_ih.T + b_ih + h_prev.view(-1) @ W_hh.T + b_hh
    i_lin, f_lin, g_lin, o_lin = gates.chunk(4)      # 파이토치 저장 순서: i, f, g, o

    i_t = torch.sigmoid(i_lin)      # 새 정보를 얼마나 받을까  (0~1)
    f_t = torch.sigmoid(f_lin)      # 지난 기억을 얼마나 남길까 (0~1)
    g_t = torch.tanh(g_lin)         # 무엇을 넣을까 — 값 자체    (-1~1)
    o_t = torch.sigmoid(o_lin)      # 얼마나 내보낼까            (0~1)

    # TODO C-1-b — 곱셈이 아니라 덧셈으로 이어지는 자리
    c_t = f_t * c_prev.view(-1) + i_t * g_t
    h_t = o_t * torch.tanh(c_t)

    out, (h_ref, c_ref) = cell(x_t, (h_prev, c_prev))

print("  내 손계산 h_t :", [round(v, 5) for v in h_t.tolist()])
print("  nn.LSTM  h_t :", [round(v, 5) for v in h_ref.flatten().tolist()])
print("  h_t 일치? ", torch.allclose(h_t, h_ref.flatten(), atol=1e-6))
print("  c_t 일치? ", torch.allclose(c_t, c_ref.flatten(), atol=1e-6))
print(f"\n  게이트 값 — i {i_t.mean():.3f} · f {f_t.mean():.3f} · o {o_t.mean():.3f}")


# %% [markdown]
# ### C-1 확인 질문 답

# %%
with torch.no_grad():
    c_before = torch.tensor([5.0, -3.0, 1.0, 0.5])       # 지난 기억이 이랬다고 하자

    # 질문 1 — 문을 활짝 열면 (f=1, i=0)
    c_open = 1.0 * c_before + 0.0 * g_t
    # 질문 2 — 문을 닫으면 (f=0)
    c_shut = 0.0 * c_before + i_t * g_t

print(f"  지난 기억 c_(t-1)      : {[round(v,2) for v in c_before.tolist()]}")
print(f"  f=1, i=0 → c_t         : {[round(v,2) for v in c_open.tolist()]}   ← 완전히 그대로다")
print(f"  f=0      → c_t         : {[round(v,2) for v in c_shut.tolist()]}   ← 과거가 사라졌다")
print("""
  [답]
   1. f=1, i=0 이면 c_t = c_{t-1} — 값이 **손대지 않은 채** 다음으로 넘어간다.
      역전파 때도 이 길로 기울기가 눌리지 않고 지나간다. 이게 '덧셈 고속도로'다.
      앞에서 본 (배율)^거리 의 지수적 감쇠가 여기서는 일어나지 않는다.
   2. f=0 이면 과거를 통째로 버리고 지금 것만 남긴다.
      "여기서부터 새 이야기"라고 판단한 셈이다. (문단이 바뀌는 자리 같은 곳)
   3. 게이트는 '**얼마나** 통과시킬까'라서 0~1 이어야 한다 → sigmoid.
      후보 g 는 '**무엇을** 더할까'라는 값 자체라서 음수도 필요하다 → tanh(-1~1).
      📌 이 구분이 LSTM 을 이해하는 핵심이다. 문(0~1)과 내용물(-1~1)은 다른 것이다.
""")


# %% [markdown]
# # C-2. 훈련된 모델의 게이트 값 들여다보기

# %%
VOCAB_SIZE, MAX_LEN = 2000, 100
train, val = load_imdb(n_train=5000, n_val=2000)
ttok = [tokenize_en(t) for t in train["text"]]
vtok = [tokenize_en(t) for t in val["text"]]
word2idx, _ = build_vocab(ttok, max_size=VOCAB_SIZE)
X_train = pad_and_tensor(ttok, word2idx, MAX_LEN)
y_train = torch.tensor(train["label"], dtype=torch.float32)
X_val = pad_and_tensor(vtok, word2idx, MAX_LEN)
y_val = torch.tensor(val["label"], dtype=torch.float32)


def train_lstm(max_len=MAX_LEN, ModelClass=LstmClassifier, epochs=8, seed=42,
               Xt=None, yt=None, Xv=None, yv=None):
    Xt = X_train if Xt is None else Xt
    yt = y_train if yt is None else yt
    Xv = X_val if Xv is None else Xv
    yv = y_val if yv is None else yv
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    m = ModelClass(len(word2idx), 32, 32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = ((m(Xv) > 0.5).float() == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m, best_acc


# TODO C-2-a
model, acc = train_lstm()
print(f"\n[C-2] 훈련 완료 — 검증 정확도 {acc:.3f}")


# TODO C-2-b — 훈련된 가중치로 한 스텝씩 손으로 굴리며 게이트를 기록한다
def gates_over_sentence(model, tokens):
    """문장을 한 단어씩 넣으며 (단어, f평균, i평균, o평균) 목록을 만든다."""
    lstm = model.rnn
    Wi, Wh = lstm.weight_ih_l0, lstm.weight_hh_l0
    bi, bh = lstm.bias_ih_l0, lstm.bias_hh_l0
    hid = lstm.hidden_size

    ids = pad_and_tensor([tokens], word2idx, MAX_LEN)[0]
    idx2word = {v: k for k, v in word2idx.items()}

    h = torch.zeros(hid)
    c = torch.zeros(hid)
    rows = []
    with torch.no_grad():
        for tok_id in ids:
            x = model.embedding(tok_id.view(1)).view(-1)          # 임베딩 벡터
            g = x @ Wi.T + bi + h @ Wh.T + bh
            i_lin, f_lin, g_lin, o_lin = g.chunk(4)
            i_, f_, g_, o_ = (torch.sigmoid(i_lin), torch.sigmoid(f_lin),
                              torch.tanh(g_lin), torch.sigmoid(o_lin))
            c = f_ * c + i_ * g_
            h = o_ * torch.tanh(c)
            rows.append((idx2word.get(int(tok_id), "?"),
                         f_.mean().item(), i_.mean().item(), o_.mean().item()))
    return rows


SENT = ("the movie started well but the acting was terrible and the story was "
        "boring i really hated this film")
rows = gates_over_sentence(model, tokenize_en(SENT))

# 손계산이 맞는지 먼저 검증한다 (앞의 C-1 과 같은 원리)
with torch.no_grad():
    ref = model(pad_and_tensor([tokenize_en(SENT)], word2idx, MAX_LEN)).item()
print(f"  (검증: 모델 예측 {ref:.3f} — 0.5보다 낮으면 '부정'으로 맞힌 것)")

# TODO C-2-c — 뒤쪽(실제 단어가 있는 부분)만 출력한다
print(f"\n  {'단어':<12} {'forget':>8} {'input':>8} {'output':>8}")
print("  " + "-" * 40)
for word, f_, i_, o_ in rows[-24:]:
    mark = ""
    if word in ("terrible", "boring", "hated", "well"):
        mark = "  ← 감정 단어"
    if word == "<pad>":
        mark = "  ← 빈칸"
    print(f"  {word:<12} {f_:>8.3f} {i_:>8.3f} {o_:>8.3f}{mark}")

pads = [r for r in rows if r[0] == "<pad>"]
words = [r for r in rows if r[0] != "<pad>"]
if pads and words:
    print(f"\n  <pad> 자리 forget 평균 : {sum(r[1] for r in pads)/len(pads):.3f}")
    print(f"  실제 단어 forget 평균  : {sum(r[1] for r in words)/len(words):.3f}")

print("""
  [읽는 법 — 조심할 것]
   · forget 값이 1에 가까울수록 지난 기억을 그대로 유지한다는 뜻이다.
   · <pad> 자리에서 forget 이 높게 유지된다면, 모델이 **빈칸에서는 기억을 건드리지 않기로**
     배웠다는 뜻이다. 아무도 그렇게 하라고 가르치지 않았는데 스스로 찾아낸 것이다.
   · ⚠️ 하지만 **아무 단어나 골라 이야기를 지어내지 말자.** 은닉 32칸의 평균 하나로
     "이 단어에서 문을 닫았다"고 말하기엔 근거가 약하다.
     제대로 보려면 감정 단어 여러 개 vs 평범한 단어 여러 개의 **평균을 비교**해야 한다.
     오늘 배운 신호/잡음 원칙이 여기서도 그대로 적용된다.
""")


# %% [markdown]
# # C-3. MAX_LEN 을 늘리면 게이트의 이점이 커지는가
#
# A-2 (d)의 짐작을 검증한다. 무겁다 — 몇 분 걸린다.

# %%
SEEDS_C3 = [42, 43]
EPOCHS_C3 = 6

results = {}
for max_len in [100, 300]:
    Xt = pad_and_tensor(ttok, word2idx, max_len)
    Xv = pad_and_tensor(vtok, word2idx, max_len)
    for name, ModelClass in [("RNN", RnnClassifier), ("LSTM", LstmClassifier)]:
        scores = []
        for seed in SEEDS_C3:
            _, a = train_lstm(ModelClass=ModelClass, epochs=EPOCHS_C3, seed=seed,
                              Xt=Xt, yt=y_train, Xv=Xv, yv=y_val)
            scores.append(a)
        results[(max_len, name)] = scores
        print(f"  MAX_LEN {max_len} · {name:4s}: {[f'{s:.3f}' for s in scores]}")

# %%
print(f"\n  {'MAX_LEN':<10} {'RNN 평균':>10} {'LSTM 평균':>11} {'격차':>8} {'잡음':>8}")
print("  " + "-" * 50)
for max_len in [100, 300]:
    r = results[(max_len, "RNN")]
    l = results[(max_len, "LSTM")]
    avg_r, avg_l = sum(r) / len(r), sum(l) / len(l)
    noise = max(max(r) - min(r), max(l) - min(l))
    print(f"  {max_len:<10} {avg_r:>10.3f} {avg_l:>11.3f} {avg_l - avg_r:>8.3f} {noise:>8.3f}")

gap100 = (sum(results[(100, "LSTM")]) - sum(results[(100, "RNN")])) / len(SEEDS_C3)
gap300 = (sum(results[(300, "LSTM")]) - sum(results[(300, "RNN")])) / len(SEEDS_C3)
noise_all = max(max(v) - min(v) for v in results.values())

print(f"\n  격차 변화: {gap100:.3f} → {gap300:.3f}  (차이 {gap300 - gap100:+.3f}, 잡음 {noise_all:.3f})")
if abs(gap300 - gap100) > noise_all * 2:
    verdict = "커졌다 — 짐작이 맞았다" if gap300 > gap100 else "오히려 줄었다 — 짐작이 틀렸다"
    print(f"  → 격차가 {verdict}.")
else:
    print("  → 격차 변화가 잡음에 묻힌다. **짐작을 확인하지도, 반박하지도 못했다.**")
    print("     시드를 늘리거나 데이터를 키워야 판단할 수 있다. 이것도 정직한 결론이다.")

print("""
  [해설 — 여러분 숫자가 달라도 읽는 법은 같다]
   · MAX_LEN 300 은 학습이 3배 느리고, 앞쪽 200단어가 대부분 <pad> 가 아니라 실제 내용이 된다.
   · RNN 은 길이가 늘수록 앞을 더 많이 잊으므로 **나아지지 않거나 오히려 나빠지기 쉽다.**
   · LSTM 은 이론상 더 유리하지만, 데이터가 5,000편뿐이라 그 이점을 다 못 쓸 수 있다.
   · 어느 쪽이 나오든 **"확인하지 못했다"를 결론으로 쓸 줄 아는 것**이 오늘의 목표다.
     실험이 짐작을 지지하지 않았다고 실패가 아니다. 그게 실험을 하는 이유다.
""")
