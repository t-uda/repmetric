import random
import numpy as np
import matplotlib.pyplot as plt

# 塩基（DNA風の配列として生成します）
NUCS = ["A", "C", "G", "T"]

# 複雑TRのパターン定義（論文の代表例）
PATTERNS = [
    # (AAAG)^i (AG)^j (CAG)^i (CAA)^j (AAAG)^i (AG)^j (AAAG)^k
    [
        ("AAAG", "i"),
        ("AG", "j"),
        ("CAG", "i"),
        ("CAA", "j"),
        ("AAAG", "i"),
        ("AG", "j"),
        ("AAAG", "k"),
    ],
    # (AAAG)^i (AG)^j (AAAG)^k (AG)^l (AAAG)^m
    [("AAAG", "i"), ("AG", "j"), ("AAAG", "k"), ("AG", "l"), ("AAAG", "m")],
    # (AGGGG)^i (AAAAGAAAGAGAGGG)^j (AGGGG)^k
    [("AGGGG", "i"), ("AAAAGAAAGAGAGGG", "j"), ("AGGGG", "k")],
]


def realize_pattern(pattern, rng):
    """パターンから純粋なTR配列を生成する"""
    reps = {}
    parts = []
    for unit, rep in pattern:
        if isinstance(rep, int):
            r = rep
        else:
            r = rng.randint(0, 50)  # 0〜50回繰り返し
            reps[rep] = r
        parts.append(unit * r)
    return "".join(parts), reps


def mutate(seq, rate, rng):
    """配列に置換・挿入・削除を導入して不純な配列を作成"""
    if rate <= 0 or not seq:
        return seq
    s = list(seq)
    n_ops = max(1, round(len(s) * rate / 100))
    for _ in range(n_ops):
        op = rng.choice(["sub", "ins", "del"]) if s else "ins"
        pos = rng.randrange(len(s) + (1 if op == "ins" else 0))
        if op == "sub":
            old = s[pos]
            choices = [x for x in NUCS if x != old]
            s[pos] = rng.choice(choices)
        elif op == "ins":
            s.insert(pos, rng.choice(NUCS))
        else:  # del
            del s[pos]
    return "".join(s)


# ==========================
# 実行例
# ==========================
rng = random.Random(123)

# 1つ目のパターンから純粋な配列を生成
seq, reps = realize_pattern(PATTERNS[0], rng)

# エラー率5%で不純な配列を作成
mut = mutate(seq, 5, rng)

print("繰り返し回数:", reps)
print("純粋配列:", seq[:120] + "...")
print("不純配列:", mut[:120] + "...")

# 実際のアミノ酸配列（例: ヒトミオグロビン）
# seq = "GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDIAAKYKELGFQG"
seq = mut

print(f"入力配列の長さ: {len(seq)}")


# ========== スライディングウィンドウ ==========
def sliding_windows(s: str, k: int, step: int = 1):
    """文字列 s から幅 k のスライディングウィンドウを step 刻みで取得"""
    subs = []
    starts = []
    i = 0
    while i + k <= len(s):
        subs.append(s[i : i + k])
        starts.append(i)
        i += step
    return subs, starts


# ========== レーベンシュタイン距離 ==========
def levenshtein(a: str, b: str) -> int:
    """レーベンシュタイン距離（挿入・削除・置換の最小回数）"""
    na, nb = len(a), len(b)
    if na < nb:
        a, b = b, a
        na, nb = nb, na
    prev = list(range(nb + 1))
    for i in range(1, na + 1):
        cur = [i] + [0] * nb
        ca = a[i - 1]
        for j in range(1, nb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost  # 削除  # 挿入  # 置換
            )
        prev = cur
    return prev[nb]


# ========== 行列を作成 ==========
window = 10  # ウィンドウ幅
step = 1  # ステップ幅
subs, starts = sliding_windows(seq, window, step)
m = len(subs)
print(f"ウィンドウ数: {m}（window={window}, step={step}）")

M = np.zeros((m, m), dtype=int)
for i in range(m):
    for j in range(i, m):
        d = levenshtein(subs[i], subs[j])
        M[i, j] = d
        M[j, i] = d

# ========== 可視化 ==========
plt.figure(figsize=(6, 6))
im = plt.imshow(M, origin="lower", cmap="viridis")
plt.title(f"Edit-distance matrix (k={window}, step={step})")
plt.xlabel("window index j")
plt.ylabel("window index i")
plt.colorbar(im, label="Levenshtein distance")
plt.tight_layout()
plt.show()

# ========== ドットプロット風（しきい値処理） ==========
threshold = 1  # 許容する編集距離
B = (M <= threshold).astype(int)

plt.figure(figsize=(6, 6))
plt.imshow(B, origin="lower", cmap="Greys")
plt.title(f"Dot-plot by threshold (distance ≤ {threshold})")
plt.xlabel("window index j")
plt.ylabel("window index i")
plt.tight_layout()
plt.show()
