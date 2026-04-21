import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

# =======================
# НАСТРОЙКИ
# =======================
INPUT_CSV = "SBER_2014_2024.csv"
SEP = ";"

# Методика: HP + MA(10) + SAX + правила k=3
LAMBDA_HP = 14400
MA_WINDOW = 10

ALPHABET_SIZE = 7   # можно 7 или 9
K = 3               # abc -> d
WINDOW = 252
STEP = 1
MIN_COUNT = 3

# Сетки подбора порогов
SUPPORT_GRID = np.array([
    0.010, 0.011, 0.012,
    0.0125, 0.013, 0.0135,
    0.014, 0.015,
    0.017, 0.020
])

CONFIDENCE_GRID = np.array([
    0.10, 0.12, 0.14,
    0.15, 0.16, 0.18,
    0.20, 0.22, 0.25
])

# Метрика для локтя: число сигналов BUY+SELL
METRIC = "n_trades"

# "Первый локоть": порог доли от максимального падения
FIRST_ELBOW_FRAC = 0.5


# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================
def hp_filter(y, lam):
    n = len(y)
    e = np.ones(n)
    D = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n), format="csc")
    A = sparse.eye(n, format="csc") + lam * (D.T @ D)
    trend = spsolve(A, y)
    cycle = y - trend
    return trend, cycle


def sax_symbolize(values, breakpoints, alphabet):
    idx = np.searchsorted(breakpoints, values, side="right")
    return alphabet[idx]


def first_big_drop_elbow(x, y, frac=0.5):
    """
    Первый локоть = первая точка, где падение y[i] -> y[i+1]
    >= frac * max_drop. Возвращает индекс точки ПОСЛЕ падения (i+1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 2:
        return 0

    drops = y[:-1] - y[1:]
    max_drop = drops.max()

    if max_drop <= 0:
        return 0

    thr = frac * max_drop

    for i, d in enumerate(drops):
        if d >= thr:
            return i + 1

    return int(np.argmax(drops)) + 1


def plot_elbow(x, y, idx, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.axvline(x[idx], linestyle="--", label="first-drop elbow")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================
# КЭШ: кандидаты правил для каждого окна
# =======================
def build_cache(df, alphabet, breakpoints, window, step, k, min_count):
    """
    Для каждого окна сохраняем список кандидатов для текущего контекста:
    [(support, confidence, directed_bool), ...]
    directed_bool = (b != last_symbol_of_context)

    Это позволяет быстро пересчитывать n_trades для разных порогов support/confidence.
    """
    cache = []
    total_windows = max(0, (len(df) - window - 1) // step + 1)

    for w_idx, start in enumerate(range(0, len(df) - window - 1, step), 1):
        w = df.iloc[start:start + window]
        s = w["cycle_hp_ma10"].values

        s_std = s.std()
        if s_std == 0 or np.isnan(s_std):
            cache.append([])
            continue

        s_norm = (s - s.mean()) / s_std
        sax = sax_symbolize(s_norm, breakpoints, alphabet)

        # n-граммы: A = контекст, B = следующий символ
        A = ["".join(sax[i - k:i]) for i in range(k, len(sax))]
        B = [sax[i] for i in range(k, len(sax))]

        if len(A) == 0:
            cache.append([])
            continue

        a_counts = pd.Series(A).value_counts()
        rules_series = pd.Series([A[i] + "->" + B[i] for i in range(len(A))]).value_counts()

        total = len(A)
        current_ctx = "".join(sax[-k:])
        last_a = current_ctx[-1]

        candidates = []
        for rule, cnt in rules_series.items():
            a, b = rule.split("->")
            if a != current_ctx:
                continue
            if cnt < min_count:
                continue

            support = cnt / total
            confidence = cnt / a_counts[a]
            directed = (b != last_a)
            candidates.append((support, confidence, directed))

        cache.append(candidates)

        if w_idx % 300 == 0:
            print(f"cache: {w_idx}/{total_windows} windows")

    return cache


def evaluate_from_cache(cache, min_support, min_confidence):
    """
    Быстро считаем метрики при данных порогах.
    Логика сигнала: если есть хотя бы одно directed правило, прошедшее пороги -> trade (BUY/SELL),
    иначе HOLD.
    """
    trades = 0

    for candidates in cache:
        if not candidates:
            continue

        ok_directed = any(
            (sup >= min_support and conf >= min_confidence and directed)
            for sup, conf, directed in candidates
        )
        if ok_directed:
            trades += 1

    hold_share = 1 - trades / (len(cache) + 1e-12)
    return trades, hold_share


# =======================
# MAIN
# =======================
def main():
    # 1) Load data
    df = pd.read_csv(INPUT_CSV, sep=SEP)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin").reset_index(drop=True)

    # 2) HP + MA(10)
    _, cycle = hp_filter(df["close"].values, LAMBDA_HP)
    df["cycle_hp_ma10"] = pd.Series(cycle).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    df = df.dropna().reset_index(drop=True)
    print("✔ HP-фильтр и MA(10) применены")

    # 3) SAX setup
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))[:ALPHABET_SIZE]
    breakpoints = norm.ppf([i / ALPHABET_SIZE for i in range(1, ALPHABET_SIZE)])

    # 4) Build cache once
    print("▶ building cache (one-time, then sweeps are fast)...")
    cache = build_cache(df, alphabet, breakpoints, WINDOW, STEP, K, MIN_COUNT)
    print("✔ cache ready. windows:", len(cache))

    # =======================
    # SUPPORT sweep
    # =======================
    best_trades_by_support = []
    best_conf_for_support = []

    for sup in SUPPORT_GRID:
        trades_list = []
        for conf in CONFIDENCE_GRID:
            trades, _ = evaluate_from_cache(cache, float(sup), float(conf))
            trades_list.append(trades)

        trades_list = np.array(trades_list)
        j_best = int(np.argmax(trades_list))

        best_trades_by_support.append(int(trades_list[j_best]))
        best_conf_for_support.append(float(CONFIDENCE_GRID[j_best]))

    best_trades_by_support = np.array(best_trades_by_support, dtype=float)

    s_idx = first_big_drop_elbow(SUPPORT_GRID, best_trades_by_support, frac=FIRST_ELBOW_FRAC)
    best_support = float(SUPPORT_GRID[s_idx])
    best_conf_at_support = float(best_conf_for_support[s_idx])

    print("\n✅ SUPPORT first elbow (confidence optimized per support):")
    for i in range(len(SUPPORT_GRID)):
        print(
            f"support={SUPPORT_GRID[i]:.4f} -> "
            f"best_conf={best_conf_for_support[i]:.2f}, "
            f"trades={int(best_trades_by_support[i])}"
        )

    print(
        f"\nFIRST-DROP ELBOW SUPPORT = {best_support:.4f} "
        f"(best_conf at that support = {best_conf_at_support:.2f})"
    )

    plot_elbow(
        SUPPORT_GRID,
        best_trades_by_support,
        s_idx,
        title="First elbow for SUPPORT",
        xlabel="min_support",
        ylabel="n_trades"
    )

    # =======================
    # CONFIDENCE sweep
    # =======================
    best_trades_by_conf = []
    best_sup_for_conf = []

    for conf in CONFIDENCE_GRID:
        trades_list = []
        for sup in SUPPORT_GRID:
            trades, _ = evaluate_from_cache(cache, float(sup), float(conf))
            trades_list.append(trades)

        trades_list = np.array(trades_list)
        j_best = int(np.argmax(trades_list))

        best_trades_by_conf.append(int(trades_list[j_best]))
        best_sup_for_conf.append(float(SUPPORT_GRID[j_best]))

    best_trades_by_conf = np.array(best_trades_by_conf, dtype=float)

    c_idx = first_big_drop_elbow(CONFIDENCE_GRID, best_trades_by_conf, frac=FIRST_ELBOW_FRAC)
    best_confidence = float(CONFIDENCE_GRID[c_idx])
    best_sup_at_conf = float(best_sup_for_conf[c_idx])

    print("\n✅ CONFIDENCE first elbow (support optimized per confidence):")
    for i in range(len(CONFIDENCE_GRID)):
        print(
            f"conf={CONFIDENCE_GRID[i]:.2f} -> "
            f"best_support={best_sup_for_conf[i]:.4f}, "
            f"trades={int(best_trades_by_conf[i])}"
        )

    print(
        f"\nFIRST-DROP ELBOW CONFIDENCE = {best_confidence:.2f} "
        f"(best_support at that conf = {best_sup_at_conf:.4f})"
    )

    plot_elbow(
        CONFIDENCE_GRID,
        best_trades_by_conf,
        c_idx,
        title="First elbow for CONFIDENCE",
        xlabel="min_confidence",
        ylabel="n_trades"
    )

    # =======================
    # Sensitivity analysis
    # =======================
    print("\n📊 Sensitivity analysis for CONFIDENCE (support fixed):")
    rows_conf = []
    fixed_support = best_support

    for conf in CONFIDENCE_GRID:
        trades, hold = evaluate_from_cache(cache, fixed_support, float(conf))
        rows_conf.append({
            "min_confidence": conf,
            "n_trades": trades,
            "hold_share": round(hold, 4)
        })

    df_conf = pd.DataFrame(rows_conf)
    print(df_conf)

    print("\n📊 Sensitivity analysis for SUPPORT (confidence fixed):")
    rows_sup = []
    fixed_conf = best_confidence

    for sup in SUPPORT_GRID:
        trades, hold = evaluate_from_cache(cache, float(sup), fixed_conf)
        rows_sup.append({
            "min_support": sup,
            "n_trades": trades,
            "hold_share": round(hold, 4)
        })

    df_sup = pd.DataFrame(rows_sup)
    print(df_sup)

    print("\n🎯 Итог:")
    print("MIN_SUPPORT    =", round(best_support, 6))
    print("MIN_CONFIDENCE =", round(best_confidence, 6))


if __name__ == "__main__":
    main()