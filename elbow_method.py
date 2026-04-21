import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm


INPUT_CSV = "SBER_2014_2024.csv"
SEP = ";"

LAMBDA_HP = 14400
MA_WINDOW = 10

ALPHABET_SIZE = 7
K = 3
WINDOW = 252
STEP = 1
MIN_COUNT = 3

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

FIRST_ELBOW_FRAC = 0.5


def hp_filter(y, lam):
    n = len(y)
    e = np.ones(n)
    d = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n), format="csc")
    a = sparse.eye(n, format="csc") + lam * (d.T @ d)
    trend = spsolve(a, y)
    cycle = y - trend
    return trend, cycle


def sax_symbolize(values, breakpoints, alphabet):
    idx = np.searchsorted(breakpoints, values, side="right")
    return alphabet[idx]


def first_big_drop_elbow(x, y, frac=0.5):
    """
    Return the index of the first point after a significant drop in y.
    A drop is considered significant if it is at least frac * max_drop.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 2:
        return 0

    drops = y[:-1] - y[1:]
    max_drop = drops.max()

    if max_drop <= 0:
        return 0

    threshold = frac * max_drop

    for i, drop in enumerate(drops):
        if drop >= threshold:
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


def build_cache(df, alphabet, breakpoints, window, step, k, min_count):
    """
    Build a cache of candidate rules for each sliding window.

    Each cache item is a list of tuples:
        (support, confidence, directed)

    directed = True if the next symbol differs from the last symbol
    of the current context.
    """
    cache = []
    total_windows = max(0, (len(df) - window - 1) // step + 1)

    for w_idx, start in enumerate(range(0, len(df) - window - 1, step), 1):
        window_df = df.iloc[start:start + window]
        series = window_df["cycle_hp_ma10"].values

        series_std = series.std()
        if series_std == 0 or np.isnan(series_std):
            cache.append([])
            continue

        series_norm = (series - series.mean()) / series_std
        sax = sax_symbolize(series_norm, breakpoints, alphabet)

        contexts = ["".join(sax[i - k:i]) for i in range(k, len(sax))]
        next_symbols = [sax[i] for i in range(k, len(sax))]

        if len(contexts) == 0:
            cache.append([])
            continue

        context_counts = pd.Series(contexts).value_counts()
        rule_counts = pd.Series(
            [contexts[i] + "->" + next_symbols[i] for i in range(len(contexts))]
        ).value_counts()

        total = len(contexts)
        current_context = "".join(sax[-k:])
        last_symbol = current_context[-1]

        candidates = []
        for rule, count in rule_counts.items():
            context, symbol = rule.split("->")

            if context != current_context:
                continue
            if count < min_count:
                continue

            support = count / total
            confidence = count / context_counts[context]
            directed = symbol != last_symbol

            candidates.append((support, confidence, directed))

        cache.append(candidates)

        if w_idx % 300 == 0:
            print(f"cache: {w_idx}/{total_windows} windows")

    return cache


def evaluate_from_cache(cache, min_support, min_confidence):
    """
    Count the number of tradable windows for a given pair of thresholds.

    A window is considered tradable if it contains at least one directed rule
    that satisfies the support and confidence thresholds.
    """
    trades = 0

    for candidates in cache:
        if not candidates:
            continue

        has_valid_rule = any(
            support >= min_support and confidence >= min_confidence and directed
            for support, confidence, directed in candidates
        )

        if has_valid_rule:
            trades += 1

    hold_share = 1 - trades / (len(cache) + 1e-12)
    return trades, hold_share


def main():
    df = pd.read_csv(INPUT_CSV, sep=SEP)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin").reset_index(drop=True)

    _, cycle = hp_filter(df["close"].values, LAMBDA_HP)
    df["cycle_hp_ma10"] = pd.Series(cycle).rolling(
        MA_WINDOW,
        min_periods=MA_WINDOW
    ).mean()
    df = df.dropna().reset_index(drop=True)

    print("HP filter and MA(10) applied")

    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))[:ALPHABET_SIZE]
    breakpoints = norm.ppf([i / ALPHABET_SIZE for i in range(1, ALPHABET_SIZE)])

    print("Building cache...")
    cache = build_cache(df, alphabet, breakpoints, WINDOW, STEP, K, MIN_COUNT)
    print("Cache ready. Windows:", len(cache))

    best_trades_by_support = []
    best_conf_for_support = []

    for support in SUPPORT_GRID:
        trades_list = []

        for confidence in CONFIDENCE_GRID:
            trades, _ = evaluate_from_cache(cache, float(support), float(confidence))
            trades_list.append(trades)

        trades_list = np.array(trades_list)
        best_idx = int(np.argmax(trades_list))

        best_trades_by_support.append(int(trades_list[best_idx]))
        best_conf_for_support.append(float(CONFIDENCE_GRID[best_idx]))

    best_trades_by_support = np.array(best_trades_by_support, dtype=float)

    support_idx = first_big_drop_elbow(
        SUPPORT_GRID,
        best_trades_by_support,
        frac=FIRST_ELBOW_FRAC
    )
    best_support = float(SUPPORT_GRID[support_idx])
    best_conf_at_support = float(best_conf_for_support[support_idx])

    print("\nSupport sweep:")
    for i in range(len(SUPPORT_GRID)):
        print(
            f"support={SUPPORT_GRID[i]:.4f} -> "
            f"best_conf={best_conf_for_support[i]:.2f}, "
            f"trades={int(best_trades_by_support[i])}"
        )

    print(
        f"\nSelected min_support = {best_support:.4f} "
        f"(best confidence at this point = {best_conf_at_support:.2f})"
    )

    plot_elbow(
        SUPPORT_GRID,
        best_trades_by_support,
        support_idx,
        title="Elbow analysis for min_support",
        xlabel="min_support",
        ylabel="n_trades"
    )

    best_trades_by_conf = []
    best_sup_for_conf = []

    for confidence in CONFIDENCE_GRID:
        trades_list = []

        for support in SUPPORT_GRID:
            trades, _ = evaluate_from_cache(cache, float(support), float(confidence))
            trades_list.append(trades)

        trades_list = np.array(trades_list)
        best_idx = int(np.argmax(trades_list))

        best_trades_by_conf.append(int(trades_list[best_idx]))
        best_sup_for_conf.append(float(SUPPORT_GRID[best_idx]))

    best_trades_by_conf = np.array(best_trades_by_conf, dtype=float)

    confidence_idx = first_big_drop_elbow(
        CONFIDENCE_GRID,
        best_trades_by_conf,
        frac=FIRST_ELBOW_FRAC
    )
    best_confidence = float(CONFIDENCE_GRID[confidence_idx])
    best_sup_at_conf = float(best_sup_for_conf[confidence_idx])

    print("\nConfidence sweep:")
    for i in range(len(CONFIDENCE_GRID)):
        print(
            f"confidence={CONFIDENCE_GRID[i]:.2f} -> "
            f"best_support={best_sup_for_conf[i]:.4f}, "
            f"trades={int(best_trades_by_conf[i])}"
        )

    print(
        f"\nSelected min_confidence = {best_confidence:.2f} "
        f"(best support at this point = {best_sup_at_conf:.4f})"
    )

    plot_elbow(
        CONFIDENCE_GRID,
        best_trades_by_conf,
        confidence_idx,
        title="Elbow analysis for min_confidence",
        xlabel="min_confidence",
        ylabel="n_trades"
    )

    print("\nSensitivity analysis for min_confidence (fixed support):")
    confidence_rows = []
    fixed_support = best_support

    for confidence in CONFIDENCE_GRID:
        trades, hold_share = evaluate_from_cache(cache, fixed_support, float(confidence))
        confidence_rows.append({
            "min_confidence": confidence,
            "n_trades": trades,
            "hold_share": round(hold_share, 4)
        })

    confidence_df = pd.DataFrame(confidence_rows)
    print(confidence_df)

    print("\nSensitivity analysis for min_support (fixed confidence):")
    support_rows = []
    fixed_confidence = best_confidence

    for support in SUPPORT_GRID:
        trades, hold_share = evaluate_from_cache(cache, float(support), fixed_confidence)
        support_rows.append({
            "min_support": support,
            "n_trades": trades,
            "hold_share": round(hold_share, 4)
        })

    support_df = pd.DataFrame(support_rows)
    print(support_df)

    print("\nFinal parameters:")
    print("MIN_SUPPORT    =", round(best_support, 6))
    print("MIN_CONFIDENCE =", round(best_confidence, 6))


if __name__ == "__main__":
    main()
