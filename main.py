import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

# =======================
# НАСТРОЙКИ
# =======================
INPUT_CSV = "SBER_2014_2024.csv"

ALPHABET_SIZE = 7
WINDOW = 252
STEP = 1
LAMBDA_HP = 14400
MA_WINDOW = 10
K = 5

MIN_SUPPORT = 0.0125
MIN_CONFIDENCE = 0.14
MIN_COUNT = 3

PROB_THRESHOLD = 0.5
H = 1


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


def rules_from_sax_k(sax, k=3):
    sax = np.asarray(sax)
    if len(sax) <= k:
        return pd.DataFrame(columns=["rule", "a", "b", "count", "support", "confidence", "lift"])

    A = ["".join(sax[i - k:i]) for i in range(k, len(sax))]
    B = [sax[i] for i in range(k, len(sax))]

    df_local = pd.DataFrame({"a": A, "b": B})
    df_local["rule"] = df_local["a"] + "->" + df_local["b"]

    total = len(df_local)

    rule_counts = df_local["rule"].value_counts()
    a_counts = df_local["a"].value_counts()
    b_counts = df_local["b"].value_counts()

    rules = rule_counts.rename("count").to_frame()
    rules["support"] = rules["count"] / total
    rules = rules.reset_index().rename(columns={"index": "rule"})

    rules["a"] = rules["rule"].str.split("->").str[0]
    rules["b"] = rules["rule"].str.split("->").str[1]

    rules["confidence"] = rules.apply(lambda r: r["count"] / a_counts[r["a"]], axis=1)
    rules["lift"] = rules.apply(
        lambda r: r["support"] / ((a_counts[r["a"]] / total) * (b_counts[r["b"]] / total)),
        axis=1
    )

    return rules


def decide_action(row, prob_threshold=PROB_THRESHOLD):
    if pd.isna(row.get("exp_ret")):
        return "HOLD"

    exp_ret = row["exp_ret"]

    if prob_threshold is None:
        if exp_ret > 0:
            return "BUY"
        elif exp_ret < 0:
            return "SELL"
        return "HOLD"

    p_up = row.get("p_up", np.nan)
    p_down = row.get("p_down", np.nan)

    if (p_up >= prob_threshold) and (exp_ret > 0):
        return "BUY"
    elif (p_down >= prob_threshold) and (exp_ret < 0):
        return "SELL"
    return "HOLD"


def calc_rule_score(row):
    edge = row.get("edge", np.nan)
    confidence = row.get("confidence", np.nan)
    count = row.get("count", np.nan)
    lift = row.get("lift", np.nan)

    if pd.isna(edge) or pd.isna(confidence) or pd.isna(count) or pd.isna(lift):
        return np.nan

    return abs(edge) * confidence * np.log1p(count) * lift


def calc_metric_stats(series: pd.Series, metric_name: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return {
            "metric": metric_name,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    mean_val = s.mean()
    std_val = s.std(ddof=1) if len(s) > 1 else 0.0
    cv_val = std_val / mean_val if mean_val != 0 else np.nan

    return {
        "metric": metric_name,
        "n": len(s),
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "median": s.median(),
        "min": s.min(),
        "max": s.max(),
    }


def evaluate_signal(signal, actual_ret):
    if pd.isna(actual_ret):
        return np.nan

    if signal == "BUY":
        return 1 if actual_ret > 0 else 0
    elif signal == "SELL":
        return 1 if actual_ret < 0 else 0
    elif signal == "HOLD":
        return np.nan

    return np.nan


def strategy_return(signal, actual_ret):
    if pd.isna(actual_ret):
        return np.nan

    if signal == "BUY":
        return actual_ret
    elif signal == "SELL":
        return -actual_ret
    elif signal == "HOLD":
        return 0.0

    return np.nan


# =======================
# 1) ЗАГРУЗКА ДАННЫХ
# =======================
df = pd.read_csv(INPUT_CSV, sep=";")
df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["begin", "close"]).sort_values("begin").reset_index(drop=True)

# целевая доходность через H дней вперёд
df["ret_fwd"] = df["close"].pct_change(H).shift(-H)

print("✔ Данные загружены")

# =======================
# 2) FFT по всему ряду (опционально, только справочно)
# =======================
x0 = df["close"].values.astype(float)
x0 = x0 - np.mean(x0)
freqs = np.fft.rfftfreq(len(x0), d=1)
fft_vals = np.abs(np.fft.rfft(x0))
fft_vals[0] = 0

dominant_freq = freqs[np.argmax(fft_vals)]
dominant_period = (1 / dominant_freq) if dominant_freq > 0 else np.nan

print(f"✔ Доминирующий цикл рынка (справочно): {dominant_period:.2f} торговых дней")

# =======================
# 3) WALK-FORWARD БЕЗ УТЕЧКИ БУДУЩЕГО
# =======================
alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))[:ALPHABET_SIZE]
breakpoints = norm.ppf([i / ALPHABET_SIZE for i in range(1, ALPHABET_SIZE)])

signals = []
window_rules = []

print("▶ Начинаем walk-forward анализ...")

# Нужно иметь:
# - WINDOW дней истории для обучения
# - H дней вперёд для проверки факта
for start in range(0, len(df) - WINDOW - H, STEP):
    train_slice = df.iloc[start:start + WINDOW].copy()
    target_row = df.iloc[start + WINDOW].copy()

    # считаем признаки только по train_slice
    trend, cycle = hp_filter(train_slice["close"].values, LAMBDA_HP)
    cycle_ma = pd.Series(cycle).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()

    train_feat = train_slice.copy()
    train_feat["cycle_hp_ma"] = cycle_ma
    train_feat = train_feat.dropna().reset_index(drop=True)

    if len(train_feat) <= K + 5:
        signals.append({
            "date": target_row["begin"],
            "signal": "HOLD",
            "context": None,
            "chosen_rule": None,
            "actual_ret": target_row["ret_fwd"],
            "correct": np.nan,
            "strategy_ret": 0.0
        })
        continue

    s = train_feat["cycle_hp_ma"].values
    s_std = s.std()

    if s_std == 0 or np.isnan(s_std):
        signals.append({
            "date": target_row["begin"],
            "signal": "HOLD",
            "context": None,
            "chosen_rule": None,
            "actual_ret": target_row["ret_fwd"],
            "correct": np.nan,
            "strategy_ret": 0.0
        })
        continue

    s_norm = (s - s.mean()) / s_std
    sax = sax_symbolize(s_norm, breakpoints, alphabet)

    rules = rules_from_sax_k(sax, k=K)

    # Доходность для статистики правил:
    # берём фактическую доходность следующих H дней
    # уже существующую в исходном train_feat через merge по index нельзя,
    # поэтому просто используем ret_fwd из train_feat
    rets_next = train_feat["ret_fwd"].values
    idx = np.arange(K, len(sax))

    A = ["".join(sax[i - K:i]) for i in idx]
    B = [sax[i] for i in idx]
    R = [rets_next[i] for i in idx]

    trans = pd.DataFrame({"a": A, "b": B, "ret_fwd": R})
    trans["rule"] = trans["a"] + "->" + trans["b"]

    stats = trans.groupby("rule")["ret_fwd"].agg(
        exp_ret="mean",
        p_up=lambda x: float((x > 0).mean()),
        p_down=lambda x: float((x < 0).mean()),
    ).reset_index()

    rules = rules.merge(stats, on="rule", how="left")

    rules = rules[
        (rules["count"] >= MIN_COUNT) &
        (rules["support"] >= MIN_SUPPORT) &
        (rules["confidence"] >= MIN_CONFIDENCE)
    ].copy()

    current_ctx = "".join(sax[-K:])
    candidates = rules[rules["a"] == current_ctx].copy()

    action = "HOLD"
    chosen_rule = None

    if not candidates.empty:
        candidates["edge"] = candidates["p_up"] - candidates["p_down"]
        candidates["action"] = candidates.apply(decide_action, axis=1)
        candidates["score"] = candidates.apply(calc_rule_score, axis=1)

        tradable = candidates[candidates["action"] != "HOLD"].copy()

        if not tradable.empty:
            best = tradable.sort_values(
                ["score", "confidence", "count", "lift"],
                ascending=[False, False, False, False]
            ).iloc[0]
        else:
            best = candidates.sort_values(
                ["score", "confidence", "count", "lift"],
                ascending=[False, False, False, False]
            ).iloc[0]

        chosen_rule = best["rule"]
        action = best["action"]

    actual_ret = target_row["ret_fwd"]
    correct = evaluate_signal(action, actual_ret)
    strat_ret = strategy_return(action, actual_ret)

    signals.append({
        "date": target_row["begin"],
        "signal": action,
        "context": current_ctx,
        "chosen_rule": chosen_rule,
        "actual_ret": actual_ret,
        "correct": correct,
        "strategy_ret": strat_ret
    })

    for _, r in candidates.iterrows():
        window_rules.append({
            "date": target_row["begin"],
            "context": r["a"],
            "rule": r["rule"],
            "action": r["action"],
            "count": int(r["count"]),
            "support": float(r["support"]),
            "confidence": float(r["confidence"]),
            "lift": float(r["lift"]),
            "exp_ret": float(r["exp_ret"]) if pd.notna(r["exp_ret"]) else np.nan,
            "p_up": float(r["p_up"]) if pd.notna(r["p_up"]) else np.nan,
            "p_down": float(r["p_down"]) if pd.notna(r["p_down"]) else np.nan,
            "edge": float(r["edge"]) if pd.notna(r["edge"]) else np.nan,
            "score": float(r["score"]) if pd.notna(r["score"]) else np.nan,
        })

print("✔ Walk-forward анализ завершён")

# =======================
# 4) СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =======================
signals_df = pd.DataFrame(signals)
rules_df = pd.DataFrame(window_rules)

signals_file = f"SBER_trading_signals_eval_k{K}_h{H}.csv"
rules_file = f"SBER_rules_with_actions_k{K}_h{H}.csv"

signals_df.to_csv(signals_file, sep=";", index=False)
rules_df.to_csv(rules_file, sep=";", index=False)

print(f"\n📌 СИГНАЛЫ СОХРАНЕНЫ: {signals_file}")
print(f"📌 ПРАВИЛА СОХРАНЕНЫ: {rules_file}")

# =======================
# 5) ОЦЕНКА КАЧЕСТВА ПРОГНОЗОВ
# =======================
print("\n📊 РАСПРЕДЕЛЕНИЕ СИГНАЛОВ:")
signal_counts = signals_df["signal"].value_counts()

buy_count = int(signal_counts.get("BUY", 0))
sell_count = int(signal_counts.get("SELL", 0))
hold_count = int(signal_counts.get("HOLD", 0))

total_signals = len(signals_df)
hold_ratio = hold_count / total_signals if total_signals > 0 else np.nan

print(signal_counts)
print(f"Доля HOLD: {hold_ratio:.6f}")

buy_df = signals_df[signals_df["signal"] == "BUY"].copy()
sell_df = signals_df[signals_df["signal"] == "SELL"].copy()
tradable_df = signals_df[signals_df["signal"].isin(["BUY", "SELL"])].copy()

buy_acc = buy_df["correct"].mean() if not buy_df.empty else np.nan
sell_acc = sell_df["correct"].mean() if not sell_df.empty else np.nan
overall_acc = tradable_df["correct"].mean() if not tradable_df.empty else np.nan

avg_buy_ret = buy_df["actual_ret"].mean() if not buy_df.empty else np.nan
avg_sell_ret = sell_df["actual_ret"].mean() if not sell_df.empty else np.nan

avg_strategy_ret = signals_df["strategy_ret"].mean() if not signals_df.empty else np.nan
cum_strategy_ret = (1 + signals_df["strategy_ret"].fillna(0)).prod() - 1 if not signals_df.empty else np.nan

print("\n🎯 ОЦЕНКА КАЧЕСТВА ПРОГНОЗА:")
print(f"BUY accuracy: {buy_acc:.6f}" if pd.notna(buy_acc) else "BUY accuracy: nan")
print(f"SELL accuracy: {sell_acc:.6f}" if pd.notna(sell_acc) else "SELL accuracy: nan")
print(f"Общая accuracy по BUY/SELL: {overall_acc:.6f}" if pd.notna(overall_acc) else "Общая accuracy: nan")
print(f"Средняя фактическая доходность после BUY: {avg_buy_ret:.6f}" if pd.notna(avg_buy_ret) else "Средняя доходность после BUY: nan")
print(f"Средняя фактическая доходность после SELL: {avg_sell_ret:.6f}" if pd.notna(avg_sell_ret) else "Средняя доходность после SELL: nan")
print(f"Средняя доходность стратегии: {avg_strategy_ret:.6f}" if pd.notna(avg_strategy_ret) else "Средняя доходность стратегии: nan")
print(f"Накопленная доходность стратегии: {cum_strategy_ret:.6f}" if pd.notna(cum_strategy_ret) else "Накопленная доходность стратегии: nan")

# =======================
# 6) АГРЕГАЦИЯ УНИКАЛЬНЫХ ПРАВИЛ
# =======================
uniq2 = rules_df[rules_df["action"] != "HOLD"].copy()

if uniq2.empty:
    best_action = pd.DataFrame(columns=[
        "context", "rule", "action", "n_windows",
        "exp_ret_mean", "p_up_mean", "p_down_mean",
        "lift_mean", "confidence_mean", "support_mean",
        "count_mean", "edge_mean", "score", "abs_exp_ret_mean"
    ])
else:
    agg = uniq2.groupby(["context", "rule", "action"], as_index=False).agg(
        n_windows=("date", "count"),
        exp_ret_mean=("exp_ret", "mean"),
        p_up_mean=("p_up", "mean"),
        p_down_mean=("p_down", "mean"),
        lift_mean=("lift", "mean"),
        confidence_mean=("confidence", "mean"),
        support_mean=("support", "mean"),
        count_mean=("count", "mean"),
        edge_mean=("edge", "mean"),
    )

    agg["score"] = (
        agg["edge_mean"].abs()
        * agg["confidence_mean"]
        * np.log1p(agg["count_mean"])
        * agg["lift_mean"]
    )

    agg = agg.sort_values(
        ["context", "score", "n_windows", "confidence_mean"],
        ascending=[True, False, False, False]
    )

    best_action = agg.groupby(["context", "rule"], as_index=False).head(1).copy()
    best_action["abs_exp_ret_mean"] = best_action["exp_ret_mean"].abs()

    best_action = best_action.sort_values(
        ["score", "n_windows", "abs_exp_ret_mean", "lift_mean"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

print("\n📌 ТОП-30 УНИКАЛЬНЫХ ПРАВИЛ:")
print(best_action.head(30))

# =======================
# 7) СТАТИСТИКА ДЛЯ ДИПЛОМА
# =======================
best_action_file = f"SBER_best_action_k{K}_h{H}.csv"
best_action.to_csv(best_action_file, sep=";", index=False)
print(f"\n📌 УНИКАЛЬНЫЕ ПРАВИЛА СОХРАНЕНЫ: {best_action_file}")

summary_df = pd.DataFrame([{
    "K": K,
    "H": H,
    "n_rules": len(best_action),
    "BUY": buy_count,
    "SELL": sell_count,
    "HOLD": hold_count,
    "HOLD_ratio": hold_ratio,
    "BUY_accuracy": buy_acc,
    "SELL_accuracy": sell_acc,
    "overall_accuracy": overall_acc,
    "avg_strategy_ret": avg_strategy_ret,
    "cum_strategy_ret": cum_strategy_ret
}])

summary_file = f"SBER_summary_eval_k{K}_h{H}.csv"
summary_df.to_csv(summary_file, sep=";", index=False)

print(f"📌 КРАТКАЯ СВОДКА СОХРАНЕНА: {summary_file}")

print("\n📌 КРАТКАЯ СВОДКА ДЛЯ ДИПЛОМА:")
print(summary_df.to_string(index=False))