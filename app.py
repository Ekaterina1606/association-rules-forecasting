# app.py
import numpy as np
import pandas as pd
import streamlit as st

# =======================
# Проверка SciPy
# =======================
try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.stats import norm
except Exception as e:
    st.set_page_config(page_title="Ошибка зависимостей", layout="wide")
    st.error("Не установлен пакет scipy — из-за этого приложение не может запуститься.")
    st.code(str(e))
    st.info("Решение: установите scipy (pip install scipy) или добавьте scipy в requirements.txt.")
    st.stop()

# =======================
# ДЕФОЛТЫ (UI)
# =======================
DEFAULTS = {
    "sep": ";",
    "alphabet_size": 7,
    "k": 5,
    "mode": "Нормально",
    "forecast_days": 10,
}

# =======================
# ВНУТРЕННИЕ ПАРАМЕТРЫ
# =======================
LAMBDA_HP = 14400
MA_WINDOW = 10
INTERNAL_WINDOW = 252
PROB_THRESHOLD = 0.5
HOLD_RET_THRESHOLD = 0.05


STRICTNESS_PRESETS = {
    "Мягко": {"min_support": 0.005, "min_confidence": 0.10, "min_count": 2},
    "Нормально": {"min_support": 0.0125, "min_confidence": 0.14, "min_count": 3},
    "Строго": {"min_support": 0.02, "min_confidence": 0.20, "min_count": 5},
}


def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_to_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# =======================
# CORE
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

    df = pd.DataFrame({"a": A, "b": B})
    df["rule"] = df["a"] + "->" + df["b"]

    total = len(df)

    rule_counts = df["rule"].value_counts()
    a_counts = df["a"].value_counts()
    b_counts = df["b"].value_counts()

    rules = rule_counts.rename("count").to_frame()
    rules["support"] = rules["count"] / total
    rules = rules.reset_index().rename(columns={"index": "rule"})

    rules["a"] = rules["rule"].str.split("->").str[0]
    rules["b"] = rules["rule"].str.split("->").str[1]

    rules["confidence"] = rules.apply(lambda r: r["count"] / a_counts[r["a"]], axis=1)

    rules["lift"] = rules.apply(
        lambda r: r["support"] / ((a_counts[r["a"]] / total) * (b_counts[r["b"]] / total)),
        axis=1,
    )
    return rules


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит названия колонок к стандартным begin / close.
    Поддерживает популярные варианты из Yahoo / Investing и др.
    """
    df = df.copy()

    rename_map = {}
    for col in df.columns:
        col_norm = str(col).strip().lower()

        if col_norm in {"begin", "date", "datetime", "time"}:
            rename_map[col] = "begin"
        elif col_norm in {"close", "price", "adj close", "adj_close", "last", "last price"}:
            rename_map[col] = "close"

    return df.rename(columns=rename_map)


def parse_close_series(series: pd.Series) -> pd.Series:
    """
    Гибко преобразует цены в числа.
    Поддерживает варианты:
    - 1234.56
    - 1,234.56
    - 1234,56
    - 1 234,56
    - 1 234.56
    """
    s = series.astype(str).str.strip()

    # убираем неразрывные пробелы и обычные пробелы
    s = s.str.replace("\u00A0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    def normalize_number(x: str) -> str:
        x = x.strip()

        if x == "" or x.lower() in {"nan", "none", "null"}:
            return ""

        # если есть и запятая, и точка:
        # считаем, что последний символ-разделитель дробной части определяет формат
        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                # формат типа 1.234,56 -> убрать точки, запятую заменить на точку
                x = x.replace(".", "")
                x = x.replace(",", ".")
            else:
                # формат типа 1,234.56 -> убрать запятые
                x = x.replace(",", "")
            return x

        # если есть только запятая
        if "," in x:
            # одна запятая и справа 1-2 цифры -> вероятно десятичная
            if x.count(",") == 1:
                left, right = x.split(",")
                if right.isdigit() and 1 <= len(right) <= 3:
                    return left + "." + right

            # иначе считаем запятую разделителем тысяч
            return x.replace(",", "")

        # если только точки — оставляем как есть
        return x

    s = s.apply(normalize_number)
    return pd.to_numeric(s, errors="coerce")


def read_uploaded_csv(uploaded_file, user_sep: str) -> pd.DataFrame:
    """
    Умеет читать CSV с разделителем ; или ,.
    Сначала пробует разделитель, указанный пользователем,
    затем пытается подобрать автоматически.
    """
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    candidate_seps = []
    if user_sep:
        candidate_seps.append(user_sep)
    for sep in [",", ";", "\t"]:
        if sep not in candidate_seps:
            candidate_seps.append(sep)

    best_df = None
    best_score = -1

    for sep in candidate_seps:
        try:
            uploaded_file.seek(0)
            df_try = pd.read_csv(uploaded_file, sep=sep)
            df_try = normalize_columns(df_try)

            score = 0
            cols = set(df_try.columns)
            if "begin" in cols:
                score += 1
            if "close" in cols:
                score += 1

            # если файл не распарсился и всё ушло в 1 колонку — это плохой вариант
            if df_try.shape[1] > 1:
                score += 1

            if score > best_score:
                best_score = score
                best_df = df_try.copy()
        except Exception:
            continue

    if best_df is None:
        raise ValueError("Не удалось прочитать CSV-файл. Проверьте разделитель и формат файла.")

    return best_df


def preprocess_df(df: pd.DataFrame, window_min: int, forecast_days: int) -> pd.DataFrame:
    df = normalize_columns(df)

    needed = {"begin", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"В файле не хватает столбцов: {', '.join(sorted(missing))}")

    df = df.copy()

    # гибкий разбор даты
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce", dayfirst=False)

    # гибкий разбор цены
    df["close"] = parse_close_series(df["close"])

    df = df.dropna(subset=["begin", "close"]).sort_values("begin").reset_index(drop=True)

    _, cycle = hp_filter(df["close"].values, LAMBDA_HP)
    df["cycle_hp_ma"] = pd.Series(cycle).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    df = df.dropna().reset_index(drop=True)

    forecast_days = int(max(1, forecast_days))
    df["ret_fwd"] = df["close"].pct_change(periods=forecast_days).shift(-forecast_days)
    df = df.dropna().reset_index(drop=True)

    if len(df) < window_min:
        raise ValueError(
            f"Слишком мало данных после предобработки.\n"
            f"Нужно хотя бы ~{window_min} строк, а получилось {len(df)}.\n"
            f"Попробуйте уменьшить 'Прогноз на сколько дней вперёд' или загрузить файл с более длинной историей."
        )

    return df


def add_rule_stats(df_window: pd.DataFrame, sax: np.ndarray, rules: pd.DataFrame, k: int) -> pd.DataFrame:
    rets_next = df_window["ret_fwd"].values
    idx = np.arange(k, len(sax))

    A = ["".join(sax[i - k:i]) for i in idx]
    B = [sax[i] for i in idx]
    R = [rets_next[i] for i in idx]

    trans = pd.DataFrame({"a": A, "b": B, "ret": R})
    trans["rule"] = trans["a"] + "->" + trans["b"]

    stats = trans.groupby("rule")["ret"].agg(
        exp_ret="mean",
        p_up=lambda x: float((x > 0).mean()),
        p_down=lambda x: float((x < 0).mean()),
    ).reset_index()

    return rules.merge(stats, on="rule", how="left")


def decide_action(row) -> str:
    if pd.isna(row.get("exp_ret")):
        return "HOLD"

    th = PROB_THRESHOLD
    exp_ret = row.get("exp_ret", np.nan)
    p_up = row.get("p_up", np.nan)
    p_down = row.get("p_down", np.nan)

    if (p_up >= th) and (exp_ret > 0):
        return "BUY"

    if (p_down >= th) and (exp_ret < 0):
        return "SELL"

    return "HOLD"


def calc_rule_score(row) -> float:
    edge = row.get("edge", np.nan)
    confidence = row.get("confidence", np.nan)
    count = row.get("count", np.nan)
    lift = row.get("lift", np.nan)

    if pd.isna(edge) or pd.isna(confidence) or pd.isna(count) or pd.isna(lift):
        return np.nan

    return abs(edge) * confidence * np.log1p(count) * lift


def build_rules_for_df(df, alphabet_size, k, min_support, min_confidence, min_count) -> pd.DataFrame:
    s = df["cycle_hp_ma"].values
    s_std = np.std(s)
    if s_std == 0 or np.isnan(s_std):
        return pd.DataFrame()

    s_norm = (s - np.mean(s)) / s_std
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))[:alphabet_size]
    breakpoints = norm.ppf([i / alphabet_size for i in range(1, alphabet_size)])

    sax = sax_symbolize(s_norm, breakpoints, alphabet)
    rules = rules_from_sax_k(sax, k=k)
    rules = add_rule_stats(df, sax, rules, k=k)

    rules = rules[
        (rules["count"] >= min_count)
        & (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
    ].copy()

    if rules.empty:
        return rules

    rules["edge"] = rules["p_up"] - rules["p_down"]
    rules["action"] = rules.apply(decide_action, axis=1)
    rules["score"] = rules.apply(calc_rule_score, axis=1)
    rules["strength"] = rules["score"]

    return rules


def predict_next_action(df, window, alphabet_size, k, min_support, min_confidence, min_count):
    df_window = df.iloc[-window:].copy()

    s = df_window["cycle_hp_ma"].values
    s_std = np.std(s)
    if s_std == 0 or np.isnan(s_std):
        return "HOLD", pd.DataFrame(), None, None

    s_norm = (s - np.mean(s)) / s_std
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))[:alphabet_size]
    breakpoints = norm.ppf([i / alphabet_size for i in range(1, alphabet_size)])

    sax = sax_symbolize(s_norm, breakpoints, alphabet)

    rules = rules_from_sax_k(sax, k=k)
    rules = add_rule_stats(df_window, sax, rules, k=k)

    rules = rules[
        (rules["count"] >= min_count)
        & (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
    ].copy()

    current_ctx = "".join(sax[-k:])
    if rules.empty:
        return "HOLD", pd.DataFrame(), None, current_ctx

    rules["edge"] = rules["p_up"] - rules["p_down"]
    rules["action"] = rules.apply(decide_action, axis=1)
    rules["score"] = rules.apply(calc_rule_score, axis=1)

    candidates = rules[rules["a"] == current_ctx].copy()

    chosen_rule = None
    action = "HOLD"

    if not candidates.empty:
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

        candidates = candidates.sort_values(
            ["score", "confidence", "count", "lift"],
            ascending=[False, False, False, False]
        )

    return action, candidates, chosen_rule, current_ctx


def render_action_badge(action: str):
    if action == "BUY":
        st.success("BUY (покупать)")
    elif action == "SELL":
        st.warning("SELL (продавать)")
    else:
        st.info("HOLD (ничего не делать)")


# =======================
# UI
# =======================
st.set_page_config(page_title="Ассоциативные правила + прогноз", layout="wide")
init_state()

st.markdown("""
<h1 style="
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 10px;
    line-height: 1.3;
">
Ассоциативные правила (SAX) и прогноз BUY/HOLD/SELL
</h1>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Настройки")

    st.selectbox(
        "Режим отбора правил",
        options=["Мягко", "Нормально", "Строго"],
        key="mode",
        help="Нормально — рекомендуемый режим. Мягко — больше сигналов, но больше шума. Строго — меньше сигналов, но надёжнее.",
    )

    st.slider(
        "Горизонт прогнозирования (H)",
        min_value=1,
        max_value=30,
        step=1,
        key="forecast_days",
        help=(
            "Определяет количество торговых дней, на которое формируется прогноз. "
            "H = 1 соответствует прогнозу на следующий торговый день. "
            "Увеличение значения H расширяет горизонт удержания позиции."
        ),
    )

    st.slider(
        "Размер окна SAX (k)",
        min_value=2,
        max_value=10,
        step=1,
        key="k",
        help=(
            "Параметр k задаёт длину подпоследовательности временного ряда, "
            "которая дискретизируется методом SAX и представляется "
            "в виде символьной последовательности."
        )
    )

    st.slider(
        "Размер алфавита SAX (S)",
        min_value=3,
        max_value=12,
        step=1,
        key="alphabet_size",
        help=(
            "Параметр S задаёт количество символов, используемых для "
            "дискретизации временного ряда. "
            "Увеличение S повышает детализацию представления, "
            "уменьшение — приводит к более грубой агрегации данных."
        ),
    )

st.info(
    "Требования к файлу:\n"
    "- Формат: CSV\n"
    "- Обязательные столбцы: `begin` (дата) и `close` (цена)\n"
    "- Данные должны идти по времени от старых к новым\n"
    "- Желательно 300+ строк данных\n"
)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

st.markdown("### Загрузка файла")
st.caption("Перетащите CSV-файл в область ниже или нажмите кнопку для выбора")

st.markdown("""
<style>
[data-testid="stFileUploader"] {
    margin-top: -40px;
}
</style>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv"])

if uploaded is None:
    st.stop()

try:
    raw = read_uploaded_csv(uploaded, st.session_state["sep"])

    st.subheader("Предпросмотр данных")
    st.dataframe(raw.head(20), use_container_width=True)

    window = INTERNAL_WINDOW
    k = int(st.session_state["k"])
    alphabet_size = int(st.session_state["alphabet_size"])
    forecast_days = int(st.session_state["forecast_days"])

    df = preprocess_df(raw, window_min=window + forecast_days + 5, forecast_days=forecast_days)
    st.success(f"Данные предобработаны. Строк после обработки: {len(df)}")

    st.subheader("График временного ряда цены закрытия")
    plot_df = df[["begin", "close"]].copy().set_index("begin")
    st.line_chart(plot_df)
    st.caption(f"Последняя дата: {df['begin'].iloc[-1]} | close: {df['close'].iloc[-1]:.6f}")

    mode = st.session_state["mode"]
    preset = STRICTNESS_PRESETS[mode]

    st.subheader("Оценка качества прогнозирования")

    results = []

    for i in range(INTERNAL_WINDOW, len(df) - forecast_days):
        df_slice = df.iloc[:i].copy()

        action, _, _, _ = predict_next_action(
            df=df_slice,
            window=INTERNAL_WINDOW,
            alphabet_size=alphabet_size,
            k=k,
            min_support=preset["min_support"],
            min_confidence=preset["min_confidence"],
            min_count=preset["min_count"],
        )

        actual_ret = df.iloc[i]["ret_fwd"]

        if pd.isna(actual_ret):
            continue

        if action == "BUY":
            correct = 1 if actual_ret > 0 else 0
            strategy_ret = actual_ret
        elif action == "SELL":
            correct = 1 if actual_ret < 0 else 0
            strategy_ret = -actual_ret
        else:  # HOLD
            correct = 1 if abs(actual_ret) <= HOLD_RET_THRESHOLD else 0
            strategy_ret = 0.0

        results.append({
            "action": action,
            "correct": correct,
            "ret": actual_ret,
            "strategy_ret": strategy_ret,
        })

    res_df = pd.DataFrame(results)

    # статистика
    # статистика
    buy_df = res_df[res_df["action"] == "BUY"]
    sell_df = res_df[res_df["action"] == "SELL"]
    hold_df = res_df[res_df["action"] == "HOLD"]
    tradable_df = res_df[res_df["action"].isin(["BUY", "SELL"])]

    n_buy = int((res_df["action"] == "BUY").sum()) if not res_df.empty else 0
    n_sell = int((res_df["action"] == "SELL").sum()) if not res_df.empty else 0
    n_hold = int((res_df["action"] == "HOLD").sum()) if not res_df.empty else 0

    # N_correct теперь включает корректные BUY, SELL и HOLD
    n_correct = int(res_df["correct"].sum()) if not res_df.empty else 0

    buy_acc = buy_df["correct"].mean() if not buy_df.empty else np.nan
    sell_acc = sell_df["correct"].mean() if not sell_df.empty else np.nan
    hold_acc = hold_df["correct"].mean() if not hold_df.empty else np.nan

    # Точность только по направленным сигналам BUY/SELL
    accuracy_dir = (
        tradable_df["correct"].mean()
        if not tradable_df.empty else np.nan
    )

    # Общая точность с учетом HOLD:
    # Accuracy_all = N_correct / (N_BUY + N_SELL + N_HOLD)
    accuracy_all = (
        n_correct / (n_buy + n_sell + n_hold)
        if (n_buy + n_sell + n_hold) > 0 else np.nan
    )

    hold_ratio = (res_df["action"] == "HOLD").mean() if not res_df.empty else np.nan

    avg_strategy_ret = res_df["strategy_ret"].mean() if not res_df.empty else np.nan
    cum_strategy_ret = (1 + res_df["strategy_ret"]).prod() - 1 if not res_df.empty else np.nan

    avg_buy_ret = buy_df["ret"].mean() if not buy_df.empty else np.nan
    avg_sell_ret = sell_df["ret"].mean() if not sell_df.empty else np.nan


    # ===== Метрики точности =====
    col1, col2 = st.columns(2)
    col1.metric(
        "Точность прогнозов BUY",
        f"{buy_acc:.3f}" if pd.notna(buy_acc) else "—"
    )
    col2.metric(
        "Точность прогнозов SELL",
        f"{sell_acc:.3f}" if pd.notna(sell_acc) else "—"
    )

    col3, col4 = st.columns(2)
    col3.metric(
        "Точность по направленным сигналам (BUY/SELL)",
        f"{accuracy_dir:.3f}" if pd.notna(accuracy_dir) else "—"
    )
    col4.metric(
        "Общая точность (включая HOLD)",
        f"{accuracy_all:.3f}" if pd.notna(accuracy_all) else "—"
    )


    col5, col6 = st.columns(2)
    col5.metric(
        "Доля сигналов HOLD",
        f"{hold_ratio:.3f}" if pd.notna(hold_ratio) else "—"
    )
    col6.metric(
        "Средняя доходность стратегии (%)",
        f"{avg_strategy_ret * 100:.2f}%" if pd.notna(avg_strategy_ret) else "—"
    )

    # распределение сигналов
    st.subheader("Распределение сигналов")
    import matplotlib.pyplot as plt

    counts = res_df["action"].value_counts()

    # фиксируем порядок + убираем NaN
    order = ["BUY", "HOLD", "SELL"]
    counts = counts.reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(counts.index, counts.values, color="#1f77b4", width=0.5)

    # подписи осей (крупные)
    ax.set_xlabel("Тип сигнала", fontsize=19)
    ax.set_ylabel("Количество сигналов", fontsize=19)

    # подписи делений
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=14)

    # сетка
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # убираем лишние рамки
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # центрируем график
    col1, col2, col3 = st.columns([1, 3.5, 1])

    with col2:
        st.pyplot(fig, use_container_width=False)

    action, candidates, chosen_rule, current_ctx = predict_next_action(
        df=df,
        window=window,
        alphabet_size=alphabet_size,
        k=k,
        min_support=preset["min_support"],
        min_confidence=preset["min_confidence"],
        min_count=preset["min_count"],
    )

    st.subheader(f"Главная рекомендация (горизонт: {forecast_days} дн.)")
    render_action_badge(action)
    st.caption(f"Режим: **{mode}** | Шаблон: **{current_ctx}** | Правило: **{chosen_rule or '—'}**")

    st.subheader("Кандидаты правил для текущего шаблона (в выбранном режиме)")
    if candidates is None or candidates.empty:
        st.write("Нет правил под текущий шаблон. Попробуйте «Мягко» или уменьшите K/ALPHABET_SIZE.")
    else:
        cols = [
            "rule", "count", "support", "confidence", "lift",
            "exp_ret", "p_up", "p_down", "edge", "score", "action"
        ]
        cols = [c for c in cols if c in candidates.columns]
        st.dataframe(candidates[cols].head(50), use_container_width=True)

    st.divider()
    st.subheader("20 ассоциативных правил")

    all_rules = build_rules_for_df(
        df=df,
        alphabet_size=alphabet_size,
        k=k,
        min_support=preset["min_support"],
        min_confidence=preset["min_confidence"],
        min_count=preset["min_count"],
    )

    if all_rules.empty:
        st.warning("Правила не найдены при текущих настройках. Попробуйте режим «Мягко» или уменьшите K.")
        st.stop()

    all_rules["abs_exp_ret"] = all_rules["exp_ret"].abs()
    top20_strong = all_rules.sort_values(
        ["score", "abs_exp_ret", "lift", "confidence"],
        ascending=[False, False, False, False]
    ).head(20)

    st.dataframe(
        top20_strong[
            ["rule", "count", "support", "confidence", "lift", "exp_ret", "p_up", "p_down", "edge", "score", "action"]
        ],
        use_container_width=True,
    )

    st.subheader("20 уникальных ассоциативных правил")
    top20_unique = (
        all_rules.sort_values(
            ["score", "abs_exp_ret", "lift", "confidence"],
            ascending=[False, False, False, False]
        )
        .drop_duplicates(subset=["a"])
        .head(20)
    )

    st.dataframe(
        top20_unique[
            ["rule", "a", "b", "count", "support", "confidence", "lift", "exp_ret", "p_up", "p_down", "edge", "score", "action"]
        ],
        use_container_width=True,
    )

except Exception as e:
    st.error(f"Ошибка обработки файла: {e}")