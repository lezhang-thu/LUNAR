import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- Config ----------
XLSX_PATH = "USD-lunar-sculp-price.xlsx"
OUT_PDF = "lunar-sculp-price-labeled.pdf"  # vector
OUT_PNG = "lunar-sculp-price-labeled.png"

type_to_metric = {
    "input_cache_hit_tokens": "cache hit",
    "input_cache_miss_tokens": "cache miss",
    "output_tokens": "output",
    "request_count": "request count",
}
metrics = ["cache hit", "cache miss", "output", "request count"]

# legend order (must be SCULP, LUNAR, AdaParser)
series_order = ["SCULP", "LUNAR", "AdaParser"]
# ---------------------------


def load_excel_with_sum(path: str):
    """
    Expected excel layout:
    - 1st column has model header rows (e.g., LUNAR/SCULP/AdaParser)
    - rows beneath have columns: type, amount, ... and a SUM row:
        amount == "SUM" and its numeric value stored in column "RMB" (per your xlsx)
    Returns:
      data: {model: {type: amount}}
      sums: {model: sum_value_float}
    """
    df = pd.read_excel(path, sheet_name=0)

    first_col = df.columns[0]  # e.g., "LUNAR" (as in your file header)
    current_model = str(first_col).strip()

    data = {}
    sums = {}

    for _, row in df.iterrows():
        cell = row.get(first_col)

        # model header line
        if isinstance(cell, str) and cell.strip() and cell.strip().lower() != "nan":
            current_model = cell.strip()
            continue

        t = row.get("type")
        amt = row.get("amount")

        # SUM row: in your file, amount column is string "SUM"
        if isinstance(amt, str) and amt.strip().upper() == "SUM":
            sv = row.get("RMB")
            try:
                sums[current_model] = float(sv)
            except Exception:
                pass
            continue

        # normal rows: need a valid type + numeric amount
        if not isinstance(t, str) or not t.strip():
            continue

        try:
            amt = float(amt)
        except Exception:
            continue

        data.setdefault(current_model, {})[t.strip()] = amt

    return data, sums


def main():
    data, sums = load_excel_with_sum(XLSX_PATH)

    # Prepare values:
    # - tokens on left axis in Millions (M)
    # - request_count on right axis in unit=1
    tokens_M = {m: [] for m in series_order}
    req = {m: np.nan for m in series_order}

    for model in series_order:
        for met in metrics:
            type_key = next(k for k, v in type_to_metric.items() if v == met)
            val = data.get(model, {}).get(type_key, np.nan)

            if met == "request count":
                req[model] = val
                tokens_M[model].append(0.0)  # placeholder on left axis
            else:
                # excel amount is token count -> convert to M
                tokens_M[model].append(val / 1e6 if pd.notna(val) else np.nan)

    # ---------- Plot ----------
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax2 = ax.twinx()

    x = np.arange(len(metrics))
    m = len(series_order)
    width = 0.22
    gap = 1.05
    offsets = (np.arange(m) - (m - 1) / 2) * width * gap

    # Tokens bars (left axis)
    colors = []
    token_bar_containers = []
    for i, model in enumerate(series_order):
        cont = ax.bar(x + offsets[i], tokens_M[model], width=width, label=model)
        token_bar_containers.append(cont)
        colors.append(cont.patches[0].get_facecolor())

    # Request count bars (right axis) only at last category
    req_x = x[-1]
    req_bar_containers = []
    for i, model in enumerate(series_order):
        cont = ax2.bar(req_x + offsets[i], req[model], width=width, color=colors[i], label="_nolegend_")
        req_bar_containers.append(cont)

    # Axes labels / ticks
    ax.set_ylabel("Tokens (M)")
    ax2.set_ylabel("")  # request unit is 1, keep unlabeled as requested
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    # ---- Headroom: increase y-limit so legend won't overlap bars ----
    all_tokens = []
    for model in series_order:
        all_tokens.extend([v for v in tokens_M[model][0:3] if pd.notna(v)])
    max_tokens = max(all_tokens) if all_tokens else 1.0
    ax.set_ylim(0, max_tokens * 1.50)

    all_req = [req[m] for m in series_order if pd.notna(req[m])]
    max_req = max(all_req) if all_req else 1.0
    ax2.set_ylim(0, max_req * 1.50)

    # ---- Value annotations on each bar top ----
    # Tokens (first 3 categories)
    for cont in token_bar_containers:
        for j, rect in enumerate(cont.patches):
            if j == len(metrics) - 1:
                continue  # skip request_count placeholder
            h = rect.get_height()
            if not np.isfinite(h) or h <= 0:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2,
                    h + max_tokens * 0.02,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10)

    # Request count (last category)
    for cont in req_bar_containers:
        rect = cont.patches[0]
        h = rect.get_height()
        if np.isfinite(h):
            ax2.text(rect.get_x() + rect.get_width() / 2,
                     h + max_req * 0.02,
                     f"{int(h) if float(h).is_integer() else h:g}",
                     ha="center", va="bottom", fontsize=10)

    # Spines: closed box on primary axis
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(1.0)
    ax2.spines["top"].set_visible(False)  # avoid double top line

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

    # -------- Legend with SUM ($xxx) --------
    legend_labels = []
    for model in series_order:
        sv = sums.get(model, np.nan)
        if np.isfinite(sv):
            legend_labels.append(f"{model} / ${sv:.3f}")
        else:
            legend_labels.append(model)

    legend_handles = [Patch(facecolor=colors[i], edgecolor="none", label=legend_labels[i])
                      for i in range(len(series_order))]

    ax.legend(handles=legend_handles,
              loc="upper left",
              bbox_to_anchor=(0.01, 0.99),
              frameon=True)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
