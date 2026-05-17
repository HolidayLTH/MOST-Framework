import os
import matplotlib.pyplot as plt
import pandas as pd
from func_model_process import (
    create_aggregated_comparison_table,
    parse_log_file
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

def _build_log_candidates(model: str, version: str, events: str) -> list:
    suffix_map = {
        "TFT-GAT": "",
        "LSTM": "_LSTM",
        "Transformer": "_Transformer",
        "STGCN": "_STGCN",
        "DCRNN": "_DCRNN",
        "GraphWaveNet": "_GraphWaveNet",
        "STTN": "_STTN",
        "GMAN": "_GMAN",
        "PI-MPN": "_PI-MPN",
        "PAG-STAN": "_PAG-STAN",
    }
    suffix = suffix_map.get(model, "")
    file_name = f"evaluation_log_SPA_{version}_LAB_{events}{suffix}.txt"

    if model == "PAG-STAN":
        folders = ["PAG-STAN", ""]
    else:
        folders = [model]

    candidates = []
    for folder in folders:
        if folder:
            candidates.append(os.path.join(BASE_DIR, "Log", folder, file_name))
        else:
            candidates.append(os.path.join(BASE_DIR, "Log", file_name))
    return candidates


def _load_loginfo(model: str, version: str, events: str) -> list:
    for path in _build_log_candidates(model, version, events):
        if os.path.exists(path):
            return parse_log_file(path)
    raise FileNotFoundError(
        f"No log file found for model={model}, version={version}, events={events}"
    )


def _select_metric_columns(model_df: pd.DataFrame, metric: str) -> str:
    metric_lower = metric.lower()
    exact = [c for c in model_df.columns if c.lower() == metric_lower]
    if exact:
        return exact[0]

    endswith = [c for c in model_df.columns if c.lower().endswith(f"_{metric_lower}")]
    if endswith:
        return endswith[0]

    contains = [c for c in model_df.columns if metric_lower in c.lower()]
    if contains:
        return contains[0]

    raise KeyError(f"Metric {metric} not found in columns: {list(model_df.columns)}")


def _build_model_df(loginfo: list, model_name: str) -> pd.DataFrame:
    agg_table = create_aggregated_comparison_table(
        loginfo,
        loginfo,
        model_A_name=model_name,
        model_B_name=f"{model_name}__dup",
    )
    if agg_table.empty:
        return pd.DataFrame()

    model_cols = [col for col in agg_table.columns if col[1] == model_name]
    model_df = agg_table[model_cols].copy()
    model_df.columns = [col[0] for col in model_cols]

    if isinstance(model_df.index, pd.MultiIndex):
        model_df = model_df.reset_index()
        if "Date" in model_df.columns:
            model_df = model_df.set_index("Date")
    return model_df


def summarize_performance(events: str, pattern: str):
    """
    Summarize model performance for the given events and pattern.

    Args:
        events: Event set, e.g. "E1_E2_E3_E4_E5".
        pattern: "OP", "AB", or "EX".
    """
    pattern = pattern.upper()
    metrics = ["MAE", "RMSE", "sMAPE", "MPIW", "PICP"]
    baselines = [
        "LSTM",
        "Transformer",
        "STGCN",
        "DCRNN",
        "GraphWaveNet",
        "STTN",
        "GMAN",
        "PI-MPN",
        "PAG-STAN",
    ]

    if pattern == "OP":
        rows = {}
        tft_log = _load_loginfo("TFT-GAT", "PHY_STA", events)
        tft_df = _build_model_df(tft_log, "TFT-GAT")
        rows["TFT-GAT"] = {
            metric: tft_df[_select_metric_columns(tft_df, metric)].mean()
            for metric in metrics
        }

        for model in baselines:
            loginfo = _load_loginfo(model, "NOPHY_NOSTA", events)
            model_df = _build_model_df(loginfo, model)
            rows[model] = {
                metric: model_df[_select_metric_columns(model_df, metric)].mean()
                for metric in metrics
            }

        return pd.DataFrame.from_dict(rows, orient="index")[metrics]

    if pattern == "AB":
        versions = {
            "PHY_STA": "MOST",
            "NOPHY_STA": "MOST w/o Ohm",
            "PHY_NOSTA": "MOST w/o KCL",
            "NOPHY_NOSTA": "MOST-Vanilla",
        }
        metric_tables = {}

        for metric in metrics:
            series_list = []
            for version, label in versions.items():
                loginfo = _load_loginfo("TFT-GAT", version, events)
                model_df = _build_model_df(loginfo, "TFT-GAT")
                metric_col = _select_metric_columns(model_df, metric)
                series = model_df[metric_col].rename(label)
                series_list.append(series)

            metric_df = pd.concat(series_list, axis=1)
            if "MOST-Vanilla" in metric_df.columns:
                metric_df = metric_df.sort_values(by="MOST-Vanilla", ascending=True)
            metric_tables[metric] = metric_df

        return metric_tables

    if pattern == "EX":
        tft_log = _load_loginfo("TFT-GAT", "PHY_STA", events)
        tft_df = _build_model_df(tft_log, "TFT-GAT")
        tft_metric_means = {
            metric: tft_df[_select_metric_columns(tft_df, metric)].mean()
            for metric in metrics
        }

        metric_tables = {}
        for metric in metrics:
            rows = {}
            for model in baselines:
                log_original = _load_loginfo(model, "NOPHY_NOSTA", events)
                df_original = _build_model_df(log_original, model)
                val_original = df_original[_select_metric_columns(df_original, metric)].mean()

                log_ov = _load_loginfo(model, "PHY_STA", events)
                df_ov = _build_model_df(log_ov, model)
                val_ov = df_ov[_select_metric_columns(df_ov, metric)].mean()

                rows[model] = {
                    "Original": val_original,
                    "OV": val_ov,
                    "MOST": tft_metric_means[metric],
                }
            metric_tables[metric] = pd.DataFrame.from_dict(rows, orient="index")

        return metric_tables

    raise ValueError(f"Unknown pattern: {pattern}")

if __name__ == "__main__":
    op_summary = summarize_performance("E1_E2_E3_E4_E5", "OP") # Overall performance
    print('Overall performance')
    print(op_summary)
    ab_summary = summarize_performance("E1_E2_E3_E4_E5", "AB") # Ablation study
    print('Ablation study')
    print(ab_summary["MAE"])
    ex_summary = summarize_performance("E6_E7_E8_E9_E10", "EX") # Extension study
    print('Extension study')
    print(ex_summary["MAE"])