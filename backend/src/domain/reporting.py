import numpy as np

from .diagnostics import (
    CORRELATION_REVIEW_LEVEL,
    RESIDUAL_REVIEW_PERCENT,
    dependence_rows,
    residual_rows,
    review_rows,
    rms,
)
from .schema import HOLDOUT, MIC_KEYS, MIC_LABELS, SPEC
from .separation import Separation


def _table(key: str, title: str, labels: dict[str, str], records: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": records}


def build_report(rows: list[dict], microphones: np.ndarray, calibration_count: int, separation: Separation) -> dict:
    components = [f"ic_{index + 1}" for index in range(separation.scores.shape[1])]
    score_labels = {key: f"成分 {index}（校准单位方差）" for index, key in enumerate(components, 1)}
    tracks = [
        {
            "id": row["id"],
            "sample_index": row["sample_index"],
            "sample_rate_hz": row["sample_rate_hz"],
            "split": row["split"],
            **{component: float(value) for component, value in zip(components, sample)},
        }
        for row, sample in zip(rows, separation.scores)
    ]
    axes = [
        {
            "channel": channel,
            "calibration_mean": float(separation.mean[mic_index]),
            **{f"mix_{key}": float(separation.mixing[mic_index, j]) for j, key in enumerate(components)},
            **{f"unmix_{key}": float(separation.unmixing[j, mic_index]) for j, key in enumerate(components)},
        }
        for mic_index, channel in enumerate(MIC_KEYS)
    ]
    dependence = dependence_rows(separation.scores, calibration_count, components)
    channel_errors = residual_rows(microphones, separation, calibration_count)
    calibration_scale = float(rms(microphones[:calibration_count] - separation.mean))
    holdout_error_pct = 100 * float(rms(separation.residual[calibration_count:])) / calibration_scale
    windows = review_rows(rows, separation.residual, calibration_count, calibration_scale)
    holdout_pairs = [row for row in dependence if row["split"] == HOLDOUT]
    max_pearson = max((abs(row["pearson"]) for row in holdout_pairs if row["pearson"] is not None), default=None)
    flagged_pairs = [
        row for row in holdout_pairs
        if any(abs(row[key]) >= CORRELATION_REVIEW_LEVEL
               for key in ("pearson", "amplitude_spearman") if row[key] is not None)
    ]

    warnings = []
    if separation.iteration_limit_reached:
        warnings.append("FastICA 触及迭代上限，成分可能尚未收敛。请提高迭代上限后重跑，当前结果仅供诊断。")
    if separation.rank < len(MIC_KEYS):
        warnings.append(f"校准四麦的有效秩为 {separation.rank}，请检查通道重复、串接和静音。")
    clipped = [label for index, label in enumerate(MIC_LABELS.values()) if np.any(np.abs(microphones[:, index]) == 1)]
    if clipped:
        warnings.append(f"{'、'.join(clipped)}存在满量程采样，请复核是否削顶。")
    if any(row["holdout_residual_pct"] is None for row in channel_errors):
        warnings.append("有麦克风在校准段几乎静音，该通道的相对残差留空，请查看绝对残差。")
    if holdout_error_pct >= RESIDUAL_REVIEW_PERCENT:
        warnings.append(f"留样相对残差达到 {RESIDUAL_REVIEW_PERCENT:g}%，请按窗口顺序复听并检查遗漏成分。")
    if flagged_pairs:
        warnings.append(f"留样有 {len(flagged_pairs)} 对成分的相关绝对值达到 {CORRELATION_REVIEW_LEVEL:g}，请复听是否串入。")
    if any(row["pearson"] is None or row["amplitude_spearman"] is None for row in holdout_pairs):
        warnings.append("留样中有成分或其绝对幅度近乎恒定，无法计算的相关项已留空。")

    next_step = "请先调整迭代上限并重跑。" if separation.iteration_limit_reached else SPEC["recommendation"]
    return {
        "summary": (
            f"用 {calibration_count} 条校准采样拟合 FastICA，拆出 {len(components)} 条校准单位方差成分，"
            f"另用 {len(rows) - calibration_count} 条留样采样核查。成分按校准时对四麦的贡献强度排列，"
            "各成分的最大绝对载荷取正。编号、极性与原声增益存在歧义，声源身份请听录音确认。"
            f"残差衡量四麦重建误差，相关性用于发现可疑串入。相关较小仍需试听核实。{next_step}"
        ),
        "metrics": {
            "保留成分数": len(components),
            "校准通道有效秩": separation.rank,
            "FastICA 迭代次数": separation.iterations,
            "留样相对残差（%）": round(holdout_error_pct, 6),
            "留样最大绝对 Pearson 相关": round(max_pearson, 6) if max_pearson is not None else None,
        },
        "tables": [
            _table("component_tracks", "逐点成分采样", {
                **{key: SPEC["fields"][key] for key in ("id", "sample_index", "sample_rate_hz", "split")},
                **score_labels,
            }, tracks),
            _table("reconstruction_model", "麦通道中心值、混合与解混系数", {
                "channel": "麦通道字段",
                "calibration_mean": "校准中心值（满量程）",
                **{f"mix_{key}": f"混合系数 {key}" for key in components},
                **{f"unmix_{key}": f"解混系数 {key}" for key in components},
            }, axes),
            _table("channel_residuals", "四麦重建残差", {
                "channel": "麦通道",
                "calibration_signal_rms": "校准信号 RMS（去中心）",
                "calibration_residual_rms": "校准残差 RMS",
                "holdout_residual_rms": "留样残差 RMS",
                "holdout_residual_pct": "留样残差／校准信号（%）",
            }, channel_errors),
            _table("component_dependence", "成分两两相关性", {
                "split": "数据段",
                "left": "成分一",
                "right": "成分二",
                "pearson": "Pearson 线性相关",
                "amplitude_spearman": "绝对幅度 Spearman 秩相关",
            }, dependence),
            _table("holdout_review", "留样试听顺序", {
                "priority": "试听次序",
                "first_id": "首条记录编号",
                "first_sample": "起始采样点",
                "last_sample": "末采样点（含）",
                "start_ms": "起点（毫秒）",
                "end_ms_exclusive": "终点（毫秒，不含）",
                "sample_count": "采样点数",
                "residual_pct": "窗口残差／校准信号（%）",
                "peak_channel": "残差 RMS 最大的麦通道",
            }, windows),
        ],
        "warnings": warnings,
    }
