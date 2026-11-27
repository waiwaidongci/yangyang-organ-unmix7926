MIC_LABELS = {
    "mic_pipe": "管口近场麦",
    "mic_bellows": "风箱侧麦",
    "mic_action": "键机侧麦",
    "mic_room": "室内参考麦",
}
MIC_KEYS = tuple(MIC_LABELS)
CALIBRATION = "calibration"
HOLDOUT = "holdout"
SPLIT_LABELS = {CALIBRATION: "校准段", HOLDOUT: "留样段"}
MIN_CALIBRATION = 48
MIN_HOLDOUT = 24
MIN_RATE = 1_000
MAX_RATE = 192_000
MAX_INDEX = 2**31 - 1

FIELDS = {
    "id": "采样记录编号",
    "sample_index": "采样点序号（从零计）",
    "sample_rate_hz": "采样率（Hz）",
    "split": "数据段（calibration 或 holdout）",
    **{key: f"{label}（归一化满量程）" for key, label in MIC_LABELS.items()},
}

SPEC = {
    "columns": list(FIELDS),
    "fields": FIELDS,
    "settings": [
        {"key": "n_components", "label": "保留成分数", "default": 3, "min": 2, "max": 4, "step": 1},
        {"key": "max_iter", "label": "迭代上限", "default": 600, "min": 5, "max": 1000, "step": 1},
    ],
    "modules": {"datasets": "麦阵采样", "runs": "拆分运行", "results": "成分与残差", "reviews": "试听复核"},
    "limitation": "FastICA 假设声源近似独立且同步线性混合。成分的编号、极性和增益由算法约定，声源身份需要试听确认。",
    "dataset_name": "LHY-G07-064_台架四麦.csv",
    "recommendation": "先核对迭代状态和留样相关性，再按残差排序复听采样窗口，最后交接成分采样及矩阵。",
}
