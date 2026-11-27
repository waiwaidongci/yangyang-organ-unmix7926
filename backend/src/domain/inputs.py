import math

from .schema import (
    CALIBRATION,
    HOLDOUT,
    MAX_INDEX,
    MAX_RATE,
    MIC_KEYS,
    MIN_CALIBRATION,
    MIN_HOLDOUT,
    MIN_RATE,
    SPEC,
)


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is int:
        result = value
    elif type(value) is float and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip().isascii() and value.strip().isdecimal():
        try:
            result = int(value)
        except ValueError:
            raise ValueError(f"{label}须在 {minimum} 至 {maximum} 之间。") from None
    else:
        raise ValueError(f"{label}须填写整数。")
    if not minimum <= result <= maximum:
        raise ValueError(f"{label}须在 {minimum} 至 {maximum} 之间。")
    return result


def _amplitude(value: object, label: str) -> float:
    if type(value) not in (int, float, str):
        raise ValueError(f"{label}须填写数值。")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{label}须填写数值。") from None
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{label}须为 −1 到 1 之间的有限数值。")
    return result


def normalize_rows(rows: list[dict]) -> list[dict]:
    expected = set(SPEC["columns"])
    records = []
    ids: set[str] = set()
    indices: set[int] = set()
    sample_rate = None
    for line, raw in enumerate(rows, 1):
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"第 {line} 条采样的字段须与模板完全相符（{', '.join(SPEC['columns'])}）。")
        record_id = raw["id"].strip() if isinstance(raw["id"], str) else ""
        if not record_id:
            raise ValueError(f"第 {line} 条采样缺少采样记录编号。")
        if record_id in ids:
            raise ValueError(f"采样记录编号重复（{record_id}）。")
        ids.add(record_id)

        prefix = f"采样 {record_id} 的"
        index = _integer(raw["sample_index"], f"{prefix}采样点序号", 0, MAX_INDEX)
        if index in indices:
            raise ValueError(f"采样点序号 {index} 重复。")
        indices.add(index)
        rate = _integer(raw["sample_rate_hz"], f"{prefix}采样率", MIN_RATE, MAX_RATE)
        if sample_rate is not None and rate != sample_rate:
            raise ValueError(f"采样 {record_id} 的采样率不同，一份文件只能使用同一采样率。")
        sample_rate = rate

        split = raw["split"].strip() if isinstance(raw["split"], str) else ""
        if split not in (CALIBRATION, HOLDOUT):
            raise ValueError(f"{prefix}数据段须填写 calibration 或 holdout。")
        records.append({
            "id": record_id,
            "sample_index": index,
            "sample_rate_hz": rate,
            "split": split,
            **{key: _amplitude(raw[key], f"{prefix}{SPEC['fields'][key]}") for key in MIC_KEYS},
        })

    calibration_count = sum(row["split"] == CALIBRATION for row in records)
    holdout_count = len(records) - calibration_count
    if calibration_count < MIN_CALIBRATION or holdout_count < MIN_HOLDOUT:
        raise ValueError(f"至少需要 {MIN_CALIBRATION} 条校准采样和 {MIN_HOLDOUT} 条留样采样。")
    records.sort(key=lambda row: row["sample_index"])
    first_index = records[0]["sample_index"]
    for offset, row in enumerate(records):
        if row["sample_index"] != first_index + offset:
            raise ValueError(f"采样点序号须连续，缺少序号 {first_index + offset}。")
        if offset < calibration_count and row["split"] != CALIBRATION:
            raise ValueError("校准段须连续排在留样段之前。")
    return records


def normalize_settings(settings: dict) -> dict[str, int]:
    return {
        field["key"]: _integer(
            settings.get(field["key"], field["default"]), field["label"], field["min"], field["max"],
        )
        for field in SPEC["settings"]
    }
