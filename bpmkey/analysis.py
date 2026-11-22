"""BPM 与调性分析：从单声道样本生成起始包络、速度、色度与调性。

公共约定
========
* 音频：``list[float]``，单声道，按显式传入的 ``sample_rate``（Hz）采样。
* hop：帧移/帧长，正整数采样点数；``hop <= 0`` 抛 :class:`AnalysisError`。
* BPM 范围：``(low, high)``，要求 ``0 < low < high``。测速结果必须落在
  该闭区间内；正常节拍信号的检测容差为 0.5 BPM（范围边界除外）。
* 调性：12 个音级，索引 0..11 依次为 C、C#、D、D#、E、F、F#、G、
  G#、A、A#、B。调性名形如 ``"C major"`` 或 ``"A minor"``。置信度是
  0..1 的浮点数：0 表示无调性或完全不可信，1 表示完全确定。
* 异常：输入非法（hop 非正、采样率非正、样本非有限、BPM 范围非法、
  样本过短等）统一抛 :class:`AnalysisError`；:func:`validate` 例外，
  它返回问题列表而不抛异常。
"""

import math


class AnalysisError(ValueError):
    """BPM/调性分析参数或音频非法时抛出的异常。"""


# ---------------------------------------------------------------------------
# 内部校验与常量
# ---------------------------------------------------------------------------

# Krumhansl–Kessler 自然小调音级模板，索引 0 为小调主音。
_KK_MINOR_PROFILE = [
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 科学音高记谱中的 C0（约 16.35 Hz），色度折叠的参考频率。
_C0_FREQ = 16.351597831287414  # 440 * 2 ** (-57 / 12)


def _is_real_number(value):
    """布尔值不算数值；int/float 且非 NaN/Inf 才是有限实数。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AnalysisError(f"{name} must be a positive integer, got {value!r}")


def _validate_positive_rate(value, name):
    if not _is_real_number(value) or value <= 0:
        raise AnalysisError(f"{name} must be a finite positive number, got {value!r}")


def _check_bpm_range(bpm_range):
    if not isinstance(bpm_range, (tuple, list)) or len(bpm_range) != 2:
        raise AnalysisError(f"invalid bpm range: {bpm_range!r}")
    low, high = bpm_range
    if not (_is_real_number(low) and _is_real_number(high)) or low <= 0 or low >= high:
        raise AnalysisError(f"invalid bpm range: {low}, {high}")
    return float(low), float(high)


def _validate_samples(samples):
    for index, value in enumerate(samples):
        if not _is_real_number(value):
            raise AnalysisError(f"non-finite sample at index {index}: {value}")


def _validate_env(env):
    for index, value in enumerate(env):
        if not _is_real_number(value) or value < 0:
            raise AnalysisError(f"non-finite or negative envelope at index {index}: {value}")


def onset_envelope(
    samples: list[float], hop: int, sample_rate: int
) -> list[float]:
    """用帧能量差分构造归一化到 0..1 的起始包络。

    参数
    ----
    samples : list[float]
        单声道样本。空列表返回 ``[]``；非空样本中的每个值都必须是
        有限数，否则抛 :class:`AnalysisError`。
    hop : int
        帧长/帧移，单位为采样点，必须是正整数。``hop <= 0`` 抛
        :class:`AnalysisError`，不能发生除零。
    sample_rate : int
        采样率，单位 Hz，必须是有限正数。

    返回
    ----
    list[float]
        长度与不重叠帧数相同（第一帧与 0 能量比较，因此首帧也可能
        产生正差分）。算法先计算每帧均方能量，再取半波整流的一阶
        差分 ``max(0, energy[i] - energy[i-1])``，最后除以最大正值，
        使结果落在 0..1 且非静音信号的最大值严格为 1.0。全零/静音
        信号返回同长度的全 0 列表；``samples`` 为空时返回 ``[]``。
        未归一化的能量差分会把响度当成节拍强度，导致 BPM 偏高。
    """
    _validate_positive_int(hop, "hop")
    _validate_positive_rate(sample_rate, "sample_rate")
    if not samples:
        return []
    _validate_samples(samples)

    n_frames = len(samples) // hop
    energies = []
    for frame in range(n_frames):
        start = frame * hop
        frame_samples = samples[start:start + hop]
        mean_energy = sum(value * value for value in frame_samples) / hop
        energies.append(mean_energy)

    # 第一帧与 0 能量比较，因此首帧也可能产生正差分。
    diffs = [max(0.0, energies[0])]
    diffs.extend(
        max(0.0, energies[index] - energies[index - 1])
        for index in range(1, n_frames)
    )

    peak = max(diffs)
    if peak <= 0.0:
        # 全零/静音信号：同长度全 0。
        return [0.0] * n_frames
    # 归一化：最大值严格为 1.0，响度差异不再影响节拍强度。
    return [value / peak for value in diffs]


def autocorr_tempo(
    env: list[float], hop: int, sample_rate: int, bpm_range: tuple
) -> float:
    """用自相关从起始包络估计 BPM，并钳制到给定范围。

    参数
    ----
    env : list[float]
        起始包络（通常来自 :func:`onset_envelope`），非负有限值。
        长度小于 2 或全零时抛 :class:`AnalysisError`。
    hop : int
        包络帧对应的采样点数，正整数。
    sample_rate : int
        采样率，单位 Hz，必须是有限正数。
    bpm_range : tuple[float, float]
        ``(low, high)``，要求 ``0 < low < high``；非法范围抛
        :class:`AnalysisError`。

    返回
    ----
    float
        估计的 BPM，单位为每分钟拍数，必须落在 ``[low, high]``。
        对每个候选整数 lag，使用按重叠样本数归一化的自相关::

            r(lag) = sum(env[i] * env[i + lag]) / (len(env) - lag)

        不能直接使用未除重叠长度的点积和，否则重叠更长的小 lag 会
        系统性胜出。只在 ``bpm_range`` 对应的 lag 范围内选最大者，
        再用 ``60 * sample_rate / (lag * hop)`` 换算；换算结果若越界
        则钳制到最近的边界。对规则的 120 BPM 节拍信号，结果与 120
        的误差不得超过 0.5 BPM。
    """
    _validate_positive_int(hop, "hop")
    _validate_positive_rate(sample_rate, "sample_rate")
    low, high = _check_bpm_range(bpm_range)
    _validate_env(env)
    if len(env) < 2 or not any(value > 0.0 for value in env):
        raise AnalysisError("envelope must contain at least 2 frames and be non-silent")

    n = len(env)
    # lag 与 BPM 的换算：bpm = 60 * sample_rate / (lag * hop)，
    # 因此 lag 与 bpm 成反比：lag 越小 bpm 越高。
    min_lag = max(1, math.floor(60.0 * sample_rate / (high * hop)))
    max_lag = min(n - 1, math.ceil(60.0 * sample_rate / (low * hop)))
    if min_lag > max_lag:
        # 包络短到连一个候选 lag 都没有：钳制到最近边界。
        return min(max(60.0 * sample_rate / hop, low), high)

    best_lag = min_lag
    best_score = -math.inf
    for lag in range(min_lag, max_lag + 1):
        overlap = n - lag
        # 按重叠长度归一，避免重叠更长的小 lag 系统性胜出。
        score = (
            sum(env[index] * env[index + lag] for index in range(overlap))
            / overlap
        )
        if score > best_score:
            best_score = score
            best_lag = lag

    bpm = 60.0 * sample_rate / (best_lag * hop)
    # 范围外钳制到最近边界。
    return min(max(bpm, low), high)


def score_pulse(
    env: list[float], bpm: float, hop: int, sample_rate: int
) -> float:
    """返回节拍网格对起始包络能量的覆盖比例，范围 0..1。

    参数
    ----
    env : list[float]
        起始包络，非负有限值。
    bpm : float
        待评分的速度，必须是有限正数。
    hop : int
        包络帧对应的采样点数，正整数。
    sample_rate : int
        采样率，单位 Hz，必须是有限正数。

    返回
    ----
    float
        从时间 0 开始、间隔 ``60 / bpm`` 秒生成节拍网格，把每个节拍
        映射到最近的包络帧 ``round(seconds * sample_rate / hop)``；
        对去重后的有效帧求包络值之和，再除以 ``sum(env)``。结果必须
        钳制在 0..1。空包络或全零包络返回 0.0。该评分用“覆盖了
        多少能量”而不是只对节拍点取平均，因此 120 BPM 网格应比
        60 BPM 网格覆盖更多 120 BPM 的脉冲能量。
    """
    _validate_positive_int(hop, "hop")
    _validate_positive_rate(sample_rate, "sample_rate")
    if not _is_real_number(bpm) or bpm <= 0:
        raise AnalysisError(f"bpm must be a finite positive number, got {bpm!r}")
    _validate_env(env)

    total = sum(env)
    if not env or total <= 0.0:
        return 0.0

    n = len(env)
    period_seconds = 60.0 / bpm
    covered_frames = set()
    beat = 0
    while True:
        frame = int(round(beat * period_seconds * sample_rate / hop))
        if frame >= n:
            break
        if 0 <= frame < n:
            covered_frames.add(frame)
        beat += 1

    covered_energy = sum(env[frame] for frame in covered_frames)
    score = covered_energy / total
    return min(max(score, 0.0), 1.0)


def chroma_vector(
    samples: list[float], sample_rate: int, n_bands: int
) -> list[float]:
    """把音频能量折叠成 n_bands 个等分八度音级的归一化色度向量。

    参数
    ----
    samples : list[float]
        单声道样本。空列表返回 ``[]``；非空样本中的值必须有限。
    sample_rate : int
        采样率，单位 Hz，必须是有限正数。
    n_bands : int
        一个八度内的音级数，必须是正整数；常用 12。索引 0 对应 C，
        ``n_bands == 12`` 时索引 0..11 对应 C、C#、...、B。440 Hz 的
        A4 必须落在索引 9，不能把所有频带混成一个平均能量。

    返回
    ----
    list[float]
        长度恰好为 ``n_bands``，非负且最大值为 1.0（静音时全 0）。
        实现应把每个频率按 ``n_bands * log2(freq / C0)`` 折叠到对应
        音级，并汇总该音级及八度等价频率的能量；不能只用全频带平均
        能量，否则不同音高的调性会变成随机结果。
    """
    _validate_positive_rate(sample_rate, "sample_rate")
    _validate_positive_int(n_bands, "n_bands")
    if not samples:
        return []
    _validate_samples(samples)

    n = len(samples)
    nyquist = sample_rate / 2.0
    chroma = [0.0] * n_bands

    # 仅标准库可用：对每个八度等价音级的中心频率（MIDI 音高）做
    # Goertzel 单频 DFT，把功率汇总到对应色度 bin。
    # MIDI 69 = A4 = 440 Hz；覆盖钢琴音域 MIDI 24..108。
    for midi in range(24, 109):
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        if freq >= nyquist:
            continue
        omega = 2.0 * math.pi * freq / sample_rate
        coeff = 2.0 * math.cos(omega)
        prev_1 = 0.0
        prev_2 = 0.0
        for value in samples:
            current = value + coeff * prev_1 - prev_2
            prev_2 = prev_1
            prev_1 = current
        power = prev_1 * prev_1 + prev_2 * prev_2 - coeff * prev_1 * prev_2
        if power <= 0.0:
            continue
        # 按 n_bands * log2(freq / C0) 折叠到八度内音级。
        band = int(round(n_bands * math.log2(freq / _C0_FREQ))) % n_bands
        chroma[band] += power

    peak = max(chroma)
    if peak <= 0.0:
        # 静音信号：全 0。
        return [0.0] * n_bands
    return [value / peak for value in chroma]


def _pearson_correlation(x, y):
    """两个等长序列的皮尔逊相关系数；零方差时返回 0。"""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((value - mean_x) ** 2 for value in x)
    var_y = sum((value - mean_y) ** 2 for value in y)
    denominator = math.sqrt(var_x * var_y)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def detect_key(chroma: list[float], major_profile: list[float]) -> tuple:
    """用色度向量和大调模板判定调性，返回 ``(调性名, 置信度)``。

    参数
    ----
    chroma : list[float]
        长度为 12 的色度向量，索引 0..11 对应 C..B。
    major_profile : list[float]
        长度为 12 的大调音级权重模板，索引 0 为主音；值必须有限且
        非负，不能全为 0。

    返回
    ----
    tuple[str, float]
        调性名与置信度。调性名由主音和模式组成，例如 ``"C major"``、
        ``"A minor"``，主音使用 C、C#、D、D#、E、F、F#、G、G#、A、
        A#、B。函数必须同时评估 12 个大调和 12 个自然小调（自然小调
        使用内置的 Krumhansl–Kessler 小调模板），并用相关系数而不是
        欧氏距离比较模板，不能把小调简单判成关系大调。置信度是
        0..1 的浮点数，建议按 ``clip(best_score - second_best_score,
        0, 1)`` 计算，平坦/无调性输入返回置信度 0.0，不能恒为 1。

    非法长度、非有限值或全零模板抛 :class:`AnalysisError`。
    """
    if not isinstance(chroma, (list, tuple)) or len(chroma) != 12:
        raise AnalysisError("chroma must be a length-12 vector")
    if not isinstance(major_profile, (list, tuple)) or len(major_profile) != 12:
        raise AnalysisError("major_profile must be a length-12 vector")
    if any(not _is_real_number(value) for value in chroma):
        raise AnalysisError("chroma contains non-finite values")
    if any(not _is_real_number(value) or value < 0 for value in major_profile):
        raise AnalysisError("major_profile values must be finite and non-negative")
    if not any(value > 0.0 for value in major_profile):
        raise AnalysisError("major_profile must not be all zeros")

    candidates = []
    for root in range(12):
        rotated_major = [major_profile[(degree - root) % 12] for degree in range(12)]
        rotated_minor = [_KK_MINOR_PROFILE[(degree - root) % 12] for degree in range(12)]
        candidates.append((
            _pearson_correlation(chroma, rotated_major),
            f"{_NOTE_NAMES[root]} major",
        ))
        candidates.append((
            _pearson_correlation(chroma, rotated_minor),
            f"{_NOTE_NAMES[root]} minor",
        ))

    # 确定性排序：分数降序，分数相同按调性名稳定排序。
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_name = candidates[0]
    second_score = candidates[1][0]

    # 置信度：最佳与次佳的相关系数差距，clip 到 0..1。
    confidence = min(max(best_score - second_score, 0.0), 1.0)
    return best_name, confidence


def half_double_fix(bpm: float, prior: float) -> float:
    """使用先验速度解决半速/倍速歧义。

    参数
    ----
    bpm : float
        自相关得到的候选速度，必须是有限正数。
    prior : float
        先验/期望速度，必须是有限正数；通常来自曲库元数据或用户
        标记。

    返回
    ----
    float
        在 ``bpm``、``2 * bpm``、``bpm / 2`` 三个候选中，选择与
        ``prior`` 绝对差最小的一个。距离相同时优先返回原 ``bpm``。
        必须确定性选择，不能随机取一个。本函数只解决半速/倍速，
        不做 BPM 范围钳制；范围由调用者在传入前保证。

    ``bpm <= 0``、``prior <= 0`` 或非有限值抛 :class:`AnalysisError`。
    """
    if not _is_real_number(bpm) or bpm <= 0:
        raise AnalysisError(f"bpm must be a finite positive number, got {bpm!r}")
    if not _is_real_number(prior) or prior <= 0:
        raise AnalysisError(f"prior must be a finite positive number, got {prior!r}")

    # 原 bpm 放在首位：距离相同时优先返回原 bpm，保证确定性。
    candidates = [bpm, 2.0 * bpm, bpm / 2.0]
    best = candidates[0]
    best_distance = abs(best - prior)
    for candidate in candidates[1:]:
        distance = abs(candidate - prior)
        if distance < best_distance:
            best_distance = distance
            best = candidate
    return best


def validate(samples: list[float], bpm_range: tuple) -> list[str]:
    """返回输入问题列表，而不是抛出异常。

    参数
    ----
    samples : list[float]
        待检查的单声道样本。
    bpm_range : tuple[float, float]
        待检查的 BPM 范围。

    返回
    ----
    list[str]
        按固定顺序排列的问题；没有问题时返回 ``[]``。问题使用稳定的
        英文前缀，便于调用方和测试检查：

        - ``"invalid bpm range: <low>, <high>"``：范围不是两个有限
          数，或 ``low <= 0``，或 ``low >= high``；
        - ``"too short samples: <n>"``：样本数少于 2，无法做能量
          差分或自相关；
        - ``"non-finite sample at index <i>: <value>"``：样本为 NaN
          或正负无穷。

        先检查范围，再检查长度，最后按索引逐样本检查。本函数不修改
        输入，也不通过抛 :class:`AnalysisError` 表示这些内容问题。
    """
    problems = []

    # 1) 先检查范围。
    if not isinstance(bpm_range, (tuple, list)) or len(bpm_range) != 2:
        problems.append(f"invalid bpm range: {bpm_range}, {bpm_range}")
        low, high = bpm_range, bpm_range
    else:
        low, high = bpm_range
        if (
            not _is_real_number(low)
            or not _is_real_number(high)
            or low <= 0
            or low >= high
        ):
            problems.append(f"invalid bpm range: {low}, {high}")

    # 2) 再检查长度。
    if len(samples) < 2:
        problems.append(f"too short samples: {len(samples)}")
        return problems

    # 3) 最后按索引逐样本检查。
    for index, value in enumerate(samples):
        if not _is_real_number(value):
            problems.append(f"non-finite sample at index {index}: {value}")

    return problems


def analyze_batch(tracks: list, config: dict) -> dict:
    """批量分析 BPM 与调性；任一轨非法时整体回滚，不返回部分结果。

    参数
    ----
    tracks : list[dict]
        有序的轨道列表，每项为::

            {"id": str,
             "samples": list[float],
             "sample_rate": int}

        ``id`` 必须是非空字符串，``sample_rate`` 必须是有限正数；
        样本规则与 :func:`validate` 相同。空列表是合法输入。
    config : dict
        共享配置::

            {"hop": int,
             "bpm_range": (low, high),
             "n_bands": int,
             "major_profile": list[float],
             "prior_bpm": float | None}

        ``prior_bpm`` 为 ``None`` 时不做半速/倍速修正；非 ``None``
        时传给 :func:`half_double_fix`。配置字段缺失或非法时抛
        :class:`AnalysisError`。

    返回
    ----
    dict
        ``{"results": [{"id": str, "bpm": float,
        "key": str, "confidence": float}, ...],
        "ids": [str, ...], "total_tracks": int}``。
        ``results`` 与 ``ids`` 的顺序必须与输入完全一致。每轨都必须
        独立计算自己的起始包络，不能复用上一轨的包络或中间变量，
        否则轨与轨之间会串数据。BPM 必须落在 ``bpm_range`` 内，正常
        节拍信号与真值的误差不超过 0.5 BPM；``confidence`` 在 0..1。

    全有或全无：先校验整个配置和所有轨道，任一轨非法就抛
    :class:`AnalysisError`，不返回部分结果，也不修改输入。空
    ``tracks`` 返回
    ``{"results": [], "ids": [], "total_tracks": 0}``。
    """
    if not isinstance(config, dict):
        raise AnalysisError("config must be a dict")

    required = ("hop", "bpm_range", "n_bands", "major_profile", "prior_bpm")
    missing = [key for key in required if key not in config]
    if missing:
        raise AnalysisError(f"missing config fields: {', '.join(missing)}")

    hop = config["hop"]
    bpm_range = config["bpm_range"]
    n_bands = config["n_bands"]
    major_profile = config["major_profile"]
    prior_bpm = config["prior_bpm"]

    # 配置整体校验（任一非法即整体失败）。
    _validate_positive_int(hop, "hop")
    low, high = _check_bpm_range(bpm_range)
    _validate_positive_int(n_bands, "n_bands")
    if not isinstance(major_profile, (list, tuple)) or len(major_profile) != 12:
        raise AnalysisError("major_profile must be a length-12 vector")
    if any(not _is_real_number(value) or value < 0 for value in major_profile):
        raise AnalysisError("major_profile values must be finite and non-negative")
    if not any(value > 0.0 for value in major_profile):
        raise AnalysisError("major_profile must not be all zeros")
    if prior_bpm is not None and (not _is_real_number(prior_bpm) or prior_bpm <= 0):
        raise AnalysisError(
            f"prior_bpm must be None or a finite positive number, got {prior_bpm!r}"
        )
    if not isinstance(tracks, list):
        raise AnalysisError("tracks must be a list")

    # 先校验所有轨道：任一轨非法就整体回滚，不进入计算阶段。
    for track in tracks:
        if not isinstance(track, dict):
            raise AnalysisError("each track must be a dict")
        if not isinstance(track.get("id"), str) or not track["id"]:
            raise AnalysisError("track id must be a non-empty string")
        sample_rate = track.get("sample_rate")
        if not _is_real_number(sample_rate) or sample_rate <= 0:
            raise AnalysisError(
                f"track {track.get('id')!r} has invalid sample_rate: {sample_rate!r}"
            )
        samples = track.get("samples")
        if not isinstance(samples, list):
            raise AnalysisError(f"track {track['id']!r} samples must be a list")
        if validate(samples, bpm_range):
            raise AnalysisError(
                f"track {track['id']!r} failed validation: "
                f"{validate(samples, bpm_range)}"
            )

    if not tracks:
        return {"results": [], "ids": [], "total_tracks": 0}

    results = []
    ids = []
    for track in tracks:
        # 每轨使用独立的局部变量计算包络，杜绝跨轨复用中间状态。
        env = onset_envelope(track["samples"], hop, track["sample_rate"])
        bpm = autocorr_tempo(env, hop, track["sample_rate"], bpm_range)
        if prior_bpm is not None:
            bpm = half_double_fix(bpm, prior_bpm)
            # half_double_fix 不做范围钳制，由调用者保证落回范围。
            bpm = min(max(bpm, low), high)
        chroma = chroma_vector(track["samples"], track["sample_rate"], n_bands)
        key, confidence = detect_key(chroma, major_profile)
        confidence = min(max(confidence, 0.0), 1.0)

        results.append(
            {"id": track["id"], "bpm": bpm, "key": key, "confidence": confidence}
        )
        ids.append(track["id"])

    return {"results": results, "ids": ids, "total_tracks": len(tracks)}
