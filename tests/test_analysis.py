"""audio-bpm-key-detect 的验收测试：15 条全红，tests 不可修改。

每个测试都直接调用 ``bpmkey/analysis.py`` 中尚未实现的桩函数，因此初始
失败原因是 ``NotImplementedError``。音频一律使用 ``list[float]``，采样率
显式传入；信号全部由 ``math`` 确定性生成，不使用随机数、不调用 sleep。
"""

import math

import pytest

from bpmkey.analysis import (
    AnalysisError,
    analyze_batch,
    autocorr_tempo,
    chroma_vector,
    detect_key,
    half_double_fix,
    onset_envelope,
    score_pulse,
    validate,
)


SAMPLE_RATE = 44100
HOP = 441
BPM_RANGE = (60.0, 180.0)

# Krumhansl–Kessler 大调音级模板，索引 0 为主音。
MAJOR_PROFILE = [
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
]


def _tone(freq, seconds=1.0, sample_rate=SAMPLE_RATE):
    """返回确定性的单频正弦信号。"""
    length = int(round(seconds * sample_rate))
    return [
        math.sin(2.0 * math.pi * freq * index / sample_rate)
        for index in range(length)
    ]


def _click_track(bpm, seconds=4.0, sample_rate=SAMPLE_RATE):
    """返回每拍一个单位脉冲的确定性节拍轨。"""
    length = int(round(seconds * sample_rate))
    samples = [0.0] * length
    period = 60.0 * sample_rate / bpm
    position = 0.0
    while position < length:
        samples[int(round(position))] = 1.0
        position += period
    return samples


def _config(**overrides):
    config = {
        "hop": HOP,
        "bpm_range": BPM_RANGE,
        "n_bands": 12,
        "major_profile": MAJOR_PROFILE,
        "prior_bpm": None,
    }
    config.update(overrides)
    return config


# --- 每个缺陷一个聚焦测试 -------------------------------------------------


def test_onset_envelope_is_normalized_to_unit_range():
    """缺陷 1：起始包络不归一，响度会被当成节拍强度，bpm 偏高。"""
    samples = [0.0] * 2000
    samples[100] = 0.25
    samples[500] = 0.125
    samples[900] = 0.0625

    env = onset_envelope(samples, 100, SAMPLE_RATE)

    assert env
    assert max(env) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in env)


def test_autocorr_tempo_normalizes_by_overlap_length():
    """缺陷 2：自相关不除重叠长度，长窗会系统性胜出。"""
    env = [0.5] * 400
    for index in range(0, 400, 100):
        env[index] = 1.5

    bpm = autocorr_tempo(env, HOP, SAMPLE_RATE, (40.0, 180.0))

    assert bpm == pytest.approx(60.0, abs=0.5)


def test_autocorr_tempo_clamps_to_bpm_range():
    """缺陷 3：BPM 范围外不钳制，会返回范围外的 300。"""
    env = [0.0] * 400
    for index in range(0, 400, 20):
        env[index] = 1.0

    bpm = autocorr_tempo(env, HOP, SAMPLE_RATE, BPM_RANGE)

    assert 60.0 <= bpm <= 180.0


def test_half_double_fix_uses_prior():
    """缺陷 4：半速/倍速不修正，随机取一个。"""
    assert half_double_fix(75.0, 120.0) == pytest.approx(150.0)
    assert half_double_fix(150.0, 75.0) == pytest.approx(75.0)
    assert half_double_fix(100.0, 110.0) == pytest.approx(100.0)


def test_chroma_vector_resolves_pitch_bands():
    """缺陷 5：色度用平均能量不分频带，调性会随机。"""
    a_chroma = chroma_vector(_tone(440.0), SAMPLE_RATE, 12)
    c_chroma = chroma_vector(_tone(261.6255653005986), SAMPLE_RATE, 12)

    assert len(a_chroma) == 12
    assert len(c_chroma) == 12
    assert max(a_chroma) == pytest.approx(1.0)
    assert max(c_chroma) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in a_chroma)
    assert a_chroma.index(max(a_chroma)) == 9
    assert c_chroma.index(max(c_chroma)) == 0


def test_detect_key_distinguishes_relative_minor():
    """缺陷 6：调性相关性用欧氏距离，小调会被判成关系大调。"""
    # A 自然小调的 Krumhansl–Kessler 音级模板，按 A 为主音排列。
    a_minor_profile = [
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
    chroma = [0.0] * 12
    for degree, weight in enumerate(a_minor_profile):
        chroma[(9 + degree) % 12] = weight

    key, confidence = detect_key(chroma, MAJOR_PROFILE)

    assert key == "A minor"
    assert 0.0 <= confidence <= 1.0


def test_detect_key_reports_uncertain_confidence():
    """缺陷 7：置信度恒为 1，误检无法过滤。"""
    flat = [1.0] * 12

    key, confidence = detect_key(flat, MAJOR_PROFILE)

    assert isinstance(key, str) and key
    assert 0.0 <= confidence < 1.0


def test_validate_reports_short_samples():
    """缺陷 8：样本过短不报错，自相关会得到全 0。"""
    short = validate([0.25], BPM_RANGE)
    assert any("short" in problem.lower() for problem in short)

    non_finite = validate([0.0, float("nan")], BPM_RANGE)
    assert any("non-finite" in problem.lower() for problem in non_finite)

    bad_range = validate([0.0, 1.0], (180.0, 60.0))
    assert any("range" in problem.lower() for problem in bad_range)


def test_analyze_batch_isolates_tracks_and_rolls_back():
    """缺陷 9：批量共用包络会让轨与轨之间串数据。"""
    tracks = [
        {"id": "slow", "samples": _click_track(100.0), "sample_rate": SAMPLE_RATE},
        {"id": "fast", "samples": _click_track(120.0), "sample_rate": SAMPLE_RATE},
    ]

    result = analyze_batch(tracks, _config())

    assert result["total_tracks"] == 2
    assert result["ids"] == ["slow", "fast"]
    assert [entry["id"] for entry in result["results"]] == ["slow", "fast"]
    assert [entry["bpm"] for entry in result["results"]] == pytest.approx(
        [100.0, 120.0], abs=0.5
    )

    snapshot = repr(tracks)
    bad = tracks + [
        {"id": "bad", "samples": [0.0, float("nan")], "sample_rate": SAMPLE_RATE}
    ]
    with pytest.raises(AnalysisError):
        analyze_batch(bad, _config())
    assert repr(tracks) == snapshot


def test_onset_envelope_rejects_zero_hop():
    """缺陷 10：hop=0 不校验，计算帧能量时会除零。"""
    with pytest.raises(AnalysisError):
        onset_envelope(_click_track(120.0), 0, SAMPLE_RATE)

    with pytest.raises(AnalysisError):
        onset_envelope(_click_track(120.0), -HOP, SAMPLE_RATE)


# --- 五个组合/契约测试 -----------------------------------------------------


def test_120_bpm_click_track_detected_within_half_bpm():
    """组合 1：120 BPM 纯节拍信号必须测得 120（容差 0.5 BPM）。"""
    samples = _click_track(120.0)
    env = onset_envelope(samples, HOP, SAMPLE_RATE)
    bpm = autocorr_tempo(env, HOP, SAMPLE_RATE, BPM_RANGE)

    assert bpm == pytest.approx(120.0, abs=0.5)

    score_120 = score_pulse(env, 120.0, HOP, SAMPLE_RATE)
    score_60 = score_pulse(env, 60.0, HOP, SAMPLE_RATE)
    assert 0.0 <= score_120 <= 1.0
    assert score_120 >= 0.75
    assert score_120 > score_60


def test_single_sine_maps_to_its_note():
    """组合 2：单一 440 Hz 正弦必须判为对应音 A。"""
    chroma = chroma_vector(_tone(440.0), SAMPLE_RATE, 12)
    key, confidence = detect_key(chroma, MAJOR_PROFILE)

    assert key.split()[0] == "A"
    assert 0.0 <= confidence <= 1.0


def test_empty_inputs_return_empty():
    """组合 3：空音频、空色度输入和空批量都返回空结果。"""
    assert onset_envelope([], HOP, SAMPLE_RATE) == []
    assert chroma_vector([], SAMPLE_RATE, 12) == []
    assert analyze_batch([], _config()) == {
        "results": [],
        "ids": [],
        "total_tracks": 0,
    }


def test_autocorr_tempo_rejects_zero_hop():
    """组合 4：autocorr_tempo 的 hop=0 也必须抛 AnalysisError。"""
    with pytest.raises(AnalysisError):
        autocorr_tempo([1.0, 0.0, 1.0], 0, SAMPLE_RATE, BPM_RANGE)


def test_analyze_batch_preserves_input_order():
    """组合 5：批量全成功时，输出顺序与输入顺序一致。"""
    tracks = [
        {"id": "bpm150", "samples": _click_track(150.0), "sample_rate": SAMPLE_RATE},
        {"id": "bpm120", "samples": _click_track(120.0), "sample_rate": SAMPLE_RATE},
        {"id": "bpm100", "samples": _click_track(100.0), "sample_rate": SAMPLE_RATE},
    ]

    result = analyze_batch(tracks, _config())

    assert result["ids"] == ["bpm150", "bpm120", "bpm100"]
    assert [entry["id"] for entry in result["results"]] == [
        "bpm150",
        "bpm120",
        "bpm100",
    ]
    assert [entry["bpm"] for entry in result["results"]] == pytest.approx(
        [150.0, 120.0, 100.0], abs=0.5
    )
