from __future__ import annotations

from dataclasses import asdict, dataclass

from .touchstone import TouchstoneData, get_sparameter, to_db


@dataclass(slots=True)
class SearchWindow:
    start_hz: float | None = None
    stop_hz: float | None = None


@dataclass(slots=True)
class BandwidthEstimate:
    left_freq_hz: float | None
    right_freq_hz: float | None
    width_hz: float | None


@dataclass(slots=True)
class FilterMetricConfig:
    peak_search_start_hz: float | None = None
    peak_search_stop_hz: float | None = None
    bandwidth_drop_db: float = 3.0
    high_side_search_start_hz: float | None = None
    high_side_search_stop_hz: float | None = None


@dataclass(slots=True)
class FilterMetrics:
    main_peak_freq_hz: float
    main_peak_s21_db: float
    main_peak_s11_db: float
    best_s11_freq_hz: float
    best_s11_db: float
    passband_left_freq_hz: float | None
    passband_right_freq_hz: float | None
    bandwidth_3db_hz: float | None
    high_side_zero_freq_hz: float | None
    high_side_zero_s21_db: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _indices_in_window(frequency_hz: list[float], search_window: SearchWindow | None) -> list[int]:
    if search_window is None:
        return list(range(len(frequency_hz)))
    return [
        index
        for index, frequency in enumerate(frequency_hz)
        if (search_window.start_hz is None or frequency >= search_window.start_hz)
        and (search_window.stop_hz is None or frequency <= search_window.stop_hz)
    ]


def _interpolate_crossing(x0: float, y0: float, x1: float, y1: float, target_y: float) -> float:
    if y1 == y0:
        return x1
    return x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)


def find_main_passband_peak(data: TouchstoneData, search_window: SearchWindow | None = None) -> int:
    candidate_indices = _indices_in_window(data.frequency_hz, search_window)
    if not candidate_indices:
        raise ValueError("No Touchstone samples fall inside the specified peak search window")

    s21_db = to_db(get_sparameter(data, "s21"))
    return max(candidate_indices, key=lambda index: s21_db[index])


def estimate_bandwidth_3db(data: TouchstoneData, peak_index: int, drop_db: float = 3.0) -> BandwidthEstimate:
    frequency_hz = data.frequency_hz
    s21_db = to_db(get_sparameter(data, "s21"))
    threshold = s21_db[peak_index] - drop_db

    left_freq_hz: float | None = None
    if peak_index > 0:
        left_index = peak_index
        while left_index > 0 and s21_db[left_index] >= threshold:
            left_index -= 1
        if left_index != 0 or s21_db[left_index] < threshold:
            left_freq_hz = _interpolate_crossing(
                frequency_hz[left_index],
                s21_db[left_index],
                frequency_hz[left_index + 1],
                s21_db[left_index + 1],
                threshold,
            )

    right_freq_hz: float | None = None
    if peak_index < len(frequency_hz) - 1:
        right_index = peak_index
        while right_index < len(frequency_hz) - 1 and s21_db[right_index] >= threshold:
            right_index += 1
        if right_index != len(frequency_hz) - 1 or s21_db[right_index] < threshold:
            right_freq_hz = _interpolate_crossing(
                frequency_hz[right_index - 1],
                s21_db[right_index - 1],
                frequency_hz[right_index],
                s21_db[right_index],
                threshold,
            )

    width_hz = None if left_freq_hz is None or right_freq_hz is None else right_freq_hz - left_freq_hz
    return BandwidthEstimate(left_freq_hz=left_freq_hz, right_freq_hz=right_freq_hz, width_hz=width_hz)


def find_high_side_transmission_zero(
    data: TouchstoneData,
    right_start_freq_hz: float,
    stop_freq_hz: float | None = None,
) -> int | None:
    s21_db = to_db(get_sparameter(data, "s21"))
    candidate_indices = [
        index
        for index, frequency in enumerate(data.frequency_hz)
        if frequency > right_start_freq_hz and (stop_freq_hz is None or frequency <= stop_freq_hz)
    ]
    if not candidate_indices:
        return None
    return min(candidate_indices, key=lambda index: s21_db[index])


def extract_filter_metrics(data: TouchstoneData, config: FilterMetricConfig | None = None) -> FilterMetrics:
    metrics_config = config or FilterMetricConfig()
    s11_db = to_db(get_sparameter(data, "s11"))
    s21_db = to_db(get_sparameter(data, "s21"))

    peak_window = SearchWindow(
        start_hz=metrics_config.peak_search_start_hz,
        stop_hz=metrics_config.peak_search_stop_hz,
    )
    peak_index = find_main_passband_peak(data, search_window=peak_window)
    bandwidth = estimate_bandwidth_3db(data, peak_index, drop_db=metrics_config.bandwidth_drop_db)

    passband_indices = [
        index
        for index, frequency in enumerate(data.frequency_hz)
        if (
            bandwidth.left_freq_hz is None
            or frequency >= bandwidth.left_freq_hz
        )
        and (
            bandwidth.right_freq_hz is None
            or frequency <= bandwidth.right_freq_hz
        )
    ]
    if not passband_indices:
        passband_indices = _indices_in_window(data.frequency_hz, peak_window) or [peak_index]

    best_s11_index = min(passband_indices, key=lambda index: s11_db[index])

    high_side_start_hz = metrics_config.high_side_search_start_hz
    if high_side_start_hz is None:
        high_side_start_hz = bandwidth.right_freq_hz if bandwidth.right_freq_hz is not None else data.frequency_hz[peak_index]
    high_side_zero_index = find_high_side_transmission_zero(
        data,
        right_start_freq_hz=high_side_start_hz,
        stop_freq_hz=metrics_config.high_side_search_stop_hz,
    )

    return FilterMetrics(
        main_peak_freq_hz=data.frequency_hz[peak_index],
        main_peak_s21_db=s21_db[peak_index],
        main_peak_s11_db=s11_db[peak_index],
        best_s11_freq_hz=data.frequency_hz[best_s11_index],
        best_s11_db=s11_db[best_s11_index],
        passband_left_freq_hz=bandwidth.left_freq_hz,
        passband_right_freq_hz=bandwidth.right_freq_hz,
        bandwidth_3db_hz=bandwidth.width_hz,
        high_side_zero_freq_hz=None if high_side_zero_index is None else data.frequency_hz[high_side_zero_index],
        high_side_zero_s21_db=None if high_side_zero_index is None else s21_db[high_side_zero_index],
    )
