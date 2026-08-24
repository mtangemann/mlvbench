"""Tests for the monitoring module."""


from mlvbench.monitoring import ThroughputMonitor


class _FakeClock:
    """Clock that advances by a fixed amount on every call."""

    def __init__(self, increment: float = 1.0):
        self.increment = increment
        self.time = 0.0

    def __call__(self) -> float:
        value = self.time
        self.time += self.increment
        return value


def _make_monitor(record_every_n_steps=1, increment=1.0) -> ThroughputMonitor:
    return ThroughputMonitor(
        "cpu",
        record_every_n_steps=record_every_n_steps,
        clock=_FakeClock(increment),
    )


def test_iterate_yields_all_items():
    monitor = _make_monitor()
    assert list(monitor.iterate(range(4))) == [0, 1, 2, 3]


def test_iterate_accounts_data_time():
    monitor = _make_monitor()

    for _ in monitor.iterate(range(3)):
        pass
    monitor.step(1)

    # Each item costs one clock increment (two clock calls per item).
    assert monitor.records[0]["t_data"] == 3.0


def test_phase_accumulates_across_batches():
    monitor = _make_monitor()

    for _ in range(2):
        with monitor.phase("features"):
            pass
    monitor.step(1)

    assert monitor.records[0]["t_features"] == 2.0


def test_phase_is_not_recorded_on_error():
    monitor = _make_monitor()

    try:
        with monitor.phase("features"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    monitor.step(1)

    assert "t_features" not in monitor.records[0]


def test_step_records_throughput_and_resets_accumulators():
    monitor = _make_monitor()

    for _ in monitor.iterate(range(2)):
        monitor.record_samples(4)
    monitor.step(1)

    # 8 samples over 2 seconds of accounted time.
    assert monitor.records[0]["step"] == 1
    assert monitor.records[0]["samples_per_sec"] == 4.0

    # The accumulators are reset after each snapshot.
    monitor.step(2)
    assert monitor.records[1]["samples_per_sec"] == 0.0
    assert "t_data" not in monitor.records[1]


def test_step_returns_the_recorded_snapshot():
    monitor = _make_monitor(record_every_n_steps=2)

    assert monitor.step(1) is None
    assert monitor.step(2) == monitor.records[-1]


def test_step_records_only_every_n_steps():
    monitor = _make_monitor(record_every_n_steps=3)

    for step in range(1, 8):
        monitor.record_samples(1)
        monitor.step(step)

    assert [record["step"] for record in monitor.records] == [3, 6]


def test_disabled_monitor_records_nothing():
    monitor = _make_monitor(record_every_n_steps=None)

    assert not monitor.enabled
    for _ in monitor.iterate(range(3)):
        with monitor.phase("features"):
            monitor.record_samples(1)

    assert monitor.step(1) is None
    assert monitor.records == []
