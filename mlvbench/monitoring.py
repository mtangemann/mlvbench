"""Monitoring utilities for training loops."""

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch


class ThroughputMonitor:
    """Monitor wall-clock time per phase, throughput and GPU memory usage.

    The monitor is driven by the training loop, which marks the phases to account for
    and registers each optimizer step:

    ```python
    monitor = ThroughputMonitor(device)

    for batch in monitor.iterate(dataloader):
        with monitor.phase("features"):
            features = model.forward_features(batch["image"])
        with monitor.phase("probes"):
            ...
        monitor.record_samples(batch["image"].shape[0])
        stats = monitor.step(step)
    ```

    Every `record_every_n_steps` steps, the accumulated timings are appended to
    `records` as a snapshot, returned by `step()`, and the accumulators are reset.
    Snapshots therefore cover disjoint intervals of the run and can be saved with
    `save()`. The monitor only collects the statistics; reporting them is up to the
    caller.

    Phase timings are attributed correctly only if the device is synchronized, so the
    monitor synchronizes CUDA devices when a phase ends. This costs time, which is why
    monitoring can be disabled entirely by passing `record_every_n_steps=None`. All
    methods then become no-ops and no synchronization is performed.
    """

    def __init__(
        self,
        device: torch.device | str,
        record_every_n_steps: int | None = 50,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """Initialize the monitor.

        Args:
            device: The device the monitored work runs on. CUDA devices are synchronized
                at phase boundaries and reported with their memory usage.
            record_every_n_steps: Record a snapshot every N steps. If None, monitoring
                is disabled entirely.
            clock: Function returning the current time in seconds. Intended for testing.
        """
        self.device = torch.device(device)
        self.record_every_n_steps = record_every_n_steps
        self.clock = clock

        self._durations: dict[str, float] = defaultdict(float)
        self._num_samples = 0
        self._records: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        """Whether the monitor accumulates timings."""
        return self.record_every_n_steps is not None

    @property
    def records(self) -> list[dict[str, Any]]:
        """The snapshots recorded so far, one per logged step."""
        return list(self._records)

    def iterate(self, iterable: Iterable[Any]) -> Iterator[Any]:
        """Iterate over `iterable`, accounting the time spent in it as `"data"`.

        Only the time needed to produce the next item is accounted for, not the time the
        loop body takes to process it.

        Args:
            iterable: The iterable to monitor, typically a data loader.

        Yields:
            The items of `iterable`, unchanged.
        """
        if not self.enabled:
            yield from iterable
            return

        iterator = iter(iterable)
        while True:
            start = self.clock()
            try:
                item = next(iterator)
            except StopIteration:
                return
            self._durations["data"] += self.clock() - start
            yield item

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Account the time spent in the enclosed block to the phase `name`.

        Nothing is recorded if the enclosed block raises.

        Args:
            name: Name of the phase. Phases are reported in the order first seen.

        Yields:
            None.
        """
        if not self.enabled:
            yield
            return

        start = self.clock()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._durations[name] += self.clock() - start

    def record_samples(self, count: int) -> None:
        """Register `count` processed samples for the throughput computation."""
        self._num_samples += count

    def step(self, step: int) -> dict[str, Any] | None:
        """Register a completed step and record a snapshot every `record_every_n_steps`.

        Args:
            step: The current step, used for reporting and to determine whether a
                snapshot is due.

        Returns:
            The recorded snapshot if one is due at this step, None otherwise. The
                snapshot holds the step, the accumulated per-phase durations (`t_<name>`
                in seconds), the throughput (`samples_per_sec`) and the current and peak
                GPU memory usage (`gpu_mem_gb`, `gpu_mem_peak_gb`).
        """
        if not self.enabled or step % self.record_every_n_steps != 0:
            return None

        total_duration = sum(self._durations.values())
        samples_per_sec = self._num_samples / total_duration if total_duration else 0.0
        memory, memory_peak = self._memory_usage()

        record = {
            "step": step,
            **{f"t_{name}": round(value, 4) for name, value in self._durations.items()},
            "samples_per_sec": round(samples_per_sec, 2),
            "gpu_mem_gb": round(memory, 3),
            "gpu_mem_peak_gb": round(memory_peak, 3),
        }
        self._records.append(record)

        self.reset()

        return record

    def reset(self) -> None:
        """Reset the accumulated timings, sample count and peak memory usage."""
        self._durations = defaultdict(float)
        self._num_samples = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _memory_usage(self) -> tuple[float, float]:
        """Return the current and peak memory usage in GB, or zeros on CPU."""
        if self.device.type != "cuda":
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated(self.device) / 1e9,
            torch.cuda.max_memory_allocated(self.device) / 1e9,
        )
