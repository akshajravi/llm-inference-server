"""Engine registry.

`make bench` sweeps this dict, so registering an engine here is the single act that
puts it into the published comparison. Stubs register too — they raise on construction
rather than being silently absent.
"""

from __future__ import annotations

from typing import Callable

from inference_server.engine.base import Engine, NotBuiltYet, Request, Result

ENGINES: dict[str, Callable[[], Engine]] = {}


def register(name: str) -> Callable[[Callable[[], Engine]], Callable[[], Engine]]:
    def wrap(factory: Callable[[], Engine]) -> Callable[[], Engine]:
        ENGINES[name] = factory
        return factory

    return wrap


def build(name: str) -> Engine:
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; registered: {sorted(ENGINES)}")
    return ENGINES[name]()


# Registration is import-order-sensitive and this is the order phases ship in.
def _naive() -> Engine:
    from inference_server.engine.naive import NaiveEngine

    return NaiveEngine()


def _manual() -> Engine:
    from inference_server.engine.manual import ManualEngine

    return ManualEngine()


def _static() -> Engine:
    from inference_server.engine.static_batch import StaticBatchEngine

    return StaticBatchEngine()


def _continuous() -> Engine:
    from inference_server.engine.continuous import ContinuousEngine

    return ContinuousEngine()


def _paged() -> Engine:
    from inference_server.engine.paged import PagedEngine

    return PagedEngine()


ENGINES.update(
    naive=_naive,
    manual=_manual,
    static=_static,
    continuous=_continuous,
    paged=_paged,
)

#: Engines that are actually implemented today. `make bench` sweeps this; the full
#: ENGINES dict is what `--engine` accepts. Move a name across as its phase lands.
IMPLEMENTED = ["naive", "manual", "static"]

__all__ = ["ENGINES", "IMPLEMENTED", "Engine", "NotBuiltYet", "Request", "Result", "build", "register"]
