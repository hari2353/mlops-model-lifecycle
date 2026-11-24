"""Edge deployment budgets: tensor arena planning and integer readiness.

Cross-pollinated from tensorflow/tflite-micro (greedy tensor-arena reuse) and
google-coral (full-integer models). TFLite Micro fits a model into a fixed
arena by reusing space between tensors whose lifetimes do not overlap; Coral
requires int8/uint8 input and output. Both checks answer one release-gate
question: will this model run on the target edge device?
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

_INT8_DTYPES = {"int8", "uint8"}


@dataclass(frozen=True, slots=True)
class Tensor:
    """One tensor allocation with an inclusive lifetime range."""

    name: str
    nbytes: int
    first_use: int
    last_use: int


def _align(value: int, alignment: int) -> int:
    if value == 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def _lifetimes_overlap(a: Tensor, b: Tensor) -> bool:
    return not (a.last_use < b.first_use or b.last_use < a.first_use)


def plan_tensor_arena(tensors: Iterable[Tensor], *, alignment: int = 16) -> dict:
    """Greedily place tensors, reusing space when lifetimes do not overlap."""
    if alignment < 1:
        raise ValueError("alignment must be >= 1")
    items = list(tensors)
    names = [tensor.name for tensor in items]
    if len(names) != len(set(names)):
        raise ValueError("tensor names must be unique")
    for tensor in items:
        if tensor.nbytes < 0:
            raise ValueError(f"{tensor.name}: nbytes must be >= 0")
        if tensor.first_use < 0 or tensor.last_use < tensor.first_use:
            raise ValueError(f"{tensor.name}: invalid lifetime")

    sizes = {tensor.name: _align(tensor.nbytes, alignment) for tensor in items}
    placed: list[tuple[int, int, Tensor]] = []
    offsets: dict[str, int] = {}
    for tensor in sorted(items, key=lambda t: (-sizes[t.name], t.name)):
        size = sizes[tensor.name]
        offset = 0
        while True:
            conflict = next(
                (
                    spot
                    for spot in placed
                    if _lifetimes_overlap(tensor, spot[2])
                    and spot[0] < offset + size
                    and offset < spot[1]
                ),
                None,
            )
            if conflict is None:
                break
            offset = conflict[1]
        placed.append((offset, offset + size, tensor))
        offsets[tensor.name] = offset

    peak = max((end for _start, end, _tensor in placed), default=0)
    total = sum(sizes.values())
    return {
        "tensors": len(items),
        "peak_bytes": peak,
        "total_bytes": total,
        "reused_bytes": total - peak,
        "offsets": dict(sorted(offsets.items())),
    }


def coral_readiness(
    input_dtype: str,
    output_dtype: str,
    ops: Iterable[str],
    *,
    unsupported_ops: Iterable[str] = (),
) -> dict:
    """Check the Coral full-integer requirements for a model graph."""
    op_list = [str(op) for op in ops]
    if not op_list:
        raise ValueError("ops must be non-empty")
    blocking: list[str] = []
    if input_dtype.lower() not in _INT8_DTYPES:
        blocking.append(f"input dtype {input_dtype} is not int8/uint8")
    if output_dtype.lower() not in _INT8_DTYPES:
        blocking.append(f"output dtype {output_dtype} is not int8/uint8")
    unsupported = sorted(set(op_list) & {str(op) for op in unsupported_ops})
    if unsupported:
        blocking.append("unsupported ops: " + ", ".join(unsupported))
    return {"ready": not blocking, "blocking": blocking, "ops": len(op_list)}
