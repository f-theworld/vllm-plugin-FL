# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU workaround for int64 arithmetic in the shared ``ModelRunnerFL``.

``vllm_fl/worker/model_runner.py`` is the plugin's cross-vendor adaptation of
vLLM's ``gpu_model_runner.py`` -- the same ``ModelRunnerFL`` runs on GCU, MUSA,
Ascend and Kunlunxin -- so a GCU-only workaround does not belong in that file.
It is applied here instead, by rewriting the single offending statement inside
``ModelRunnerFL._prepare_inputs``::

    self.positions[:total_num_scheduled_tokens] = (
        self.num_computed_tokens[req_indices_gpu].to(torch.int64)
        + self.query_pos.gpu[:total_num_scheduled_tokens]
    )

torch_gcu native ATen has no int64 ``add`` kernel for this pattern: the call
aborts inside ``topsatenAdd`` ("This binary op pattern not support!",
``TOPSATEN_STATUS_NOT_SUPPORT``) and takes the worker down through
``torch_gcu/csrc/gcu/gcu_utils.cpp:74`` -- a hard abort, not a catchable
exception.  Both operands are token positions bounded by ``max_model_len``
(far below 2**31), so doing the add in int32 and widening on store is lossless.

The statement is rewritten as source text rather than by shipping a copy of the
whole method: ``_prepare_inputs`` is ~330 lines long, and a copy kept here
would silently rot the next time the shared runner changes.  If the statement
stops matching, we log a warning and leave the original code in place.
"""

from __future__ import annotations

import inspect
import logging
import re
import textwrap

logger = logging.getLogger(__name__)

# Whitespace-tolerant so a pure reformat of the shared file does not silently
# disable the fix, but anchored on the whole statement so it cannot match some
# other int64 add.
_STATEMENT = re.compile(
    r"self\.positions\[:total_num_scheduled_tokens\]\s*=\s*\(\s*"
    r"self\.num_computed_tokens\[req_indices_gpu\]\.to\(torch\.int64\)\s*"
    r"\+\s*self\.query_pos\.gpu\[:total_num_scheduled_tokens\]\s*\)"
)

_REPLACEMENT = (
    "self.positions[:total_num_scheduled_tokens] = (\n"
    "            self.num_computed_tokens[req_indices_gpu]\n"
    "            + self.query_pos.gpu[:total_num_scheduled_tokens].to(torch.int32)\n"
    "        ).to(self.positions.dtype)"
)


def apply_runner_compat_patches() -> None:
    """Rewrite the int64 positions add in ``ModelRunnerFL``. Idempotent."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL
    except Exception as exc:
        logger.warning(
            "runner_compat: ModelRunnerFL not importable (%s); the int64 "
            "positions add is left in place",
            exc,
        )
        return

    fn = ModelRunnerFL._prepare_inputs
    if getattr(fn, "_gcu_i32_positions", False):
        return

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except Exception as exc:
        logger.warning(
            "runner_compat: cannot read _prepare_inputs source (%s)", exc
        )
        return

    rewritten, matches = _STATEMENT.subn(_REPLACEMENT, source)
    if matches != 1:
        logger.warning(
            "runner_compat: int64 positions add not found in "
            "ModelRunnerFL._prepare_inputs (%d matches); leaving it untouched. "
            "GCU aborts with TOPSATEN_STATUS_NOT_SUPPORT if it is reached.",
            matches,
        )
        return

    # ``getsource`` returns the method starting at its own line 1; pad the
    # source back to the original offset so tracebacks (and py-spy) keep
    # pointing at the real lines of the shared file.
    padded = "\n" * (fn.__code__.co_firstlineno - 1) + rewritten

    try:
        namespace: dict = {}
        code = compile(
            padded, inspect.getsourcefile(fn) or "<model_runner>", "exec"
        )
        exec(code, fn.__globals__, namespace)
        patched = namespace["_prepare_inputs"]
    except Exception as exc:
        logger.warning("runner_compat: rewriting _prepare_inputs failed: %s", exc)
        return

    patched.__qualname__ = fn.__qualname__
    patched.__doc__ = fn.__doc__
    patched._gcu_i32_positions = True
    ModelRunnerFL._prepare_inputs = patched
    logger.info(
        "runner_compat: rewrote the int64 positions add in "
        "ModelRunnerFL._prepare_inputs to int32 (no int64 add kernel on GCU)"
    )
