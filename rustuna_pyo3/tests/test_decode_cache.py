"""Regression tests for decode-cache admission under oversized scans.

Use fresh interpreters so the process-global cache and cached json.loads callable are
isolated. Counting decodes checks reuse directly, without timing-dependent assertions.
"""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("workload", ["hot_value", "oversized_scan"])
def test_decode_cache_retains_reused_values(workload: str) -> None:
    script = """
        import collections
        import json
        import rustuna

        loads = json.loads
        counts = collections.Counter()

        def counting_loads(text, *args, **kwargs):
            counts[text] += 1
            return loads(text, *args, **kwargs)

        json.loads = counting_loads
    """
    if workload == "hot_value":
        script += """
        hot = rustuna.create_trial(value=0.0, user_attrs={"value": ["frequently-read-value"]})
        for i in range(70000):
            assert hot.user_attrs["value"] == ["frequently-read-value"]
            cold = rustuna.create_trial(value=0.0, user_attrs={"value": f"unique-{i}"})
            assert cold.user_attrs["value"] == f"unique-{i}"
        assert hot.user_attrs["value"] == ["frequently-read-value"]
        assert counts[json.dumps(["frequently-read-value"])] == 1
        """
    else:
        script += """
        trials = [rustuna.create_trial(value=0.0, user_attrs={"value": f"unique-{i}"})
                  for i in range(80000)]
        for _ in range(2):
            for i, trial in enumerate(trials):
                assert trial.user_attrs["value"] == f"unique-{i}"
        counts.clear()
        for i, trial in enumerate(trials):
            assert trial.user_attrs["value"] == f"unique-{i}"
        # LRU / clear-on-full re-decodes the entire oversized working set.
        assert sum(counts.values()) < len(trials) // 2, sum(counts.values())
        """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], check=True, timeout=60
    )
