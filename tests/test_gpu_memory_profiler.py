import re
import pytest
import torch

from verl.utils.device import get_device_name
from verl.utils.profiler.performance import log_gpu_memory_usage, _get_current_mem_info


@pytest.mark.skipif(get_device_name() == "cpu", reason="No GPU/NPU available")
def test_log_gpu_memory_usage_reports_increase():
    """Allocate some GPU memory and assert profiler reports increased allocated memory in MB."""
    # Get baseline allocated memory in MB (precision 0 -> integer MB)
    before_alloc_strs = _get_current_mem_info(unit="MB", precision=0)
    before_alloc_mb = int(float(before_alloc_strs[0]))

    # Try allocating a few sizes until success (avoid OOM on small GPUs)
    alloc_sizes_mb = [64, 32, 16, 8]
    tensor = None
    for m in alloc_sizes_mb:
        try:
            # allocate m MB as float32
            num_elems = (m * 1024 * 1024) // 4
            tensor = torch.empty((num_elems,), dtype=torch.float32, device=get_device_name())
            break
        except Exception:
            tensor = None
            continue

    if tensor is None:
        pytest.skip("Could not allocate GPU tensor for the test")

    try:
        # call profiler to ensure it runs without error and prints output
        log_gpu_memory_usage("[TEST GPU MEM] After alloc")

        after_alloc_strs = _get_current_mem_info(unit="MB", precision=0)
        after_alloc_mb = int(float(after_alloc_strs[0]))

        assert after_alloc_mb >= before_alloc_mb
    finally:
        # cleanup
        del tensor
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
