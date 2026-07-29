# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from tests.v1.gpu_connector.bench_e2e import (
    build_vllm_command,
    validate_external_cache_hits,
)


def test_build_mp_vllm_command_disables_hybrid_cache_manager() -> None:
    """vLLM 0.20.1 MP mode disables its incompatible hybrid KV manager."""
    command = build_vllm_command()

    assert "--disable-hybrid-kv-cache-manager" in command
    kv_config = command[command.index("--kv-transfer-config") + 1]
    assert '"kv_connector": "LMCacheMPConnector"' in kv_config


def test_build_high_load_vllm_command() -> None:
    """High-load limits and cached-token reporting are explicit."""
    command = build_vllm_command(
        model="/models/qwen",
        max_model_len=40960,
        max_num_seqs=256,
    )

    assert command[2] == "/models/qwen"
    assert command[command.index("--max-model-len") + 1] == "40960"
    assert command[command.index("--max-num-seqs") + 1] == "256"
    assert "--no-enable-prefix-caching" in command
    assert "--enable-prompt-tokens-details" in command


def test_external_cache_hit_validation() -> None:
    """Every SSD-only hot request must report a complete external hit."""
    validate_external_cache_hits([1024, 1024], 1024, "c2")

    with pytest.raises(RuntimeError, match="expected 1024"):
        validate_external_cache_hits([1024, None], 1024, "c2")
