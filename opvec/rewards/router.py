"""Task-to-reward adapter routing."""

from __future__ import annotations

from .bfcl import BFCLToolRewardAdapter
from .simple import adapter_for_task


class RewardRouter:
    def score(self, prompt_record: dict, output_text: str) -> dict:
        reference = prompt_record.get("reference", {})
        if prompt_record.get("task") == "tool" and isinstance(reference, dict) and reference.get("bfcl"):
            adapter = BFCLToolRewardAdapter()
        else:
            adapter = adapter_for_task(str(prompt_record.get("task", "")))
        return adapter.score(prompt_record, output_text).as_dict()
