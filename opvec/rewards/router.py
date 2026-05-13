"""Task-to-reward adapter routing."""

from __future__ import annotations

from collections.abc import Sequence

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

    def batch_score(self, prompt_records: Sequence[dict] | dict, output_texts: Sequence[str]) -> list[dict]:
        """Score a batch without changing per-task official reward semantics.

        ``prompt_records`` may be a same-length sequence or a single prompt
        record to broadcast across multiple samples from the same prompt.
        """

        if isinstance(prompt_records, dict):
            records = [prompt_records for _ in output_texts]
        else:
            records = list(prompt_records)
        outputs = list(output_texts)
        if len(records) != len(outputs):
            raise ValueError(f"batch_score length mismatch: {len(records)} records vs {len(outputs)} outputs")
        return [self.score(record, output) for record, output in zip(records, outputs)]
