import unittest

import torch

from scripts.attention_pauh.core import (
    LayerEnergy,
    apply_coefficient_floors_preserve_mean,
    entry_activation_energy,
    gate_values_from_layer_coefficients,
    layer_coefficients_from_scores,
    linear_delta_probe,
    transform_layer_scores,
)
from scripts.attention_pauh.build_signature_preserving_gates import (
    build_spre_gates,
    should_shrink_family,
)
from scripts.attention_pauh.build_structured_capability_gates import (
    CapabilityProfile,
    build_structured_capability_gates,
    coefficient_for_entry,
    parse_int_set,
)
from scripts.attention_pauh.build_response_conditioned_residual_filtering_gates import (
    RcrfProfile,
    build_rcrf_gates,
)
from scripts.attention_pauh.probe_signed_utility import (
    build_probe_token_mask,
    response_char_intervals,
    response_span_token_mask,
)
from scripts.attention_pauh.summarize_gate_structure import (
    expand_gate_rows,
    summarize_groups,
)


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            def __init__(self, size):
                self.input_ids = list(range(size))

        return Encoded(len(text))


class AttentionPauhCoreTest(unittest.TestCase):
    def test_constant_scores_fall_back_to_alpha(self):
        scores = {
            "tool": {
                0: LayerEnergy(utility=1.0, harm=1.0, raw_score=0.0, score=0.0),
                1: LayerEnergy(utility=1.0, harm=1.0, raw_score=0.0, score=0.0),
            }
        }

        coeffs = layer_coefficients_from_scores(
            scores,
            alpha_by_expert={"tool": 0.75},
            beta=0.7,
            min_coeff=0.25,
            max_coeff=1.25,
        )

        self.assertAlmostEqual(coeffs["tool"][0], 0.75, places=6)
        self.assertAlmostEqual(coeffs["tool"][1], 0.75, places=6)

    def test_coefficients_preserve_expert_budget_after_clipping(self):
        scores = {
            "code": {
                0: LayerEnergy(utility=10.0, harm=1.0, raw_score=2.0, score=3.0),
                1: LayerEnergy(utility=1.0, harm=1.0, raw_score=0.0, score=0.0),
                2: LayerEnergy(utility=0.1, harm=1.0, raw_score=-2.0, score=-3.0),
            }
        }

        coeffs = layer_coefficients_from_scores(
            scores,
            alpha_by_expert={"code": 0.75},
            beta=2.0,
            min_coeff=0.25,
            max_coeff=1.25,
        )["code"]

        values = list(coeffs.values())
        self.assertGreaterEqual(min(values), 0.25)
        self.assertLessEqual(max(values), 1.25)
        self.assertAlmostEqual(sum(values) / len(values), 0.75, places=6)

    def test_gate_values_scope_attn_only_skips_mlp(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool"},
            ]
        }

        gates = gate_values_from_layer_coefficients(
            manifest,
            {"tool": {0: 0.75}},
            scope="attn-only",
        )

        self.assertEqual(gates, {"model.layers.0.self_attn.q_proj.weight::tool": 0.75})

    def test_gate_values_scope_layer_all_keeps_mlp(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool"},
            ]
        }

        gates = gate_values_from_layer_coefficients(
            manifest,
            {"tool": {0: 0.75}},
            scope="layer-all",
        )

        self.assertEqual(
            gates,
            {
                "model.layers.0.self_attn.q_proj.weight::tool": 0.75,
                "model.layers.0.mlp.down_proj.weight::tool": 0.75,
            },
        )

    def test_entry_activation_energy_delta_norm_normalization(self):
        delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        diag = torch.tensor([2.0, 5.0])

        raw = entry_activation_energy(delta, diag, normalization="none")
        normalized = entry_activation_energy(delta, diag, normalization="delta-norm")

        # Column norms are [10, 20], so raw = 10 * 2 + 20 * 5 = 120.
        self.assertAlmostEqual(raw, 120.0)
        self.assertAlmostEqual(normalized, 120.0 / 30.0)

    def test_linear_delta_probe_matches_first_order_effect(self):
        delta = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        inputs = torch.tensor([[[3.0, 5.0], [7.0, 11.0]]])
        output_grads = torch.tensor([[[2.0, -1.0], [1.0, 4.0]]])
        token_mask = torch.tensor([True, False])

        expression, signed_effect, mean_update = linear_delta_probe(
            delta=delta,
            inputs=inputs,
            output_grads=output_grads,
            token_mask=token_mask,
        )

        # First token induced update is [3, 10].
        self.assertAlmostEqual(expression, 3.0**2 + 10.0**2)
        self.assertAlmostEqual(signed_effect, -((2.0 * 3.0) + (-1.0 * 10.0)))
        self.assertTrue(torch.equal(mean_update, torch.tensor([3.0, 10.0])))

    def test_transform_inverse_and_smooth_scores(self):
        scores = {"memory": {0: 1.0, 1: 3.0, 2: 5.0}}

        inverted = transform_layer_scores(scores, transform="inverse")
        smoothed = transform_layer_scores(scores, transform="smooth", smooth_radius=1)

        self.assertEqual(inverted["memory"], {0: -1.0, 1: -3.0, 2: -5.0})
        self.assertEqual(smoothed["memory"], {0: 2.0, 1: 3.0, 2: 4.0})

    def test_coefficient_floors_preserve_mean_when_feasible(self):
        coeffs = {"memory": {0: 1.4, 1: 0.2, 2: 0.4, 3: 1.0}}

        adjusted = apply_coefficient_floors_preserve_mean(
            coeffs,
            alpha_by_expert={"memory": 1.0},
            floors={"memory": {1: 0.8, 2: 0.8}},
            min_coeff=0.2,
            max_coeff=1.6,
        )["memory"]

        self.assertGreaterEqual(adjusted[1], 0.8)
        self.assertGreaterEqual(adjusted[2], 0.8)
        self.assertAlmostEqual(sum(adjusted.values()) / len(adjusted), 1.0, places=6)

    def test_gate_values_scope_hybrid_scales_mlp_toward_alpha(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool"},
            ]
        }

        gates = gate_values_from_layer_coefficients(
            manifest,
            {"tool": {0: 1.4}},
            scope="hybrid",
            alpha_by_expert={"tool": 1.0},
            mlp_residual_scale=0.5,
        )

        self.assertAlmostEqual(gates["model.layers.0.self_attn.q_proj.weight::tool"], 1.4)
        self.assertAlmostEqual(gates["model.layers.0.mlp.down_proj.weight::tool"], 1.2)

    def test_spre_shrinks_code_attention_but_not_mlp(self):
        self.assertTrue(
            should_shrink_family(
                expert="code",
                family="attn_q",
                ratios={"prompt": 0.80, "response": 0.74},
                shrink_threshold=0.85,
            )
        )

    def test_spre_memory_attn_calm_isolates_memory_attention(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "memory"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory"},
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "code"},
            ]
        }

        gates, _, decisions = build_spre_gates(
            manifest=manifest,
            exposure=None,
            default_alpha=1.0,
            shrink_threshold=0.85,
            min_coeff=0.70,
            method="memory-attn-calm",
            code_attn_qkv_scale=0.75,
            code_attn_o_scale=0.90,
            attention_calm_scale=0.60,
        )

        self.assertAlmostEqual(gates["model.layers.0.self_attn.q_proj.weight::memory"], 0.60)
        self.assertAlmostEqual(gates["model.layers.0.mlp.down_proj.weight::memory"], 1.00)
        self.assertAlmostEqual(gates["model.layers.0.self_attn.q_proj.weight::tool"], 1.00)
        self.assertAlmostEqual(gates["model.layers.0.self_attn.q_proj.weight::code"], 1.00)
        reasons = {row["param_name"] + "::" + row["expert"]: row["reason"] for row in decisions}
        self.assertEqual(reasons["model.layers.0.self_attn.q_proj.weight::memory"], "memory_attention_calm")

    def test_structured_capability_gate_rules(self):
        profile = CapabilityProfile(name="toy", description="toy")
        mid_layers = parse_int_set("8-20")
        conflict_layers = parse_int_set("24,27")

        self.assertEqual(parse_int_set("1,3-5"), {1, 3, 4, 5})
        self.assertEqual(
            coefficient_for_entry(
                expert="memory",
                layer=16,
                family="attn_q",
                profile=profile,
                code_mid_layers=mid_layers,
                code_conflict_layers=conflict_layers,
            ),
            (1.0, "preserve_memory_attention_and_mlp"),
        )
        self.assertEqual(
            coefficient_for_entry(
                expert="code",
                layer=16,
                family="mlp_up",
                profile=profile,
                code_mid_layers=mid_layers,
                code_conflict_layers=conflict_layers,
            ),
            (1.0, "open_code_mid_positive_family"),
        )
        self.assertEqual(
            coefficient_for_entry(
                expert="code",
                layer=16,
                family="mlp_down",
                profile=profile,
                code_mid_layers=mid_layers,
                code_conflict_layers=conflict_layers,
            ),
            (0.85, "guard_code_mid_weak_family"),
        )
        self.assertEqual(
            coefficient_for_entry(
                expert="code",
                layer=27,
                family="mlp_up",
                profile=profile,
                code_mid_layers=mid_layers,
                code_conflict_layers=conflict_layers,
            ),
            (0.5, "suppress_code_late_conflict"),
        )

    def test_structured_capability_gate_materializes_manifest(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.16.mlp.up_proj.weight", "expert": "code"},
                {"param_name": "model.layers.16.mlp.down_proj.weight", "expert": "code"},
                {"param_name": "model.layers.27.mlp.up_proj.weight", "expert": "code"},
                {"param_name": "model.layers.16.self_attn.q_proj.weight", "expert": "memory"},
                {"param_name": "model.layers.16.self_attn.q_proj.weight", "expert": "tool"},
            ]
        }

        gates, decisions = build_structured_capability_gates(
            manifest=manifest,
            profile=CapabilityProfile(name="toy", description="toy"),
            code_mid_layers={16},
            code_conflict_layers={27},
        )

        self.assertAlmostEqual(gates["model.layers.16.mlp.up_proj.weight::code"], 1.0)
        self.assertAlmostEqual(gates["model.layers.16.mlp.down_proj.weight::code"], 0.85)
        self.assertAlmostEqual(gates["model.layers.27.mlp.up_proj.weight::code"], 0.5)
        self.assertAlmostEqual(gates["model.layers.16.self_attn.q_proj.weight::memory"], 1.0)
        self.assertAlmostEqual(gates["model.layers.16.self_attn.q_proj.weight::tool"], 1.0)
        self.assertEqual(len(decisions), len(manifest["basis_entries"]))

    def test_response_char_intervals_find_tool_call_and_code_block(self):
        tool_text = "<think>x</think> <tool_call>{\"name\":\"a\"}</tool_call> done"
        code_text = "reasoning\n```python\nprint(1)\n```\nfinal"

        tool_intervals = response_char_intervals(tool_text, span="tool-call")
        code_intervals = response_char_intervals(code_text, span="code-block")

        self.assertEqual(tool_text[tool_intervals[0][0] : tool_intervals[0][1]], '<tool_call>{"name":"a"}</tool_call>')
        self.assertEqual(code_text[code_intervals[0][0] : code_intervals[0][1]], "```python\nprint(1)\n```")

    def test_signature_span_uses_causal_shift_for_tool_call(self):
        tokenizer = CharTokenizer()
        response = "abc<tool_call>x</tool_call>z"
        response_ids = tokenizer(response, add_special_tokens=False).input_ids
        prompt_len = 5

        mask = build_probe_token_mask(
            tokenizer=tokenizer,
            response=response,
            response_ids=response_ids,
            seq_len=prompt_len + len(response_ids),
            prompt_len=prompt_len,
            span="signature",
            response_tail_tokens=0,
            task="tool",
        )
        response_mask = response_span_token_mask(
            tokenizer=tokenizer,
            response=response,
            response_ids=response_ids,
            span="tool-call",
        )
        local_indices = torch.nonzero(response_mask, as_tuple=False).view(-1)
        expected_first = prompt_len + int(local_indices[0]) - 1
        expected_last = prompt_len + int(local_indices[-1]) - 1

        self.assertTrue(bool(mask[expected_first]))
        self.assertTrue(bool(mask[expected_last]))
        self.assertFalse(bool(mask[prompt_len + 1]))

    def test_expand_layer_gate_rows_with_manifest_families(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool"},
                {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory"},
            ]
        }

        rows = expand_gate_rows({"layer0.tool": 1.2}, manifest=manifest)
        groups = summarize_groups(rows)

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(groups["tool|attention"]["mean"], 1.2)
        self.assertAlmostEqual(groups["tool|mlp"]["mean"], 1.2)
        self.assertFalse(
            should_shrink_family(
                expert="code",
                family="mlp_up",
                ratios={"prompt": 0.80, "response": 0.74},
                shrink_threshold=0.85,
            )
        )
        self.assertFalse(
            should_shrink_family(
                expert="memory",
                family="attn_o",
                ratios={"prompt": 0.80, "response": 0.74},
                shrink_threshold=0.85,
            )
        )

    def test_rcrf_amplifies_stable_owner_residual(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.mlp.up_proj.weight", "expert": "memory"},
            ]
        }
        signed_summary = {
            "module_summary": {
                "memory": {
                    "model.layers.0.mlp.up_proj.weight": {
                        "memory": {
                            "signed_effect_mean": 2.0,
                            "expression_mean": 1.0,
                            "positive_fraction": 1.0,
                            "harm_mean": 0.0,
                        },
                        "tool": {
                            "signed_effect_mean": 1.0,
                            "expression_mean": 1.0,
                            "positive_fraction": 1.0,
                            "harm_mean": 0.0,
                        },
                    }
                }
            },
            "conflict_summary": {"memory": {}},
        }

        gates, decisions = build_rcrf_gates(
            manifest=manifest,
            signed_summary=signed_summary,
            stats_index={
                "memory": {
                    "mlp_up": {
                        0: signed_summary["module_summary"]["memory"]["model.layers.0.mlp.up_proj.weight"]
                    }
                }
            },
            scale_index={"memory": {"owner_effect_scale": 1.0, "expression_scale": 1.0}},
            profile=RcrfProfile(name="toy", description="toy"),
        )

        self.assertGreater(gates["model.layers.0.mlp.up_proj.weight::memory"], 1.0)
        self.assertEqual(decisions[0]["reason"], "amplify_cross_task_agreement")

    def test_rcrf_suppresses_conflicting_routing_residual(self):
        manifest = {
            "basis_entries": [
                {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
            ]
        }
        signed_summary = {
            "module_summary": {
                "tool": {
                    "model.layers.0.self_attn.q_proj.weight": {
                        "tool": {
                            "signed_effect_mean": -0.5,
                            "expression_mean": 1.0,
                            "positive_fraction": 0.25,
                            "harm_mean": 0.5,
                        },
                        "memory": {
                            "signed_effect_mean": -0.5,
                            "expression_mean": 1.0,
                            "positive_fraction": 0.25,
                            "harm_mean": 0.5,
                        },
                    }
                }
            },
            "conflict_summary": {
                "tool": {
                    "layer_0:memory|tool": {
                        "cosine_mean": -0.2,
                        "negative_fraction": 1.0,
                    }
                }
            },
        }

        gates, decisions = build_rcrf_gates(
            manifest=manifest,
            signed_summary=signed_summary,
            stats_index={
                "tool": {
                    "attn_q": {
                        0: signed_summary["module_summary"]["tool"]["model.layers.0.self_attn.q_proj.weight"]
                    }
                }
            },
            scale_index={"tool": {"owner_effect_scale": 1.0, "expression_scale": 1.0}},
            profile=RcrfProfile(name="toy", description="toy"),
        )

        self.assertLess(gates["model.layers.0.self_attn.q_proj.weight::tool"], 1.0)
        self.assertEqual(decisions[0]["reason"], "suppress_low_energy_or_unstable_residual")


if __name__ == "__main__":
    unittest.main()
