"""OP-VEC gate-training adapters for verl experiments."""

from .gate_actor import export_effective_hf_state_dict, install_opvec_gate_actor
from .reward_fn import compute_score

__all__ = ["compute_score", "export_effective_hf_state_dict", "install_opvec_gate_actor"]
