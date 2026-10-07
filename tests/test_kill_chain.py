import pandas as pd

from kill_chain import KillChainEngine


def _chain_df():
    return pd.DataFrame([
        {"chain_id": "c1", "label": "brute_force", "timestamp": "2026-01-01T00:00:00"},
        {"chain_id": "c1", "label": "lateral_movement", "timestamp": "2026-01-01T04:00:00"},
        {"chain_id": "c1", "label": "low_and_slow_exfil", "timestamp": "2026-01-01T20:00:00"},
        {"chain_id": "", "label": "normal", "timestamp": "2026-01-01T01:00:00"},
    ])


def test_fit_learns_a_transition_matrix_from_chain_sequences():
    engine = KillChainEngine().fit(_chain_df())
    assert "initial_access" in engine.transition_matrix
    probs = engine.transition_matrix["initial_access"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_update_advances_stage_forward_but_not_backward():
    engine = KillChainEngine().fit(_chain_df())
    engine.update("e1", "lateral_movement", "2026-01-01T00:00:00")
    assert engine.entity_states["e1"]["stage"] == "lateral_movement"

    engine.update("e1", "brute_force", "2026-01-01T01:00:00")
    assert engine.entity_states["e1"]["stage"] == "lateral_movement"


def test_get_full_state_for_unseen_entity_is_dormant_with_no_warning():
    engine = KillChainEngine().fit(_chain_df())
    state = engine.get_full_state("never_seen")
    assert state["current_stage"] == "dormant"
    assert state["escalation_warning"] is False


def test_escalation_warning_fires_once_lateral_movement_reached():
    engine = KillChainEngine().fit(_chain_df())
    engine.update("e1", "lateral_movement", "2026-01-01T00:00:00")
    state = engine.get_full_state("e1")
    assert state["next_stage"] == "exfiltration"
    assert state["escalation_warning"] is True
