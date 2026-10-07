from mitre import get_mitre, MITRE_MAP, UNKNOWN_TECHNIQUE


def test_known_type_maps_to_its_technique():
    info = get_mitre("brute_force")
    assert info["technique_id"] == "T1110.001"
    assert info["tactic"] == "Credential Access"


def test_like_suffix_from_rule_based_fallback_still_resolves():
    assert get_mitre("lateral_movement_like") == MITRE_MAP["lateral_movement"]


def test_unrecognized_type_falls_back_to_unknown_technique():
    assert get_mitre("something_never_seen") == UNKNOWN_TECHNIQUE


def test_every_mapped_entry_has_required_fields():
    for technique in MITRE_MAP.values():
        assert {"technique_id", "technique_name", "tactic"} <= technique.keys()
