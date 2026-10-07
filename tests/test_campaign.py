from campaign import _ip_subnet24, _classify_campaign


def test_ip_subnet24_truncates_to_first_three_octets():
    assert _ip_subnet24("10.20.30.40") == "10.20.30"


def test_ip_subnet24_handles_malformed_input():
    assert _ip_subnet24("not-an-ip") == ""
    assert _ip_subnet24("") == ""


def test_classify_campaign_recognizes_breach_and_pivot():
    assert _classify_campaign({"brute_force", "lateral_movement"}) == "Breach-and-Pivot Campaign"


def test_classify_campaign_strips_like_suffix_from_rule_based_labels():
    assert _classify_campaign({"brute_force_like", "credential_stuffing_like"}) == "Credential Spray Campaign"


def test_classify_campaign_falls_back_to_multi_vector_apt():
    assert _classify_campaign({"totally_novel_attack"}) == "Multi-Vector APT Campaign"
