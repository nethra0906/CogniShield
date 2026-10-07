import pandas as pd

from severity import compute_severity, identify_sensitive_resources


def test_low_anomaly_score_yields_low_tier():
    score, tier = compute_severity(
        anomaly_score=0.0,
        score_95th=1.0,
        entity_type="user",
        resource="/resource/1",
        sensitive_resources=set(),
        off_hours=False,
        new_geo=False,
        auth_failed=False,
    )
    assert score == 0.0
    assert tier == "Low"


def test_max_base_with_every_multiplier_stays_under_the_100_cap():
    # base is capped at 60 (anomaly_score/score_95th maxes out at 1.0), and the
    # multiplier tops out at 1.6 with every bonus active, so the ceiling here
    # is 96, not 100 - this pins that ceiling down as a regression guard.
    score, tier = compute_severity(
        anomaly_score=10.0,
        score_95th=1.0,
        entity_type="service_account",
        resource="/resource/1",
        sensitive_resources={"/resource/1"},
        off_hours=True,
        new_geo=True,
        auth_failed=True,
    )
    assert score == 96.0
    assert tier == "Critical"




def test_service_account_multiplier_raises_score_over_plain_user():
    user_score, _ = compute_severity(
        anomaly_score=0.8, score_95th=1.0, entity_type="user", resource="x",
        sensitive_resources=set(), off_hours=False, new_geo=False, auth_failed=False,
    )
    svc_score, _ = compute_severity(
        anomaly_score=0.8, score_95th=1.0, entity_type="service_account", resource="x",
        sensitive_resources=set(), off_hours=False, new_geo=False, auth_failed=False,
    )
    assert svc_score > user_score


def test_off_hours_and_new_geo_must_combine_to_add_bonus():
    off_hours_only, _ = compute_severity(
        anomaly_score=0.8, score_95th=1.0, entity_type="user", resource="x",
        sensitive_resources=set(), off_hours=True, new_geo=False, auth_failed=False,
    )
    both, _ = compute_severity(
        anomaly_score=0.8, score_95th=1.0, entity_type="user", resource="x",
        sensitive_resources=set(), off_hours=True, new_geo=True, auth_failed=False,
    )
    assert both > off_hours_only


def test_tier_boundaries():
    assert compute_severity(1.0, 1.0, "user", "x", set(), False, False, False)[1] == "High"
    assert compute_severity(0.6, 1.0, "user", "x", set(), False, False, False)[1] == "Medium"
    assert compute_severity(0.2, 1.0, "user", "x", set(), False, False, False)[1] == "Low"


def test_identify_sensitive_resources_returns_top_pct_by_frequency():
    df = pd.DataFrame({
        "resource_accessed": ["a"] * 10 + ["b"] * 5 + ["c"] * 1 + ["d"] * 1 + ["e"] * 1,
    })
    sensitive = identify_sensitive_resources(df, top_pct=0.20)
    assert "a" in sensitive
    assert len(sensitive) == 1


def test_identify_sensitive_resources_never_returns_empty_set():
    df = pd.DataFrame({"resource_accessed": ["only_one"]})
    assert identify_sensitive_resources(df) == {"only_one"}
