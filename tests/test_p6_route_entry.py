import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_p5_route_v2_score as score_audit
import prepare_p6_static_scenarios as scenarios
import run_p6_static as p6_runner


def test_interactive_goal_heading_is_opt_in_but_fixed_runs_remain_strict():
    class Args:
        require_goal_heading = False

    args = Args()
    args.stage = "interactive"
    assert not p6_runner.goal_heading_required(args)
    args.require_goal_heading = True
    assert p6_runner.goal_heading_required(args)
    args.require_goal_heading = False
    args.stage = "shadow"
    assert p6_runner.goal_heading_required(args)


def test_route_v2_p6_config_is_explicitly_fusion_only():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_static_route_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    scenarios.validate_config(config)
    assert config["enabled_modalities"] == ["fusion"]
    assert set(config["checkpoints"]) == {"fusion"}
    assert config["checkpoints"]["fusion"]["shadow_entry_acceptance"].endswith(
        ".score_shadow_acceptance.json"
    )


def test_score_shadow_contract_is_validation_only_and_never_active():
    contract, _ = score_audit.load_contract()
    assert contract["population"]["split"] == "validation"
    assert contract["population"]["maximum_samples"] == 0
    assert contract["authority"]["test_split_sealed"] is True
    assert contract["scope"] == "P6_SHADOW_ONLY"


def test_manifest_rebind_preserves_frozen_scenarios_and_archives_old_bytes(
    tmp_path, monkeypatch
):
    world = tmp_path / "map.world"
    map_yaml = tmp_path / "map.yaml"
    source = tmp_path / "source_task_manifest.json"
    manifest_path = tmp_path / "scenario_manifest.json"
    world.write_text("world", encoding="utf-8")
    map_yaml.write_text("image: map.png\n", encoding="utf-8")
    source.write_text(
        json.dumps({"task_manifest_sha256": "source-hash"}), encoding="utf-8"
    )
    frozen = [{
        "scenario_id": "p6_fixed",
        "world": str(world),
        "world_sha256": scenarios.sha256_file(world),
        "map_yaml": str(map_yaml),
        "map_yaml_sha256": scenarios.sha256_file(map_yaml),
        "start": [0.0, 0.0, 0.0],
        "goal": [1.0, 0.0, 0.0],
    }]
    manifest = {
        "schema": scenarios.SCHEMA,
        "config": "old.yaml",
        "config_semantic_sha256": "old-config",
        "source_task_manifest": str(source),
        "source_task_manifest_sha256": "source-hash",
        "checkpoints": {"fusion": {"checkpoint_sha256": "old"}},
        "scenarios": frozen,
    }
    manifest["scenario_manifest_sha256"] = scenarios.canonical_sha256(manifest)
    original = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        scenarios,
        "checkpoint_matrix",
        lambda _config: {"fusion": {"checkpoint_sha256": "new"}},
    )
    monkeypatch.setattr(scenarios, "verify_manifest", lambda *_args, **_kwargs: None)
    config = {"enabled_modalities": ["fusion"], "checkpoints": {"fusion": {}}}
    report = scenarios.rebind_manifest(
        manifest_path, tmp_path / "p6.yaml", config
    )
    rebound = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert rebound["scenarios"] == frozen
    assert rebound["enabled_modalities"] == ["fusion"]
    assert rebound["checkpoints"]["fusion"]["checkpoint_sha256"] == "new"
    assert Path(report["legacy_manifest"]).read_text(encoding="utf-8") == original
