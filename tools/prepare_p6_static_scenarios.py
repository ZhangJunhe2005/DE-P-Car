#!/usr/bin/env python3
"""Generate and freeze unseen, seed-reproducible P6 static scenarios."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.model.checkpoint import verify_checkpoint
from dep_car.runtime.p6_contract import canonical_sha256, sha256_file
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car.runtime.start_robustness import audit_start_robustness


SCHEMA = "DEPCarP6StaticScenarioManifestV1"


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_grid(folder):
    metadata = yaml.safe_load((folder / "map.yaml").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(str(folder / metadata["image"])).convert("L"))
    data = np.flipud(np.where(image < 128, 100, 0).astype(np.int16))
    return RuntimeOccupancyGrid2D(
        data, float(metadata["resolution"]), tuple(metadata["origin"][:2])
    )


def validate_config(config):
    if config.get("schema") != "DEPCarP6StaticConfigV1":
        raise ValueError("unknown P6 static config schema")
    if int(config["maps"]["generated_count"]) < int(config["maps"]["selected_count"]):
        raise ValueError("generated_count must cover selected_count")
    quotas = config["tasks"]["maneuver_quotas"]
    if sum(int(value) for value in quotas.values()) != int(config["tasks"]["total"]):
        raise ValueError("P6 maneuver quotas must sum to tasks.total")
    required = {
        "NORMAL", "SHARP_TURN", "NARROW_CORRIDOR", "U_TURN",
        "DEAD_END_ESCAPE", "REVERSE_EXIT", "THREE_POINT_TURN",
    }
    if set(quotas) != required:
        raise ValueError("P6 scenario config must cover all seven maneuvers")
    enabled = config.get(
        "enabled_modalities", ["depth_only", "lidar_only", "fusion"]
    )
    if (
        not isinstance(enabled, list)
        or not enabled
        or len(enabled) != len(set(enabled))
        or any(value not in ("depth_only", "lidar_only", "fusion") for value in enabled)
    ):
        raise ValueError("enabled_modalities is invalid")
    if set(config.get("checkpoints", {})) != set(enabled):
        raise ValueError("checkpoint matrix must exactly match enabled_modalities")
    return config


def checkpoint_matrix(config):
    output = {}
    enabled = config.get(
        "enabled_modalities", ["depth_only", "lidar_only", "fusion"]
    )
    for modality in enabled:
        row = config["checkpoints"][modality]
        checkpoint, contract = resolve(row["checkpoint"]), resolve(row["contract"])
        contract_document = json.loads(contract.read_text(encoding="utf-8"))
        if contract_document.get("architecture_id") == "dep_car_multimodal_v2_route_ackermann_3x5":
            payload = __import__("torch").load(
                checkpoint, map_location="cpu", weights_only=True
            )
            verified = dict(contract_document)
            if (
                verified.get("schema") != "DEPCarRouteV2ArtifactContractV1"
                or verified.get("checkpoint_sha256") != sha256_file(checkpoint)
                or payload.get("architecture_id") != verified.get("architecture_id")
                or payload.get("completed_epochs", 0) < 40
                or payload.get("partial_epoch") is not False
                or payload.get("run_completed") is not True
                or contract_document.get("run_completed") is not True
            ):
                raise ValueError("%s V2 checkpoint identity is invalid" % modality)
            acceptance = resolve(
                row.get(
                    "shadow_entry_acceptance",
                    checkpoint.with_suffix(".score_shadow_acceptance.json"),
                )
            )
            evidence = json.loads(acceptance.read_text(encoding="utf-8"))
            acceptance_tool = resolve(evidence.get("acceptance_tool", ""))
            expected_acceptance_tool = (
                ROOT / "tools/audit_p5_route_v2_score.py"
            ).resolve()
            if (
                evidence.get("schema")
                != "DEPCarRouteV2ScoreShadowAcceptanceV1"
                or evidence.get("status") != "PASS"
                or evidence.get("gate_passed") is not True
                or evidence.get("scope") != "P6_SHADOW_ONLY"
                or evidence.get("active_control_authorized") is not False
                or evidence.get("production_qualified") is not False
                or evidence.get("test_split_accessed") is not False
                or evidence.get("checkpoint_sha256") != sha256_file(checkpoint)
                or evidence.get("checkpoint_contract_sha256")
                != sha256_file(contract)
                or evidence.get("acceptance_tool_sha256")
                != sha256_file(acceptance_tool)
                or acceptance_tool != expected_acceptance_tool
            ):
                raise ValueError("%s Score shadow acceptance is invalid" % modality)
        else:
            verified = verify_checkpoint(checkpoint, contract, allow_untrained=True)
        expected = {
            "training_stage": "score_calibration",
            "artifact_role": "best",
            "status": "TRAINED_UNQUALIFIED",
            "production_qualified": False,
            "modality": modality,
        }
        mismatches = [key for key, value in expected.items() if verified.get(key) != value]
        if mismatches:
            raise ValueError("%s checkpoint mismatch: %s" % (modality, ",".join(mismatches)))
        output[modality] = {
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "contract": relative(contract),
            "contract_sha256": sha256_file(contract),
            "architecture_id": verified.get("architecture_id"),
        }
        if verified.get("architecture_id") == "dep_car_multimodal_v2_route_ackermann_3x5":
            output[modality].update({
                "shadow_entry_acceptance": relative(acceptance),
                "shadow_entry_acceptance_sha256": sha256_file(acceptance),
            })
    return output


def commands(config_path, config, root, workers):
    maps = config["maps"]
    map_command = [
        "/usr/bin/python3",
        str(ROOT / "ros/dep_car_dataset/scripts/generate_static_maps.py"),
        "--output", str(root / "maps"),
        "--type", str(maps["type"]),
        "--count", str(maps["generated_count"]),
        "--seed", str(maps["generator_seed"]),
        "--width", str(maps["width"]),
        "--height", str(maps["height"]),
        "--resolution", str(maps["resolution_m"]),
        "--corridor-radius", str(maps["corridor_radius_cells"]),
        "--iterations", str(maps["iterations"]),
        "--workers", str(workers),
        "--resume",
    ]
    task_command = [
        "/usr/bin/python3",
        str(ROOT / "ros/dep_car_dataset/scripts/generate_pilot_tasks.py"),
        "--config", str(config_path),
        "--maps", str(root / "maps"),
        "--output", str(root / "source_task_manifest.json"),
    ]
    return map_command, task_command


def run(command):
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)


def freeze_manifest(config_path, config, root):
    source_path = root / "source_task_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_copy = dict(source)
    expected_source_hash = source_copy.pop("task_manifest_sha256", "")
    if expected_source_hash != canonical_sha256(source_copy):
        raise ValueError("source task manifest identity mismatch")
    checkpoints = checkpoint_matrix(config)
    scenarios = []
    gazebo_seed = int(config["runtime"]["gazebo_seed"])
    grid_cache = {}
    for index, task in enumerate(source["tasks"]):
        world, map_yaml = resolve(task["world"]), resolve(task["map_yaml"])
        if world.parent not in grid_cache:
            grid_cache[world.parent] = load_grid(world.parent)
        grid = grid_cache[world.parent]
        start_robustness = audit_start_robustness(grid, tuple(task["start"]))
        if start_robustness["status"] != "PASS":
            raise ValueError(
                "scenario source start failed perturbation gate: %s" % task["task_id"]
            )
        identity = {
            "map_uuid": task["map_uuid"],
            "map_seed": int(task["map_seed"]),
            "task_seed": int(task["task_seed"]),
            "start": task["start"],
            "goal": task["goal"],
            "maneuver_mode": task["maneuver_mode"],
        }
        scenarios.append({
            "scenario_id": "p6_" + canonical_sha256(identity)[:16],
            "cohort": "holdout" if task["map_split"] == "test" else "development",
            "source_map_split": task["map_split"],
            "maneuver_mode": task["maneuver_mode"],
            "map_name": task["map_name"],
            "map_uuid": task["map_uuid"],
            "map_seed": int(task["map_seed"]),
            "map_occupancy_sha256": task["map_occupancy_sha256"],
            "world": relative(world),
            "world_sha256": sha256_file(world),
            "map_yaml": relative(map_yaml),
            "map_yaml_sha256": sha256_file(map_yaml),
            "start": [float(value) for value in task["start"]],
            "goal": [float(value) for value in task["goal"]],
            "task_seed": int(task["task_seed"]),
            "gazebo_seed": gazebo_seed + index,
            "route_evidence": task["route_evidence"],
            "start_robustness": start_robustness,
        })
    payload = {
        "schema": SCHEMA,
        "config": relative(config_path),
        "config_semantic_sha256": canonical_sha256(config),
        "source_task_manifest": relative(source_path),
        "source_task_manifest_sha256": source["task_manifest_sha256"],
        "arena_tools_commit": "664b950d88c91b34e3cdce62512c8864b574b9d4",
        "checkpoints": checkpoints,
        "scenarios": scenarios,
    }
    payload["scenario_manifest_sha256"] = canonical_sha256(payload)
    output = root / "scenario_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload, output


def verify_manifest(path, config, require_start_audit=True):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    expected = content.pop("scenario_manifest_sha256", "")
    if manifest.get("schema") != SCHEMA or canonical_sha256(content) != expected:
        raise ValueError("P6 scenario manifest identity mismatch")
    if manifest.get("config_semantic_sha256") != canonical_sha256(config):
        raise ValueError("P6 scenario manifest belongs to a different config")
    observed_checkpoints = checkpoint_matrix(config)
    if manifest.get("checkpoints") != observed_checkpoints:
        raise ValueError("P6 checkpoint matrix changed after scenario freezing")
    grid_cache = {}
    for scenario in manifest.get("scenarios", []):
        for key, hash_key in (("world", "world_sha256"), ("map_yaml", "map_yaml_sha256")):
            path_value = resolve(scenario[key])
            if sha256_file(path_value) != scenario[hash_key]:
                raise ValueError("scenario file hash mismatch: " + str(path_value))
        map_manifest = json.loads((resolve(scenario["world"]).parent / "manifest.json").read_text(encoding="utf-8"))
        if (
            map_manifest.get("map_uuid") != scenario["map_uuid"]
            or int(map_manifest.get("seed", -1)) != int(scenario["map_seed"])
            or map_manifest.get("occupancy_sha256") != scenario["map_occupancy_sha256"]
        ):
            raise ValueError("map seed/UUID/occupancy identity mismatch")
        if require_start_audit:
            folder = resolve(scenario["world"]).parent
            if folder not in grid_cache:
                grid_cache[folder] = load_grid(folder)
            grid = grid_cache[folder]
            observed = audit_start_robustness(grid, tuple(scenario["start"]))
            if scenario.get("start_robustness") != observed:
                raise ValueError(
                    "scenario start robustness evidence changed: "
                    + scenario["scenario_id"]
                )
    return manifest


def rebind_manifest(path, config_path, config):
    """Bind frozen scenarios to a new checkpoint matrix without regenerating maps."""

    original_bytes = path.read_bytes()
    manifest = json.loads(original_bytes.decode("utf-8"))
    content = dict(manifest)
    old_hash = content.pop("scenario_manifest_sha256", "")
    if manifest.get("schema") != SCHEMA or canonical_sha256(content) != old_hash:
        raise ValueError("P6 scenario manifest identity mismatch before rebind")
    old_scenarios_sha256 = canonical_sha256(manifest.get("scenarios", []))
    source_path = resolve(manifest.get("source_task_manifest", ""))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("task_manifest_sha256") != manifest.get(
        "source_task_manifest_sha256"
    ):
        raise ValueError("P6 source task manifest changed before rebind")

    # Verify every frozen world/map identity before changing authority fields.
    for scenario in manifest.get("scenarios", []):
        for key, hash_key in (("world", "world_sha256"), ("map_yaml", "map_yaml_sha256")):
            if sha256_file(resolve(scenario[key])) != scenario[hash_key]:
                raise ValueError("scenario file changed before rebind: " + scenario["scenario_id"])
    backup = path.with_name("scenario_manifest.legacy_%s.json" % old_hash[:12])
    if backup.exists() and backup.read_bytes() != original_bytes:
        raise ValueError("P6 legacy manifest backup path contains different bytes")
    if not backup.exists():
        backup.write_bytes(original_bytes)

    manifest["config"] = relative(config_path)
    manifest["config_semantic_sha256"] = canonical_sha256(config)
    manifest["enabled_modalities"] = list(config.get(
        "enabled_modalities", ["depth_only", "lidar_only", "fusion"]
    ))
    manifest["checkpoints"] = checkpoint_matrix(config)
    if canonical_sha256(manifest.get("scenarios", [])) != old_scenarios_sha256:
        raise RuntimeError("scenario payload changed during checkpoint rebind")
    manifest.pop("scenario_manifest_sha256", None)
    manifest["scenario_manifest_sha256"] = canonical_sha256(manifest)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    verify_manifest(path, config)
    return {
        "schema": "DEPCarP6ManifestRebindV1",
        "status": "PASS",
        "old_scenario_manifest_sha256": old_hash,
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        "scenarios_sha256": old_scenarios_sha256,
        "scenarios": len(manifest.get("scenarios", [])),
        "enabled_modalities": manifest["enabled_modalities"],
        "legacy_manifest": str(backup),
        "manifest": str(path),
    }


def reaudit_starts(path, config):
    manifest = verify_manifest(path, config, require_start_audit=False)
    old_hash = manifest["scenario_manifest_sha256"]
    grid_cache = {}
    results = []
    for scenario in manifest["scenarios"]:
        folder = resolve(scenario["world"]).parent
        if folder not in grid_cache:
            grid_cache[folder] = load_grid(folder)
        grid = grid_cache[folder]
        evidence = audit_start_robustness(grid, tuple(scenario["start"]))
        scenario["start_robustness"] = evidence
        results.append({
            "scenario_id": scenario["scenario_id"],
            "cohort": scenario["cohort"],
            "maneuver_mode": scenario["maneuver_mode"],
            **evidence,
        })
    content = dict(manifest)
    content.pop("scenario_manifest_sha256", None)
    manifest["scenario_manifest_sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    verify_manifest(path, config)
    failed = [row["scenario_id"] for row in results if row["status"] != "PASS"]
    report = {
        "schema": "DEPCarP6StartRobustnessAuditV1",
        "status": "PASS" if not failed else "PARTIAL",
        "scenarios": len(results),
        "robust": len(results) - len(failed),
        "not_robust": len(failed),
        "failed_scenario_ids": failed,
        "old_scenario_manifest_sha256": old_hash,
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        "results": results,
    }
    output = ROOT / "reports/p6_start_robustness_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p6_static.yaml")
    parser.add_argument("--root", type=Path, default=ROOT / "data/p6_static")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reaudit-starts", action="store_true")
    parser.add_argument("--rebind-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config_path = args.config.resolve()
    config = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    root = args.root.resolve()
    matrix = checkpoint_matrix(config)
    map_command, task_command = commands(config_path, config, root, args.workers)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "parallel_map_workers": args.workers,
            "checkpoints": matrix,
            "commands": [map_command, task_command],
            "output": str(root / "scenario_manifest.json"),
        }, indent=2, sort_keys=True))
        return
    if args.verify:
        manifest = verify_manifest(root / "scenario_manifest.json", config)
        print(json.dumps({
            "status": "PASS",
            "scenarios": len(manifest["scenarios"]),
            "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        }, indent=2, sort_keys=True))
        return
    if args.rebind_checkpoints:
        report = rebind_manifest(
            root / "scenario_manifest.json", config_path, config
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.reaudit_starts:
        report, output = reaudit_starts(root / "scenario_manifest.json", config)
        print(json.dumps({
            "status": report["status"],
            "scenarios": report["scenarios"],
            "robust": report["robust"],
            "not_robust": report["not_robust"],
            "failed_scenario_ids": report["failed_scenario_ids"],
            "scenario_manifest_sha256": report["scenario_manifest_sha256"],
            "report": str(output),
        }, indent=2, sort_keys=True))
        return
    root.mkdir(parents=True, exist_ok=True)
    run(map_command)
    run(task_command)
    manifest, output = freeze_manifest(config_path, config, root)
    verify_manifest(output, config)
    print(json.dumps({
        "status": "PASS",
        "scenarios": len(manifest["scenarios"]),
        "development": sum(row["cohort"] == "development" for row in manifest["scenarios"]),
        "holdout": sum(row["cohort"] == "holdout" for row in manifest["scenarios"]),
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
