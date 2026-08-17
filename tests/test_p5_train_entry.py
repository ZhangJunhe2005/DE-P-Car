import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import yaml
import cv2


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_dep_car as trainer
import accept_p5_candidate as acceptance_tool
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.training.dataset import map_split
from dep_car.training.p4_dataset import (
    TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
    TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
    training_index_content_aggregate,
)


REAL_STATE_DICT_VALIDATOR = trainer._validate_model_state_dict_structure


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checkpoint(path, payload, contract):
    payload = dict(payload)
    implementation = build_p4_implementation_contract(ROOT)
    payload.setdefault(
        "implementation_aggregate_sha256", implementation["aggregate_sha256"]
    )
    torch.save(payload, path)
    contract = dict(contract)
    contract.update({
        "architecture_id": trainer.P4_ARCHITECTURE_ID,
        "checkpoint_sha256": sha256(path),
        "production_qualified": False,
        "implementation_contract": implementation,
    })
    path.with_suffix(".contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return path


def full_contract(**updates):
    contract = json.loads(
        trainer.DEFAULT_CANDIDATE_INITIALIZATION.with_suffix(
            ".contract.json"
        ).read_text(encoding="utf-8")
    )
    contract["implementation_contract"] = build_p4_implementation_contract(ROOT)
    contract["training_contract"]["objective_id"] = trainer.DEPCarObjectiveV1.objective_id
    contract["training_contract"]["objective_revision"] = (
        trainer.DEPCarObjectiveV1.objective_revision
    )
    contract.update(updates)
    return contract


@pytest.fixture
def authority(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer, "_validate_model_state_dict_structure", lambda _state: None
    )
    monkeypatch.setattr(
        trainer, "_authority_path", lambda value, _name: Path(value).resolve()
    )
    monkeypatch.setattr(
        trainer,
        "verify_checkpoint",
        lambda _checkpoint, contract, **_kwargs: json.loads(
            Path(contract).read_text(encoding="utf-8")
        ),
    )
    samples = tmp_path / "samples"
    maps = tmp_path / "maps"
    samples.mkdir()
    maps.mkdir()
    index = tmp_path / "training_index.json"
    entries = []
    for split, filename in (("train", "a.npz"), ("validation", "b.npz")):
        map_uuid = next(
            "p5-index-%s-%04d" % (split, value)
            for value in range(10000)
            if map_split("p5-index-%s-%04d" % (split, value)) == split
        )
        folder = samples / map_uuid
        folder.mkdir()
        sample = folder / filename
        sample.write_bytes((split + "-content-authority").encode("utf-8"))
        map_folder = maps / map_uuid
        map_folder.mkdir()
        pixels = np.full((3, 4), 254 if split == "train" else 205, dtype=np.uint8)
        assert cv2.imwrite(str(map_folder / "map.png"), pixels)
        (map_folder / "manifest.json").write_text(json.dumps({
            "map_uuid": map_uuid,
            "occupancy_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }), encoding="utf-8")
        (map_folder / "map.yaml").write_text(yaml.safe_dump({
            "image": "map.png",
            "resolution": 0.1,
            "origin": [-1.0, -1.0, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }), encoding="utf-8")
        stat = sample.stat()
        entries.append({
            "path": str(sample.relative_to(samples)),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "content_sha256": sha256(sample),
            "map_uuid": map_uuid,
            "split": split,
            "maneuver_mode": "NORMAL",
            "sample_id": sample.stem,
            "task_id": "synthetic-p5",
            "candidate_context": "MISSION",
        })
    index.write_text(json.dumps({
        "schema": trainer.TRAINING_INDEX_SCHEMA,
        "training_view": trainer.TRAINING_VIEW_SCHEMA,
        "content_hash_algorithm": TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
        "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "content_aggregate_sha256": training_index_content_aggregate(entries),
        "sample_root": str(samples.resolve()),
        "maps_root": str(maps.resolve()),
        "splits": ["train", "validation"],
        "sensor_authority": "urban_car_depth_vlp16_sim",
        "bev_preprocessing_sha256": trainer.EXPECTED_BEV_PREPROCESSING_SHA256,
        "samples": len(entries),
        "counts_by_split": {"train": 1, "validation": 1},
        "counts_by_mode": {"NORMAL": 2},
        "entries": entries,
    }), encoding="utf-8")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    map_contract = trainer._map_contract_aggregate(maps, index_payload)
    training_document = yaml.safe_load(trainer.DEFAULT_TRAINING_CONFIG.read_text())
    training_document["dataset"].update({
        "root": str(samples.resolve()),
        "maps": str(maps.resolve()),
        "index": str(index.resolve()),
        "content_aggregate_sha256": index_payload["content_aggregate_sha256"],
        "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
    })
    training_document["qualification"].update({
        "corrected_footprint_p3_status": "PASS",
        "p5_formal_training_allowed": True,
        "blocked_gates": [],
    })
    training_config = tmp_path / "training.yaml"
    training_config.write_text(yaml.safe_dump(training_document), encoding="utf-8")
    monkeypatch.setattr(trainer, "DEFAULT_TRAINING_CONFIG", training_config)
    training_authority = trainer._training_config_authority()
    initialization = write_checkpoint(
        tmp_path / "depth_init.pth",
        {
            "checkpoint_version": trainer.FORMAL_INITIALIZATION_VERSION,
            "architecture_id": trainer.P4_ARCHITECTURE_ID,
            "model_state_dict": {"sentinel": torch.zeros(1)},
            "status": trainer.FORMAL_INITIALIZATION_STATUS,
            "production_qualified": False,
        },
        full_contract(**{
            "status": trainer.FORMAL_INITIALIZATION_STATUS,
            "dataset_provenance": {
                **full_contract()["dataset_provenance"],
                "p3_preprocessing_sha256": trainer.EXPECTED_BEV_PREPROCESSING_SHA256,
            },
            "training_contract": {
                **full_contract()["training_contract"],
                "loss_config_sha256": training_authority["loss_config_sha256"],
            },
        }),
    )
    monkeypatch.setattr(trainer, "DEFAULT_CANDIDATE_INITIALIZATION", initialization)
    return {
        "samples": samples,
        "maps": maps,
        "index": index,
        "initialization": initialization,
        "output": tmp_path / "output.pth",
    }


def arguments(authority, *extra):
    return trainer.build_parser().parse_args([
        "--data", str(authority["samples"]),
        "--maps", str(authority["maps"]),
        "--index", str(authority["index"]),
        "--output", str(authority["output"]),
        *extra,
    ])


def candidate_checkpoint(
    path, authority, *, modality="fusion", stage="candidate_capacity", steps=10000,
    smoke=False, partial=False,
):
    config = trainer._training_config_authority()
    implementation = build_p4_implementation_contract(ROOT)
    def metric_row(frames):
        return {
            "frames": frames,
            "zero_feasible_rate": 0.01,
            "mean_feasible_candidates": 10.0,
            "kinematic_violation_rate": 0.01,
        }
    validation = {
        "total": 1.0,
        "geometry_valid_fraction": 0.95,
        "candidate_metrics": {
            "overall": {
                "frames": 350,
                "zero_feasible_rate": 0.01,
                "mean_feasible_candidates": 10.0,
                "kinematic_violation_rate": 0.01,
            },
            "by_maneuver": {
                mode: metric_row(50)
                for mode in config["raw"]["qualification"]["candidate_acceptance"]["required_maneuvers"]
            },
            "by_requested_gear": {
                "FORWARD": metric_row(175), "REVERSE": metric_row(175),
            },
            "by_candidate_context": {
                "MISSION": metric_row(175), "RECOVERY": metric_row(175),
            },
        },
    }
    metrics_payload = {"validation": validation}
    index_payload = json.loads(authority["index"].read_text(encoding="utf-8"))
    map_contract = trainer._map_contract_aggregate(authority["maps"], index_payload)
    effective_contract = {
        "stage": stage,
        "modality": modality,
        "batch_size": 16,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
        "gradient_clip": 5.0,
        "sensor_dropout_probability": 0.10 if modality == "fusion" else 0.0,
        "amp_requested": True,
        "seed": 49105,
        "torch_threads": 8,
        "workers": 8,
        "epochs": 40,
        "max_samples": None,
        "max_steps": None,
        "data": str(authority["samples"].resolve()),
        "maps": str(authority["maps"].resolve()),
        "index": str(authority["index"].resolve()),
    }
    checkpoint = write_checkpoint(
        path,
        {
            "schema": trainer.CHECKPOINT_SCHEMA,
            "checkpoint_version": trainer.CHECKPOINT_VERSION,
            "architecture_id": trainer.P4_ARCHITECTURE_ID,
            "model_state_dict": {"sentinel": torch.zeros(1)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "grad_scaler_state_dict": {},
            "status": "TRAINED_UNQUALIFIED",
            "qualification_status": "UNQUALIFIED",
            "production_qualified": False,
            "training_stage": stage,
            "modality": modality,
            "global_step": steps,
            "completed_epochs": 40,
            "partial_epoch": partial,
            "smoke_lineage": smoke,
            "training_index_sha256": sha256(authority["index"]),
            "training_config_sha256": config["file_sha256"],
            "loss_config_sha256": config["loss_config_sha256"],
            "trainer_sha256": sha256(ROOT / "tools/train_dep_car.py"),
            "training_index_content_sha256": index_payload["content_aggregate_sha256"],
            "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
            "implementation_aggregate_sha256": implementation["aggregate_sha256"],
            "effective_training_contract": effective_contract,
            "metrics": metrics_payload,
        },
        full_contract(**{
            "status": "TRAINED_UNQUALIFIED",
            "qualification_status": "UNQUALIFIED",
            "training_stage": stage,
            "modality": modality,
            "training_run": {
                "smoke_limited": smoke,
                "smoke_lineage": smoke,
                "partial_epoch": partial,
                "training_config_sha256": config["file_sha256"],
                "loss_config_sha256": config["loss_config_sha256"],
                "trainer_sha256": sha256(ROOT / "tools/train_dep_car.py"),
                "implementation_aggregate_sha256": implementation["aggregate_sha256"],
            },
            "p3_footprint_gate": {"passed": not smoke},
            "index_content_gate": {"passed": not smoke},
            "dataset_authority_gate": {"passed": not smoke},
            "validation_coverage_gate": {"passed": not smoke},
            "training_yaml_qualification_gate": {"passed": not smoke},
            "dataset_provenance": {
                **full_contract()["dataset_provenance"],
                "index_sha256": sha256(authority["index"]),
                "content_aggregate_sha256": index_payload["content_aggregate_sha256"],
                "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
            },
            "training_contract": {
                **full_contract()["training_contract"],
                "loss_config_sha256": config["loss_config_sha256"],
            },
        }),
    )
    metrics = checkpoint.with_suffix(".metrics.json")
    metrics.write_text(json.dumps({
        "schema": "DEPCarP5TrainingMetricsV1",
        "architecture_id": trainer.P4_ARCHITECTURE_ID,
        "training_stage": stage,
        "modality": modality,
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": 40,
        "global_step": steps,
        "partial_epoch": partial,
        "metrics": metrics_payload,
    }), encoding="utf-8")
    contract_path = checkpoint.with_suffix(".contract.json")
    contract = json.loads(contract_path.read_text())
    contract["artifacts"] = {
        "metrics": metrics.name,
        "metrics_sha256": sha256(metrics),
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return checkpoint


def accept_candidate(path):
    result = acceptance_tool.evaluate_candidate(path)
    assert result["gate_passed"] is True


def test_cli_defaults_to_formal_stage_and_eight_workers(authority):
    args = arguments(authority, "--init", str(authority["initialization"]))
    assert args.stage == "candidate_capacity"
    assert args.modality == "fusion"
    assert args.workers == 8
    assert args.torch_threads == 8
    plan = trainer.build_training_plan(args)
    assert plan["splits"] == ("train", "validation")
    assert plan["source"]["kind"] == "formal_depth_v483_initialization"


def test_source_identity_is_verified_before_training_pickle_load(authority, monkeypatch):
    events = []
    original_verify = trainer._verify_sidecar_identity
    original_load = trainer._load_checkpoint

    def verify_first(*args, **kwargs):
        events.append("verify")
        return original_verify(*args, **kwargs)

    def load_second(*args, **kwargs):
        events.append("load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(trainer, "_verify_sidecar_identity", verify_first)
    monkeypatch.setattr(trainer, "_load_checkpoint", load_second)
    args = arguments(authority, "--init", str(authority["initialization"]))
    trainer._inspect_source(args)
    assert events[:2] == ["verify", "load"]


def test_candidate_output_may_not_overwrite_initialization(authority):
    args = arguments(
        authority, "--init", str(authority["initialization"])
    )
    args.output = authority["initialization"]
    with pytest.raises(trainer.TrainingConfigurationError, match="must not overwrite"):
        trainer._inspect_source(args)


def test_resume_output_may_not_overwrite_source(authority):
    source = candidate_checkpoint(authority["output"].parent / "resume.pth", authority)
    args = arguments(authority, "--resume", str(source))
    args.output = source
    with pytest.raises(trainer.TrainingConfigurationError, match="must not overwrite"):
        trainer._inspect_source(args)


def test_score_output_may_not_overwrite_candidate_source(authority):
    source = candidate_checkpoint(authority["output"].parent / "score-source.pth", authority)
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(source),
    )
    args.output = source
    with pytest.raises(trainer.TrainingConfigurationError, match="must not overwrite"):
        trainer._inspect_source(args)


def test_output_sidecar_may_not_alias_source_checkpoint(authority):
    args = arguments(
        authority,
        "--init", str(authority["output"].with_suffix(".contract.json")),
    )
    with pytest.raises(trainer.TrainingConfigurationError, match="sidecars"):
        trainer._inspect_source(args)


def test_existing_output_sidecar_is_rejected_before_training(authority):
    authority["output"].with_suffix(".history.json").write_text(
        "existing", encoding="utf-8"
    )
    args = arguments(
        authority, "--init", str(authority["initialization"])
    )
    with pytest.raises(trainer.TrainingConfigurationError, match="already exist"):
        trainer.build_training_plan(args)


def test_arguments_changed_after_plan_are_rejected(authority):
    args = arguments(
        authority, "--init", str(authority["initialization"])
    )
    plan = trainer.build_training_plan(args)
    args.learning_rate *= 2.0
    args.max_steps = 1
    args.max_samples = 1
    with pytest.raises(
        trainer.TrainingConfigurationError,
        match="arguments_changed_after_training_plan",
    ):
        trainer._require_training_gate(args, plan)


def test_verified_checkpoint_bytes_are_reused_if_path_is_swapped(
    authority, monkeypatch
):
    source = authority["initialization"]
    original_bytes = source.read_bytes()
    contract_document = json.loads(
        source.with_suffix(".contract.json").read_text(encoding="utf-8")
    )

    def swap_after_verification(checkpoint, _contract, **kwargs):
        assert kwargs["checkpoint_bytes"] == original_bytes
        replacement = {
            "architecture_id": trainer.P4_ARCHITECTURE_ID,
            "model_state_dict": {"sentinel": torch.ones(1)},
            "status": "SWAPPED",
            "production_qualified": False,
        }
        torch.save(replacement, checkpoint)
        return contract_document

    monkeypatch.setattr(trainer, "verify_checkpoint", swap_after_verification)
    contract, verified_bytes, _contract_sha = trainer._verify_sidecar_identity(source)
    payload = trainer._load_checkpoint(source, verified_bytes)
    assert contract == contract_document
    assert payload["status"] == trainer.FORMAL_INITIALIZATION_STATUS
    assert verified_bytes == original_bytes
    assert source.read_bytes() != verified_bytes


def test_checkpoint_loader_is_weights_only(authority):
    unsafe = authority["output"].parent / "unsafe.pth"
    torch.save({
        "architecture_id": trainer.P4_ARCHITECTURE_ID,
        "model_state_dict": {"sentinel": torch.zeros(1)},
        "production_qualified": False,
        "unsafe_numpy": np.asarray([1, 2, 3]),
    }, unsafe)
    with pytest.raises(trainer.TrainingConfigurationError, match="cannot load checkpoint"):
        trainer._load_checkpoint(unsafe, unsafe.read_bytes())


def test_candidate_initialization_override_is_permanent_smoke(authority):
    args = arguments(
        authority, "--init", str(authority["output"].parent / "other-init.pth")
    )
    assert "candidate_initialization_override" in trainer._smoke_reasons(args)
    assert trainer._explicit_smoke_run(args) is True


@pytest.mark.parametrize(
    "cap_args", (("--max-samples", "33"), ("--max-samples", "350"), ("--max-steps", "11"))
)
def test_any_training_cap_is_permanent_smoke(authority, cap_args):
    args = arguments(
        authority, "--init", str(authority["initialization"]), *cap_args
    )
    assert trainer._explicit_smoke_run(args) is True


@pytest.mark.parametrize(
    "override,reason",
    (
        (("--epochs", "41"), "training_parameter_override_epochs"),
        (("--batch-size", "17"), "training_parameter_override_batch_size"),
        (("--learning-rate", "0.0002"), "training_parameter_override_learning_rate"),
        (("--weight-decay", "0.00002"), "training_parameter_override_weight_decay"),
        (("--gradient-clip", "4.0"), "training_parameter_override_gradient_clip"),
        (("--sensor-dropout-probability", "0.2"), "training_parameter_override_sensor_dropout_probability"),
        (("--seed", "7"), "training_parameter_override_seed"),
        (("--no-amp",), "training_parameter_override_mixed_precision"),
        (("--workers", "7"), "training_parameter_override_workers"),
        (("--torch-threads", "7"), "training_parameter_override_torch_threads"),
    ),
)
def test_formal_training_parameter_overrides_are_permanent_smoke(
    authority, override, reason
):
    args = arguments(
        authority, "--init", str(authority["initialization"]), *override
    )
    assert reason in trainer._smoke_reasons(args)
    assert trainer._explicit_smoke_run(args) is True
    assert trainer._bounded_smoke_run(args) is False


@pytest.mark.parametrize(
    "caps",
    (
        ("--max-steps", "10"),
        ("--max-samples", "32"),
        ("--max-steps", "11", "--max-samples", "32"),
        ("--max-steps", "10", "--max-samples", "33"),
    ),
)
def test_unbounded_or_oversized_smoke_cannot_bypass_formal_gates(authority, caps):
    args = arguments(
        authority, "--init", str(authority["initialization"]), *caps
    )
    plan = trainer.build_training_plan(args)
    plan["p3_footprint_gate"] = {"passed": False, "errors": ["synthetic_fail"]}
    with pytest.raises(trainer.TrainingConfigurationError, match="formal P5 training"):
        trainer._require_training_gate(args, plan)


def test_double_capped_smoke_can_run_but_remains_permanent_smoke(authority):
    args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--max-steps", "10",
        "--max-samples", "32",
    )
    plan = trainer.build_training_plan(args)
    plan["p3_footprint_gate"] = {"passed": False, "errors": ["synthetic_fail"]}
    trainer._require_training_gate(args, plan)
    assert plan["permanent_smoke"] is True
    assert plan["bounded_smoke_authorized"] is True


def test_map_contract_rejects_manifest_that_disagrees_with_decoded_png(authority):
    map_folder = next(path for path in authority["maps"].iterdir() if path.is_dir())
    manifest_path = map_folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["occupancy_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = json.loads(authority["index"].read_text(encoding="utf-8"))
    with pytest.raises(trainer.TrainingConfigurationError, match="decoded occupancy"):
        trainer._map_contract_aggregate(authority["maps"], index)


def test_trainer_map_contract_rejects_nonzero_origin_yaw(authority):
    map_folder = next(path for path in authority["maps"].iterdir() if path.is_dir())
    yaml_path = map_folder / "map.yaml"
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata["origin"][2] = 0.01
    yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    index = json.loads(authority["index"].read_text(encoding="utf-8"))
    with pytest.raises(trainer.TrainingConfigurationError, match="map contract"):
        trainer._map_contract_aggregate(authority["maps"], index)


def test_non_authority_index_override_can_only_create_permanent_smoke(authority):
    alternate = authority["index"].with_name("alternate-index.json")
    alternate.write_bytes(authority["index"].read_bytes())
    args = trainer.build_parser().parse_args([
        "--data", str(authority["samples"]),
        "--maps", str(authority["maps"]),
        "--index", str(alternate),
        "--output", str(authority["output"]),
        "--init", str(authority["initialization"]),
    ])
    plan = trainer.build_training_plan(args)
    assert plan["dataset_authority_gate"]["passed"] is False
    assert plan["permanent_smoke"] is True
    assert "dataset_path_override_index" in plan["smoke_reasons"]


def test_checkpoint_structure_rejects_nonempty_sentinel_state_dict():
    with pytest.raises(trainer.TrainingConfigurationError, match="keys mismatch"):
        REAL_STATE_DICT_VALIDATOR({"sentinel": torch.zeros(1)})


@pytest.mark.parametrize("unsafe", ("/tmp/p5-outside", "../p5-parent"))
def test_training_authority_rejects_absolute_or_parent_paths(
    unsafe, tmp_path, monkeypatch
):
    document = yaml.safe_load(trainer.DEFAULT_TRAINING_CONFIG.read_text())
    document["dataset"]["root"] = unsafe
    path = tmp_path / "unsafe-training.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(trainer, "DEFAULT_TRAINING_CONFIG", path)
    with pytest.raises(trainer.TrainingConfigurationError, match="project-relative"):
        trainer._training_config_authority()


@pytest.mark.parametrize("mutation", ("missing", "unknown"))
def test_training_yaml_loss_contract_is_complete_and_closed(mutation, tmp_path, monkeypatch):
    document = yaml.safe_load(trainer.DEFAULT_TRAINING_CONFIG.read_text())
    if mutation == "missing":
        document["loss"].pop("safety_cvar_fraction")
    else:
        document["loss"]["unreviewed_loss_knob"] = 1.0
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(trainer, "DEFAULT_TRAINING_CONFIG", path)
    with pytest.raises(trainer.TrainingConfigurationError, match="missing or unknown"):
        trainer._training_config_authority()


def test_training_yaml_stage_partition_labels_are_frozen(tmp_path, monkeypatch):
    document = yaml.safe_load(trainer.DEFAULT_TRAINING_CONFIG.read_text())
    document["training"]["candidate_capacity"]["train"].append(
        "candidate_queries"
    )
    path = tmp_path / "unsafe-stage-partition.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(trainer, "DEFAULT_TRAINING_CONFIG", path)
    with pytest.raises(trainer.TrainingConfigurationError, match="parameter partitions"):
        trainer._training_config_authority()


@pytest.mark.parametrize(
    "field,value",
    (
        ("required_candidate_contexts", []),
        ("maximum_validation_zero_feasible_rate", 1.0),
        ("minimum_completed_epochs", 39),
    ),
)
def test_candidate_acceptance_policy_rejects_unsafe_ranges(
    field, value, tmp_path, monkeypatch
):
    document = yaml.safe_load(trainer.DEFAULT_TRAINING_CONFIG.read_text())
    document["qualification"]["candidate_acceptance"][field] = value
    path = tmp_path / "unsafe-acceptance.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(trainer, "DEFAULT_TRAINING_CONFIG", path)
    with pytest.raises(trainer.TrainingConfigurationError, match="candidate acceptance"):
        trainer._training_config_authority()


def test_real_curated_validation_index_passes_context_and_maneuver_coverage():
    authority = trainer._training_config_authority()
    index = json.loads(authority["authority_paths"]["index"].read_text(encoding="utf-8"))
    gate = trainer._validation_coverage_gate(index, authority)
    assert gate["passed"] is True
    assert gate["validation_frames"] == 3624
    assert gate["candidate_context"]["counts"] == {
        "MISSION": 3364,
        "RECOVERY": 260,
    }
    assert gate["candidate_context"]["unexpected"] == {}
    assert gate["maneuver"]["counts"] == {
        "DEAD_END_ESCAPE": 291,
        "NARROW_CORRIDOR": 569,
        "NORMAL": 291,
        "REVERSE_EXIT": 118,
        "SHARP_TURN": 1077,
        "THREE_POINT_TURN": 1137,
        "U_TURN": 141,
    }
    assert gate["errors"] == []
    assert gate["requested_gear"]["status"] == (
        "DEFERRED_TO_LOADER_AND_CANDIDATE_ACCEPTANCE"
    )
    assert gate["test_samples_opened"] is False


def test_real_p3_v3_report_exposes_its_current_feasibility_gates():
    report = json.loads(
        trainer.DEFAULT_P3_FOOTPRINT_REAUDIT.read_text(encoding="utf-8")
    )
    authority = report["training_authority"]
    gate = trainer._p3_footprint_gate(
        {},
        index_sha256=authority["index_sha256"],
        content_aggregate_sha256=authority["content_aggregate_sha256"],
        map_contract_aggregate_sha256=authority[
            "map_contract_aggregate_sha256"
        ],
    )
    assert gate["passed"] is (
        report.get("status") == "PASS" and not report.get("errors")
    )
    assert gate["overall_zero_feasible_rate"] == pytest.approx(
        report["statistics"]["overall"]["new"]["zero_feasible_rate"]
    )
    assert gate["overall_median_feasible_candidates"] == pytest.approx(
        report["statistics"]["overall"]["new"]["feasible_candidates_median"]
    )
    for name in report.get("errors", ()):
        assert name in gate["errors"]
    assert "reaudit_rollout_contract_mismatch" not in gate["errors"]


def test_candidate_dry_run_writes_nothing(authority, capsys):
    result = trainer.main([
        "--data", str(authority["samples"]),
        "--maps", str(authority["maps"]),
        "--index", str(authority["index"]),
        "--output", str(authority["output"]),
        "--init", str(authority["initialization"]),
        "--max-samples", "1",
        "--max-steps", "1",
        "--dry-run",
    ])
    assert result == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "BLOCKED"
    assert document["sealed_test_split"] is True
    assert document["formal_training_authorized"] is False
    assert document["workers"] == 8
    assert not authority["output"].exists()


def test_score_stage_requires_explicit_trained_candidate(authority):
    args = arguments(authority, "--stage", "score_calibration")
    with pytest.raises(trainer.TrainingConfigurationError, match="explicit --init"):
        trainer.build_training_plan(args)


def test_failed_corrected_footprint_gate_blocks_formal_but_allows_bounded_smoke(
    authority,
):
    formal_args = arguments(
        authority, "--init", str(authority["initialization"])
    )
    formal_plan = trainer.build_training_plan(formal_args)
    formal_plan["p3_footprint_gate"] = {
        "passed": False,
        "errors": ["overall_zero_feasible_rate_lt_0_10"],
    }
    with pytest.raises(trainer.TrainingConfigurationError, match="formal P5 training"):
        trainer._require_training_gate(formal_args, formal_plan)

    smoke_args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--max-steps", "1",
        "--max-samples", "1",
    )
    smoke_plan = trainer.build_training_plan(smoke_args)
    smoke_plan["p3_footprint_gate"] = {
        "passed": False,
        "errors": ["overall_zero_feasible_rate_lt_0_10"],
    }
    trainer._require_training_gate(smoke_args, smoke_plan)


def test_trainer_streams_hard_veto_oracle_and_maneuver_metrics():
    accumulator = trainer._MetricAccumulator()
    accumulator.note_geometry(2, 2)
    scores = torch.tensor([[0.0, 10.0, 20.0] + [30.0] * 12, [0.0] * 15])
    clearance = torch.tensor([[-1.0, 0.2, 0.1] + [-1.0] * 12, [-1.0] * 15])
    cost = torch.tensor([[0.0, 2.0, 1.0] + [5.0] * 12, list(map(float, range(15)))])
    losses = {
        name: torch.tensor(1.0) for name in accumulator.scalar_loss_names
    }
    losses.update({
        "minimum_clearance": clearance,
        "candidate_cost": cost,
        "kinematic_per_candidate": torch.zeros_like(clearance),
    })
    accumulator.update(
        SimpleNamespace(scores=scores),
        losses,
        2,
        {
            "maneuver": ("NORMAL", "REVERSE_EXIT"),
            "candidate_context": ("MISSION", "UNKNOWN"),
            "requested_gear": ("FORWARD", "REVERSE"),
        },
    )
    summary = accumulator.result()["candidate_metrics"]
    assert summary["overall"]["zero_feasible_rate"] == 0.5
    assert summary["overall"]["mean_oracle_regret"] == pytest.approx(1.0)
    assert summary["overall"]["kinematic_violation_rate"] == 0.0
    assert set(summary["by_maneuver"]) == {"NORMAL", "REVERSE_EXIT"}
    assert set(summary["by_candidate_context"]) == {"MISSION", "UNKNOWN"}
    assert set(summary["by_requested_gear"]) == {"FORWARD", "REVERSE"}


@pytest.mark.parametrize(
    "mode,frozen_module,frozen_token,active_module",
    (
        ("depth_only", "lidar_encoder", "lidar_missing_token", "depth_encoder"),
        ("lidar_only", "depth_encoder", "depth_missing_token", "lidar_encoder"),
    ),
)
def test_modality_optimizer_excludes_inactive_sensor_and_freezes_its_bn(
    mode, frozen_module, frozen_token, active_module
):
    model = DEPCarNetV1()
    optimizer, ownership = trainer._build_effective_optimizer(
        model, "candidate_capacity", mode, 1.0e-4, 1.0e-5
    )
    trainer._set_training_mode(model, "candidate_capacity", mode, True)
    frozen = getattr(model, frozen_module)
    active = getattr(model, active_module)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert ownership["frozen_sensor_partition"] in {"depth", "lidar"}
    assert not frozen.training
    assert active.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(id(parameter) not in optimizer_ids for parameter in frozen.parameters())
    token = getattr(model, frozen_token)
    assert not token.requires_grad and id(token) not in optimizer_ids


@pytest.mark.parametrize(
    "mode,frozen_module,frozen_token",
    (
        ("depth_only", "lidar_encoder", "lidar_missing_token"),
        ("lidar_only", "depth_encoder", "depth_missing_token"),
    ),
)
def test_inactive_sensor_parameters_and_buffers_remain_bitwise_unchanged_after_step(
    mode, frozen_module, frozen_token
):
    torch.manual_seed(9)
    model = DEPCarNetV1()
    optimizer, _ = trainer._build_effective_optimizer(
        model, "candidate_capacity", mode, 1.0e-4, 1.0e-5
    )
    trainer._set_training_mode(model, "candidate_capacity", mode, True)
    frozen = getattr(model, frozen_module)
    before = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
    token = getattr(model, frozen_token)
    token_before = token.detach().clone()
    depth = torch.rand(1, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.rand(1, 6, 160, 160)
    state = torch.zeros(1, 9)
    state[:, 7] = 1.0
    gear = torch.ones(1, dtype=torch.int64)
    mask = trainer.modality_mask(1, mode)
    output = model(depth, lidar, state, gear, modality_mask=mask)
    loss = output.raw_residuals.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = frozen.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert torch.equal(token_before, token.detach())


def test_amp_encoder_guard_supports_mixed_depth_lidar_and_fusion_rows():
    torch.manual_seed(19)
    model = DEPCarNetV1().train()
    depth = torch.rand(3, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.rand(3, 6, 160, 160)
    state = torch.zeros(3, 9)
    state[:, 7] = 1.0
    gear = torch.tensor([1, -1, 1], dtype=torch.int64)
    mask = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    depth_hooks_before = len(model.depth_encoder._forward_hooks)
    lidar_hooks_before = len(model.lidar_encoder._forward_hooks)

    with trainer._amp_encoder_output_fp32(model, True):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(depth, lidar, state, gear, modality_mask=mask)

    assert all(torch.isfinite(value).all() for value in output)
    assert model.depth_missing_token.dtype == torch.float32
    assert model.lidar_missing_token.dtype == torch.float32
    assert len(model.depth_encoder._forward_hooks) == depth_hooks_before
    assert len(model.lidar_encoder._forward_hooks) == lidar_hooks_before


def test_score_stage_rejects_direct_transfer_initialization(authority):
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(authority["initialization"]),
    )
    with pytest.raises(trainer.TrainingConfigurationError, match="transfer initialization"):
        trainer.build_training_plan(args)


def test_score_stage_accepts_trained_candidate_checkpoint(authority):
    candidate = candidate_checkpoint(
        authority["output"].parent / "candidate.pth", authority
    )
    accept_candidate(candidate)
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(candidate),
    )
    plan = trainer.build_training_plan(args)
    assert plan["source"]["kind"] == "accepted_candidate_capacity_checkpoint"


def test_score_recomputes_metrics_and_rejects_handwritten_pass(authority):
    candidate = candidate_checkpoint(
        authority["output"].parent / "forged_pass.pth", authority
    )
    accept_candidate(candidate)
    metrics_path = candidate.with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["validation"]["candidate_metrics"]["overall"][
        "zero_feasible_rate"
    ] = 0.99
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    contract_path = candidate.with_suffix(".contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["artifacts"]["metrics_sha256"] = sha256(metrics_path)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    recorded_path = candidate.with_suffix(".candidate_acceptance.json")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    recorded.update({
        "status": "PASS",
        "gate_passed": True,
        "contract_sha256": sha256(contract_path),
        "metrics_sha256": sha256(metrics_path),
    })
    recorded_path.write_text(json.dumps(recorded), encoding="utf-8")
    args = arguments(
        authority, "--stage", "score_calibration", "--init", str(candidate)
    )
    with pytest.raises(trainer.TrainingConfigurationError, match="live recomputation"):
        trainer.build_training_plan(args)


def test_score_rejects_one_step_candidate_without_smoke_lineage(authority):
    candidate = candidate_checkpoint(
        authority["output"].parent / "one_step.pth", authority, steps=1
    )
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(candidate),
    )
    with pytest.raises(trainer.TrainingConfigurationError, match="one-step"):
        trainer.build_training_plan(args)


def test_allow_smoke_source_requires_and_preserves_bounded_smoke_lineage(authority):
    candidate = candidate_checkpoint(
        authority["output"].parent / "candidate_smoke.pth",
        authority,
        smoke=True,
        partial=True,
    )
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(candidate),
        "--allow-smoke-source",
        "--max-steps", "1",
        "--max-samples", "1",
    )
    plan = trainer.build_training_plan(args)
    assert plan["source"]["kind"] == "candidate_smoke_checkpoint"
    assert trainer._explicit_smoke_run(args) is True


@pytest.mark.parametrize(
    "field",
    (
        "acceptance_tool_sha256",
        "trainer_sha256",
        "training_config_sha256",
        "loss_config_sha256",
        "metrics_sha256",
    ),
)
def test_score_rejects_tampered_candidate_acceptance_hash(authority, field):
    candidate = candidate_checkpoint(
        authority["output"].parent / ("tampered_" + field + ".pth"),
        authority,
    )
    accept_candidate(candidate)
    acceptance_path = candidate.with_suffix(".candidate_acceptance.json")
    document = json.loads(acceptance_path.read_text())
    document[field] = "0" * 64
    acceptance_path.write_text(json.dumps(document), encoding="utf-8")
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(candidate),
    )
    with pytest.raises(
        trainer.TrainingConfigurationError,
        match="cannot read JSON|acceptance gate",
    ):
        trainer.build_training_plan(args)


def test_score_rejects_metrics_changed_after_candidate_acceptance(authority):
    candidate = candidate_checkpoint(
        authority["output"].parent / "tampered_metrics.pth", authority
    )
    accept_candidate(candidate)
    candidate.with_suffix(".metrics.json").write_text("tampered", encoding="utf-8")
    args = arguments(
        authority,
        "--stage", "score_calibration",
        "--init", str(candidate),
    )
    with pytest.raises(
        trainer.TrainingConfigurationError,
        match="cannot read JSON|acceptance gate",
    ):
        trainer.build_training_plan(args)


@pytest.mark.parametrize(
    "stage,modality,message",
    (
        ("score_calibration", "fusion", "stage"),
        ("candidate_capacity", "depth_only", "modality"),
    ),
)
def test_resume_requires_same_stage_and_modality(
    authority, stage, modality, message
):
    checkpoint = candidate_checkpoint(
        authority["output"].parent / "resume.pth", authority
    )
    args = arguments(
        authority,
        "--stage", stage,
        "--modality", modality,
        "--resume", str(checkpoint),
    )
    with pytest.raises(trainer.TrainingConfigurationError, match=message):
        trainer.build_training_plan(args)


@pytest.mark.parametrize(
    "changed", (("--max-samples", "10"), ("--max-steps", "2"), ("--epochs", "41"))
)
def test_resume_binds_caps_and_epoch_budget(authority, changed):
    checkpoint = candidate_checkpoint(
        authority["output"].parent / (changed[0][2:] + ".pth"), authority
    )
    args = arguments(authority, "--resume", str(checkpoint), *changed)
    with pytest.raises(trainer.TrainingConfigurationError, match="effective training"):
        trainer.build_training_plan(args)


def test_parser_rejects_init_and_resume_together(authority):
    with pytest.raises(SystemExit):
        arguments(
            authority,
            "--init", str(authority["initialization"]),
            "--resume", str(authority["initialization"]),
        )


def test_training_sidecars_are_explicitly_unqualified(authority):
    args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--workers", "0",
        "--max-samples", "1",
        "--max-steps", "1",
    )
    plan = trainer.build_training_plan(args)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = trainer._make_grad_scaler(False)
    paths = trainer._write_artifacts(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        plan=plan,
        history=[{"epoch": 1}],
        metrics={"validation": {"total": 1.0}},
        completed_epochs=1,
        global_step=1,
        partial_epoch=False,
    )
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["qualification_status"] == "UNQUALIFIED"
    assert checkpoint["production_qualified"] is False
    assert checkpoint["training_config_sha256"] == plan["training_config"]["file_sha256"]
    assert checkpoint["loss_config_sha256"] == plan["training_config"]["loss_config_sha256"]
    assert checkpoint["effective_optimizer_hyperparameters"]
    assert set(checkpoint["rng_state"]) == {
        "python", "numpy", "torch_cpu", "torch_cuda"
    }
    assert checkpoint["sampler_state"]["strategy"] == "epoch_seed_v1"
    for name in ("contract", "history", "metrics"):
        document = json.loads(Path(paths[name]).read_text(encoding="utf-8"))
        assert document["qualification_status"] == "UNQUALIFIED"
        assert document["production_qualified"] is False
    contract = json.loads(Path(paths["contract"]).read_text(encoding="utf-8"))
    assert contract["dataset_provenance"]["splits_used"] == ["train", "validation"]
    assert contract["dataset_provenance"]["test_split_used"] is False
    assert contract["objective_execution"]["geometry_authority"] == [
        "physical_state",
        "physical_route_pose",
        "map_distance_field",
        "chassis_to_map",
    ]
    candidate_acceptance = json.loads(
        Path(paths["candidate_acceptance"]).read_text(encoding="utf-8")
    )
    assert candidate_acceptance["status"] == "SMOKE_SOURCE_ONLY"
    assert candidate_acceptance["gate_passed"] is False


def test_rng_state_round_trip_is_exact():
    trainer._seed_everything(77)
    state = trainer._capture_rng_state()
    expected = (
        torch.rand(4),
        np.random.random(4),
        random.random(),
    )
    trainer._seed_everything(99)
    trainer._restore_rng_state(state)
    actual = (
        torch.rand(4),
        np.random.random(4),
        random.random(),
    )
    assert torch.equal(actual[0], expected[0])
    assert (actual[1] == expected[1]).all()
    assert actual[2] == expected[2]


def test_two_epoch_smoke_checkpoint_updates_do_not_reject_owned_artifacts(authority):
    args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--workers", "0",
        "--max-samples", "1",
        "--max-steps", "2",
    )
    plan = trainer.build_training_plan(args)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = trainer._make_grad_scaler(False)
    paths = None
    for epoch in (1, 2):
        paths = trainer._write_artifacts(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            plan=plan,
            history=[{"epoch": value} for value in range(1, epoch + 1)],
            metrics={"validation": {"total": 1.0}},
            completed_epochs=epoch,
            global_step=epoch,
            partial_epoch=False,
        )
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    assert checkpoint["completed_epochs"] == 2


def test_config_change_after_plan_is_rejected_before_artifact_signing(authority):
    args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--max-samples", "1",
        "--max-steps", "1",
    )
    plan = trainer.build_training_plan(args)
    plan["training_config"]["path"].write_text(
        plan["training_config"]["path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    model = torch.nn.Linear(1, 1)
    with pytest.raises(trainer.TrainingConfigurationError, match="authority changed"):
        trainer._write_artifacts(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=trainer._make_grad_scaler(False),
            args=args,
            plan=plan,
            history=[],
            metrics={},
            completed_epochs=1,
            global_step=1,
            partial_epoch=False,
        )


@pytest.mark.parametrize("mutation", ("trainer", "implementation"))
def test_code_change_after_plan_is_rejected_before_artifact_signing(
    authority, monkeypatch, mutation
):
    args = arguments(
        authority,
        "--init", str(authority["initialization"]),
        "--max-samples", "1",
        "--max-steps", "1",
    )
    plan = trainer.build_training_plan(args)
    if mutation == "trainer":
        original_sha256 = trainer.sha256_file

        def changed_trainer_hash(path):
            if Path(path).resolve() == trainer.TRAINER_PATH.resolve():
                return "0" * 64
            return original_sha256(path)

        monkeypatch.setattr(trainer, "sha256_file", changed_trainer_hash)
    else:
        changed = dict(plan["implementation_contract"])
        changed["aggregate_sha256"] = "0" * 64
        monkeypatch.setattr(
            trainer, "build_p4_implementation_contract", lambda _root: changed
        )
    model = torch.nn.Linear(1, 1)
    with pytest.raises(trainer.TrainingConfigurationError, match="implementation changed"):
        trainer._write_artifacts(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=trainer._make_grad_scaler(False),
            args=args,
            plan=plan,
            history=[],
            metrics={},
            completed_epochs=1,
            global_step=1,
            partial_epoch=False,
        )
