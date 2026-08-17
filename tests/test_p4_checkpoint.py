import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import migrate_v491_checkpoint as migration
from dep_car.model.checkpoint import (
    LEGACY_ARCHITECTURE_ID,
    P4_ARCHITECTURE_ID,
    load_contract,
    verify_checkpoint,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def migrated_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("p4_checkpoint")
    checkpoint = directory / "initialization_a.pth"
    checkpoint, contract_path, contract = migration.migrate_checkpoint(
        migration.DEFAULT_SOURCE, checkpoint
    )
    return checkpoint, contract_path, contract, directory


def test_migration_is_an_exact_246_tensor_depth_allowlist(migrated_checkpoint):
    checkpoint, _, contract, _ = migrated_checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload = torch.load(
        migration.DEFAULT_SOURCE, map_location="cpu", weights_only=False
    )
    source_state = source_payload["model"]
    target_state = payload["model_state_dict"]
    transfers = payload["transfers"]

    assert len(transfers) == migration.EXPECTED_TRANSFER_TENSORS == 246
    assert contract["transfer"]["tensor_count"] == 246
    assert contract["transfer"]["partial_transfer_allowed"] is False
    assert contract["transfer"]["head_transfer_allowed"] is False
    assert contract["input_contract"]["validity_fusion"].startswith("independent_learned")
    assert contract["footprint_contract"]["runtime_grid_allowance"] == "half_cell_diagonal"
    assert contract["footprint_contract"]["differentiable_training_grid_allowance"] == "one_cell_diagonal"
    assert all(row["mode"] == "exact" for row in transfers)
    assert all(row["source"].startswith(migration.SOURCE_PREFIX) for row in transfers)
    assert all(row["target"].startswith(migration.TARGET_PREFIX) for row in transfers)
    assert not any("head" in row["source"] or "head" in row["target"] for row in transfers)

    expected_targets = {
        migration.TARGET_PREFIX + name[len(migration.SOURCE_PREFIX):]
        for name in source_state if name.startswith(migration.SOURCE_PREFIX)
    }
    assert {row["target"] for row in transfers} == expected_targets
    for row in transfers:
        source = source_state[row["source"]]
        target = target_state[row["target"]]
        assert source.shape == target.shape
        assert source.dtype == target.dtype
        assert torch.equal(source, target)


def test_fixed_seed_produces_byte_identical_initialization(migrated_checkpoint):
    checkpoint_a, contract_a, _, directory = migrated_checkpoint
    checkpoint_b = directory / "initialization_b.pth"
    checkpoint_b, contract_b, _ = migration.migrate_checkpoint(
        migration.DEFAULT_SOURCE, checkpoint_b
    )

    assert file_sha256(checkpoint_a) == file_sha256(checkpoint_b)
    assert checkpoint_a.read_bytes() == checkpoint_b.read_bytes()
    assert contract_a.read_text(encoding="utf-8") == contract_b.read_text(encoding="utf-8")


def test_contract_verifies_identity_and_optional_data_hashes(migrated_checkpoint):
    checkpoint, contract_path, contract, _ = migrated_checkpoint
    dataset = contract["dataset_provenance"]
    verified = verify_checkpoint(
        checkpoint,
        contract_path,
        allow_untrained=True,
        expected_preprocessing_sha256=dataset["p3_preprocessing_sha256"],
        expected_dataset_sha256=dataset["p3_task_manifest_sha256"],
    )
    assert verified["architecture_id"] == P4_ARCHITECTURE_ID
    assert verified["production_qualified"] is False

    with pytest.raises(ValueError, match="not production-qualified"):
        verify_checkpoint(checkpoint, contract_path)
    with pytest.raises(ValueError, match="preprocessing SHA-256 mismatch"):
        verify_checkpoint(
            checkpoint,
            contract_path,
            allow_untrained=True,
            expected_preprocessing_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        verify_checkpoint(
            checkpoint,
            contract_path,
            allow_untrained=True,
            expected_dataset_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("state_dim", "state dimension"),
        ("gear_prediction", "must not predict gear"),
        ("partial_transfer", "partial/head transfer"),
        ("backbone_aggregate", "aggregate SHA-256 mismatch"),
        ("implementation_aggregate", "implementation aggregate SHA-256 mismatch"),
    ),
)
def test_strict_contract_rejects_semantic_mutation(
    migrated_checkpoint, tmp_path, mutation, message
):
    _, _, original, _ = migrated_checkpoint
    contract = copy.deepcopy(original)
    if mutation == "state_dim":
        contract["state_contract"]["dimension"] = 8
    elif mutation == "gear_prediction":
        contract["gear_contract"]["network_predicts_gear"] = True
    elif mutation == "partial_transfer":
        contract["transfer"]["partial_transfer_allowed"] = True
    elif mutation == "backbone_aggregate":
        contract["source"]["backbone_source"]["aggregate_sha256"] = "0" * 64
    elif mutation == "implementation_aggregate":
        contract["implementation_contract"]["aggregate_sha256"] = "0" * 64
    path = tmp_path / (mutation + ".json")
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_contract(path)


def test_contract_boolean_cannot_promote_unqualified_checkpoint(
    migrated_checkpoint, tmp_path
):
    checkpoint, _, original, _ = migrated_checkpoint
    forged = copy.deepcopy(original)
    forged["production_qualified"] = True
    path = tmp_path / "forged-production.contract.json"
    path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ValueError, match="production qualification mismatch"):
        verify_checkpoint(checkpoint, path)


def test_exact_transfer_rejects_shape_mismatch(migrated_checkpoint):
    checkpoint, _, _, _ = migrated_checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload = torch.load(
        migration.DEFAULT_SOURCE, map_location="cpu", weights_only=False
    )
    target = dict(payload["model_state_dict"])
    first_target = next(
        name for name in target if name.startswith(migration.TARGET_PREFIX)
    )
    target[first_target] = torch.empty((1,), dtype=target[first_target].dtype)
    with pytest.raises(ValueError, match="exact transfer shape mismatch"):
        migration.exact_backbone_transfer(source_payload["model"], target)


def test_legacy_checkpoint_requires_explicit_architecture(tmp_path):
    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"legacy-checkpoint-test")
    contract_path = tmp_path / "legacy.contract.json"
    contract_path.write_text(json.dumps({
        "architecture_id": LEGACY_ARCHITECTURE_ID,
        "state_dim": 8,
        "lidar_shape": [2, 16, 440],
        "checkpoint_sha256": file_sha256(checkpoint),
        "production_qualified": False,
    }), encoding="utf-8")

    verified = verify_checkpoint(
        checkpoint,
        contract_path,
        architecture_id=LEGACY_ARCHITECTURE_ID,
        allow_untrained=True,
    )
    assert verified["architecture_id"] == LEGACY_ARCHITECTURE_ID
    with pytest.raises(ValueError, match="architecture mismatch"):
        verify_checkpoint(checkpoint, contract_path, allow_untrained=True)
