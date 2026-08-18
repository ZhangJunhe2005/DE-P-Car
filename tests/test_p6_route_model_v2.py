import torch
import hashlib
import json
from pathlib import Path

from dep_car.model.dep_car_net import DEPCarNetConfig, DEPCarNetV1
from dep_car.model.dep_car_net_v2 import DEPCarNetV2, mirror_route_corridor
from dep_car.training.stages import configure_training_stage
from dep_car.runtime.p6_policy import P6PolicyRuntime


def batch(size=2):
    depth = torch.rand(size, 2, 96, 160)
    lidar = torch.rand(size, 6, 160, 160)
    state = torch.zeros(size, 9)
    gear = torch.ones(size, dtype=torch.long)
    route = torch.zeros(size, 12, 3)
    route[:, :, 0] = torch.linspace(0.1, 2.5, 12)
    mask = torch.ones(size, 12, dtype=torch.bool)
    modality = torch.ones(size, 2)
    return depth, lidar, state, gear, route, mask, modality


def test_v2_route_conditioned_model_shapes_and_stage_ownership():
    model = DEPCarNetV2(
        DEPCarNetConfig(enforce_reflection_equivariance=False)
    ).eval()
    output = model(*batch())
    assert output.trajectories.shape == (2, 15, 11, 6)
    assert output.scores.shape == (2, 15)
    candidate = configure_training_stage(model, "candidate_capacity")
    assert candidate["candidate_trainable"] > 0
    assert candidate["score_trainable"] == 0
    score = configure_training_stage(model, "score_calibration")
    assert score["candidate_trainable"] == 0
    assert score["score_trainable"] > 0


def test_v2_mirror_route_changes_lateral_and_yaw_only():
    route = torch.tensor([[[1.0, 0.5, 0.4], [2.0, -0.2, -0.1]]])
    mirrored = mirror_route_corridor(route)
    torch.testing.assert_close(mirrored[..., 0], route[..., 0])
    torch.testing.assert_close(mirrored[..., 1:], -route[..., 1:])


def test_v2_v1_initialization_preserves_candidate_outputs_before_route_training():
    torch.manual_seed(7)
    config = DEPCarNetConfig(enforce_reflection_equivariance=False)
    v1 = DEPCarNetV1(config).eval()
    v2 = DEPCarNetV2(config).eval()
    v2.initialize_from_v1(v1.state_dict())
    values = batch(size=1)
    with torch.no_grad():
        old = v1(values[0], values[1], values[2], values[3], values[6])
        new = v2(*values)
    torch.testing.assert_close(new.raw_residuals, old.raw_residuals)
    torch.testing.assert_close(new.scores, old.scores)


def test_p6_runtime_loads_v2_and_requires_fixed_route_contract(tmp_path):
    source = tmp_path / "candidate.pth"
    source.write_bytes(b"accepted-v2-candidate-lineage")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_contract = source.with_suffix(".contract.json")
    source_contract.write_text('{"status":"candidate"}')
    source_contract_sha = hashlib.sha256(source_contract.read_bytes()).hexdigest()
    source_acceptance = source.with_suffix(".candidate_acceptance.json")
    source_acceptance.write_text('{"status":"PASS"}')
    source_acceptance_sha = hashlib.sha256(source_acceptance.read_bytes()).hexdigest()
    source_report = tmp_path / "candidate_acceptance_report.json"
    source_report.write_bytes(source_acceptance.read_bytes())
    source_report_sha = hashlib.sha256(source_report.read_bytes()).hexdigest()
    source_gate = {
        "schema": "DEPCarRouteV2ScoreSourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(source),
        "checkpoint_sha256": source_sha,
        "checkpoint_contract": str(source_contract),
        "checkpoint_contract_sha256": source_contract_sha,
        "acceptance_sidecar": str(source_acceptance),
        "acceptance_sidecar_sha256": source_acceptance_sha,
        "formal_acceptance_report": str(source_report),
        "formal_acceptance_report_sha256": source_report_sha,
        "test_split_accessed": False,
    }
    model = DEPCarNetV2(
        DEPCarNetConfig(enforce_reflection_equivariance=False)
    )
    checkpoint = tmp_path / "score.best.pth"
    torch.save(
        {
            "architecture_id": model.architecture_id,
            "training_stage": "score_calibration",
            "modality": "fusion",
            "artifact_role": "best",
            "completed_epochs": 40,
            "partial_epoch": False,
            "run_completed": True,
            "source_checkpoint_sha256": source_sha,
            "source_acceptance_gate": source_gate,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = tmp_path / "score.best.contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "DEPCarRouteV2ArtifactContractV1",
                "architecture_id": model.architecture_id,
                "checkpoint_sha256": checkpoint_sha,
                "training_stage": "score_calibration",
                "modality": "fusion",
                "artifact_role": "best",
                "status": "TRAINED_UNQUALIFIED",
                "qualification_status": "UNQUALIFIED",
                "production_qualified": False,
                "run_completed": True,
                "formal_training_authority_gate": {"passed": True},
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": source_sha,
                "source_acceptance_gate": source_gate,
            }
        )
    )
    acceptance_tool = (
        Path(__file__).resolve().parents[1]
        / "tools/audit_p5_route_v2_score.py"
    )
    acceptance = checkpoint.with_suffix(".score_shadow_acceptance.json")
    acceptance.write_text(json.dumps({
        "schema": "DEPCarRouteV2ScoreShadowAcceptanceV1",
        "status": "PASS",
        "gate_passed": True,
        "scope": "P6_SHADOW_ONLY",
        "active_control_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_contract_sha256": hashlib.sha256(
            contract.read_bytes()
        ).hexdigest(),
        "acceptance_tool": str(acceptance_tool),
        "acceptance_tool_sha256": hashlib.sha256(
            acceptance_tool.read_bytes()
        ).hexdigest(),
        "candidate_source_gate": source_gate,
    }))
    runtime = P6PolicyRuntime(
        checkpoint, contract, modality="fusion", device="cpu", mode="shadow"
    )
    depth, lidar, state, gear, route, mask, _ = batch(size=1)
    trajectories, controls, scores = runtime.infer(
        depth[0].numpy(), lidar[0].numpy(), state[0].numpy(), int(gear[0]),
        route_pose=torch.nn.functional.pad(route[0], (0, 0, 0, 68)).numpy(),
        route_mask=torch.nn.functional.pad(mask[0], (0, 68)).numpy(),
    )
    assert trajectories.shape == (15, 11, 6)
    assert controls.shape == (15, 4)
    assert scores.shape == (15,)
