"""Strict DE-P-Car checkpoint identity and compatibility checks."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from dep_car.core.occupancy import (
    FIVE_CIRCLE_FOOTPRINT_SCHEMA,
    FOOTPRINT_ALLOWANCE_SCHEMA,
    RUNTIME_HALF_DIAGONAL_MULTIPLIER,
    SWEPT_INTERPOLATION_SCHEMA,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    TRAINING_ONE_DIAGONAL_MULTIPLIER,
)

from .ackermann_rollout import (
    ACKERMANN_ROLLOUT_SCHEMA,
    LONGITUDINAL_LIMITS_FRAME,
)
from .implementation_contract import (
    P4_IMPLEMENTATION_FILES,
    P4_IMPLEMENTATION_SCHEMA,
    verify_p4_implementation_contract,
)


P4_ARCHITECTURE_ID = "dep_car_multimodal_v1_ackermann_3x5"
LEGACY_ARCHITECTURE_ID = "dep_car_lidar_v1_3x5_mobilenetv3_v483"
P4_CONTRACT_SCHEMA = "DEPCarCheckpointContractV2"

P4_STATE_FIELDS = (
    "signed_speed",
    "longitudinal_acceleration",
    "steering",
    "yaw_rate",
    "subgoal_x",
    "subgoal_y",
    "sin_heading_error",
    "cos_heading_error",
    "reference_curvature",
)
P4_BEV_CHANNELS = (
    "occupancy",
    "log_density",
    "minimum_height",
    "maximum_height",
    "nearest_range",
    "observed",
)
P4_RESIDUAL_FIELDS = (
    "steering_mid", "steering_end", "speed_end", "duration"
)
P4_TRAJECTORY_FIELDS = (
    "t", "x", "y", "yaw", "signed_speed", "steering"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_mapping(value, name):
    _require(isinstance(value, dict), f"checkpoint contract {name} must be a mapping")
    return value


def _require_sha256(value, name):
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"checkpoint contract {name} must be a lowercase SHA-256",
    )


def _require_keys(mapping, required, name):
    missing = set(required).difference(mapping)
    _require(
        not missing,
        f"checkpoint contract {name} is missing: {', '.join(sorted(missing))}",
    )


def _validate_legacy_contract(contract):
    _require_keys(
        contract,
        {"architecture_id", "state_dim", "lidar_shape", "checkpoint_sha256"},
        "legacy root",
    )
    _require(contract["state_dim"] == 8, "legacy checkpoint state_dim must be 8")
    _require(
        contract["lidar_shape"] == [2, 16, 440],
        "legacy checkpoint LiDAR shape must be [2,16,440]",
    )
    _require_sha256(contract["checkpoint_sha256"], "checkpoint_sha256")
    if "production_qualified" in contract:
        _require(
            isinstance(contract["production_qualified"], bool),
            "legacy production_qualified must be boolean",
        )


def _validate_p4_contract(contract):
    required_root = {
        "schema", "contract_version", "architecture_id", "checkpoint_version",
        "checkpoint_sha256", "status", "production_qualified",
        "initialization_seed", "input_contract", "state_contract",
        "gear_contract", "output_contract", "lattice_contract",
        "rollout_contract", "footprint_contract", "source", "transfer",
        "dataset_provenance", "training_contract", "implementation_contract",
    }
    _require_keys(contract, required_root, "P4 root")
    _require(contract["schema"] == P4_CONTRACT_SCHEMA, "unknown P4 checkpoint schema")
    _require(contract["contract_version"] == 2, "P4 contract version must be 2")
    _require_sha256(contract["checkpoint_sha256"], "checkpoint_sha256")
    _require(
        isinstance(contract["production_qualified"], bool),
        "P4 production_qualified must be boolean",
    )
    _require(
        isinstance(contract["initialization_seed"], int)
        and not isinstance(contract["initialization_seed"], bool),
        "P4 initialization_seed must be an integer",
    )

    implementation = _require_mapping(
        contract["implementation_contract"], "implementation_contract"
    )
    _require(
        implementation.get("schema") == P4_IMPLEMENTATION_SCHEMA,
        "P4 implementation schema mismatch",
    )
    implementation_files = _require_mapping(
        implementation.get("files"), "implementation_contract.files"
    )
    _require(
        tuple(sorted(implementation_files)) == tuple(sorted(P4_IMPLEMENTATION_FILES)),
        "P4 implementation file allowlist mismatch",
    )
    for name, value in implementation_files.items():
        _require_sha256(value, "implementation_contract.files.%s" % name)
    _require_sha256(
        implementation.get("aggregate_sha256"),
        "implementation_contract.aggregate_sha256",
    )
    _require(
        implementation["aggregate_sha256"] == _canonical_sha256(implementation_files),
        "P4 implementation aggregate SHA-256 mismatch",
    )

    inputs = _require_mapping(contract["input_contract"], "input_contract")
    _require(
        inputs.get("depth_metric_and_validity_shape") == [2, 480, 640],
        "P4 depth sample shape must be [2,480,640]",
    )
    _require(
        inputs.get("depth_network_input_shape") == [2, 96, 160]
        and inputs.get("depth_backbone_input_shape") == [1, 96, 160],
        "P4 depth network/backbone shapes mismatch",
    )
    _require(
        inputs.get("depth_fields") == ["metric_depth_normalized_by_10m", "validity"]
        and inputs.get("depth_normalization_divisor_m") == 10.0,
        "P4 depth field order mismatch",
    )
    _require(
        inputs.get("validity_fusion")
        == "independent_learned_encoder_added_to_depth_backbone_feature",
        "P4 validity must remain distinguishable from far depth",
    )
    _require(
        inputs.get("lidar_bev_shape") == [6, 160, 160],
        "P4 LiDAR BEV shape must be [6,160,160]",
    )
    _require(
        tuple(inputs.get("lidar_bev_channels", ())) == P4_BEV_CHANNELS,
        "P4 LiDAR BEV channel order mismatch",
    )
    _require(
        inputs.get("modality_mask_shape") == [2]
        and inputs.get("modality_mask_order") == ["depth", "lidar"],
        "P4 modality mask contract mismatch",
    )

    state = _require_mapping(contract["state_contract"], "state_contract")
    _require(state.get("dimension") == 9, "P4 state dimension must be 9")
    _require(
        tuple(state.get("fields", ())) == P4_STATE_FIELDS,
        "P4 state field order mismatch",
    )
    _require(
        isinstance(state.get("normalization_scale"), list)
        and len(state["normalization_scale"]) == 9
        and all(isinstance(value, (int, float)) and value > 0 for value in state["normalization_scale"]),
        "P4 state normalization must contain nine positive scales",
    )

    gear = _require_mapping(contract["gear_contract"], "gear_contract")
    _require(gear.get("allowed_values") == [-1, 1], "P4 requested gear values mismatch")
    _require(gear.get("input") == "requested_gear", "P4 requested gear input mismatch")
    _require(
        gear.get("network_predicts_gear") is False,
        "P4 network must not predict gear",
    )
    _require(
        gear.get("one_constant_gear_per_candidate_bank") is True,
        "P4 candidate bank must keep one constant gear",
    )

    output = _require_mapping(contract["output_contract"], "output_contract")
    _require(output.get("candidate_count") == 15, "P4 output must contain 15 candidates")
    _require(
        output.get("raw_residuals_shape") == [15, 4]
        and output.get("residuals_shape") == [15, 4],
        "P4 residual output shape mismatch",
    )
    _require(
        tuple(output.get("residual_fields", ())) == P4_RESIDUAL_FIELDS,
        "P4 residual field order mismatch",
    )
    _require(
        output.get("trajectory_shape") == [15, 11, 6]
        and tuple(output.get("trajectory_fields", ())) == P4_TRAJECTORY_FIELDS,
        "P4 trajectory output contract mismatch",
    )
    _require(
        output.get("scores_shape") == [15]
        and output.get("scores_nonnegative") is True,
        "P4 score output contract mismatch",
    )

    lattice = _require_mapping(contract["lattice_contract"], "lattice_contract")
    _require(lattice.get("candidate_count") == 15, "P4 lattice must contain 15 candidates")
    _require(
        lattice.get("order") == "speed_major_3x5_steering_minor",
        "P4 lattice ordering mismatch",
    )
    _require(
        lattice.get("forward_speed_anchors_mps") == [0.6, 1.2, 2.0]
        and lattice.get("reverse_speed_anchors_mps") == [0.2, 0.35, 0.5]
        and lattice.get("steering_anchors_rad") == [-0.52, -0.26, 0.0, 0.26, 0.52],
        "P4 lattice anchors mismatch",
    )

    rollout = _require_mapping(contract["rollout_contract"], "rollout_contract")
    rollout_config = _require_mapping(rollout.get("config"), "rollout_contract.config")
    _require(
        rollout.get("schema") == ACKERMANN_ROLLOUT_SCHEMA,
        "P4 rollout schema mismatch",
    )
    _require(
        rollout.get("gear_conditioned") is True
        and rollout.get("differentiable") is True,
        "P4 rollout must be differentiable and gear-conditioned",
    )
    _require(
        rollout.get("longitudinal_limits_frame") == LONGITUDINAL_LIMITS_FRAME
        and rollout.get("directed_acceleration_definition")
        == "requested_gear * signed_dv_dt",
        "P4 rollout longitudinal-limit frame mismatch",
    )
    _require(
        rollout_config.get("steps") == 11
        and rollout_config.get("horizon_s") == 1.0
        and rollout_config.get("wheelbase_m", 0) > 0,
        "P4 rollout timing or wheelbase mismatch",
    )
    _require(
        rollout_config.get("forward_speed_anchors_mps")
        == lattice["forward_speed_anchors_mps"]
        and rollout_config.get("reverse_speed_anchors_mps")
        == lattice["reverse_speed_anchors_mps"]
        and rollout_config.get("steering_anchors_rad")
        == lattice["steering_anchors_rad"],
        "P4 rollout and lattice anchors disagree",
    )

    footprint = _require_mapping(contract["footprint_contract"], "footprint_contract")
    _require(
        footprint.get("schema") == FIVE_CIRCLE_FOOTPRINT_SCHEMA
        and footprint.get("circle_count") == 5,
        "P4 swept-footprint schema mismatch",
    )
    _require(
        all(footprint.get(name, 0) > 0 for name in (
            "length_m", "width_m", "safety_margin_m", "circle_radius_m"
        )),
        "P4 footprint dimensions must be positive",
    )
    _require(
        isinstance(footprint.get("longitudinal_offsets_m"), list)
        and len(footprint["longitudinal_offsets_m"]) == 5,
        "P4 footprint must record five longitudinal offsets",
    )
    _require(
        footprint.get("runtime_grid_allowance") == "half_cell_diagonal"
        and footprint.get("differentiable_training_grid_allowance")
        == "one_cell_diagonal",
        "P4 footprint grid allowances mismatch",
    )
    _require(
        footprint.get("allowance_schema") == FOOTPRINT_ALLOWANCE_SCHEMA
        and footprint.get("runtime_allowance_multiplier")
        == RUNTIME_HALF_DIAGONAL_MULTIPLIER
        and footprint.get("training_allowance_multiplier")
        == TRAINING_ONE_DIAGONAL_MULTIPLIER,
        "P4 footprint allowance contract mismatch",
    )
    _require(
        footprint.get("sweep_interpolation_schema")
        == SWEPT_INTERPOLATION_SCHEMA
        and footprint.get("sweep_substeps_per_source_segment")
        == SWEPT_SUBSTEPS_PER_SEGMENT
        and footprint.get("source_trajectory_rows") == 11
        and footprint.get("safety_query_rows")
        == (11 - 1) * SWEPT_SUBSTEPS_PER_SEGMENT + 1,
        "P4 continuous-sweep contract mismatch",
    )

    training = _require_mapping(contract["training_contract"], "training_contract")
    _require(
        training.get("objective_id")
        == "dep_car_objective_v4_fp32_physics_all_candidate_kinematic_margin"
        and training.get("objective_revision") == 4
        and training.get("sdf_schema")
        == "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe"
        and training.get("stage_order") == ["candidate_capacity", "score_calibration"],
        "P4 training objective/stage order mismatch",
    )
    loss_config = _require_mapping(training.get("loss_config"), "training_contract.loss_config")
    _require_sha256(training.get("loss_config_sha256"), "training_contract.loss_config_sha256")
    _require(
        training["loss_config_sha256"] == _canonical_sha256(loss_config),
        "P4 loss config SHA-256 mismatch",
    )

    source = _require_mapping(contract["source"], "source")
    _require_sha256(source.get("checkpoint_sha256"), "source.checkpoint_sha256")
    backbone = _require_mapping(source.get("backbone_source"), "source.backbone_source")
    _require_sha256(
        backbone.get("aggregate_sha256"), "source.backbone_source.aggregate_sha256"
    )
    backbone_files = _require_mapping(
        backbone.get("files"), "source.backbone_source.files"
    )
    _require(
        set(backbone_files) == {
            "policy/models/backbone.py",
            "policy/models/MobileNetV3.py",
            "policy/backbone_variant.py",
        },
        "P4 backbone source file allowlist mismatch",
    )
    for name, value in backbone_files.items():
        _require_sha256(value, f"source.backbone_source.files.{name}")
    _require(
        backbone["aggregate_sha256"] == _canonical_sha256(backbone_files),
        "P4 backbone source aggregate SHA-256 mismatch",
    )
    _require(
        source.get("network_weights_modified_by_v4_9_1") is False,
        "P4 source must preserve the frozen V4.8.3 weights",
    )

    transfer = _require_mapping(contract["transfer"], "transfer")
    _require(
        transfer.get("mode") == "exact_depth_backbone_only"
        and transfer.get("source_prefix") == "network.image_backbone."
        and transfer.get("target_prefix") == "depth_encoder.image_backbone."
        and transfer.get("tensor_count") == 246,
        "P4 transfer is not the exact 246-tensor depth allowlist",
    )
    _require(
        transfer.get("partial_transfer_allowed") is False
        and transfer.get("head_transfer_allowed") is False,
        "P4 partial/head transfer must remain forbidden",
    )
    _require_sha256(transfer.get("manifest_sha256"), "transfer.manifest_sha256")

    dataset = _require_mapping(contract["dataset_provenance"], "dataset_provenance")
    _require_sha256(
        dataset.get("p3_preprocessing_sha256"),
        "dataset_provenance.p3_preprocessing_sha256",
    )
    _require_sha256(
        dataset.get("p3_task_manifest_sha256"),
        "dataset_provenance.p3_task_manifest_sha256",
    )
    _require_sha256(
        dataset.get("p3_acceptance_sha256"),
        "dataset_provenance.p3_acceptance_sha256",
    )


def _validate_contract_document(contract):
    _require(isinstance(contract, dict), "checkpoint contract root must be a mapping")
    architecture_id = contract.get("architecture_id")
    if architecture_id == P4_ARCHITECTURE_ID:
        _validate_p4_contract(contract)
    elif architecture_id == LEGACY_ARCHITECTURE_ID:
        _validate_legacy_contract(contract)
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture_id!r}")
    return contract


def load_contract(path):
    with open(path, "r", encoding="utf-8") as stream:
        contract = json.load(stream)
    return _validate_contract_document(contract)


def _load_p4_checkpoint_payload(checkpoint_bytes):
    """Load trusted local metadata needed to cross-check the JSON sidecar.

    The sidecar alone is not a qualification signature.  Requiring the binary
    payload to carry the same architecture/status/qualification state prevents
    an initialization or P5 checkpoint from being enabled by flipping one JSON
    boolean.  Callers must still treat arbitrary pickle checkpoints as
    untrusted input and only use artifacts from the recorded local lineage.
    """

    try:
        import torch

        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError("unable to read P4 checkpoint metadata: %s" % exc) from exc
    _require(isinstance(payload, dict), "P4 checkpoint payload must be a mapping")
    _require(
        payload.get("architecture_id") == P4_ARCHITECTURE_ID,
        "P4 checkpoint payload architecture mismatch",
    )
    _require(
        isinstance(payload.get("model_state_dict"), dict)
        and bool(payload["model_state_dict"]),
        "P4 checkpoint payload has no model_state_dict",
    )
    _require(
        isinstance(payload.get("production_qualified"), bool),
        "P4 checkpoint payload production_qualified must be boolean",
    )
    _require(
        isinstance(payload.get("status"), str) and bool(payload["status"]),
        "P4 checkpoint payload status is missing",
    )
    _require_sha256(
        payload.get("implementation_aggregate_sha256"),
        "checkpoint payload implementation_aggregate_sha256",
    )
    return payload


def _verify_production_attestation(contract, contract_path):
    """Require a P8 report bound to this exact checkpoint and implementation.

    This is an anti-misconfiguration boundary, not a claim that files writable
    by the same OS account form a cryptographic trust root.  A future release
    may additionally sign this report; until P8 creates the complete report,
    every P4/P5 artifact remains rejected by the default loader.
    """

    attestation = _require_mapping(
        contract.get("qualification_attestation"), "qualification_attestation"
    )
    _require(
        attestation.get("schema") == "DEPCarP8QualificationAttestationV1"
        and attestation.get("status") == "PASS",
        "P8 qualification attestation is missing or has not passed",
    )
    _require(
        attestation.get("checkpoint_sha256") == contract["checkpoint_sha256"],
        "P8 qualification checkpoint identity mismatch",
    )
    implementation_sha = contract["implementation_contract"]["aggregate_sha256"]
    _require(
        attestation.get("implementation_aggregate_sha256") == implementation_sha,
        "P8 qualification implementation identity mismatch",
    )
    _require_sha256(attestation.get("report_sha256"), "qualification report_sha256")
    relative = Path(str(attestation.get("report_path", "")))
    _require(
        bool(relative.parts) and not relative.is_absolute() and ".." not in relative.parts,
        "P8 qualification report path must be project-relative",
    )
    root = Path(__file__).resolve().parents[4]
    report_path = (root / relative).resolve()
    _require(root in report_path.parents, "P8 qualification report escapes project root")
    _require(report_path.is_file(), "P8 qualification report is missing")
    _require(
        sha256_file(report_path) == attestation["report_sha256"],
        "P8 qualification report SHA-256 mismatch",
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read P8 qualification report: %s" % exc) from exc
    _require(
        isinstance(report, dict)
        and report.get("schema") == "DEPCarP8QualificationReportV1"
        and report.get("status") == "PASS",
        "P8 qualification report has not passed",
    )
    _require(
        report.get("checkpoint_sha256") == contract["checkpoint_sha256"]
        and report.get("implementation_aggregate_sha256") == implementation_sha,
        "P8 qualification report provenance mismatch",
    )


def verify_checkpoint(
    checkpoint_path,
    contract_path,
    architecture_id=P4_ARCHITECTURE_ID,
    allow_untrained=False,
    expected_preprocessing_sha256=None,
    expected_dataset_sha256=None,
    checkpoint_bytes=None,
    contract_document=None,
):
    """Verify artifact identity plus optional data/preprocessing provenance.

    The formal P4 architecture is the default.  The retired range-image model
    remains verifiable only when callers explicitly pass
    :data:`LEGACY_ARCHITECTURE_ID`.
    """

    contract = (
        load_contract(contract_path)
        if contract_document is None
        else _validate_contract_document(contract_document)
    )
    if contract["architecture_id"] != architecture_id:
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"contract={contract['architecture_id']!r}, expected={architecture_id!r}"
        )
    if checkpoint_bytes is None:
        try:
            checkpoint_bytes = Path(checkpoint_path).read_bytes()
        except OSError as exc:
            raise ValueError("unable to read checkpoint bytes: %s" % exc) from exc
    if not isinstance(checkpoint_bytes, bytes):
        raise ValueError("checkpoint_bytes must be exact bytes")
    actual = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")

    payload = None
    if architecture_id == P4_ARCHITECTURE_ID:
        verify_p4_implementation_contract(contract["implementation_contract"])
        payload = _load_p4_checkpoint_payload(checkpoint_bytes)
        if (
            payload["implementation_aggregate_sha256"]
            != contract["implementation_contract"]["aggregate_sha256"]
        ):
            raise ValueError("checkpoint/contract implementation identity mismatch")
        if payload["production_qualified"] != contract["production_qualified"]:
            raise ValueError("checkpoint/contract production qualification mismatch")
        if payload["status"] != contract["status"]:
            raise ValueError("checkpoint/contract status mismatch")
        payload_qualification = payload.get("qualification_status")
        contract_qualification = contract.get("qualification_status")
        if (
            payload_qualification is not None
            or contract_qualification is not None
        ) and payload_qualification != contract_qualification:
            raise ValueError("checkpoint/contract qualification status mismatch")

    dataset = contract.get("dataset_provenance", {})
    if expected_preprocessing_sha256 is not None:
        _require_sha256(expected_preprocessing_sha256, "expected_preprocessing_sha256")
        if dataset.get("p3_preprocessing_sha256") != expected_preprocessing_sha256:
            raise ValueError("checkpoint preprocessing SHA-256 mismatch")
    if expected_dataset_sha256 is not None:
        _require_sha256(expected_dataset_sha256, "expected_dataset_sha256")
        if dataset.get("p3_task_manifest_sha256") != expected_dataset_sha256:
            raise ValueError("checkpoint dataset SHA-256 mismatch")

    if not allow_untrained:
        if not contract.get("production_qualified", False):
            raise ValueError("checkpoint is not production-qualified")
        if architecture_id == P4_ARCHITECTURE_ID:
            if (
                contract.get("status") != "PRODUCTION_QUALIFIED"
                or payload.get("qualification_status") != "QUALIFIED"
            ):
                raise ValueError("checkpoint has no production qualification state")
            _verify_production_attestation(contract, contract_path)
    return contract
