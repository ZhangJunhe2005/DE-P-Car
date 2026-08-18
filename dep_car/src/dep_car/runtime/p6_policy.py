"""Strict loader and inference wrapper for P5 Score Head artifacts in P6."""

import json
import time
from pathlib import Path

import numpy as np

from dep_car.model.checkpoint import P4_ARCHITECTURE_ID, verify_checkpoint
from dep_car.model.dep_car_net import DEPCarNetV1

from .p6_contract import sha256_file, verify_p6_shadow_acceptance


class PolicyArtifactError(RuntimeError):
    pass


class P6PolicyRuntime:
    """Load a best Score artifact without granting it production authority."""

    MODALITIES = ("depth_only", "lidar_only", "fusion")
    FUSION_SENSOR_MODES = ("normal", "drop_depth", "drop_lidar")

    def __init__(
        self,
        checkpoint,
        contract,
        *,
        modality,
        device="cuda",
        mode="shadow",
        p6_authority="",
        fusion_sensor_mode="normal",
    ):
        try:
            import torch
        except ImportError as exc:
            raise PolicyArtifactError("PyTorch is unavailable in the policy process") from exc
        self.torch = torch
        self.checkpoint_path = Path(checkpoint).resolve()
        self.contract_path = Path(contract).resolve()
        self.modality = str(modality)
        self.mode = str(mode)
        self.fusion_sensor_mode = str(fusion_sensor_mode)
        if self.modality not in self.MODALITIES:
            raise PolicyArtifactError("unknown policy modality: " + self.modality)
        if self.mode not in ("shadow", "active"):
            raise PolicyArtifactError("policy mode must be shadow or active")
        if self.fusion_sensor_mode not in self.FUSION_SENSOR_MODES:
            raise PolicyArtifactError("unknown fusion sensor mode")
        if self.modality != "fusion" and self.fusion_sensor_mode != "normal":
            raise PolicyArtifactError("sensor-drop validation applies only to fusion")
        try:
            verified = verify_checkpoint(
                self.checkpoint_path,
                self.contract_path,
                architecture_id=P4_ARCHITECTURE_ID,
                allow_untrained=True,
            )
        except Exception as exc:
            raise PolicyArtifactError("P5 checkpoint identity verification failed: %s" % exc) from exc
        required = {
            "training_stage": "score_calibration",
            "artifact_role": "best",
            "status": "TRAINED_UNQUALIFIED",
            "qualification_status": "UNQUALIFIED",
            "production_qualified": False,
            "modality": self.modality,
        }
        mismatches = [key for key, value in required.items() if verified.get(key) != value]
        if mismatches:
            raise PolicyArtifactError(
                "checkpoint is not an accepted P6 Score artifact: " + ",".join(mismatches)
            )
        training_run = verified.get("training_run", {})
        formal_gates = (
            "formal_dataset_authority_gate_passed",
            "formal_index_content_gate_passed",
            "formal_p3_footprint_gate_passed",
            "formal_training_yaml_gate_passed",
            "formal_validation_coverage_gate_passed",
        )
        if (
            training_run.get("completed_epochs", 0) < 40
            or training_run.get("partial_epoch") is not False
            or any(training_run.get(gate) is not True for gate in formal_gates)
        ):
            raise PolicyArtifactError("checkpoint training/coverage evidence is incomplete")
        self._verify_accepted_candidate_source(verified.get("training_source", {}))
        self.contract = verified
        self.checkpoint_sha256 = verified["checkpoint_sha256"]
        self.contract_sha256 = sha256_file(self.contract_path)
        self.control_authorized = self.mode == "active"
        if self.mode == "active":
            if not p6_authority:
                raise PolicyArtifactError("active mode requires a passed P6 shadow authority")
            try:
                verify_p6_shadow_acceptance(
                    p6_authority,
                    checkpoint_sha256=self.checkpoint_sha256,
                    checkpoint_contract_sha256=self.contract_sha256,
                    modality=self.modality,
                )
            except Exception as exc:
                raise PolicyArtifactError("P6 active authority rejected: %s" % exc) from exc
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise PolicyArtifactError("CUDA policy device requested but unavailable")
        self.device = torch.device(device)
        try:
            payload = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            model = DEPCarNetV1()
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except Exception as exc:
            raise PolicyArtifactError("unable to materialize DEPCarNetV1: %s" % exc) from exc
        self.model = model.to(self.device).eval()
        self._amp_output_handles = []
        if self.device.type == "cuda":
            # Match the signed P5 training path: convolutions/towers stay in
            # AMP, while encoder outputs join the FP32 learned missing tokens
            # before index_copy.  Keeping this adapter outside DEPCarNetV1
            # preserves the checkpoint's frozen implementation identity.
            def output_fp32(_module, _arguments, output):
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError("sensor encoder output must be a tensor")
                return output.float()

            self._amp_output_handles = [
                self.model.depth_encoder.register_forward_hook(output_fp32),
                self.model.lidar_encoder.register_forward_hook(output_fp32),
            ]
        self.last_latency_ms = 0.0

    @staticmethod
    def _verify_accepted_candidate_source(source):
        """Verify the Candidate checkpoint and its signed PASS sidecar.

        Score contracts identify their accepted Candidate parent with hashes;
        the Candidate acceptance result is a separate sidecar rather than a
        synthetic ``training_source.passed`` field.  P6 checks the real
        lineage instead of weakening that frozen P5 contract.
        """

        if source.get("kind") != "accepted_candidate_capacity_checkpoint":
            raise PolicyArtifactError("Score checkpoint has no accepted Candidate lineage")
        required = ("checkpoint", "checkpoint_sha256", "contract_sha256", "candidate_acceptance_sha256")
        if any(not source.get(key) for key in required):
            raise PolicyArtifactError("Candidate lineage is incomplete")
        checkpoint = Path(source["checkpoint"]).resolve()
        contract = checkpoint.with_suffix(".contract.json")
        acceptance = checkpoint.with_suffix(".candidate_acceptance.json")
        try:
            if sha256_file(checkpoint) != source["checkpoint_sha256"]:
                raise PolicyArtifactError("Candidate checkpoint hash changed")
            if sha256_file(contract) != source["contract_sha256"]:
                raise PolicyArtifactError("Candidate contract hash changed")
            if sha256_file(acceptance) != source["candidate_acceptance_sha256"]:
                raise PolicyArtifactError("Candidate acceptance hash changed")
            evidence = json.loads(acceptance.read_text(encoding="utf-8"))
        except PolicyArtifactError:
            raise
        except Exception as exc:
            raise PolicyArtifactError("unable to verify Candidate lineage: %s" % exc) from exc
        gears = evidence.get("coverage", {}).get("requested_gear", {})
        if (
            evidence.get("schema") != "DEPCarP5CandidateAcceptanceV1"
            or evidence.get("status") != "PASS"
            or evidence.get("gate_passed") is not True
            or evidence.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or evidence.get("contract_sha256") != source["contract_sha256"]
            or int(gears.get("FORWARD", 0)) <= 0
            or int(gears.get("REVERSE", 0)) <= 0
        ):
            raise PolicyArtifactError("Candidate acceptance evidence is invalid")

    @property
    def modality_mask(self):
        if self.modality == "depth_only":
            return (1.0, 0.0)
        if self.modality == "lidar_only":
            return (0.0, 1.0)
        if self.fusion_sensor_mode == "drop_depth":
            return (0.0, 1.0)
        if self.fusion_sensor_mode == "drop_lidar":
            return (1.0, 0.0)
        return (1.0, 1.0)

    def infer(self, depth, lidar_bev, vehicle_state, requested_gear):
        torch = self.torch
        depth = np.asarray(depth, dtype=np.float32)
        lidar_bev = np.asarray(lidar_bev, dtype=np.float32)
        vehicle_state = np.asarray(vehicle_state, dtype=np.float32)
        if depth.shape != (2, 96, 160):
            raise ValueError("policy depth must be [2,96,160]")
        if lidar_bev.shape != (6, 160, 160):
            raise ValueError("policy LiDAR BEV must be [6,160,160]")
        if vehicle_state.shape != (9,) or not np.all(np.isfinite(vehicle_state)):
            raise ValueError("policy vehicle state must be finite [9]")
        if int(requested_gear) not in (-1, 1):
            raise ValueError("policy requested gear must be -1 or +1")
        tensors = (
            torch.from_numpy(depth[None]).to(self.device, non_blocking=True),
            torch.from_numpy(lidar_bev[None]).to(self.device, non_blocking=True),
            torch.from_numpy(vehicle_state[None]).to(self.device, non_blocking=True),
            torch.tensor([int(requested_gear)], dtype=torch.int64, device=self.device),
            torch.tensor([self.modality_mask], dtype=torch.float32, device=self.device),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(*tensors)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        trajectories = output.trajectories[0].detach().cpu().numpy().astype(np.float64)
        controls = output.controls[0].detach().cpu().numpy().astype(np.float64)
        scores = output.scores[0].detach().cpu().numpy().astype(np.float64)
        if trajectories.shape != (15, 11, 6) or controls.shape != (15, 4) or scores.shape != (15,):
            raise RuntimeError("DEPCarNetV1 output shape changed")
        if not all(np.all(np.isfinite(values)) for values in (trajectories, controls, scores)):
            raise RuntimeError("DEPCarNetV1 produced non-finite output")
        return trajectories, controls, scores


__all__ = ["P6PolicyRuntime", "PolicyArtifactError"]
