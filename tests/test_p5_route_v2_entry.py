from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_dep_car_route_v2 as trainer


def test_route_v2_formal_contract_is_fusion_only_and_keeps_ablation_diagnostic():
    config, _sha256 = trainer.load_config()
    authorization = config["authorization"]
    assert authorization["formal_modalities"] == ["fusion"]
    assert set(authorization["diagnostic_only_modalities"]) == {
        "depth_only",
        "lidar_only",
    }
    assert authorization["test_split_sealed"] is True
    assert config["training"]["sensor_dropout_probability"] == 0.10
    assert config["training"]["workers"] == 8
    assert config["training"]["batch_size"] == 128


def test_route_v2_best_selection_prioritizes_candidate_feasibility():
    weak = {
        "zero_feasible_rate": 0.02,
        "future_capable_rate": 0.99,
        "mean_loss": 0.01,
    }
    safe = {
        "zero_feasible_rate": 0.01,
        "future_capable_rate": 0.80,
        "mean_loss": 1.0,
    }
    assert trainer.selection_key("candidate_capacity", safe) < trainer.selection_key(
        "candidate_capacity", weak
    )


def test_route_v2_score_selection_prioritizes_selected_safety_then_regret():
    unsafe_low_loss = {
        "selected_hard_feasible_rate": 0.80,
        "selected_future_capable_rate": 0.75,
        "mean_oracle_regret": 0.01,
        "mean_loss": 0.01,
    }
    safe = {
        "selected_hard_feasible_rate": 0.90,
        "selected_future_capable_rate": 0.80,
        "mean_oracle_regret": 0.20,
        "mean_loss": 1.0,
    }
    assert trainer.selection_key("score_calibration", safe) < trainer.selection_key(
        "score_calibration", unsafe_low_loss
    )


def test_route_v2_score_training_keeps_all_batchnorm_in_eval_mode():
    model = trainer.DEPCarNetV2()
    trainer.configure_training_stage(model, "score_calibration")
    trainer.set_stage_mode(model, "score_calibration", training=True)
    batch_norms = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert any(parameter.requires_grad for parameter in model.score_parameters())
    assert all(not parameter.requires_grad for parameter in model.candidate_parameters())


def test_finalize_best_preserves_selected_epoch_and_seals_full_run(tmp_path):
    checkpoint = tmp_path / "fusion.best.pth"
    torch.save(
        {
            "completed_epochs": 7,
            "selected_epoch": 7,
            "global_step": 70,
            "partial_epoch": False,
            "history": [],
        },
        checkpoint,
    )
    checkpoint.with_suffix(".contract.json").write_text(
        '{"checkpoint_sha256": "old", "partial_epoch": false}\n',
        encoding="utf-8",
    )
    history = [{"epoch": number} for number in range(1, 41)]
    trainer.finalize_best_artifact(
        checkpoint, completed_epochs=40, global_step=400, history=history
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = trainer.read_json(checkpoint.with_suffix(".contract.json"))
    assert payload["completed_epochs"] == 40
    assert payload["selected_epoch"] == 7
    assert payload["selected_global_step"] == 70
    assert payload["run_completed"] is True
    assert len(payload["history"]) == 40
    assert contract["checkpoint_sha256"] == trainer.sha256_file(checkpoint)
    assert contract["completed_epochs"] == 40
    assert contract["selected_epoch"] == 7
