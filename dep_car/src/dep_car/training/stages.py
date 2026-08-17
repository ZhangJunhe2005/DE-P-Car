"""P5-ready parameter ownership for the two-stage training protocol."""

import torch


TRAINING_STAGES = ("candidate_capacity", "score_calibration", "joint_smoke")
MODALITY_MODES = ("fusion", "depth_only", "lidar_only")


def _unique(parameters):
    seen = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def parameter_partitions(model):
    candidate = _unique(model.candidate_parameters())
    score = _unique(model.score_parameters())
    overlap = {id(parameter) for parameter in candidate}.intersection(
        id(parameter) for parameter in score
    )
    if overlap:
        raise RuntimeError("candidate and score parameter partitions overlap")
    owned = {id(parameter) for parameter in candidate + score}
    unowned = [name for name, parameter in model.named_parameters() if id(parameter) not in owned]
    if unowned:
        raise RuntimeError("trainable parameters are missing stage ownership: " + ", ".join(unowned))
    return {"candidate": candidate, "score": score}


def configure_training_stage(model, stage):
    if stage not in TRAINING_STAGES:
        raise ValueError(f"training stage must be one of {TRAINING_STAGES}")
    groups = parameter_partitions(model)
    candidate_active = stage in ("candidate_capacity", "joint_smoke")
    score_active = stage in ("score_calibration", "joint_smoke")
    for parameter in groups["candidate"]:
        parameter.requires_grad_(candidate_active)
    for parameter in groups["score"]:
        parameter.requires_grad_(score_active)
    return {
        "stage": stage,
        "candidate_trainable": sum(p.numel() for p in groups["candidate"] if p.requires_grad),
        "score_trainable": sum(p.numel() for p in groups["score"] if p.requires_grad),
    }


def build_optimizer(model, stage, learning_rate=1.0e-4, weight_decay=1.0e-5):
    configure_training_stage(model, stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("selected stage has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def modality_mask(batch_size, mode, device=None, dtype=torch.float32):
    if mode not in MODALITY_MODES:
        raise ValueError(f"modality mode must be one of {MODALITY_MODES}")
    values = {
        "fusion": (1.0, 1.0),
        "depth_only": (1.0, 0.0),
        "lidar_only": (0.0, 1.0),
    }[mode]
    return torch.tensor(values, device=device, dtype=dtype)[None, :].expand(batch_size, -1)


def apply_sensor_dropout(mask, probability, *, generator=None):
    """Drop at most one modality; a sample is never left sensorless."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("sensor dropout probability must be in [0,1]")
    if mask.ndim != 2 or mask.shape[1] != 2:
        raise ValueError("modality mask must have shape [B,2]")
    result = mask.clone()
    draw = torch.rand((len(mask),), device=mask.device, generator=generator)
    side = torch.randint(0, 2, (len(mask),), device=mask.device, generator=generator)
    for index in range(len(mask)):
        if draw[index] < probability and bool(result[index].all()):
            result[index, side[index]] = 0
    if bool(torch.any(result.sum(dim=1) < 1)):
        raise RuntimeError("sensor dropout produced an invalid sensorless sample")
    return result

