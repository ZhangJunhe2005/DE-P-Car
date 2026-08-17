import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import reissue_p4_acceptance as reissue


def test_proposal_application_requires_every_exact_value():
    proposal = {
        "schema": "DEPCarP3V3TrainingAuthorityProposalV1",
        "status": "READY_FOR_EXPLICIT_P5_CONFIG_APPROVAL",
        "formal_training_started": False,
        "training_yaml_changes": {
            "dataset.root": "data/curated/samples",
            "qualification.p5_formal_training_allowed": True,
            "qualification.blocked_gates": [],
        },
    }
    training = {
        "dataset": {"root": "data/curated/samples"},
        "qualification": {
            "p5_formal_training_allowed": True,
            "blocked_gates": [],
        },
    }

    result = reissue.validate_proposal_application(proposal, training)

    assert result["status"] == "APPLIED_EXACTLY"
    assert result["mismatches"] == {}

    training["qualification"]["p5_formal_training_allowed"] = False
    with pytest.raises(RuntimeError, match="not applied exactly"):
        reissue.validate_proposal_application(proposal, training)


def test_dry_run_gate_requires_all_five_gates():
    names = (
        "p3_footprint_gate",
        "index_content_gate",
        "dataset_authority_gate",
        "validation_coverage_gate",
        "training_yaml_qualification_gate",
    )
    dry_run = {name: {"passed": True} for name in names}
    assert reissue.all_dry_run_gates_pass(dry_run)

    dry_run["validation_coverage_gate"]["passed"] = False
    assert not reissue.all_dry_run_gates_pass(dry_run)


def test_index_summary_enforces_development_splits():
    index = {
        "samples": 3,
        "entries": [
            {"split": "train", "map_uuid": "map-a"},
            {"split": "train", "map_uuid": "map-b"},
            {"split": "validation", "map_uuid": "map-c"},
        ],
    }

    summary = reissue.index_summary(index)

    assert summary == {
        "samples": 3,
        "counts_by_split": {"train": 2, "validation": 1},
        "maps_by_split": {"train": 2, "validation": 1},
        "total_maps": 3,
    }

    index["entries"][2]["split"] = "test"
    with pytest.raises(RuntimeError, match="unexpected split"):
        reissue.index_summary(index)
