import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_collection_module():
    path = ROOT / "ros/dep_car_dataset/scripts/run_pilot_collection.py"
    spec = importlib.util.spec_from_file_location("dep_car_p3_collection", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eight_worker_ports_are_unique_and_numerical_threads_are_bounded():
    module = load_collection_module()
    manifest = {"task_manifest_sha256": "a" * 64}
    collection = {"ros_master_port": 11321, "gazebo_master_port": 11351}
    pairs = []
    for worker_index in range(8):
        environment, ros_port, gazebo_port = module.worker_environment(manifest, collection, worker_index)
        pairs.append((ros_port, gazebo_port))
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["DEP_CAR_P3_TASK_MANIFEST_SHA256"] == "a" * 64
    assert len({port for pair in pairs for port in pair}) == 16
    assert pairs[0] == (11321, 11351)
    assert pairs[-1] == (11328, 11358)


def test_map_affinity_prevents_same_map_from_running_on_two_workers():
    module = load_collection_module()
    tasks = [
        {"task_id": "task_%03d" % index, "map_uuid": "map_%02d" % (index % 30)}
        for index in range(150)
    ]
    buckets = module.shard_tasks_by_map(tasks, 8)
    locations = {}
    for worker_index, bucket in enumerate(buckets):
        for task in bucket:
            locations.setdefault(task["map_uuid"], set()).add(worker_index)
    assert sum(map(len, buckets)) == 150
    assert all(len(workers) == 1 for workers in locations.values())
    assert max(map(len, buckets)) - min(map(len, buckets)) <= 5


def test_lidar_costmap_does_not_duplicate_planning_inflation():
    config = ROOT / "ros/dep_car_perception/config/lidar.yaml"
    content = config.read_text(encoding="utf-8")
    assert "inflation_radius: 0.0" in content
    assert "local_map_resolution: 0.05" in content


def test_retry_policy_can_select_failed_and_zero_feasible_complete_tasks():
    module = load_collection_module()
    task = {"task_id": "task"}
    assert module.task_is_pending(task, {"tasks": {}})
    assert not module.task_is_pending(task, {"tasks": {"task": {"status": "FAILED"}}})
    assert module.task_is_pending(task, {"tasks": {"task": {"status": "FAILED"}}}, retry_failed=True)
    complete = {
        "tasks": {"task": {
            "status": "COMPLETE", "candidate_messages": 100, "zero_feasible_messages": 26,
        }}
    }
    assert not module.task_is_pending(task, complete, zero_rate_above=0.30)
    assert module.task_is_pending(task, complete, zero_rate_above=0.25)
    assert module.task_is_pending(task, complete, rerun_all_complete=True)


def test_scaled_indoor_lattice_uses_one_second_receding_horizon():
    from dep_car.core.lattice import LatticeConfig

    assert LatticeConfig().horizon == 1.0
