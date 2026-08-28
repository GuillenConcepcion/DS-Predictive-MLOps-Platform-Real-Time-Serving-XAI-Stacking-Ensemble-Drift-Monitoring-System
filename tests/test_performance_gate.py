from src.models.performance_gate import run_performance_gate


def test_performance_gate_execution():
    report = run_performance_gate()
    assert report["status"] == "PASSED"
    assert report["metrics"]["cv_roc_auc"] >= 0.880
    assert report["latency_benchmark"]["p95_ms"] <= 250.0
