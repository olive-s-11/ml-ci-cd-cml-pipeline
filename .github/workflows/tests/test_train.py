from train import train


def test_training_writes_artifacts_and_reasonable_r2(tmp_path):
    metrics = train(output_dir=tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["r2"] > 0.3
