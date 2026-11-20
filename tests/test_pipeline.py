"""End-to-end check on a tiny generated scene: train, resume, evaluate, video (~30 s on a CPU)."""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_train_resume_evaluate_and_render(tmp_path):
    scene = tmp_path / "scene"
    run("scripts/make_toy_scene.py", "--out", scene, "--res", "16", "--n_train", "6",
        "--n_val", "1", "--n_test", "2")
    base = ["train.py", "--config", "configs/smoke.yaml", "--device", "cpu", "--set",
            f"datadir={scene}", f"logdir={tmp_path / 'logs'}", "netwidth=32", "n_samples=8",
            "n_importance=8", "n_rand=64", "i_img=10", "i_weights=10", "i_print=5"]
    run(*base, "n_iters=10")
    run(*base, "n_iters=20")                  # same command, larger budget: resumes at step 10
    expdir = tmp_path / "logs" / "smoke"
    names = [p.name for p in sorted((expdir / "checkpoints").glob("step_*.pt"))]
    assert names == ["step_0000010.pt", "step_0000020.pt"]

    run("evaluate.py", "--expdir", expdir, "--lpips", "none", "--device", "cpu")
    evaldir, = expdir.glob("eval_test_0000020_*")
    report = json.loads((evaldir / "metrics.json").read_text())
    assert report["num_images"] == 2 and math.isfinite(report["mean"]["psnr"])
    assert report["testskip"] == 1 and report["lpips_backbone"] is None
    assert len(report["checkpoint_sha256"]) == len(report["split_manifest_sha256"]) == 64
    assert (evaldir / "r_0.png").exists()

    run("evaluate.py", "--expdir", expdir, "--lpips", "none", "--device", "cpu",
        "--testskip", "2")
    assert len(list(expdir.glob("eval_test_0000020_*"))) == 2

    run("render_video.py", "--expdir", expdir, "--frames", "3", "--device", "cpu")
    assert list(expdir.glob("video_0000020.*"))
