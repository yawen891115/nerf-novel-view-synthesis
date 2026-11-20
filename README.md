# NeRF: Novel View Synthesis

A PyTorch implementation of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934) for Blender-format scenes. Given calibrated RGB images of a scene, it learns a continuous 3D representation and renders RGB and depth from new camera poses.

- Positional encoding and an eight-layer MLP predict volume density and view-dependent color.
- Stratified samples feed a coarse model; importance sampling adds points where the coarse model predicts more contribution. The renderer integrates color and density along each camera ray.
- The training loop tracks coarse and fine losses, validation images, and learning rate in TensorBoard. Evaluation produces held-out renders, depth maps, and image-quality metrics. A separate script renders a camera orbit.

## Quick start

Use Python 3.10+ and install the requirements in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Generate a small local scene and verify the training/evaluation workflow.
python scripts/make_toy_scene.py --out data/toy_scene
python train.py --config configs/smoke.yaml --device cpu
python evaluate.py --expdir logs/smoke --lpips none --device cpu
```

## Blender Lego

Put the Lego scene under `data/nerf_synthetic/lego`, or use the download helper below. The scene directory must contain `transforms_train.json`, `transforms_val.json`, and `transforms_test.json` along with the image files referenced by those manifests.

```bash
bash scripts/download_blender.sh
python train.py --config configs/lego_paper.yaml
python evaluate.py --expdir logs/lego_paper --split test --testskip 1 --lpips vgg
python render_video.py --expdir logs/lego_paper --frames 120
tensorboard --logdir logs
```

| Config | Resolution | Coarse + additional fine samples | Steps |
|---|---:|---:|---:|
| `configs/lego_paper.yaml` | 800 × 800 | 64 + 128 | 500,000 |
| `configs/lego_quick.yaml` | 400 × 400 | 64 + 64 | 100,000 |

The full-resolution config uses 1,024 rays per optimization step, as in the authors’ [released Blender configuration](https://github.com/bmild/nerf/blob/master/paper_configs/blender_config.txt). This implementation also center-crops ray sampling for the first 500 steps, uses PyTorch's default layer initialization, and enables TF32 where supported when `tf32: true`. The quick config is a separate setting, not a substitute for the full-resolution protocol. Ray and MLP point chunk sizes are configurable to bound the working batches; input tensors and accumulated outputs still consume memory.

### Checkpointing and evaluation

- Re-run the same training command to resume from the latest completed checkpoint. Changing the data, model, or optimization recipe requires a new `expname`. You can increase `n_iters` without changing the learning-rate schedule. Work after the last checkpoint may be lost if you interrupt training.
- Checkpoints contain model, optimizer, step, and random number generator state. New checkpoints use tensors and primitives so they can be loaded with PyTorch's restricted loader. Prefer checkpoints you produced yourself; never use `--trust-legacy-checkpoint` on files from untrusted sources.
- `evaluate.py` creates a separate output directory per invocation. Its `metrics.json` contains mean and per-image PSNR, SSIM, and optionally LPIPS, along with resolution, split/stride, effective config, device, checkpoint hash, manifest hash, and LPIPS backbone. Choose `--lpips alex` to use AlexNet or `--lpips none` to skip it. LPIPS may download its backbone weights on first use.
- For a full Lego test-set evaluation use `--split test --testskip 1`; the official split contains 200 views. The [original NeRF paper, Table 4](https://arxiv.org/pdf/2003.08934), reports 32.54 PSNR, 0.961 SSIM, and 0.050 LPIPS for Lego. Those are the paper authors' reference values, not measurements from this implementation.

## Code map

| Component | Location |
|---|---|
| Blender data and camera rays | [`nerf/data.py`](nerf/data.py) |
| Positional encoding and radiance-field MLP | [`nerf/encoding.py`](nerf/encoding.py), [`nerf/model.py`](nerf/model.py) |
| Stratified and hierarchical sampling | [`nerf/sampling.py`](nerf/sampling.py) |
| Differentiable volume rendering | [`nerf/rendering.py`](nerf/rendering.py) |
| Training and checkpointing | [`train.py`](train.py), [`nerf/checkpoint.py`](nerf/checkpoint.py) |
| Held-out image metrics and camera orbit | [`evaluate.py`](evaluate.py), [`render_video.py`](render_video.py) |

Run `python -m pytest -q` for checks of ray conventions, sampling, volume rendering, checkpoint resume, and the generated-scene workflow.

## References

- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020. [Paper](https://arxiv.org/abs/2003.08934) · [Authors' code](https://github.com/bmild/nerf)
- [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), a PyTorch reference used to cross-check conventions.

The [earlier implementation and its original instructions](https://github.com/yawen891115/nerf-novel-view-synthesis/tree/25fe4d69078ff7581e478629b3da192b1c854fca) remain available. The root commands above are for the configurable implementation. See [Third-party notices](THIRD_PARTY_NOTICES.md) for upstream attribution.
