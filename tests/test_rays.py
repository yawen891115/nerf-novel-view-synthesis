import json
import math

import numpy as np
import torch

import make_toy_scene as toy
from nerf.data import RaySampler, get_rays, load_blender_split, pose_spherical


def official_get_rays(H, W, K, c2w):
    """yenchenlin/nerf-pytorch run_nerf_helpers.get_rays (port of the authors' code)."""
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij")
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def test_get_rays_matches_official_code():
    H, W, focal = 6, 8, 7.5
    c2w = torch.from_numpy(pose_spherical(30.0, -30.0, 4.0))
    K = [[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]]
    o1, d1 = get_rays(H, W, focal, c2w)
    o2, d2 = official_get_rays(H, W, K, c2w)
    assert torch.allclose(o1, o2) and torch.allclose(d1, d2, atol=1e-6)


def test_centre_ray_points_down_camera_minus_z():
    c2w = torch.from_numpy(toy.look_at(np.array([1.0, 2.0, 3.0]))).float()
    _, d = get_rays(10, 10, 20.0, c2w)
    centre = d[5, 5] / d[5, 5].norm()                    # pixel (H/2, W/2)
    assert torch.allclose(centre, -c2w[:3, 2], atol=1e-6)
    assert torch.allclose(centre, -c2w[:3, 3] / c2w[:3, 3].norm(), atol=1e-6)  # looks at origin


def test_rays_hit_the_toy_objects_exactly_where_alpha_is_set(tmp_path):
    """Independent check of the camera convention: cast our rays into the toy scene's
    analytic geometry and compare the hit mask with the rendered alpha channel."""
    res = 32
    c2w = toy.look_at(np.array([2.5, -2.0, 2.4]))
    rgba = toy.render(c2w, res, supersample=1)
    o, d = get_rays(res, res, 0.5 * res / math.tan(0.5 * toy.CAMERA_ANGLE_X),
                    torch.from_numpy(c2w))
    dirs = d.reshape(-1, 3).numpy()
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    hit, _ = toy.trace(o.reshape(-1, 3).numpy(), dirs)
    assert hit.mean() > 0.1
    assert np.array_equal(hit.reshape(res, res), rgba[..., 3] > 127)


def test_loader_reads_blender_format(tmp_path):
    frames = []
    (tmp_path / "test").mkdir()
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[2:6, 2:6] = [255, 0, 0, 255]                      # opaque red square, rest transparent
    import imageio.v2 as imageio
    imageio.imwrite(tmp_path / "test" / "r_0.png", rgba)
    frames.append({"file_path": "./test/r_0", "transform_matrix": np.eye(4).tolist()})
    with open(tmp_path / "transforms_test.json", "w") as f:
        json.dump({"camera_angle_x": 0.5, "frames": frames}, f)

    imgs, poses, H, W, focal, names = load_blender_split(tmp_path, "test")
    assert imgs.shape == (1, 8, 8, 3) and names == ["r_0"]
    assert math.isclose(focal, 0.5 * 8 / math.tan(0.25), rel_tol=1e-6)
    assert np.allclose(imgs[0, 0, 0], 1.0)                 # transparent -> white background
    assert np.allclose(imgs[0, 3, 3], [1.0, 0.0, 0.0])

    half, _, H2, W2, focal2, _ = load_blender_split(tmp_path, "test", half_res=True)
    assert (H2, W2) == (4, 4) and math.isclose(focal2, focal / 2)
    assert np.allclose(half[0, 1, 1], [1.0, 0.0, 0.0])     # 2x2 block fully inside the square


def test_sampler_batches_and_precrop():
    imgs = np.random.rand(3, 20, 30, 3).astype(np.float32)
    poses = np.stack([np.eye(4, dtype=np.float32)] * 3)
    s = RaySampler(imgs, poses, 20, 30, 25.0, n_rand=64, precrop_iters=10, precrop_frac=0.5)
    o, d, rgb = s.sample(step=1)
    assert o.shape == d.shape == rgb.shape == (64, 3)
    # During precrop every ray goes through the central half of the image.
    x, y = d[:, 0] * 25.0 + 15, -d[:, 1] * 25.0 + 10     # invert the pixel -> ray mapping
    assert x.min() >= 15 - 7.5 - 1e-4 and x.max() < 15 + 7.5
    assert y.min() >= 10 - 5 - 1e-4 and y.max() < 10 + 5
