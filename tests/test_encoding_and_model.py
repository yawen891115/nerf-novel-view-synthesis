import torch

from nerf.encoding import PositionalEncoding
from nerf.model import NeRF


def official_embed(x, num_freqs):
    """The embedder of the authors' code: [x, sin(1x), cos(1x), sin(2x), cos(2x), ...]."""
    out = [x]
    for freq in 2.0 ** torch.arange(num_freqs):
        out += [torch.sin(x * freq), torch.cos(x * freq)]
    return torch.cat(out, -1)


def test_encoding_matches_official_layout():
    x = torch.randn(5, 3)
    for L in (4, 10):
        enc = PositionalEncoding(L)
        assert enc.out_dim(3) == 3 + 3 * 2 * L
        assert torch.allclose(enc(x), official_embed(x, L), atol=1e-6)


def test_encoding_dims_used_by_the_paper():
    assert PositionalEncoding(10).out_dim(3) == 63
    assert PositionalEncoding(4).out_dim(3) == 27


def test_mlp_shapes_and_parameter_count():
    net = NeRF(depth=8, width=256, pos_dim=63, dir_dim=27, skips=(4,))
    out = net(torch.randn(7, 63), torch.randn(7, 27))
    assert out.shape == (7, 4)
    expected = (63 * 256 + 256) + 4 * (256 * 256 + 256) + ((256 + 63) * 256 + 256) \
        + 2 * (256 * 256 + 256) + (256 + 1) + (256 * 256 + 256) + ((256 + 27) * 128 + 128) \
        + (128 * 3 + 3)
    assert sum(p.numel() for p in net.parameters()) == expected == 595_844


def test_density_does_not_depend_on_view_direction():
    net = NeRF(depth=4, width=32, pos_dim=63, dir_dim=27, skips=(2,))
    x = torch.randn(3, 63)
    a = net(x, torch.randn(3, 27))
    b = net(x, torch.randn(3, 27))
    assert torch.allclose(a[:, 3], b[:, 3])          # sigma is view independent
    assert not torch.allclose(a[:, :3], b[:, :3])    # colour is view dependent
