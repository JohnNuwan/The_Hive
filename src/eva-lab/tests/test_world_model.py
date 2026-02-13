
import pytest
import torch
import torch.nn as nn
from eva_lab.world_model import WorldModel, FSQ

def test_fsq_initialization():
    levels = [8, 5, 5, 5]
    fsq = FSQ(levels)
    assert fsq.levels == levels
    assert fsq.basis.shape == (4,)

def test_fsq_forward_pass():
    levels = [8, 5, 5, 5]
    fsq = FSQ(levels)
    latent_dim = len(levels)
    batch_size = 2

    # Input tensor
    z = torch.randn(batch_size, latent_dim)

    # Forward pass
    z_quant = fsq(z)

    # Check output shape
    assert z_quant.shape == (batch_size, latent_dim)

    # Check output values are within expected range [-1, 1] (roughly)
    # Since FSQ scales to [-1, 1] then quantizes
    assert torch.all(z_quant >= -1.0)
    assert torch.all(z_quant <= 1.0)

def test_fsq_gradients():
    levels = [8, 5, 5, 5]
    fsq = FSQ(levels)
    latent_dim = len(levels)
    z = torch.randn(2, latent_dim, requires_grad=True)

    z_quant = fsq(z)
    loss = z_quant.sum()
    loss.backward()

    # Verify gradients flow back to input
    assert z.grad is not None
    assert torch.any(z.grad != 0)

def test_world_model_initialization():
    obs_dim = 10
    action_dim = 2
    levels = [8, 5, 5, 5]
    latent_dim = len(levels) # 4

    model = WorldModel(obs_dim, action_dim, latent_dim, levels)
    assert model.encoder is not None
    assert model.quantizer is not None
    assert model.dynamics is not None
    assert model.predictor is not None

def test_world_model_inconsistent_dims_raises_error():
    obs_dim = 10
    action_dim = 2
    levels = [8, 5, 5, 5]
    latent_dim = 512 # Inconsistent with levels length (4)

    # Expect ValueError when initializing
    with pytest.raises(ValueError, match=r"latent_dim must match len\(levels\)"):
        WorldModel(obs_dim, action_dim, latent_dim, levels)

def test_world_model_step():
    obs_dim = 10
    action_dim = 2
    levels = [8, 5, 5, 5]
    latent_dim = len(levels)

    model = WorldModel(obs_dim, action_dim, latent_dim, levels)

    batch_size = 2
    obs = torch.randn(batch_size, obs_dim)
    # Initialize prev_latent within [-1, 1] to ensure GRU output stays bounded (if intended)
    # or just use zeros
    prev_latent = torch.zeros(batch_size, latent_dim)
    action = torch.randn(batch_size, action_dim)

    next_latent = model.step(obs, prev_latent, action)

    assert next_latent.shape == (batch_size, latent_dim)
    # GRU hidden state initialized at 0 with bounded inputs should stay in (-1, 1)
    assert torch.all(next_latent >= -1.0)
    assert torch.all(next_latent <= 1.0)

def test_world_model_symlog_loss():
    obs_dim = 10
    action_dim = 2
    levels = [8, 5, 5, 5]
    latent_dim = len(levels)

    model = WorldModel(obs_dim, action_dim, latent_dim, levels)

    pred = torch.randn(5, 5)
    target = torch.randn(5, 5)

    loss = model.symlog_loss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # scalar
    assert loss >= 0
