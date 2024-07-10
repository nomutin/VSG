"""Tests of `core.py`."""

import pytest
from torch import Tensor, nn, rand

from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    DETERMINISTIC_SIZE,
    HIDDEN_SIZE,
    OBS_EMBED_SIZE,
    SEQ_LEN,
    STOCHASTIC_SIZE,
)
from vsg.core import VSG
from vsg.networks import (
    RepresentationV1,
    RepresentationV2,
    TransitionV1,
    TransitionV2,
)
from vsg.state import State


class DummyEncoder(nn.Module):
    """A dummy encoder for testing."""

    def forward(self, observation: Tensor) -> Tensor:
        """Encode observation([*B, C, H, W] -> [*B, D])."""
        if observation.ndim == 4:
            return rand(BATCH_SIZE, OBS_EMBED_SIZE)
        return rand(BATCH_SIZE, SEQ_LEN, OBS_EMBED_SIZE)


class DummyDecoder(nn.Module):
    """A dummy decoder for testing."""

    def forward(self, feature: Tensor) -> Tensor:
        """Decode feature([*B, D] -> [*B, C, H, W])."""
        if feature.ndim == 2:
            return rand(BATCH_SIZE, 3, 64, 64)
        return rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)


@pytest.fixture
def continuous_vsg() -> VSG:
    """Create a continuous VSG instance."""
    representation = RepresentationV1(
        deterministic_size=DETERMINISTIC_SIZE,
        stochastic_size=STOCHASTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        activation_name="ReLU",
    )
    transition = TransitionV1(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        stochastic_size=STOCHASTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        activation_name="ReLU",
    )
    init_proj = nn.Linear(OBS_EMBED_SIZE, DETERMINISTIC_SIZE)
    return VSG(
        representation=representation,
        transition=transition,
        encoder=DummyEncoder(),
        decoder=DummyDecoder(),
        init_proj=init_proj,
        kl_coeff=1.0,
        use_kl_balancing=False,
        sparsity_coeff=0.1,
        gate_prob=0.3,
    )


@pytest.fixture
def discrete_vsg() -> VSG:
    """Create a discrete VSG instance."""
    representation = RepresentationV2(
        deterministic_size=DETERMINISTIC_SIZE,
        category_size=CATEGORY_SIZE,
        class_size=CLASS_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        activation_name="ReLU",
    )
    transition = TransitionV2(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        category_size=CATEGORY_SIZE,
        class_size=CLASS_SIZE,
        hidden_size=HIDDEN_SIZE,
        activation_name="ReLU",
    )
    init_proj = nn.Linear(OBS_EMBED_SIZE, DETERMINISTIC_SIZE)
    return VSG(
        representation=representation,
        transition=transition,
        encoder=DummyEncoder(),
        decoder=DummyDecoder(),
        init_proj=init_proj,
        kl_coeff=1.0,
        use_kl_balancing=True,
        sparsity_coeff=0.1,
        gate_prob=0.3,
    )


def test__initial_state(
    continuous_vsg: VSG,
    discrete_vsg: VSG,
    observation_bd: Tensor,
) -> None:
    """Test `initial_state` method."""
    state = continuous_vsg.initial_state(observation_bd)
    assert state.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert state.stoch.shape == (BATCH_SIZE, STOCHASTIC_SIZE)

    state = discrete_vsg.initial_state(observation_bd)
    assert state.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert state.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__rollout_representation(
    action_bld: Tensor,
    observation_bld: Tensor,
    state_bd: State,
    continuous_vsg: VSG,
    discrete_vsg: VSG,
) -> None:
    """Test `rollout_representation` method."""
    prior, posterior, _ = continuous_vsg.rollout_representation(
        actions=action_bld,
        observations=observation_bld,
        prev_state=state_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, SEQ_LEN, STOCHASTIC_SIZE)
    assert posterior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, SEQ_LEN, STOCHASTIC_SIZE)

    prior, posterior, _ = discrete_vsg.rollout_representation(
        actions=action_bld,
        observations=observation_bld,
        prev_state=state_bd,
    )
    feature_size = CATEGORY_SIZE * CLASS_SIZE
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, SEQ_LEN, feature_size)
    assert posterior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, SEQ_LEN, feature_size)


def test__rollout_transition(
    action_bld: Tensor,
    state_bd: State,
    continuous_vsg: VSG,
    discrete_vsg: VSG,
) -> None:
    """Test `rollout_transition` method."""
    prior = continuous_vsg.rollout_transition(
        actions=action_bld,
        prev_state=state_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, SEQ_LEN, STOCHASTIC_SIZE)

    prior = discrete_vsg.rollout_transition(
        actions=action_bld,
        prev_state=state_bd,
    )
    feature_size = CATEGORY_SIZE * CLASS_SIZE
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, SEQ_LEN, feature_size)


def test__training_step(
    action_bld: Tensor,
    observation_bld: Tensor,
    continuous_vsg: VSG,
    discrete_vsg: VSG,
) -> None:
    """Test `training_step` method."""
    batch = (action_bld, observation_bld, action_bld, observation_bld)
    loss = continuous_vsg.training_step(batch)
    assert "loss" in loss
    assert "kl" in loss
    assert "recon" in loss
    assert "sparsity" in loss

    loss = discrete_vsg.training_step(batch)
    assert "loss" in loss
    assert "kl" in loss
    assert "recon" in loss
    assert "sparsity" in loss


def test__validation_step(
    action_bld: Tensor,
    observation_bld: Tensor,
    continuous_vsg: VSG,
    discrete_vsg: VSG,
) -> None:
    """Test `validation_step` method."""
    batch = (action_bld, observation_bld, action_bld, observation_bld)
    loss = continuous_vsg.validation_step(batch, 0)
    assert "val_loss" in loss
    assert "val_kl" in loss
    assert "val_recon" in loss
    assert "val_sparsity" in loss

    loss = discrete_vsg.validation_step(batch, 0)
    assert "val_loss" in loss
    assert "val_kl" in loss
    assert "val_recon" in loss
    assert "val_sparsity" in loss
