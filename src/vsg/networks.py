"""Networks for VSG."""

import torch
from distribution_extension import (
    BernoulliStraightThrough,
    MultiOneHotFactory,
    NormalFactory,
)
from torch import Tensor, nn
from torchrl.modules import MLP

from vsg.state import State

__all__ = [
    "Representation",
    "RepresentationV1",
    "RepresentationV2",
    "Transition",
    "TransitionV1",
    "TransitionV2",
    "VSGCell",
]


class VSGCell(nn.Module):
    """
    Variational Sparce Gating (VSG).

    References
    ----------
    [1] https://arnavkj1995.github.io/pubs/Jain22.pdf
    [2] https://github.com/arnavkj1995/VSG

    Parameters
    ----------
    input_size : int
        入力(x)の次元数.
    hidden_size : int
        隠れ状態(h)の次元数.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.update_bias = -1
        self.layer = nn.Linear(input_size + hidden_size, 3 * hidden_size)

    def forward(self, x: Tensor, hx: Tensor) -> Tensor:
        """
        順伝播.

        Architecture
        ------------
        inputs    := concat(x, h)
        update    := Bernoulli(Sigmoid(f_u(inputs)))
        reset     := Sigmoid(f_r(inputs))
        candidate := Tanh(reset * f_c(inputs))
        output    := update * candidate + (1 - update) * h

        Parameters
        ----------
        x : torch.Tensor
            入力. shape: [Batch, InputSize]
        h : torch.Tensor
            隠れ状態. shape: [Batch, HiddenSize]

        Returns
        -------
        torch.Tensor
            出力. shape: [Batch, HiddenSize]
        """
        parts = self.layer(torch.cat([x, hx], dim=-1))
        reset, candidate, update = torch.chunk(parts, 3, dim=-1)
        update_p = torch.sigmoid(update + self.update_bias)
        self.update = BernoulliStraightThrough(probs=update_p)
        update = self.update.rsample()
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        return update * candidate + (1 - update) * hx


class RepresentationV1(nn.Module):
    """
    Representation model for RSSM V1.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(
        self,
        *,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_to_post_projector = MLP(
            in_features=obs_embed_size + deterministic_size,
            out_features=stochastic_size * 2,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
        )
        self.distribution_factory = NormalFactory()

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """
        Single step transition, includes prior transition.

        Parameters
        ----------
        obs_embed : Tensor
            Embedding of observation. Shape: (batch_size, obs_embed_size)
        prior_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Approximate posterior state.

        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = self.rnn_to_post_projector(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)


class TransitionV1(nn.Module):
    """
    RSSM V1 Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(
        self,
        *,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        action_size: int,
        activation_name: str,
    ) -> None:
        super().__init__()

        self.rnn_cell = VSGCell(
            input_size=hidden_size,
            hidden_size=deterministic_size,
        )
        self.action_state_projector = MLP(
            in_features=action_size + stochastic_size,
            out_features=hidden_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.rnn_to_prior_projector = MLP(
            in_features=deterministic_size,
            out_features=stochastic_size * 2,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = NormalFactory()

    def forward(self, action: Tensor, prev_state: State) -> State:
        """
        Single step transition, includes deterministic transitions by GRUs.

        Parameters
        ----------
        action : Tensor
            (Prev) aciton of agent or robot. Shape: (batch_size, action_size)
        prev_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Prior state.

        """
        projector_input = torch.cat([action, prev_state.stoch], dim=-1)
        action_state = self.action_state_projector(projector_input)
        deter = self.rnn_cell.forward(action_state, hx=prev_state.deter)
        stoch_source = self.rnn_to_prior_projector(deter)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=deter, distribution=distribution)


class RepresentationV2(nn.Module):
    """
    Representation model for RSSM V2.

    ```
    stochastic = MLP(Transition.deterministic, obs_embed)
    ```
    """

    def __init__(
        self,
        *,
        deterministic_size: int,
        hidden_size: int,
        obs_embed_size: int,
        class_size: int,
        category_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_to_post_projector = MLP(
            in_features=obs_embed_size + deterministic_size,
            out_features=class_size * category_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )

    def forward(self, obs_embed: Tensor, prior_state: State) -> State:
        """
        Single step transition, includes prior transition.

        Parameters
        ----------
        obs_embed : Tensor
            Embedding of observation. Shape: (batch_size, obs_embed_size)
        prior_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Approximate posterior state.

        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = self.rnn_to_post_projector(projector_input)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=prior_state.deter, distribution=distribution)


class TransitionV2(nn.Module):
    """
    RSSM V2 Transition Model.

    ```
    deterministic = GRU(prev_action, prev_deterministic, prev_stochastic)
    stochastic = MLP(deterministic)
    ```
    """

    def __init__(
        self,
        *,
        deterministic_size: int,
        hidden_size: int,
        action_size: int,
        class_size: int,
        category_size: int,
        activation_name: str,
    ) -> None:
        """Set components."""
        super().__init__()

        self.rnn_cell = VSGCell(
            input_size=hidden_size,
            hidden_size=deterministic_size,
        )
        self.action_state_projector = MLP(
            in_features=action_size + class_size * category_size,
            out_features=hidden_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.rnn_to_prior_projector = MLP(
            in_features=deterministic_size,
            out_features=class_size * category_size,
            num_cells=hidden_size,
            depth=1,
            activation_class=getattr(torch.nn, activation_name),
            activate_last_layer=False,
        )
        self.distribution_factory = MultiOneHotFactory(
            class_size=class_size,
            category_size=category_size,
        )

    def forward(self, action: Tensor, prev_state: State) -> State:
        """
        Single step transition, includes deterministic transitions by GRUs.

        Parameters
        ----------
        action : Tensor
            (Prev) aciton of agent or robot. Shape: (batch_size, action_size)
        prev_state : State
            Previous state. Shape: (batch_size, action_size)

        Returns
        -------
        State
            Prior state.

        """
        projector_input = torch.cat([action, prev_state.stoch], dim=-1)
        action_state = self.action_state_projector(projector_input)
        deter = self.rnn_cell.forward(action_state, hx=prev_state.deter)
        stoch_source = self.rnn_to_prior_projector(deter)
        distribution = self.distribution_factory.forward(stoch_source)
        return State(deter=deter, distribution=distribution)


Representation = RepresentationV1 | RepresentationV2
Transition = TransitionV1 | TransitionV2
