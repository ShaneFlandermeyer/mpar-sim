from typing import Dict, List, Tuple
import numpy as np

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import copy

torch, nn = try_import_torch()


class LSTMActorCritic(TorchRNN, nn.Module):
  def __init__(
      self,
      # TorchRNN args
      obs_space,
      action_space,
      num_outputs,
      model_config,
      name,
      # Custom model args

  ):
    # TODO: Replace nn.Linear with SlimFC
    nn.Module.__init__(self)
    super().__init__(obs_space, action_space, num_outputs, model_config, name)

    self.obs_size = get_preprocessor(obs_space)(obs_space).size
    self.lstm_state_size = model_config["lstm_cell_size"]
    self.use_prev_action = model_config["lstm_use_prev_action"]
    self.use_prev_reward = model_config["lstm_use_prev_reward"]

    # Set up the pre-LSTM layers
    fcnet_hiddens = [self.obs_size] + model_config["fcnet_hiddens"]
    fc = []
    for i in range(1, len(fcnet_hiddens)):
      fc += [nn.Linear(fcnet_hiddens[i - 1], fcnet_hiddens[i]), nn.Tanh()]
    self.actor_layers = nn.Sequential(*fc)
    self.critic_layers = copy.deepcopy(self.actor_layers)

    # Compute LSTM input size
    self.action_dim = gym.spaces.utils.flatdim(action_space)
    self.lstm_input_size = fcnet_hiddens[-1]
    if self.use_prev_action:
      self.lstm_input_size += self.action_dim
    # if self.use_prev_reward:
    #   self.lstm_input_size += 1

    self.actor_lstm = nn.LSTM(
        self.lstm_input_size, self.lstm_state_size, batch_first=True)
    self.critic_lstm = copy.deepcopy(self.actor_lstm)
    self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
    self.value_branch = nn.Linear(self.lstm_state_size, 1)

    # TODO: Override model forward() to include previous action/reward
    if model_config["lstm_use_prev_action"]:
      self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
          SampleBatch.ACTIONS, space=action_space, shift=-1)
    # if model_config["lstm_use_prev_reward"]:
    #   self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
    #       SampleBatch.REWARDS, shift=-1)

    # Holds the current "base" output (before logits layer).
    self._features = None
    self._values = None

  @override(ModelV2)
  def get_initial_state(self):
    h = [
        self.actor_layers[-2].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0),
        self.actor_layers[-2].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0),
        self.critic_layers[-2].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0),
        self.critic_layers[-2].weight.new(
            1, self.lstm_state_size).zero_().squeeze(0)
    ]
    return h

  @override(ModelV2)
  def value_function(self):
    assert self._values is not None, "must call forward() first"
    return torch.reshape(self.value_branch(self._values), [-1])

  @override(RecurrentNetwork)
  def forward(
      self,
      input_dict: Dict[str, TensorType],
      state: List[TensorType],
      seq_lens: TensorType,
  ) -> Tuple[TensorType, List[TensorType]]:
    assert seq_lens is not None

    # Concat. prev-action/reward if required.
    prev_a_r = []
    if self.model_config["lstm_use_prev_action"]:
      prev_a = input_dict[SampleBatch.PREV_ACTIONS].float().view(-1, self.action_dim)
      prev_a_r.append(prev_a)
    # if self.model_config["lstm_use_prev_reward"]:
    #   prev_r = input_dict[SampleBatch.PREV_REWARDS].float().view(-1, 1)
    #   prev_a_r.append(prev_r)
      
    input_dict["obs_flat"] = torch.cat((input_dict["obs_flat"], *prev_a_r), dim=1)
    
    return super().forward(input_dict, state, seq_lens)

  @override(TorchRNN)
  def forward_rnn(self, inputs, state, seq_lens):
    """Feeds `inputs` (B x T x ..) through the Gru Unit.

    Returns the resulting outputs as a sequence (B x T x ...).
    Values are stored in self._cur_value in simple (B) shape (where B
    contains both the B and T dims!).

    Returns:
        NN Outputs (B x T x ...) as sequence.
        The state batches as a List of two items (c- and h-states).
    """
    # TODO: This currently only works if use_prev_action/reward is true!
    obs = inputs[:, :, :-(self.action_dim)]
    prev_actions = inputs[:, :, -self.action_dim:]
    # prev_rewards = inputs[:, :, -1].unsqueeze(-1)
    
        
    actor_features = self.actor_layers(obs)
    lstm_in = torch.cat((actor_features, prev_actions), dim=-1)
    self._features, [h1, c1] = self.actor_lstm(
        lstm_in,
        [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
    action_out = self.action_branch(self._features)

    critic_features = self.critic_layers(obs)
    lstm_in = torch.cat((critic_features, prev_actions), dim=-1)
    self._values, [h2, c2] = self.critic_lstm(
        lstm_in, 
        [torch.unsqueeze(state[2], 0), torch.unsqueeze(state[3], 0)])

    return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]
