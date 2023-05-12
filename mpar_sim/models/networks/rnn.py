import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import copy

tf1, tf, tfv = try_import_tf()
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
    nn.Module.__init__(self)
    super().__init__(obs_space, action_space, num_outputs, model_config, name)

    self.obs_size = get_preprocessor(obs_space)(obs_space).size
    self.lstm_state_size = model_config["lstm_cell_size"]
    fcnet_hiddens = [self.obs_size] + model_config["fcnet_hiddens"]

    fc = []
    for i in range(1, len(fcnet_hiddens)):
      fc += [nn.Linear(fcnet_hiddens[i - 1], fcnet_hiddens[i]), nn.Tanh()]
    self.actor_layers = nn.Sequential(*fc)
    self.critic_layers = copy.deepcopy(self.actor_layers)

    self.actor_lstm = nn.LSTM(
        fcnet_hiddens[-1], self.lstm_state_size, batch_first=True)
    self.critic_lstm = nn.LSTM(
        fcnet_hiddens[-1], self.lstm_state_size, batch_first=True)
    self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
    self.value_branch = nn.Linear(self.lstm_state_size, 1)

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
    self._features, [h1, c1] = self.actor_lstm(
        self.actor_layers(inputs), [torch.unsqueeze(state[0], 0),
                                    torch.unsqueeze(state[1], 0)])
    action_out = self.action_branch(self._features)

    self._values, [h2, c2] = self.critic_lstm(
        self.critic_layers(inputs), [torch.unsqueeze(state[2], 0),
                                     torch.unsqueeze(state[3], 0)])

    return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]
