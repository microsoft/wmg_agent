# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import numpy as np

from agents.networks.shared.transformer import Transformer
from agents.networks.shared.general import LinearLayer
from agents.networks.shared.general import SeparateActorCriticLayers
from agents.networks.shared.general import SharedActorCriticLayers
from utils.graph import Graph

from utils.spec_reader import spec
V2 = spec.val("V2")
WMG_ATTENTION_HEAD_SIZE = spec.val("WMG_ATTENTION_HEAD_SIZE")
WMG_NUM_ATTENTION_HEADS = spec.val("WMG_NUM_ATTENTION_HEADS")
WMG_NUM_LAYERS = spec.val("WMG_NUM_LAYERS")
WMG_HIDDEN_SIZE = spec.val("WMG_HIDDEN_SIZE")
AC_HIDDEN_LAYER_SIZE = spec.val("AC_HIDDEN_LAYER_SIZE")
WMG_MAX_OBS = spec.val("WMG_MAX_OBS")
WMG_MAX_MEMOS = spec.val("WMG_MAX_MEMOS")

# Set WMG_MAX_MEMOS > 0 for attention over Memos, stored in a StateMatrix.
# Set WMG_MAX_OBS > 0 for attention over past observations, stored in a StateMatrix.
# Attention over both would require two separate instances of StateMatrix.
# The WMG experiments did not explore this combination, so there's only one StateMatrix.
if WMG_MAX_MEMOS:
    # StateMatrix contains Memos.
    S = WMG_MAX_MEMOS  # Maximum number of state vectors stored in the matrix.
    WMG_MEMO_SIZE = spec.val("WMG_MEMO_SIZE")
    assert WMG_MAX_OBS == 0
elif WMG_MAX_OBS:
    # StateMatrix contains past observations.
    S = WMG_MAX_OBS
else:
    # No StateMatrix.
    S = 0

# The WMG code was refactored in minor ways before the Sokoban experiments.
#   V1:  Used for Pathfinding and BabyAI.
#        Supports a single factor type.
#        Actor and critic networks are separate.
#        The Memo matrix is initialized full of zero vectors.
#        Pre-embedded age vectors are 1-hot.
#   V2:  Used for Sokoban.
#        Supports multiple factor types, one of which is the Core vector.
#        Actor and critic networks share their initial layer.
#        The Memo matrix is initialized empty.
#        Pre-embedded age vectors are 1-hot, concatenated with the normalized age scalar.


class WMG_Network(nn.Module):
    ''' Working Memory Graph '''
    def __init__(self, observation_space, action_space):
        super(WMG_Network, self).__init__()
        self.factored_observations = isinstance(observation_space, tuple) or isinstance(observation_space, Graph)
        self.tfm_vec_size = WMG_NUM_ATTENTION_HEADS * WMG_ATTENTION_HEAD_SIZE
        if WMG_MAX_OBS:
            assert not self.factored_observations
            self.state_vector_len = observation_space
        else:
            self.state_vector_len = WMG_MEMO_SIZE
        self.prepare_vector_embedding_layers(observation_space)
        self.prepare_age_encodings()
        self.tfm = Transformer(WMG_NUM_ATTENTION_HEADS, WMG_ATTENTION_HEAD_SIZE, WMG_NUM_LAYERS, WMG_HIDDEN_SIZE)
        if V2:
            self.actor_critic_layers = SharedActorCriticLayers(self.tfm_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space)
        else:
            self.actor_critic_layers = SeparateActorCriticLayers(self.tfm_vec_size, 2, AC_HIDDEN_LAYER_SIZE, action_space)
        if WMG_MAX_MEMOS > 0:
            self.memo_creation_layer = LinearLayer(self.tfm_vec_size, WMG_MEMO_SIZE)

    def forward(self, x, old_matrix):
        embedded_vec_list = []

        # Embed the Core and any Factors.
        if self.factored_observations:
            if V2:
                for entity in x.entities:
                    embedded_vec_list.append(self.embedding_layers[entity.type](torch.tensor(entity.data)))
            else:
                embedded_vec_list.append(self.core_embedding_layer(torch.tensor(np.float32(x[0]))))
                for factor_vec in x[1:]:
                    embedded_vec_list.append(self.factor_embedding_layer(torch.tensor(np.float32(factor_vec))))
        else:
            x_tens = torch.tensor(np.float32(x))
            embedded_vec_list.append(self.core_embedding_layer(x_tens))

        # Embed the state vectors.
        for i in range(old_matrix.num_vectors):
            if V2:
                embedded_vector = self.add_age_embedding(self.state_embedding_layer(old_matrix.get_vector(i)), i)
            else:
                embedded_vector = self.state_embedding_layer(torch.cat((old_matrix.get_vector(i), self.age[i])))
            embedded_vec_list.append(embedded_vector)

        # Apply the Transformer.
        tfm_input = torch.cat(embedded_vec_list).view(1, len(embedded_vec_list), self.tfm_vec_size)
        tfm_output = self.tfm(tfm_input)[:,0,:]  # Take only the Core column's output vector.

        # Update the state.
        if S:
            if WMG_MAX_MEMOS:
                # Store a new Memo.
                new_vector = torch.tanh(self.memo_creation_layer(tfm_output))[0]
            else:
                # Store a new observation.
                new_vector = torch.tanh(x_tens)
            new_matrix = old_matrix.clone()
            new_matrix.add_vector(new_vector)
        else:
            new_matrix = None

        # Return the actor-critic logits, and the state matrix.
        policy, value_est = self.actor_critic_layers(tfm_output)
        return policy, value_est, new_matrix

    def prepare_vector_embedding_layers(self, observation_space):
        if V2:
            assert self.factored_observations
            self.factor_sizes = observation_space.entity_type_sizes
            self.embedding_layers = nn.ModuleList()
            for factor_size in self.factor_sizes:
                self.embedding_layers.append(LinearLayer(factor_size, self.tfm_vec_size))
            if S:
                self.state_embedding_layer = LinearLayer(self.state_vector_len, self.tfm_vec_size)
        else:
            if self.factored_observations:
                self.core_vec_size = observation_space[0]
                self.factor_vec_size = observation_space[1]
            else:
                self.core_vec_size = observation_space
            self.core_embedding_layer = LinearLayer(self.core_vec_size, self.tfm_vec_size)
            if self.factored_observations:
                self.factor_embedding_layer = LinearLayer(self.factor_vec_size, self.tfm_vec_size)
            if S:
                self.state_embedding_layer = LinearLayer(self.state_vector_len + S, self.tfm_vec_size)

    def prepare_age_encodings(self):
        self.age = []
        if V2:
            # self.age contains a total of S+2 vectors, each of size S+2.
            # self.age[0] is for the current time step.
            # self.age[S] is for the oldest time step in the history.
            # self.age[S+1] is for out-of-range ages.
            # One additional element is copied into any age vector to hold the raw age value itself.
            # Each vector is of size S+2: a 1-hot vector of size S+1, and the age scalar.
            self.max_age_index = S + 1
            age_vector_size = S + 2
            for i in range(self.max_age_index):
                pos = np.zeros(age_vector_size, np.float32)
                pos[i] = 1.
                self.age.append(pos)
            self.age.append(np.zeros(age_vector_size, np.float32))
            self.age_embedding_layer = LinearLayer(age_vector_size, self.tfm_vec_size)
        else:
            for i in range(S):
                pos = np.zeros(S, np.float32)
                pos[i] = 1.
                self.age.append(torch.tensor(pos))

    def add_age_embedding(self, vector_in, age):
        age_vec = self.age[min(age, self.max_age_index)]
        age_vec[-1] = age / S  # Include the normalized age scalar to convey order.
        age_embedding = self.age_embedding_layer(torch.tensor(age_vec))
        return vector_in + age_embedding

    def init_state(self):
        if S:
            return StateMatrix(S, self.state_vector_len)
        else:
            return None

    def detach_from_history(self, matrix):
        if matrix is not None:
            matrix.detach_from_history()
        return matrix


class StateMatrix(object):
    ''' Used to store either Memos, or past observations (as factors). '''
    def __init__(self, max_vectors, vector_len):
        self.vector_len = vector_len
        self.max_vectors = max_vectors
        if V2:
            self.num_vectors = 0  # The matrix starts out empty.
        else:
            self.num_vectors = max_vectors  # The matrix starts out full.
        self.vectors = [torch.tensor(np.zeros(self.vector_len, np.float32)) for n in range(self.max_vectors)]
        self.next_index = 0  # Where to store the next vector, overwriting the oldest.

    def add_vector(self, vector):
        self.vectors[self.next_index] = vector
        self.next_index = (self.next_index + 1) % self.max_vectors
        if self.num_vectors < self.max_vectors:
            self.num_vectors += 1

    def get_vector(self, i):
        assert (i >= 0) and (i < self.max_vectors)
        if V2:
            i += 1  # Map i=0 to the most recent Memo.
        return self.vectors[(self.next_index - i + self.max_vectors) % self.max_vectors]

    def clone(self):
        g = StateMatrix(S, self.vector_len)
        for i in range(self.max_vectors):
            g.vectors[i] = self.vectors[i].clone()
        g.next_index = self.next_index
        g.num_vectors = self.num_vectors
        return g

    def detach_from_history(self):
        for i in range(self.max_vectors):
            self.vectors[i] = self.vectors[i].detach()
