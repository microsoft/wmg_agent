# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import gym
import babyai  # This registers the 19 MiniGrid levels.
import numpy as np

from utils.spec_reader import spec
BABYAI_ENV_LEVEL = spec.val("BABYAI_ENV_LEVEL")
USE_SUCCESS_RATE = spec.val("USE_SUCCESS_RATE")  # Used by post-processing files.
SUCCESS_RATE_THRESHOLD = spec.val("SUCCESS_RATE_THRESHOLD")
HELDOUT_TESTING = spec.val("HELDOUT_TESTING")
NUM_TEST_EPISODES = spec.val("NUM_TEST_EPISODES")
BINARY_REWARD = spec.val("BINARY_REWARD")
OBS_ENCODER = spec.val("OBS_ENCODER")

assert USE_SUCCESS_RATE

color_list = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
color_index_dict = {}
for i in range(len(color_list)):
    color_index_dict[color_list[i]] = i

cell_object_base_index = 4

action_list = ['go', 'pick', 'put', 'open']
action_index_dict = {}
for i in range(len(action_list)):
    action_index_dict[action_list[i]] = i

objtype_list = ['door', 'key', 'ball', 'box']
objtype_index_dict = {}
for i in range(len(objtype_list)):
    objtype_index_dict[objtype_list[i]] = i

article_list = ['the', 'a']
article_index_dict = {}
for i in range(len(article_list)):
    article_index_dict[article_list[i]] = i

loc_list = [['behind', 'you'], ['in', 'front', 'of', 'you'], ['on', 'your', 'left'], ['on', 'your', 'right']]
loc_index_dict = {}
for i in range(len(loc_list)):
    loc_index_dict[' '.join(loc_list[i])] = i

object_char = ['.', '.', 'X', '?', 'd', 'k', 'b', '#']


def encode_1_hot(i, size, vector, id):  # integer => 1-hot vector
    assert (i < size)
    vector[id + i] = 1.
    id += size
    return id


def decode_1_hot(size, vector, id):  # 1-hot vector => integer
    idx = None
    for i in range(size):
        if vector[id + i] != 0.:
            idx = i
            break
    id += size
    return idx, id


def print_obs(obs):
    # sz = "pointed "
    # dir = obs['direction']
    # if dir == 0:
    #     sz += 'right'
    # elif dir == 1:
    #     sz += 'down'
    # elif dir == 2:
    #     sz += 'left'
    # elif dir == 3:
    #     sz += 'up'
    # else:
    #     print("where?")
    # print(sz)

    image = obs['image']
    for row in range(7):
        sz = ""
        for col in range(7):
            sz += " {}".format(object_char[image[col][row][0]])
        # sz += '    '
        # for col in range(7):
        #     sz += "{}  ".format(image[col][row][1])
        # sz += '    '
        # for col in range(7):
        #     sz += "{}  ".format(image[col][row][2])
        print(sz)
    print()


class ClauseEncoder(object):
    def __init__(self):
        # Encoding translates a string into a real vector.
        # Decoding (for debugging only) does the opposite.
        self.last_instruction_encoded = ""
        self.num_actions = len(action_list)
        self.num_articles = len(article_list)
        self.num_colors = len(color_list)
        self.num_objtypes = len(objtype_list)
        self.num_locs = len(loc_list)
        object_vector_length = self.num_articles + self.num_colors + self.num_objtypes + self.num_locs
        self.single_object_vector_length = self.num_actions + object_vector_length
        self.double_object_vector_length = self.num_actions + 2 * object_vector_length
        self.encoded_vector = np.zeros(self.double_object_vector_length)
        self.word_idx = 0
        self.real_idx = 0

    def encode(self, instruction_str):
        if instruction_str == self.last_instruction_encoded:
            return self.encoded_vector
        self.last_instruction_encoded = instruction_str
        self.encoded_vector = np.zeros(self.double_object_vector_length)
        self.word_list = instruction_str.split(' ')
        self.num_words = len(self.word_list)
        self.word_idx = 0
        self.real_idx = 0

        word = self.read_word()
        assert word in action_index_dict.keys()
        action_id = action_index_dict[word]
        self.real_idx = encode_1_hot(action_id, self.num_actions, self.encoded_vector, self.real_idx)

        if word == 'go':  # go to *
            self.read_word('to')
            self.encode_object()
            assert self.real_idx == self.single_object_vector_length
        elif word == 'pick':  # pick up *
            self.read_word('up')
            self.encode_object()
            assert self.real_idx == self.single_object_vector_length
        elif word == 'put':  # put * next to *
            self.encode_object()
            self.read_word('next')
            self.read_word('to')
            self.encode_object()
            assert self.real_idx == self.double_object_vector_length
        elif word == 'open':  # open *
            self.encode_object()
            assert self.real_idx == self.single_object_vector_length
        else:
            assert False  # Unrecognized action.

        return self.encoded_vector

    def decode(self, encoded_vector):
        self.encoded_vector = encoded_vector[:]
        self.word_list = []
        self.real_idx = 0
        action_id, self.real_idx = decode_1_hot(self.num_actions, self.encoded_vector, self.real_idx)
        action = action_list[action_id]

        if action == 'go':  # go to *
            self.word_list.append('go')
            self.word_list.append('to')
            self.decode_object()
            assert self.real_idx == self.single_object_vector_length
        elif action == 'pick':  # pick up *
            self.word_list.append('pick')
            self.word_list.append('up')
            self.decode_object()
            assert self.real_idx == self.single_object_vector_length
        elif action == 'put':  # put * next to *
            self.word_list.append('put')
            self.decode_object()
            self.word_list.append('next')
            self.word_list.append('to')
            self.decode_object()
            assert self.real_idx == self.double_object_vector_length
        elif action == 'open':  # open *
            self.word_list.append('open')
            self.decode_object()
            assert self.real_idx == self.single_object_vector_length
        else:
            assert False  # Unrecognized action.

        while self.real_idx < self.double_object_vector_length:
            assert self.encoded_vector[self.real_idx] == 0.
            self.real_idx += 1
        assert self.real_idx == self.double_object_vector_length

        return ' '.join(self.word_list)

    def read_word(self, expected_word=None):
        assert (self.word_idx < self.num_words)
        word = self.word_list[self.word_idx]
        if expected_word is not None:
            assert word == expected_word
        self.word_idx += 1
        return word

    def read_phrase(self, num_words_to_read, expected_phrase=None):
        assert (self.word_idx + num_words_to_read <= self.num_words)
        word_list = []
        for i in range(num_words_to_read):
            word_list.append(self.word_list[self.word_idx])
            self.word_idx += 1
        phrase = ' '.join(word_list)
        if expected_phrase is not None:
            assert phrase == expected_phrase
        return phrase

    def peek_at_word(self):
        if self.word_idx >= self.num_words:
            return None
        word = self.word_list[self.word_idx]
        return word

    def encode_object(self):
        self.encode_article()
        self.encode_color()
        self.encode_objtype()
        self.encode_loc()

    def decode_object(self):
        self.decode_article()
        self.decode_color()
        self.decode_objtype()
        self.decode_loc()

    def encode_article(self):
        word = self.read_word()
        assert (word in article_index_dict.keys())
        index = article_index_dict[word]
        self.real_idx = encode_1_hot(index, self.num_articles, self.encoded_vector, self.real_idx)

    def encode_color(self):
        word = self.peek_at_word()
        if word in color_index_dict.keys():  # Color is optional.
            self.read_word()
            index = color_index_dict[word]
            self.real_idx = encode_1_hot(index, self.num_colors, self.encoded_vector, self.real_idx)
        else:
            self.real_idx += self.num_colors

    def encode_objtype(self):
        word = self.read_word()
        assert (word in objtype_index_dict.keys())
        index = objtype_index_dict[word]
        self.real_idx = encode_1_hot(index, self.num_objtypes, self.encoded_vector, self.real_idx)

    def encode_loc(self):
        word = self.peek_at_word()
        if (word == 'behind') or (word == 'in') or (word == 'on'):
            phrase = None
            if word == 'behind':
                phrase = self.read_phrase(2, 'behind you')
            elif word == 'in':
                phrase = self.read_phrase(4, 'in front of you')
            elif word == 'on':
                phrase = self.read_phrase(3)
                assert (phrase == 'on your left') or (phrase == 'on your right')
            index = loc_index_dict[phrase]
            self.real_idx = encode_1_hot(index, self.num_locs, self.encoded_vector, self.real_idx)
        else:
            self.real_idx += self.num_locs

    def decode_article(self):
        article_id, self.real_idx = decode_1_hot(self.num_articles, self.encoded_vector, self.real_idx)
        word = article_list[article_id]
        self.word_list.append(word)

    def decode_color(self):
        color_id, self.real_idx = decode_1_hot(self.num_colors, self.encoded_vector, self.real_idx)
        if color_id is not None:
            word = color_list[color_id]
            self.word_list.append(word)

    def decode_objtype(self):
        objtype_id, self.real_idx = decode_1_hot(self.num_objtypes, self.encoded_vector, self.real_idx)
        word = objtype_list[objtype_id]
        self.word_list.append(word)

    def decode_loc(self):
        loc_id, self.real_idx = decode_1_hot(self.num_locs, self.encoded_vector, self.real_idx)
        if loc_id is not None:
            for word in loc_list[loc_id]:
                self.word_list.append(word)


class BabyAI_Env(object):
    def __init__(self, seed=None):
        if not seed:
            seed = 25912
        level = BABYAI_ENV_LEVEL

        self.env = gym.make(level)
        self.env.seed(seed)

        self.num_orientations = 4
        self.retina_width = 7
        self.num_cell_types = 9
        self.encoder = ClauseEncoder()
        self.num_object_types = self.encoder.num_objtypes
        self.num_colors = self.encoder.num_colors
        self.num_door_states = 3
        self.action_space = 7
        self.max_num_factors = 12
        if OBS_ENCODER == 'Flat':
            self.observation_space = self.action_space + self.num_orientations  # Last action, and current direction.
            self.observation_space += self.retina_width * self.retina_width * (self.num_cell_types + self.num_colors + self.num_door_states)  # Image.
            self.observation_space += self.encoder.double_object_vector_length
        elif OBS_ENCODER == 'CnnFlat':
            self.observation_space = self.action_space + self.num_orientations  # Last action, and current direction.
            self.observation_space += self.encoder.double_object_vector_length
        elif OBS_ENCODER == 'Factored':
            self.observation_global_vector_size = self.action_space + self.num_orientations + 2 * (1 + self.retina_width)
            self.observation_global_vector_size += self.encoder.double_object_vector_length
            self.observation_factor_vector_size = self.num_object_types + self.num_colors + 2 * (1 + self.retina_width)
            self.observation_factor_vector_size += self.num_door_states
            self.observation_space = (self.observation_global_vector_size, self.observation_factor_vector_size)
        elif OBS_ENCODER == 'FactoredThenFlattened':
            self.observation_global_vector_size = self.action_space + self.num_orientations + 2 * (1 + self.retina_width)
            self.observation_global_vector_size += self.encoder.double_object_vector_length
            self.observation_factor_vector_size = self.num_object_types + self.num_colors + 2 * (1 + self.retina_width)
            self.observation_factor_vector_size += self.num_door_states
            self.observation_space = self.observation_global_vector_size + self.max_num_factors * self.observation_factor_vector_size

        self.reset_online_test_sums()
        self.reward = 0.
        self.score = 0.

        # Metrics
        self.rsr_win_len = 100
        self.success_sum = 0.
        self.success_buf = []
        self.success_buf_id = 0
        self.running_success_rate = 0.
        self.sample_efficiency = 0.
        self.rsr_threshold_reached = False
        self.num_local_steps = 0
        self.prev_num_local_steps = 0
        self.prev_sr = 0
        self.sum_sr = 0.
        self.num_sr = 0.

    def assemble_current_observation(self, obs, last_action=None):
        if OBS_ENCODER == 'Flat':
            return self.encode_flat(obs, last_action)
        if OBS_ENCODER == 'CnnFlat':
            return self.encode_cnn_flat(obs, last_action)
        elif OBS_ENCODER == 'Factored':
            return self.encode_factored(obs, last_action)
        elif OBS_ENCODER == 'FactoredThenFlattened':
            return self.flatten(self.encode_factored(obs, last_action))

    def flatten(self, obs_list):
        self.observation = np.zeros(self.observation_space)
        id = 0
        for vec in obs_list:
            self.observation[id:id+len(vec)] = vec[:]
            id += len(vec)
        assert id <= self.observation_space
        return self.observation

    def mission_object_and_color_indices(self, obs):
        # Supports missions in the following format:  "blah blah blah <color> <object>"
        # Handled: GoToObj, GoToRedBallGrey, GoToRedBall, GoToLocal, Open, GoToObjMaze, GoTo, Pickup, UnblockPickup, Unlock
        # Not handled: PutNextLocal, PutNext, Synth, SynthLoc, GoToSeq, SynthSeq, GoToImpUnlock, BossLevel
        mission = obs['mission'].split(' ')
        object_id = objtype_index_dict[mission[-1]]
        color_id = color_index_dict[mission[-2]]
        assert (object_id < self.num_object_types)
        assert (color_id < self.num_colors)
        return object_id, color_id

    def encode_factored(self, obs, last_action):
        global_vec_size = self.observation_global_vector_size
        factor_vec_size = self.observation_factor_vector_size

        id = 0
        global_vector = np.zeros(global_vec_size)
        obs_list = [global_vector]

        # Encode the last action.
        if last_action != None:
            global_vector[id + last_action] = 1.
        id += self.action_space

        # Encode the current orientation.
        orientation = obs['direction']
        assert (orientation < self.num_orientations)
        id = encode_1_hot(orientation, self.num_orientations, global_vector, id)

        # Encode the mission.
        inst_vec = self.encoder.encode(obs['mission'])
        for i in range(self.encoder.double_object_vector_length):
            global_vector[id + i] = inst_vec[i]  # There must be a more pythonic way.
        id += self.encoder.double_object_vector_length

        # Encode the walls.
        image = obs['image']
        wall_x, wall_y = self.find_walls(image)
        id = self.encode_x_coordinate(wall_x, global_vector, id)
        id = self.encode_y_coordinate(wall_y, global_vector, id)
        assert(id == global_vec_size)

        # Encode objects objects as separate factor vectors.
        for y in range(7):
            for x in range(7):
                cell = image[x][y]
                object_type = cell[0]
                if object_type >= cell_object_base_index:
                    obs_list.append(self.encode_object_factor(cell, x, y, factor_vec_size, id))

        return obs_list

    def encode_object_factor(self, cell, x, y, factor_vec_size, id):
        id = 0
        factor_vector = np.zeros(factor_vec_size)
        id = encode_1_hot(cell[0]-cell_object_base_index, self.num_object_types, factor_vector, id)
        id = encode_1_hot(cell[1], self.num_colors, factor_vector, id)
        id = encode_1_hot(cell[2], self.num_door_states, factor_vector, id)
        id = self.encode_x_coordinate(x, factor_vector, id)
        id = self.encode_y_coordinate(y, factor_vector, id)
        assert id == factor_vec_size
        return factor_vector

    def find_walls(self, image):
        wall_x = None
        wall_y = None
        for y in range(7):
            c = 0
            for x in range(7):
                cell = image[x][y]
                object_type = cell[0]
                if (object_type == 2) or (object_type == 4):
                    c += 1
                    if c == 2:
                        wall_y = y
                        break
            if c == 2:
                break
        for x in range(7):
            c = 0
            for y in range(7):
                cell = image[x][y]
                object_type = cell[0]
                if (object_type == 2) or (object_type == 4):
                    c += 1
                    if c == 2:
                        wall_x = x
                        break
            if c == 2:
                break
        return wall_x, wall_y

    def encode_x_coordinate(self, x, vector, id):
        if x is None:
            vector[id] = 0
            id += 1 + self.retina_width
            return id
        # In its frame of reference, the agent always sits at the origin, pointing up, in the positive Y direction.
        vector[id] = x - 3.  # X ranging from -3 to +3
        id += 1
        id = encode_1_hot(x, self.retina_width, vector, id)
        return id

    def encode_y_coordinate(self, y, vector, id):
        if y is None:
            vector[id] = 0
            id += 1 + self.retina_width
            return id
        vector[id] = 6. - y  # Y ranging from 0 to 6
        id += 1
        id = encode_1_hot(y, self.retina_width, vector, id)
        return id

    def encode_flat(self, obs, last_action):
        self.observation = np.zeros(self.observation_space)
        id = 0

        # Encode the last action.
        if last_action != None:
            self.observation[id + last_action] = 1.
        id += self.action_space

        # Encode the agent's current direction.
        orientation = obs['direction']
        assert (orientation < self.num_orientations)
        self.observation[id + orientation] = 1.
        id += self.num_orientations

        # Encode the image.
        image = obs['image']
        for row in range(7):
            for col in range(7):
                object_type = image[col][row][0]
                assert (object_type < self.num_cell_types)
                self.observation[id + object_type] = 1.
                id += self.num_cell_types
        for row in range(7):
            for col in range(7):
                color = image[col][row][1]
                assert (color < self.num_colors)
                self.observation[id + color] = 1.
                id += self.num_colors
        for row in range(7):
            for col in range(7):
                door_state = image[col][row][2]
                assert (door_state < self.num_door_states)
                self.observation[id + door_state] = 1.
                id += self.num_door_states

        # Encode the mission.
        inst_vec = self.encoder.encode(obs['mission'])
        for i in range(self.encoder.double_object_vector_length):
            self.observation[id + i] = inst_vec[i]  # There must be a more pythonic way.
        id += self.encoder.double_object_vector_length

        assert (id == self.observation_space)
        return self.observation

    def encode_cnn_flat(self, obs, last_action):
        self.observation = np.zeros(self.observation_space)
        id = 0

        # Encode the last action.
        if last_action != None:
            self.observation[id + last_action] = 1.
        id += self.action_space

        # Encode the agent's current direction.
        orientation = obs['direction']
        assert (orientation < self.num_orientations)
        self.observation[id + orientation] = 1.
        id += self.num_orientations

        # Encode the mission.
        inst_vec = self.encoder.encode(obs['mission'])
        for i in range(self.encoder.double_object_vector_length):
            self.observation[id + i] = inst_vec[i]  # There must be a more pythonic way.
        id += self.encoder.double_object_vector_length

        assert (id == self.observation_space)
        return [self.observation, obs['image']]

    def reset(self, repeat=False, episode_id = None):
        if episode_id is not None:
            self.env.seed(episode_id + 1000000)
        obs = self.env.reset()
        return self.assemble_current_observation(obs)

    def translate_key_to_action(self, key):
        action = -1
        if key == 'Up':
            action = 1
        elif key == 'Left':
            action = 2
        elif key == 'Down':
            action = 3
        elif key == 'Right':
            action = 0
        else:
            print(("Key not found"))
        return action

    def step(self, action):
        obs, reward, done, env_info = self.env.step(action)
        if BINARY_REWARD:
            if reward > 0.:
                reward = 1.
        self.update_online_test_sums(reward, done)
        ret = self.assemble_current_observation(obs, action), reward, done
        return ret

    # Online test support.
    # In online testing, each training step is also used for testing the agent.
    # This is permissible only in the infinite data case (like games and simulations), where separate train and test sets are not required.
    # The environment must define its own (real-valued) online test metric, which may be as simple as accumulated reward.
    # To smooth out the reported test results, online testing is divided into contiguous reporting periods of many time steps.

    def reset_online_test_sums(self):
        # Called only by the environment itself.
        self.step_sum = 0
        self.reward_sum = 0.
        self.num_episodes = 0
        self.num_successful_episodes = 0
        self.need_to_reset_sums = False

    def update_online_test_sums(self, reward, done):
        # Called only by the environment itself.
        if self.need_to_reset_sums:
            # Another thread recently called reduce_online_test_sums(), so the previous counts are stale.
            self.reset_online_test_sums()
        # If another thread happens to call reduce_online_test_sums near this point in time,
        # one sample from this agent might get dropped. But that's a small price to pay to avoid locking.
        self.step_sum += 1
        self.num_local_steps += 1
        self.reward_sum += reward
        if done:
            self.num_episodes += 1
            success = 0.
            if reward > 0.:
                success = 1.
            self.num_successful_episodes += success
            # For success rate
            self.success_sum += success
            if len(self.success_buf) == self.rsr_win_len:
                self.success_sum -= self.success_buf[self.success_buf_id]
                self.success_buf[self.success_buf_id] = success
            else:
                self.success_buf.append(success)
            self.success_buf_id = (self.success_buf_id + 1) % self.rsr_win_len
            self.running_success_rate = self.success_sum / len(self.success_buf)

    def report_online_test_metric(self):
        if HELDOUT_TESTING:
            return self.heldout_test()
        else:
            return self.online_test()

    def heldout_test(self):
        # Called by the reporting manager only.

        # Evaluate the current model on a random set of environments.
        agent = self.test_agent
        env = self.test_environment
        max_episodes = NUM_TEST_EPISODES
        num_solutions_required = int(max_episodes * SUCCESS_RATE_THRESHOLD)
        num_failures_allowed = max_episodes - num_solutions_required
        num_solved = 0
        num_failed = 0
        for i in range(max_episodes):
            agent.reset_state()
            observation = env.reset()
            done = False
            while not done:
                action = agent.step(observation)
                observation, reward, done = env.step(action)
                agent.last_reward = reward
            if reward > 0.:
                num_solved += 1
            else:
                num_failed += 1
            if num_failed > num_failures_allowed:
                break
        sr = num_solved / (num_solved + num_failed)

        # Update the final reported metrics.
        if sr < SUCCESS_RATE_THRESHOLD:
            # Extrapolate forward to the point where the threshold would be crossed.
            self.sum_sr += sr
            self.num_sr += 1
            mean_sr = self.sum_sr / self.num_sr
            se = SUCCESS_RATE_THRESHOLD * self.num_local_steps / max(0.1/max_episodes, mean_sr)  # Should be global steps
        else:
            # Interpolate back to the point where the threshold was crossed.
            full_delta_steps = self.num_local_steps - self.prev_num_local_steps
            sufficient_delta_success = SUCCESS_RATE_THRESHOLD - self.prev_sr
            full_delta_success = sr - self.prev_sr
            se = self.prev_num_local_steps + full_delta_steps * sufficient_delta_success / full_delta_success
        self.prev_sr = sr
        self.prev_num_local_steps = self.num_local_steps

        # Assemble the tuple to be returned.
        metrics = []
        metrics.append((sr, "{:6.4f}".format(sr), "Success rate"))
        metrics.append((se, "{:2.0f}".format(se), "Projected sample efficiency"))
        ret = (self.step_sum, self.num_episodes, sr, metrics, sr >= SUCCESS_RATE_THRESHOLD)

        # Reset the global sums.
        self.reset_online_test_sums()

        return ret

    def online_test(self):
        # Called by the reporting manager only.
        # Combine the online sums from all threads.
        sum_RSR = self.running_success_rate
        num_RSR = 1

        # Calculate the final metric for this test period.
        # steps_per_episode = self.step_sum / self.num_episodes
        reward_per_step = self.reward_sum / self.step_sum
        success_rate = self.num_successful_episodes / self.num_episodes
        running_success_rate = sum_RSR / num_RSR

        # Update the final reported metrics.
        if not self.rsr_threshold_reached:
            if running_success_rate >= SUCCESS_RATE_THRESHOLD:
                self.rsr_threshold_reached = True
                full_delta_steps = self.num_local_steps - self.sample_efficiency
                sufficient_delta_success = SUCCESS_RATE_THRESHOLD - self.prev_sr
                full_delta_success = running_success_rate - self.prev_sr
                self.sample_efficiency += full_delta_steps * sufficient_delta_success / full_delta_success
            else:
                self.sample_efficiency = self.num_local_steps
                self.prev_sr = running_success_rate

        # Assemble the tuple to be returned.
        #   1. The number of steps in the period. (This will be a bit different for each running thread.)
        #   2. The actual metric value (must be negated if lower is better).
        #   3. A string containing the formatted metric.
        #   4. A string containing the metric's units for display.
        # ret = (self.step_sum, -steps_per_episode, "{:6.1f}".format(steps_per_episode), "steps per episode")
        # ret = (self.step_sum, reward_per_step, "{:6.3f}".format(reward_per_step), "reward per step")
        # ret = (self.step_sum, success_rate, "{:6.4f}".format(success_rate), "success rate")
        # ret = (self.step_sum, reward_per_step, "{:6.4f}  {:8.5f}  {:6.1f}".format(success_rate, reward_per_step, steps_per_episode), "     success rate, reward per step, steps per episode")

        # ret = (self.step_sum, reward_per_step, "{:6.4f} success rate  {:8.5f} reward per step  {:6.1f} steps per episode".format(success_rate, reward_per_step, steps_per_episode))

        metrics = []
        metrics.append((reward_per_step, "{:7.5f}".format(reward_per_step), "reward"))
        metrics.append((success_rate, "{:6.4f}".format(success_rate), "SR"))
        metrics.append((running_success_rate, "{:6.4f}".format(running_success_rate), "Success rate"))
        metrics.append((self.sample_efficiency, "{:2.0f}".format(self.sample_efficiency), "Projected sample efficiency"))
        # metrics.append((running_success_rate, "{:6.4f}".format(running_success_rate), "metric 1 current Success rate"))
        # metrics.append((self.sample_efficiency, "{:2.0f}".format(self.sample_efficiency), "metric 1 final Sample efficiency"))
        ret = (self.step_sum, self.num_episodes, reward_per_step, metrics, False)

        # Reset the global sums.
        self.reset_online_test_sums()

        return ret
