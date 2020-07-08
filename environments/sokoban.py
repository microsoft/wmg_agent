# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Adapted from https://github.com/mpSchrader/gym-sokoban
# Max-Philipp B. Schrader, 2018.

import os
from os import listdir
from os import path
from os.path import isfile, join
import random
import numpy as np
from utils.graph import Graph
from utils.graph import Entity
import zipfile

from utils.spec_reader import spec
SOKOBAN_MAX_STEPS = spec.val("SOKOBAN_MAX_STEPS")
SOKOBAN_DIFFICULTY = spec.val("SOKOBAN_DIFFICULTY")
SOKOBAN_SPLIT = spec.val("SOKOBAN_SPLIT")
SOKOBAN_ROOM_OVERRIDE = spec.val("SOKOBAN_ROOM_OVERRIDE")
SOKOBAN_BOXES_REQUIRED = spec.val("SOKOBAN_BOXES_REQUIRED")
SOKOBAN_OBSERVATION_FORMAT = spec.val("SOKOBAN_OBSERVATION_FORMAT")
SOKOBAN_REWARD_PER_STEP = spec.val("SOKOBAN_REWARD_PER_STEP")
SOKOBAN_REWARD_SUCCESS = spec.val("SOKOBAN_REWARD_SUCCESS")

PIXELS_PER_TILE = 6  # Each tile is one pixel in the original Sokoban images, 8 pixels per cell, and 10x10 cells in a puzzle.
TILES_PER_CELL = 8
PUZZLE_SCALE = PIXELS_PER_TILE * TILES_PER_CELL
PUZZLE_SIZE = 10

# Cell state codes
WALL = 0
FLOOR = 1
TARGET = 2
BOX_ON_TARGET = 3
BOX_ON_FLOOR = 4
AGENT_ON_FLOOR = 5
AGENT_ON_TARGET = 6

CHANGE_COORDINATES = {
    0: (-1, 0), # 0: Move up
    1: (1, 0),  # 1: Move down
    2: (0, -1), # 2: Move left
    3: (0, 1)   # 3: Move right
}

ACTION_NAMES = ['Ponder', 'Up', 'Down', 'Left', 'Right']


class Sokoban_Env(object):
    def __init__(self, seed):
        self.rand = random.Random(seed)
        self.num_boxes = 4
        self.boxes_on_target = 0
        self.max_steps_per_episode = SOKOBAN_MAX_STEPS * SOKOBAN_BOXES_REQUIRED / self.num_boxes

        # Penalties and Rewards
        self.penalty_for_step = SOKOBAN_REWARD_PER_STEP
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = SOKOBAN_REWARD_SUCCESS * SOKOBAN_BOXES_REQUIRED / self.num_boxes
        self.reward_last = 0

        # Other Settings
        self.action_space = 5
        if SOKOBAN_OBSERVATION_FORMAT == 'grid':
            self.observation_space = 400
            self.observation = np.zeros((10,10,4), dtype=np.uint8)
        elif SOKOBAN_OBSERVATION_FORMAT == 'factored':
            self.observation = Graph()
            self.num_factor_positions = 15
            self.factor_position_offset = (self.num_factor_positions - 1) // 2
            self.cell_factor_size = 0
            self.cell_factor_size += 2 * (self.num_factor_positions + 1) # Cell position.
            self.cell_factor_size += 3 # Cell identity.
            self.cell_factor_size += 4
            self.core_obs_size = self.action_space + 1
            self.core_obs_size += 1 + 4  # On target, four walls.
            self.observation.entity_type_sizes.append(self.core_obs_size)
            self.observation.entity_type_sizes.append(self.cell_factor_size)
            self.observation_space = self.observation
        self.use_display = False
        self.num_cols_or_rows = PUZZLE_SIZE
        self.pix_per_cell = PUZZLE_SCALE
        self.wid = self.num_cols_or_rows
        self.x_orig = -self.wid * self.pix_per_cell / 2
        self.y_orig = self.wid * self.pix_per_cell / 2
        self.agent_col = None
        self.agent_row = None
        self.reset_online_test_sums()
        self.score = 0.
        self.reward = 0.
        self.action = None

        # Cell channel encodings.
        self.encodings = np.array(((1,0,0,0),(0,0,0,0),(0,1,0,0),(0,1,1,0),(0,0,1,0),(0,0,0,1),(0,1,0,1)), dtype=np.uint8)

        self.total_steps = 0
        self.total_reward = 0.
        self.total_episodes = 0
        self.total_episodes_won = 0.

    def reset(self, repeat=False, episode_id = None):
        self.train_data_dir = os.path.join('data', 'boxoban-levels-master', SOKOBAN_DIFFICULTY, SOKOBAN_SPLIT)
        self.select_room(repeat, episode_id)
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.agent_col = None
        self.agent_row = None
        self.action = None
        self.draw()
        return self.assemble_current_observation(0, 0.)

    def step(self, action):
        self.action = action
        self.num_env_steps += 1
        self.new_box_position = None
        self.old_box_position = None
        moved_box = False
        if action == 0:
            moved_player = False
        else:
            moved_player, moved_box = self._push(action)
        self._calc_reward()
        self.done = self._check_if_done()
        self.reward = self.reward_last
        self.score += self.reward

        self.update_display()
        self.update_online_test_sums(self.reward, self.done)
        ret = self.assemble_current_observation(action, self.reward), self.reward, self.done
        self.draw_text()
        return ret

    def add_wall_bits(self, row, col, vec, id):
        if self.cell_state(row, col - 1) == WALL:
            vec[id + 0] = 1.
        if self.cell_state(row - 1, col) == WALL:
            vec[id + 1] = 1.
        if self.cell_state(row, col + 1) == WALL:
            vec[id + 2] = 1.
        if self.cell_state(row + 1, col) == WALL:
            vec[id + 3] = 1.
        return id + 4

    def assemble_current_observation(self, action, reward):
        if SOKOBAN_OBSERVATION_FORMAT == 'grid':
            for row in range(10):
                for col in range(10):
                    state = self.cell_state(row, col)
                    self.observation[row][col][:] = self.encodings[state]
        elif SOKOBAN_OBSERVATION_FORMAT == 'factored':
            self.assemble_observation_graph(action, reward)
        return self.observation

    def assemble_observation_graph(self, action, reward):
        self.observation.clear()
        self.agent_row = self.player_position[0]
        self.agent_col = self.player_position[1]

        # Handle the core.
        core_entity = Entity(0, self.agent_row, self.agent_col)
        core_entity.data = np.zeros(self.core_obs_size, np.float32)
        i = 0
        core_entity.data[i + action] = 1.
        i += self.action_space
        core_entity.data[i] = reward
        i += 1
        if self.cell_state(self.agent_row, self.agent_col) == AGENT_ON_TARGET:
            core_entity.data[i] = 1.
        i += 1
        i = self.add_wall_bits(self.agent_row, self.agent_col, core_entity.data, i)
        assert i == self.core_obs_size
        self.observation.entities.append(core_entity)

        # Handle the factors/percepts.
        for row in range(self.num_cols_or_rows):
            for col in range(self.num_cols_or_rows):
                state = self.cell_state(row, col)
                if (state == WALL) or (state == AGENT_ON_TARGET) or (state == AGENT_ON_FLOOR):
                    continue
                factor = Entity(1, row, col)
                factor.data = np.zeros(self.cell_factor_size, np.float32)
                i = 0

                # Encode the cell's row and col positions with respect to the agent.
                i = self.encode_position(col - self.agent_col, factor.data, i)
                i = self.encode_position(row - self.agent_row, factor.data, i)

                # Cell identity.
                bits = self.encodings[state]
                factor.data[i] = bits[1]
                i += 1
                factor.data[i] = bits[2]
                i += 1
                factor.data[i] = bits[3]
                i += 1

                i = self.add_wall_bits(row, col, factor.data, i)

                assert i == self.cell_factor_size
                self.observation.entities.append(factor)

    def encode_position(self, pos, buf, i):
        buf[i] = pos / self.factor_position_offset
        i += 1
        buf[i + self.factor_position_offset + pos] = 1.
        i += self.num_factor_positions
        return i

    def x_pix_from_col(self, col):
        return self.x_orig + col * self.pix_per_cell

    def y_pix_from_row(self, row):
        return self.y_orig - row * self.pix_per_cell

    def draw_line(self, x1, y1, x2, y2, color):
        self.t.color(color)
        self.t.setpos(x1, y1)
        self.t.pendown()
        self.t.goto(x2, y2)
        self.t.penup()

    def draw_rect(self, x1, y1, x2, y2, color):
        self.t.pensize(1)
        self.t.begin_fill()
        self.t.color(color)
        self.t.setpos(x1, y1)
        self.t.pendown()
        self.t.goto(x1, y2)
        self.t.goto(x2, y2)
        self.t.goto(x2, y1)
        self.t.goto(x1, y1)
        self.t.penup()
        self.t.end_fill()

    def cell_state(self, row, col):
        state = self.room_state[row][col]
        if (state == AGENT_ON_FLOOR) and (self.room_fixed[row][col] == TARGET):
            state = AGENT_ON_TARGET  # Patch up this one missing case.
        return state

    def render_cell(self, row, col):
        state = self.cell_state(row, col)
        draw_ball = False
        draw_x = False
        rad = self.pix_per_cell / 2
        wall_color = 'gray'
        floor_color = 'black'
        target_color = '#a00000'
        if state == WALL:
            background_color = wall_color
        elif state == FLOOR:
            background_color = floor_color
        elif state == TARGET:
            background_color = target_color
        elif state == BOX_ON_TARGET:
            background_color = target_color
            draw_x = True
        elif state == BOX_ON_FLOOR:
            background_color = floor_color
            draw_x = True
        elif state == AGENT_ON_FLOOR:
            background_color = floor_color
            draw_ball = True
        elif state == AGENT_ON_TARGET:
            background_color = target_color
            draw_ball = True
        x = self.x_pix_from_col(col)
        y = self.y_pix_from_row(row)
        self.t.color(background_color)
        self.draw_rect(x - rad, y - rad, x + rad - 1, y + rad - 1, background_color)
        if draw_ball:
            self.t.setpos(x, y)
            self.t.pensize(rad)
            self.t.dot(self.pix_per_cell, '#00ff00')
        if draw_x:
            line_wid = 3
            margin = line_wid - 1
            self.t.pensize(line_wid)
            xl = x-rad+margin
            xr = x+rad-margin
            yt = y+rad-margin
            yb = y-rad+margin
            box_color = 'yellow'
            self.draw_line(xl, yb, xr, yt, box_color)
            self.draw_line(xl, yt, xr, yb, box_color)
            self.draw_line(xl, yb, xl, yt, box_color)
            self.draw_line(xl, yt, xr, yt, box_color)
            self.draw_line(xr, yt, xr, yb, box_color)
            self.draw_line(xr, yb, xl, yb, box_color)

    def draw_text(self):
        if not self.use_display:
            return
        rad = self.num_cols_or_rows * self.pix_per_cell / 2

        # Draw text below.
        self.draw_rect(-rad, -rad - 27, -rad + 800, -rad + 12, 'light gray')
        self.t.color('black')
        self.t.setpos(-rad, -rad - 26)
        self.t.write('Last reward:  {:4.1f}       Total reward:{:5.1f}'.format(self.reward, self.score), font=("FixedSys", 16, "normal"))
        self.t.setpos(-rad, -rad - 6)
        if self.action is None:
            action_name = 'None'
        else:
            action_name = ACTION_NAMES[self.action]
        self.t.write('Last action:  {:10s}  Steps taken:  {}'.format(action_name, self.num_env_steps), font=("FixedSys", 16, "normal"))

        # self.draw_rect(-rad, rad + 71, -rad + 800, rad + 110, 'light gray')
        # self.t.setpos(-rad, rad + 70)
        # self.t.color('black')
        # self.t.write('Observation:  {}'.format(self.observation), font=("Arial", 16, "normal"))

    def draw(self):
        if self.use_display:
            self.t.shape('square')
            self.agent_row = self.player_position[0]
            self.agent_col = self.player_position[1]
            for row in range(self.num_cols_or_rows):
                for col in range(self.num_cols_or_rows):
                    self.render_cell(row, col)
            self.t._update()
            self.draw_text()

    def update_display(self):
        if self.use_display:
            # Did the agent just move?
            old_agent_row = self.agent_row
            old_agent_col = self.agent_col
            new_agent_row = self.player_position[0]
            new_agent_col = self.player_position[1]
            if (new_agent_row != old_agent_row) or (new_agent_col != old_agent_col):
                # Yes. Render the two cells involved.
                self.render_cell(old_agent_row, old_agent_col)
                self.render_cell(new_agent_row, new_agent_col)
                # Also render the next cell if it contains a box, in case it was just pushed there.
                next_row = new_agent_row + (new_agent_row - old_agent_row)
                next_col = new_agent_col + (new_agent_col - old_agent_col)
                if (next_row >= 0) and (next_row < self.num_cols_or_rows) and \
                    (next_col >= 0) and (next_col < self.num_cols_or_rows):
                    if (self.cell_state(next_row, next_col) == BOX_ON_FLOOR) or \
                        (self.cell_state(next_row, next_col) == BOX_ON_TARGET):
                        self.render_cell(next_row, next_col)
                # Update the agent location.
                self.agent_row = new_agent_row
                self.agent_col = new_agent_col

    def translate_key_to_action(self, key):
        key = key.lower()
        action = -1
        if key == 'up':
            action = 1
        elif key == 'left':
            action = 3
        elif key == 'down':
            action = 2
        elif key == 'right':
            action = 4
        elif key == 'space':
            action = None
        elif key == 'delete':
            self.reset()
        elif key == 'r':
            self.reset(True)
        elif key == 'n':
            self.reset()
        else:
            print(("Key not found"))
        #print("action = {}".format(action))
        return action

    def select_room(self, repeat=False, episode_id = None):
        if not repeat:
            generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
            if self.total_steps == 0:
                print("{} puzzle files found.".format(len(generated_files)))
            generated_files.sort()
            if SOKOBAN_ROOM_OVERRIDE is None:
                if episode_id is None:
                    map_file = self.rand.choice(generated_files)
                else:
                    map_file = generated_files[episode_id // 1000]
            else:
                map_file = generated_files[0]
            source_file = join(self.train_data_dir, map_file)
            maps = []
            current_map = []
            with open(source_file, 'r') as sf:
                for line in sf.readlines():
                    if ';' in line and current_map:
                        maps.append(current_map)
                        current_map = []
                    if '#' == line[0]:
                        current_map.append(line.strip())
            maps.append(current_map)
            if SOKOBAN_ROOM_OVERRIDE is None:
                if episode_id is None:
                    self.selected_map = self.rand.choice(maps)
                else:
                    self.selected_map = maps[episode_id % 1000]
            else:
                self.selected_map = maps[SOKOBAN_ROOM_OVERRIDE]
        self.room_fixed, self.room_state = self.generate_room(self.selected_map)

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []
        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []
            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)
                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)
                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)
                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)
                else:
                    room_f.append(1)
                    room_s.append(1)
            room_fixed.append(room_f)
            room_state.append(room_s)
        return np.array(room_fixed), np.array(room_state)

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False

        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]
            return True
        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        # game_won = self._check_if_enough_boxes_on_target()
        self.game_won = (current_boxes_on_target == SOKOBAN_BOXES_REQUIRED)
        if self.game_won:
            self.reward_last += self.reward_finished

        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        #return self._check_if_enough_boxes_on_target() or self._check_if_maxsteps()
        return self.game_won or self._check_if_maxsteps()

    def _check_if_enough_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps_per_episode == self.num_env_steps)

    # Online test support.
    # In online testing, each training step is first used for testing the agent.
    # The environment must define its own (real-valued) online test metric, which may be as simple as accumulated reward.
    # To smooth out the reported test results, online testing is divided into contiguous reporting periods of many time steps.

    def reset_online_test_sums(self):
        # Called only by the environment itself.
        self.step_sum = 0
        self.reward_sum = 0.
        self.num_episodes = 0
        self.num_episodes_won = 0

    def update_online_test_sums(self, reward, done):
        # Called only by the environment itself.
        self.step_sum += 1
        self.reward_sum += reward
        self.total_steps += 1
        self.total_reward += reward
        if done:
            self.num_episodes += 1
            self.total_episodes += 1
            if reward > 0.:
                self.num_episodes_won += 1
                self.total_episodes_won += 1

    def report_online_test_metric(self):
        # Called by the Worker only.

        # Calculate the final metric for this test period.
        self.reward_per_step = self.reward_sum / self.step_sum
        if self.num_episodes > 0.:
            self.success_rate = 100. * self.num_episodes_won / self.num_episodes
        else:
            self.success_rate = 0.

        # Assemble the tuple to be returned.
        metrics = []
        metrics.append((self.reward_per_step, "{:7.5f}".format(self.reward_per_step), "reward"))
        metrics.append((self.success_rate, "{:7.3f}".format(self.success_rate), "success rate"))
        ret = (self.step_sum, self.num_episodes, self.success_rate, metrics, False)

        # Reset the global sums.
        self.reset_online_test_sums()

        return ret
