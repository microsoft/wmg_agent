# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import time
import datetime
import pytz
import platform
import torch
import numpy as np

from utils.spec_reader import spec
TYPE_OF_RUN = spec.val("TYPE_OF_RUN")
ENV = spec.val("ENV")
LOAD_MODEL_FROM = spec.val("LOAD_MODEL_FROM")
SAVE_MODELS_TO = spec.val("SAVE_MODELS_TO")
ENV_RANDOM_SEED = spec.val("ENV_RANDOM_SEED")
AGENT_RANDOM_SEED = spec.val("AGENT_RANDOM_SEED")
REPORTING_INTERVAL = spec.val("REPORTING_INTERVAL")
TOTAL_STEPS = spec.val("TOTAL_STEPS")
ANNEAL_LR = spec.val("ANNEAL_LR")
if ANNEAL_LR:
    ANNEALING_START = spec.val("ANNEALING_START")

from agents.a3c import A3cAgent


class Worker(object):
    def __init__(self):
        torch.manual_seed(AGENT_RANDOM_SEED)
        self.start_time = time.time()
        self.heldout_testing = False
        self.environment = self.create_environment(ENV_RANDOM_SEED)
        self.agent = A3cAgent(self.observation_space, self.action_space)
        if self.heldout_testing:
            self.environment.test_environment = self.create_environment(ENV_RANDOM_SEED + 1000000)
            self.environment.test_agent = A3cAgent(self.observation_space, self.action_space)
            self.environment.test_agent.network = self.agent.network
        self.step_num = 0
        self.total_reward = 0.
        self.num_steps = 0
        self.num_episodes = 0
        self.action = None
        self.reward = 0.
        self.done = False
        self.best_metric_value = np.NINF
        self.t = None
        self.output_filename = None
        if LOAD_MODEL_FROM is not None:
            self.agent.load_model(LOAD_MODEL_FROM)

    def execute(self):
        if TYPE_OF_RUN == 'train':
            self.train()
        elif TYPE_OF_RUN == 'test':
            self.test()
        elif TYPE_OF_RUN == 'test_episodes':
            self.test_episodes()
        elif TYPE_OF_RUN == 'render':
            self.render()
        else:
            print('Run type "{}" not recognized.'.format(TYPE_OF_RUN))

    def create_results_output_file(self):
        # Output files are written to the results directory.
        server_name = '{}'.format(platform.uname()[1])
        datetime_string = pytz.utc.localize(
            datetime.datetime.utcnow()).astimezone(pytz.timezone("PST8PDT")).strftime("%y-%m-%d_%H-%M-%S")
        code_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(code_dir), 'results')
        self.output_filename = os.path.join(results_dir, 'out_{}_{}.txt'.format(server_name, datetime_string))
        file = open(self.output_filename, 'w')
        file.close()

    def create_environment(self, seed=None):
        # Each new environment should be listed here.
        if ENV == "Pathfinding_Env":
            from environments.pathfinding import Pathfinding_Env
            environment = Pathfinding_Env(seed)
        elif ENV == "BabyAI_Env":
            from environments.babyai import BabyAI_Env
            environment = BabyAI_Env(seed)
            HELDOUT_TESTING = spec.val("HELDOUT_TESTING")
            self.heldout_testing = HELDOUT_TESTING
        elif ENV == "Sokoban_Env":
            from environments.sokoban import Sokoban_Env
            environment = Sokoban_Env(seed)
        else:
            print("Environment {} not found.".format(ENV))
            exit(0)
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        return environment

    def output(self, sz):
        if self.output_filename:
            print(sz)
            file = open(self.output_filename, 'a')
            file.write(sz + '\n')
            file.close()

    def train(self):
        self.create_results_output_file()
        self.init_episode()
        spec.output_to_file(self.output_filename)
        self.take_n_steps(TOTAL_STEPS, None, True)
        self.output("{:8.6f} overall reward per step".format(self.total_reward / self.step_num))

    def test(self):
        self.create_results_output_file()
        self.init_episode()
        spec.output_to_file(self.output_filename)
        self.take_n_steps(TOTAL_STEPS, None, False)
        self.output("{:8.6f} overall reward per step".format(self.total_reward / self.step_num))

    def test_episodes(self):
        # Test the model on all episodes.
        # Success is determined by positive reward on the final step,
        # which works for BabyAI and Sokoban, but is not appropriate for many environments.
        NUM_EPISODES_TO_TEST = spec.val("NUM_EPISODES_TO_TEST")
        MIN_FINAL_REWARD_FOR_SUCCESS = spec.val("MIN_FINAL_REWARD_FOR_SUCCESS")
        self.create_results_output_file()
        spec.output_to_file(self.output_filename)
        num_wins = 0
        num_episodes_tested = 0
        self.output("Testing on {} episodes.".format(NUM_EPISODES_TO_TEST))
        start_time = time.time()
        for episode_id in range(NUM_EPISODES_TO_TEST):
            torch.manual_seed(AGENT_RANDOM_SEED)
            final_reward, steps = self.test_on_episode(episode_id)
            if final_reward >= MIN_FINAL_REWARD_FOR_SUCCESS:
                num_wins += 1
            num_episodes_tested += 1
            if (num_episodes_tested % (NUM_EPISODES_TO_TEST / 10) == 0):
                self.output('{:4d} / {:5d}  =  {:5.1f}%'.format(num_wins, num_episodes_tested, 100.0 * num_wins / num_episodes_tested))
        self.output("Time: {:3.1f} min".format((time.time() - start_time)/60.))
        self.output("Success rate = {} / {} episodes = {:5.1f}%".format(num_wins, num_episodes_tested, 100.0 * num_wins / num_episodes_tested))

    def test_on_episode(self, episode_id):
        steps_taken = 0
        action = None
        self.init_episode(episode_id)
        while True:
            action = self.agent.step(self.observation)
            self.observation, reward, done = self.environment.step(action)
            steps_taken += 1
            if done:
                return reward, steps_taken

    def render(self):
        self.init_episode()
        self.init_turtle()
        self.environment.use_display = True
        self.draw()
        self.wnd.mainloop()  # After this call, the program runs until the user closes the window.

    def init_turtle(self):
        import turtle
        self.t = turtle.Turtle()
        self.environment.t = self.t
        self.t.hideturtle()
        self.t.speed('fastest')
        self.t.screen.tracer(0, 0)
        self.t.penup()
        self.wnd = turtle.Screen()
        # self.wnd.setup(1902, 990, 0, 0)
        # self.wnd.bgcolor("#808080")
        self.wnd.bgcolor("light gray")
        self.wnd.setup(1000, 800, 0, 0)
        # Cancel (the Break key), BackSpace, Tab, Return(the Enter key), Shift_L (any Shift key),
        # Control_L (any Control key), Alt_L (any Alt key), Pause, Caps_Lock, Escape,
        # Prior (Page Up), Next (Page Down), End, Home, Left, Up, Right, Down, Print, Insert, Delete,
        # F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Num_Lock, Scroll_Lock
        self.wnd.onkey(self.on_up_key, "Up")
        self.wnd.onkey(self.on_down_key, "Down")
        self.wnd.onkey(self.on_left_key, "Left")
        self.wnd.onkey(self.on_right_key, "Right")
        self.wnd.onkey(self.on_space_key, " ")
        self.wnd.onkey(self.on_escape_key, "Escape")
        self.wnd.onkey(self.on_delete_key, "Delete")
        self.wnd.onkey(self.on_r_key, "r")
        self.wnd.onkey(self.on_n_key, "n")
        self.wnd.listen()

    def on_up_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Up'), False)

    def on_down_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Down'), False)

    def on_left_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Left'), False)

    def on_right_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Right'), False)

    def on_space_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Space'), False)

    def on_delete_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Delete'), False)

    def on_r_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('r'), False)

    def on_n_key(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('n'), False)

    def on_escape_key(self):
        exit(0)

    def take_n_steps(self, max_steps, manual_action_override, train_agent):
        if manual_action_override == -1:
            return
        self.action = None
        for i in range(max_steps):
            # Get an action.
            if manual_action_override is None:
                self.action = self.agent.step(self.observation)
            else:
                self.action = manual_action_override

            # Apply the action to the environment.
            self.observation, self.reward, self.done = self.environment.step(self.action)

            # Let the agent adapt to the effects of the action.
            if (manual_action_override is None) and train_agent:
                self.agent.adapt(self.reward, self.done, self.observation)

            # Report status periodically.
            terminate = self.monitor(self.reward)
            if terminate:
                break

            # After an episode ends, start a new one.
            if self.done:
                self.init_episode()

    def init_episode(self, episode_id = None):
        self.agent.reset_state()
        self.observation = self.environment.reset(repeat=False, episode_id=episode_id)
        self.draw()

    def draw(self):
        if self.t:
            self.t.clear()
            self.environment.draw()

    def monitor(self, reward):
        self.step_num += 1
        self.total_reward += reward
        terminate = False

        # Report results periodically.
        if (self.step_num % REPORTING_INTERVAL) == 0:
            if ANNEAL_LR:
                if self.step_num > ANNEALING_START:
                    self.agent.anneal_lr()
            sz = "{:10.2f} sec  {:12,d} steps".format(
                time.time() - self.start_time, self.step_num)
            if hasattr(self.environment, 'report_online_test_metric'):
                num_steps, num_episodes, metric_value, metric_tuple, terminate = \
                    self.environment.report_online_test_metric()
                self.num_steps += num_steps
                self.num_episodes += num_episodes

                saved = False
                # Is this the best metric so far?
                if metric_value > self.best_metric_value:
                    self.best_metric_value = metric_value
                    if SAVE_MODELS_TO is not None:
                        self.agent.save_model(SAVE_MODELS_TO)
                        saved = True

                # Report one line.
                sz += "  {:11,d} episodes".format(self.num_episodes)
                for metric in metric_tuple:
                    formatted_value = metric[1]
                    label = metric[2]
                    sz += "      {} {}".format(formatted_value, label)
                if saved:
                    sz += "  SAVED"
                else:
                    sz += "       "
            self.output(sz)

        return terminate
