# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import random

from utils.spec_reader import spec
NUM_PATTERNS = spec.val("NUM_PATTERNS")

PATTERN_LENGTH = 7


class Link(object):
    def __init__(self, graph, src, tar):
        self.graph = graph
        self.src = src
        self.tar = tar
        assert (graph.path_len[src][tar] == 0)
        graph.path_len[src][tar] = 1

    def output(self):
        print("Link  {} -> {}".format(self.src, self.tar))


class Graph(object):
    def __init__(self, rand):
        self.rand = rand

    def reset(self):
        # Links are represented by objects, but nodes are represented by indices.
        # Keep in mind that the nodes in this graph do not (necessarily) correspond to transformer nodes.

        # Allocate the pattern set.
        self.patterns = [[0. for element in range(PATTERN_LENGTH)] for pattern in range(NUM_PATTERNS)]

        # Define the node patterns.
        for i, p in enumerate(self.patterns):
            for j in range(PATTERN_LENGTH):
                p[j] = self.rand.random() * 2. - 1.

        # Allocate the matrix for tracking path lengths. Zero indicates absence of a path.
        self.path_len = [[0 for src in range(NUM_PATTERNS)] for tar in range(NUM_PATTERNS)]

        # Initialize the graph with one node, and no links.
        self.num_nodes = 1
        self.links = []

    def add_node(self):
        # Add one node, linked to exactly one existing node in the graph, to build a polytree (singly-connected DAG).
        new_node = self.num_nodes

        # As a connection point, choose a random node that's already in the graph.
        cnct_node = self.rand.randint(0, self.num_nodes-1)

        # Choose a random direction for the connecting link.
        if self.rand.randint(0, 1) == 1:
            # Run the link from the new node to the connection node in the graph,
            link = Link(self, new_node, cnct_node)
            # and update path lengths to other nodes in the graph.
            for node in range(self.num_nodes):
                if (node != cnct_node) and (self.path_len[cnct_node][node] > 0):
                    self.path_len[new_node][node] = self.path_len[cnct_node][node] + 1
        else:
            # Run the link from the connection node in the graph to the new node,
            link = Link(self, cnct_node, new_node)
            # and update path lengths to other nodes in the graph.
            for node in range(self.num_nodes):
                if (node != cnct_node) and (self.path_len[node][cnct_node] > 0):
                    self.path_len[node][new_node] = self.path_len[node][cnct_node] + 1
        self.links.append(link)
        self.num_nodes += 1

    def mean_path_ratio(self):
        num_paths = 0
        num_node_pairs = 0
        for s in range(NUM_PATTERNS):
            for t in range(NUM_PATTERNS):
                if s != t:
                    if self.path_len[s][t] > 0:
                        num_paths += 1
                    num_node_pairs += 1
        return num_paths / num_node_pairs

    def mean_path_len(self):
        path_len_sum = 0
        num_paths = 0
        for s in range(NUM_PATTERNS):
            for t in range(NUM_PATTERNS):
                if s != t:
                    if self.path_len[s][t] > 0:
                        path_len_sum += self.path_len[s][t]
                        num_paths += 1
        return path_len_sum / num_paths

    def max_path_len(self):
        max_len = 0
        for s in range(NUM_PATTERNS):
            for t in range(NUM_PATTERNS):
                if s != t:
                    if self.path_len[s][t] > max_len:
                        max_len = self.path_len[s][t]
        return max_len


class Pathfinding_Env(object):
    def __init__(self, seed=None):
        if seed:
            self.seed = seed
        else:
            self.seed = 257
        self.rand = random.Random(self.seed)

        self.use_display = False
        self.reward = 0
        self.done = False
        self.correct_output = 0  # This initial value doesn't matter.
        self.action_space = 2
        self.observation_space = 2 * PATTERN_LENGTH + 1
        self.observation = np.zeros(self.observation_space)

        #self.test_graphs()
        self.graph = Graph(self.rand)
        self.reset_online_test_sums()
        self.cumulative_counts = [[0., 0.] for pattern in range(NUM_PATTERNS-1)]

        self.total_reward = 0.
        self.total_steps = 0

    def test_graphs(self):
        num_graphs = 10000
        sum_ratios = 0.
        sum_lens = 0.
        sum_max_lens = 0.
        for i in range(num_graphs):
            self.rand = random.Random(self.seed + i)
            self.graph = Graph(self.rand)
            self.graph.reset()
            while self.graph.num_nodes < NUM_PATTERNS:
                self.graph.add_node()

            # print(self.graph.patterns)
            # print(self.graph.path_len)
            # for link in self.graph.links:
            #     link.output()
            # print()

            sum_ratios += self.graph.mean_path_ratio()
            sum_lens += self.graph.mean_path_len()
            sum_max_lens += self.graph.max_path_len()
        print('\n{:8.4f} % of paths are reachable'.format(100 * sum_ratios / num_graphs))
        print('{:8.4f} mean path length'.format(sum_lens / num_graphs))
        print('{:8.4f} max path length'.format(sum_max_lens / num_graphs))
        exit(0)

    def assemble_current_observation(self):
        if self.quiz_agent_on_next_step:
            # Decide whether the correct answer should be 0 or 1.
            self.correct_output = self.rand.randint(0, 1)

            # Find a random A-B key-lock pair that satisfies the answer.
            while True:
                node_A = self.rand.randint(0, self.graph.num_nodes - 1)
                node_B = self.rand.randint(0, self.graph.num_nodes - 1)
                if node_A == node_B:
                    continue
                if (self.graph.path_len[node_A][node_B] > 0) == (self.correct_output == 1):
                    break

            # Show patterns A and B to the agent.
            self.observation = self.graph.patterns[node_A] + self.graph.patterns[node_B] + [1.]
        else:
            # Add one node to the graph.
            if self.graph.num_nodes < NUM_PATTERNS:
                self.graph.add_node()

            # Reveal the latest key-lock pattern pair to the agent.
            link = self.graph.links[-1]
            self.observation = self.graph.patterns[link.src] + self.graph.patterns[link.tar] + [0.]

        self.quiz_agent_on_next_step = not self.quiz_agent_on_next_step
        return self.observation

    def reset(self, repeat=False, episode_id = None):
        self.graph.reset()
        self.quiz_agent_on_next_step = False
        return self.assemble_current_observation()

    def translate_key_to_action(self, key):
        action = -1
        if key == 'Up':
            action = 1
        elif key == 'Right':
            action = 0
        else:
            print(("Key not found"))
        return action

    def step(self, action):
        self.reward = 0
        self.max_reward = 0.
        self.done = False
        if not self.quiz_agent_on_next_step:
            quiz_id = self.graph.num_nodes - 2
            if action == self.correct_output:
                self.reward = 1.
                self.cumulative_counts[quiz_id][1] += 1.
            else:
                self.cumulative_counts[quiz_id][0] += 1.
            self.max_reward += 1.
            if self.graph.num_nodes == NUM_PATTERNS:
                self.done = True
        self.update_online_test_sums(self.reward, self.done)
        ret = self.assemble_current_observation(), self.reward, self.done
        return ret

    def log_settings(self, summary_file):
        return

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
        self.need_to_reset_sums = False

    def update_online_test_sums(self, reward, done):
        # Called only by the environment itself.
        if self.need_to_reset_sums:
            # Another thread recently called reduce_online_test_sums(), so the previous counts are stale.
            self.reset_online_test_sums()
        # If another thread happens to call reduce_online_test_sums near this point in time,
        # one sample from this agent might get dropped. But that's a small price to avoid locking.
        self.step_sum += 1
        self.reward_sum += reward
        self.total_steps += 1
        self.total_reward += reward
        if done:
            self.num_episodes += 1

    def report_online_test_metric(self):
        # Called by the reporting manager only.

        # Calculate the final metric for this test period.
        self.reward_percentage = 200.0 * self.reward_sum / self.step_sum  # Reward available on every other step.

        # Assemble the tuple to be returned.
        #   1. The number of steps in the period. (This will be a bit different for each running thread.)
        #   2. The actual metric value (must be negated if lower is better).
        #   3. A string containing the formatted metric.
        #   4. A string containing the metric's units for display.
        #ret = (self.step_sum, self.num_episodes, self.reward_percentage, "{:7.3f}".format(self.reward_percentage), "reward percentage", False)

        metrics = []
        metrics.append((self.reward_percentage, "{:7.3f}".format(self.reward_percentage), "% reward"))
        ret = (self.step_sum, self.num_episodes, self.reward_percentage, metrics, False)

        # Reset the global sums.
        self.reset_online_test_sums()

        return ret
