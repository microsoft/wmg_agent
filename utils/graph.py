# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Graph(object):
    def __init__(self):
        self.entities = []
        self.entity_type_sizes = []
        self.relations = None
        self.relation_size = 0

    def clear(self):
        self.entities = []


class Entity(object):
    def __init__(self, type, row, col):
        self.data = None
        self.type = type
        self.age = None
        # The following should not be used directly by the agent.
        # It's used by the environment in assembling the observation.
        self.row = row
        self.col = col
