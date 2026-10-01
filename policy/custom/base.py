"""Common lifecycle for a model using RoboPoly's per-robot inference interface."""
from abc import ABC, abstractmethod

import numpy as np


class DistributedPolicy(ABC):
    def __init__(self, *, config):
        self.config = config
        self.model = self.load_model(config)

    @abstractmethod
    def load_model(self, config):
        """Return a model with infer(observation) -> {'actions': own_actions}."""

    def reset(self, *, seed, robot_id):
        self.robot_id = robot_id
        self.rng = np.random.default_rng(seed)
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'set_rng_seed'):
            self.model.set_rng_seed(seed)

    def act(self, observation):
        """Forward native images/state/prompt; action validation stays in the evaluator."""
        return self.model.infer(observation)

    def close(self):
        if hasattr(self.model, 'close'):
            self.model.close()
