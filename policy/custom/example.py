"""Minimal inference model for interface checks, not a task-solving policy."""
import numpy as np
from policy.custom import DistributedPolicy, register_policy


class ExampleModel:
    def infer(self, observation):
        # Absolute joint targets, NOT a no-op.
        return {'actions': np.zeros(8, dtype=np.float32),
                'instruction': 'Example joint targets'}


@register_policy('example')
class ExamplePolicy(DistributedPolicy):
    def load_model(self, config):
        return ExampleModel()
