"""Load your checkpoint; DistributedPolicy provides the evaluation lifecycle."""
from policy.custom import DistributedPolicy, register_policy


@register_policy('my_algorithm')
class MyPolicy(DistributedPolicy):
    def load_model(self, config):
        # TODO: return your model loaded from config['checkpoint'].
        # It must implement infer(observation) -> {'actions': own_actions}.
        # Input: images.head_camera + images.wrist_camera (own RGB, CHW),
        #        state (own 8 values), prompt (high-level task instruction).
        # Output: own actions (8,) or (T, 8), never both robots' actions.
        raise NotImplementedError('Load your checkpoint in MyPolicy.load_model')
