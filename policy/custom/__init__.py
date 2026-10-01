"""Registration for user-owned RoboPoly controllers (no DuoMind dependencies)."""
import importlib

from policy.custom.base import DistributedPolicy

_REGISTRY = {}


def register_policy(name):
    """Decorate a factory/class accepting the keyword argument ``config``."""
    def register(factory):
        if not name or name in _REGISTRY:
            raise ValueError(f'Duplicate or empty custom policy name: {name!r}')
        _REGISTRY[name] = factory
        return factory
    return register


def load_policy(spec, config):
    """Import ``module:registered_name`` and construct its registered adapter."""
    module, separator, name = spec.partition(':')
    if not separator or not module or not name:
        raise ValueError('--custom-policy must be module:registered_name')
    importlib.import_module(module)
    if name not in _REGISTRY:
        raise ValueError(f'{name!r} was not registered by {module!r}')
    policy = _REGISTRY[name](config=config)
    if not all(callable(getattr(policy, method, None)) for method in ('reset', 'act')):
        raise TypeError('Custom policies must implement reset(...) and act(observation)')
    return policy
