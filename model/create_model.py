import sys

_model_entrypoints = {}


def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_entrypoints[model_name] = fn
    return fn


def is_model(model_name):
    return model_name in _model_entrypoints


def model_entrypoints(model_name):
    return _model_entrypoints[model_name]


def create_model(model_name, pretrained=False, checkpoint_path="", **kwargs):
    """Create a model
    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if is_model(model_name):
        create_fn = model_entrypoints(model_name)
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)

    model = create_fn(**kwargs)

    return model
