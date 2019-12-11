import numpy as np
from easydict import EasyDict as edict

root = edict()
cfg = root

root.batch_size = 32
root.learning_rate = 2e-4
root.num_epochs = 100
root.num_iters = 10000

root.is_Train = 1
root.grad_clip = 1
root.num_units = 512
root.embed_dim = 512
root.num_categories = 1354
root.cat_dim = 300
root.num_brands = 6312
root.brand_dim = 300
root.std = 0.02
root.restore = 0
root.mode = "train"
root.dataset = "mercari-price-prediction"

root.host = '0.0.0.0'
root.port = '5000'

root.brand_model_dir = "saved_models/brand_encoding.pickle"
root.category_model_dir = "saved_models/category_encoding.pickle"

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, root)