from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.CANCONV = CN()

_C.MODEL.CANCONV.K_MEANS_N_CLUSTERS = 16 # Default, will be overridden
_C.MODEL.CANCONV.K_MEANS_MAX_ITER = 100
_C.MODEL.CANCONV.K_MEANS_TOLERANCE = 1e-4
_C.MODEL.CANCONV.ENABLE_PATCH_WISE_FC = False # True by default, True: fxy -> Linear -> fxy_reduced. False: fxy -> mean_pool -> fxy_original_channel
_C.MODEL.CANCONV.PATCH_FC_OUT_CHANNELS = 32 # Default 32, only used if ENABLE_PATCH_WISE_FC is True
_C.MODEL.CANCONV.FILTER_THRESHOLD = 0.0 # threshold to filter clusters. 0.0 means no filtering.
_C.MODEL.CANCONV.HC_N_CLUSTERS = None # Default, will be overridden. Can be int or list/tuple of ints for different layers.
_C.MODEL.CANCONV.HC_DISTANCE_THRESHOLD = None # For HC, distance threshold for cutting the dendrogram. Overrides HC_N_CLUSTERS if set.
_C.MODEL.CANCONV.ENABLE_DYNAMIC_K = True # For K-Means, whether to dynamically adjust K based on feature variance. 

# It's good practice to also define other top-level config groups if they exist globally
# e.g., _C.TRAIN = CN(), _C.DATASET = CN() etc.
# For now, only defining what's minimally required based on usage.

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values."""
  return _C.clone() 