from edm2impl.sampler_run import Sampler
from .guidances.classifier_guidance import CGSampler
from .guidances.no_guidance import NoGuidSampler
from .guidances.map_ddg_chen import MapNegDDGSampler
# from .guidances.map_ddg import MapNegDDGSampler
from .guidances.time_offset_guidance import Time_Offset_Sampler
from .guidances.random_ddg import Random_DDG_Sampler
from .guidances.map_ddg_x0pred import MapNegDDGSampler_x0pred
from .guidances.map_stats import MapStatsSampler_x0pred
from .guidances.noguid_stats import StatsSampler
from .guidances.cfg_stats import CFGStatsSampler
from .guidances.cfg_adaptive import CFGAdaptiveSampler
from .guidances.cfg_general_adaptive import CFGAdaptiveSamplerGeneral
from .guidances.map_adaptive import MAPAdaptiveSampler
from .guidances.cfg_adaptive_v9 import CFGAdaptiveSamplerV9
from .guidances.cfg_adaptive_v10 import CFGAdaptiveSamplerV10
from .guidances.map_adaptive_v10 import MAPAdaptiveSamplerV10
from .guidances.map_ddg_x0pred_checking import MapNegDDGSampler_x0pred_checking
from .guidances.cfg_adaptive_v11 import CFGAdaptiveSamplerV11
from .guidances.cfg_adaptive_v12 import CFGAdaptiveSamplerV12
from .guidances.cfg_adaptive_v13 import CFGAdaptiveSamplerV13
guidance_config = {
    'noguid': NoGuidSampler,
    'cfg': Sampler,
    'cg': CGSampler,
    'map_neg_ddg': MapNegDDGSampler,
    'time_offset':Time_Offset_Sampler,
    'random_ddg':Random_DDG_Sampler,
    'map_x0pred': MapNegDDGSampler_x0pred,
    'map_stats':MapStatsSampler_x0pred,
    'noguid_stats':StatsSampler,
    'cfg_stats':CFGStatsSampler,
    'cfg_adaptive':CFGAdaptiveSampler,
    'cfg_adaptive_general':CFGAdaptiveSamplerGeneral,
    'map_adaptive':MAPAdaptiveSampler,
    'cfg_adaptive_v9':CFGAdaptiveSamplerV9,
    'cfg_adaptive_v10':CFGAdaptiveSamplerV10,
    'map_adaptive_v10':MAPAdaptiveSamplerV10,
    'map_ddg_x0pred_checking':MapNegDDGSampler_x0pred_checking,
    'cfg_adaptive_v11':CFGAdaptiveSamplerV11,
    'cfg_adaptive_v12':CFGAdaptiveSamplerV12,
    'cfg_adaptive_v13':CFGAdaptiveSamplerV13,
}