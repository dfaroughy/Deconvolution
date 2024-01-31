from deconvolution.configs.utils import Configs

from deconvolution.datamodules.gaia.configs import Gaia_Configs
from deconvolution.models.configs import MAF_RQS_Configs, Couplings_RQS_Configs
from deconvolution.dynamics.nf.configs import NormFlow_Configs
from deconvolution.pipelines.configs import  NormFlows_Sampler_Configs

Gaia_MAF_RQS_NormFlow = Configs(data = Gaia_Configs,
                                model = MAF_RQS_Configs, 
                                dynamics = NormFlow_Configs, 
                                pipeline = NormFlows_Sampler_Configs)

Gaia_Couplings_RQS_NormFlow = Configs(data = Gaia_Configs,
                                     model = Couplings_RQS_Configs, 
                                     dynamics = NormFlow_Configs, 
                                     pipeline = NormFlows_Sampler_Configs)