from deconvolution.configs.utils import Configs

from deconvolution.datamodules.gaia.configs import Gaia_Configs, Deconvolve_Gaia_Configs
from deconvolution.models.configs import MAF_RQS_Configs, Couplings_RQS_Configs
from deconvolution.dynamics.nf.configs import NormFlow_Configs, Deconvolution_Configs
from deconvolution.pipelines.configs import  NormFlows_Sampler_Configs


#...REGISTER CONFIG CARD FOR EACH EXPERIMENT:


Gaia_MAF_RQS_NormFlow = Configs(experiment_description = 'density estimation for Gaia phase-space data', 
                                data = Gaia_Configs,
                                model = MAF_RQS_Configs, 
                                dynamics = NormFlow_Configs, 
                                pipeline = NormFlows_Sampler_Configs)

Gaia_Couplings_RQS_NormFlow = Configs(experiment_description = 'density estimation for Gaia phase-space data', 
                                      data = Gaia_Configs,
                                      model = Couplings_RQS_Configs, 
                                      dynamics = NormFlow_Configs, 
                                      pipeline = NormFlows_Sampler_Configs)

Deconvolve_Gaia_MAF_RQS_NormFlow = Configs(experiment_description = 'error deconvolution for Gaia phase-space data', 
                                           data = Deconvolve_Gaia_Configs,
                                           model = MAF_RQS_Configs, 
                                           dynamics = Deconvolution_Configs, 
                                           pipeline = NormFlows_Sampler_Configs)

Deconvolve_Gaia_Couplings_RQS_NormFlow = Configs(experiment_description = 'error deconvolution for Gaia phase-space data', 
                                                 data = Deconvolve_Gaia_Configs,
                                                 model = Couplings_RQS_Configs, 
                                                 dynamics = Deconvolution_Configs, 
                                                 pipeline = NormFlows_Sampler_Configs)