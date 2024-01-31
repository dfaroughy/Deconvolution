
from deconvolution.configs.utils import Configs
from deconvolution.datamodules.toys.configs import Deconvolution_Gauss_Configs
from deconvolution.models.configs import MAF_RQS_Configs, MAF_Affine_Configs, Couplings_RQS_Configs
from deconvolution.dynamics.nf.configs import Deconvolution_Configs
from deconvolution.pipelines.configs import NormFlows_Sampler_Configs

#...REGISTER CONFIG CARD FOR EXPERIMENTS

Deconvolution_Gauss_MAF_Affine_NormFlow = Configs(data = Deconvolution_Gauss_Configs,
                                                  model = MAF_Affine_Configs, 
                                                  dynamics = Deconvolution_Configs, 
                                                  pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss_MAF_RQS_NormFlow = Configs(data = Deconvolution_Gauss_Configs,
                                                model = MAF_RQS_Configs, 
                                                dynamics = Deconvolution_Configs, 
                                                pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss_Couplings_RQS_NormFlow = Configs(data = Deconvolution_Gauss_Configs,
                                                model = Couplings_RQS_Configs, 
                                                dynamics = Deconvolution_Configs, 
                                                pipeline = NormFlows_Sampler_Configs)

