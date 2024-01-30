
from deconvolution.configs.utils import Configs
from deconvolution.datamodules.deconvolution.configs import Deconvolution_Gauss1D_Configs,  Deconvolution_Gauss2D_Configs
from deconvolution.models.configs import MLP_Configs, MAF_RQS_Configs, MAF_Affine_Configs, Couplings_RQS_Configs
from deconvolution.dynamics.cnf.configs import FlowMatch_Configs
from deconvolution.dynamics.nf.configs import Deconvolution_NormFlow_Configs
from deconvolution.pipelines.configs import NormFlows_Sampler_Configs, NeuralODE_Sampler_Configs


Deconvolution_Gauss_MLP_FlowMatch = Configs(data = Deconvolution_Gauss2D_Configs,
                                            model = MLP_Configs, 
                                            dynamics = FlowMatch_Configs, 
                                            pipeline = NeuralODE_Sampler_Configs)


Deconvolution_Gauss_MAF_Affine_NormFlow = Configs(data = Deconvolution_Gauss2D_Configs,
                                                  model = MAF_Affine_Configs, 
                                                  dynamics = Deconvolution_NormFlow_Configs, 
                                                  pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss_MAF_RQS_NormFlow = Configs(data = Deconvolution_Gauss2D_Configs,
                                                model = MAF_RQS_Configs, 
                                                dynamics = Deconvolution_NormFlow_Configs, 
                                                pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss_Couplings_RQS_NormFlow = Configs(data = Deconvolution_Gauss2D_Configs,
                                                model = Couplings_RQS_Configs, 
                                                dynamics = Deconvolution_NormFlow_Configs, 
                                                pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss1D_MAF_RQS_NormFlow = Configs(data = Deconvolution_Gauss1D_Configs,
                                                model = MAF_RQS_Configs, 
                                                dynamics = Deconvolution_NormFlow_Configs, 
                                                pipeline = NormFlows_Sampler_Configs)