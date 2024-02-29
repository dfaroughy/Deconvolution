import sys
from utils import plot_gaussians
from deconvolution.trainer.trainer import Trainer
from deconvolution.datamodules.toys.datasets import SmearedGaussDataset
from deconvolution.datamodules.toys.dataloader import ToysDataLoader 
from deconvolution.models.nflow_nets import CouplingsPiecewiseRQS
from deconvolution.dynamics.nf.normflows import DeconvolutionFlow
from deconvolution.pipelines.SamplingPipeline import NormFlowPipeline 
from deconvolution.configs.toys_configs import Deconvolve_Toys_Couplings_RQS_NormFlow as Configs

#...set configurations:

configs = Configs(DATA = 'smeared_gaussians',
                NUM_POINTS = 100000,
                DATA_SPLIT_FRACS = [0.5, 0.5, 0.0],
                DATA_LOC = 0,
                DATA_SCALE = 1,
                PREPROCESS = None, 
                NOISE_COV = [[0.1, 0],[0, 1]],
                BATCH_SIZE = 512,
                EPOCHS = 120,
                EARLY_STOPPING = 20,
                MIN_EPOCHS = 20,
                NUM_TRANSFORMS = 5,
                NUM_GEN_SAMPLES = 12000,
                DIM_HIDDEN = int(sys.argv[1]), 
                NUM_NOISE_DRAWS = 50,
                COUPLING_MASK = 'mid-split',
                NUM_BLOCKS = 2,
                USE_RESIDUAL_BLOCKS = False,
                USE_BATCH_NORM = True,               
                NUM_RQS_BINS = 20,
                TAIL_BOUND = 10,
                PRINT_EPOCHS = 5,
                LR = float(sys.argv[2]),
                DEVICE = 'cpu')

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)


if __name__ == '__main__':

    #...define dataset and train the model:

    gaussians = SmearedGaussDataset(configs)
    dataloader = ToysDataLoader(gaussians, configs)

    flow = Trainer(dynamics = DeconvolutionFlow(configs), 
                   model = CouplingsPiecewiseRQS(configs), 
                   dataloader =  dataloader,
                   configs = configs)

    #...train the model:

    flow.train()

    #...generate samples from the trained model:

    pipeline = NormFlowPipeline(trained_model=flow, best_epoch_model=True)
    pipeline.generate_samples(num=configs.NUM_GEN_SAMPLES)

    #...plot the results:

    plot_gaussians(gaussians.target, title='smeared data', num_points=configs.NUM_GEN_SAMPLES, xlim=(-5,5), ylim=(-5,5), save_path=configs.WORKDIR + '/smeared_data.png')
    plot_gaussians(gaussians.truth, title='truth data', num_points=configs.NUM_GEN_SAMPLES, xlim=(-5,5), ylim=(-5,5), save_path=configs.WORKDIR + '/truth_data.png')
    plot_gaussians(pipeline.target, title='deconvoluted data', num_points=configs.NUM_GEN_SAMPLES, xlim=(-5,5), ylim=(-5,5), save_path=configs.WORKDIR + '/deconvolution_result.png')
