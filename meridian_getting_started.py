"""
fifi_inference_torch.py

Uses a pretrained model to reclassify service requests submitted to the city of Seattle
using the FindItFixIt application.

TORCH VERSION

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az
import argparse
import logging
import time

import IPython

from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter

# check if GPU is available
from psutil import virtual_memory

#def main(fileout):
def main():
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    # Load the data
    coord_to_columns = load.CoordToColumns(
        time='time',
        geo='geo',
        controls=['GQV', 'Competitor_Sales'],
        population='population',
        kpi='conversions',
        revenue_per_kpi='revenue_per_conversion',
        media=[
            'Channel0_impression',
            'Channel1_impression',
            'Channel2_impression',
            'Channel3_impression',
            'Channel4_impression',
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
            'Channel2_spend',
            'Channel3_spend',
            'Channel4_spend',
        ],
        organic_media=['Organic_channel0_impression'],
        non_media_treatments=['Promo'],
    )

    # Map media variables and spends to designated channel names
    correct_media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
        'Channel2_impression': 'Channel_2',
        'Channel3_impression': 'Channel_3',
        'Channel4_impression': 'Channel_4',
    }
    correct_media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
        'Channel2_spend': 'Channel_2',
        'Channel3_spend': 'Channel_3',
        'Channel4_spend': 'Channel_4',
    }

    # Load the CSV data
    loader = load.CsvDataLoader(
        csv_path="https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=correct_media_to_channel,
        media_spend_to_channel=correct_media_spend_to_channel,
    )
    data = loader.load()

    # Configure the model
    roi_mu = 0.2     # Mu for ROI prior for each media channel.
    roi_sigma = 0.9  # Sigma for ROI prior for each media channel.
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior)

    mmm = model.Meridian(input_data=data, model_spec=model_spec)

    # Use the sample_prior() and sample_posterior() methods to to obtain samples.  Log run times
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Getting prior samples...")
    mmm.sample_prior(500)
    logging.info("Prior sampling complete.")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Getting posterior samples...")
    mmm.sample_posterior(n_chains=7, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
    logging.info("Posterior sampling complete.")

    # Save the model
    file_path='~/saved_mmm.pkl'
    model.save_mmm(mmm, file_path)
    print('Model saved to', file_path) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument("--fileout", type=str, required=False, help="Enter name of output file", default='fifi_inference_torch_output.txt')

#    args = parser.parse_args()
#    print('Running classification with fileout = ', args.fileout)
#    main(args.fileout)
    main()