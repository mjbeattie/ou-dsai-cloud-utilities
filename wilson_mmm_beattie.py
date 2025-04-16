"""
wilson_mmm_beattie.py

Runs Google Meridian on Wilson marketing data and saves the resulting model to a pickle file.

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

from google.cloud import bigquery


# check if GPU is available
from psutil import virtual_memory

#def main(fileout):
def main():
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    # Initialize BigQuery client (automatically picks up credentials)
    client = bigquery.Client()

    # Define a public dataset query
    query = "SELECT * FROM `ou-dsa5900.mmm_spring2025.wilson_mmm_view`"

    # Run the query and load into pandas dataframe
    query_job = client.query(query)
    result = query_job.result()
    wilsonpdf = result.to_dataframe()

    # Convert the date field to a string for Meridian
    wilsonpdf['date'] = wilsonpdf['date'].astype(str)

    # Map dataframe columns to Meridian variables.  Note, we can ignore media impressions by setting media equal to spend.
    coord_to_columns = load.CoordToColumns(
        time='date',
        kpi='revenue',
        media_spend=[
            'pmax_cost',
            'search_cost',
            'shopping_cost',
            'youtube_cost',
            'demandgen_cost',
            'meta_cost'
        ],
        media=[
            'pmax_cost',
            'search_cost',
            'shopping_cost',
            'youtube_cost',
            'demandgen_cost',
            'meta_cost'
        ],
    controls=['seppromo'],
    )

    media_spend_to_channel = {
        'pmax_cost' : 'pmax',
        'search_cost' : 'search',
        'shopping_cost' : 'shopping',
        'youtube_cost' : 'youtube',
        'demandgen_cost' : 'demandgen',
        'meta_cost' : 'meta',
    }

    media_to_channel = {
        'pmax_cost' : 'pmax',
        'search_cost' : 'search',
        'shopping_cost' : 'shopping',
        'youtube_cost' : 'youtube',
        'demandgen_cost' : 'demandgen',
        'meta_cost' : 'meta',
    }

    loader = load.DataFrameDataLoader(
        df=wilsonpdf,
        kpi_type='revenue',
        coord_to_columns=coord_to_columns,
        media_spend_to_channel=media_spend_to_channel,
        media_to_channel=media_to_channel,
    )
    data = loader.load()

    # Set a custom prior for the September promotion period
    gamma_c_mu = 0.5  # Expected impact (mean) for control variable
    gamma_c_sigma = 0.1  # Uncertainty (standard deviation) for control variable
    prior = prior_distribution.PriorDistribution(
        gamma_c=tfp.distributions.Normal(gamma_c_mu, gamma_c_sigma, name="GAMMA_C")
        )

    model_spec = spec.ModelSpec(
        prior=prior,
        knots=12
        )


    # Configure the Meridian model using default priors
    mmm = model.Meridian(input_data=data)

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
    file_path='wilson_mmm_control_12knots_priors.pkl'
    model.save_mmm(mmm, file_path)
    print('Model saved to', file_path) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument("--fileout", type=str, required=False, help="Enter name of output file", default='fifi_inference_torch_output.txt')

#    args = parser.parse_args()
#    print('Running classification with fileout = ', args.fileout)
#    main(args.fileout)
    main()