"""
Meridian_Getting_Started.py
August 3, 2025

This script is an excerpt of Google's Meridian Getting Started guide from August 2025, a
Jupyter notebook that introduces users to Meridian, a tool for Bayesian data analysis.
This version leaves off many of the print out features.

"""

import arviz as az
import IPython
from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import optimizer
from meridian.analysis import summarizer
from meridian.analysis import visualizer
from meridian.data import data_frame_input_data_builder
from meridian.data import test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import pandas as pd

# check if GPU is available
from psutil import virtual_memory
import tensorflow as tf
import tensorflow_probability as tfp

def main():
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print(
        'Num GPUs Available: ',
        len(tf.config.experimental.list_physical_devices('GPU')),
    )
    print(
        'Num CPUs Available: ',
        len(tf.config.experimental.list_physical_devices('CPU')),)

    df = pd.read_csv(
        "https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv"
    )

    # Create a DataFrameInputDataBuilder instance.
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type='non_revenue',
        default_kpi_column='conversions',
        default_revenue_per_kpi_column='revenue_per_conversion',
    )

    # Offer components to the builder.
    builder = (
        builder.with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(
            df, control_cols=["sentiment_score_control", "competitor_sales_control"]
        )
    )

    channels = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4"]
    builder = builder.with_media(
        df,
        media_cols=[f"{channel}_impression" for channel in channels],
        media_spend_cols=[f"{channel}_spend" for channel in channels],
        media_channels=channels,
    )

    data = builder.build()

    # Configure the model
    roi_mu = 0.2  # Mu for ROI prior for each media channel.
    roi_sigma = 0.9  # Sigma for ROI prior for each media channel.
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior)

    mmm = model.Meridian(input_data=data, model_spec=model_spec)

    # Using sample prior and posteriors, construct the model.
    %%time
    mmm.sample_prior(500)
    mmm.sample_posterior(
        n_chains=10, n_adapt=2000, n_burnin=500, n_keep=1000, seed=0
    )

    # Run model diagnostics.  First check for convergence then plot predicted vs actuals.
    model_diagnostics = visualizer.ModelDiagnostics(mmm)
    model_diagnostics.plot_rhat_boxplot()

    model_fit = visualizer.ModelFit(mmm)
    model_fit.plot_model_fit()

    # Summarize the model results.
    mmm_summarizer = summarizer.Summarizer(mmm)

    filepath = '/content/drive/MyDrive'
    start_date = '2021-01-25'
    end_date = '2024-01-15'
    mmm_summarizer.output_model_results_summary(
        'summary_output.html', filepath, start_date, end_date
    )

if __name__ == "__main__":
    main()