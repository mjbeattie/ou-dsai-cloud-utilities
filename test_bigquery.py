"""
test_bigquery.py

Tests access to BigQuery from a GCP VM

"""


from google.cloud import bigquery
import argparse

def main():
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

    # Print header from pdf
    print(wilsonpdf.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument("--fileout", type=str, required=False, help="Enter name of output file", default='fifi_inference_torch_output.txt')

#    args = parser.parse_args()
#    print('Running classification with fileout = ', args.fileout)
#    main(args.fileout)
    main()
