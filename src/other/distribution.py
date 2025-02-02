import os
import pandas as pd
from distfit import distfit

# Define directory paths
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, "Data")

# Ensure the directory exists
if not os.path.exists(data_directory):
    raise FileNotFoundError(f"Directory '{data_directory}' does not exist.")

# Initialize an empty list to store results
results = []

# Iterate over each CSV file in the data directory
for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    
    if filename.endswith(".csv"):
        try:
            # Read the CSV file and extract the 'units' column
            data = pd.read_csv(file_path, usecols=["units"])["units"].dropna().values

            # Ensure data is not empty before fitting
            if data.size == 0:
                print(f"Skipping empty file: {filename}")
                continue
            
            # Fit theoretical distributions
            dfit = distfit(todf=True)
            dfit.fit_transform(data)

            # Extract best distribution details
            best_dist = dfit.model.get('name', 'N/A')
            rss_score = dfit.model.get('score', 'N/A')
            params = dfit.model.get('params', 'N/A')

            # Append results
            results.append({
                "File": filename,
                "Best Distribution": best_dist,
                "RSS": rss_score,
                "Parameters": params
            })

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Export results to a CSV file
output_file = "distfit_results.csv"
output_path = os.path.join(data_directory, output_file)
results_df.to_csv(output_path, index=False)

print(f"Distfit results exported to: {output_path}")
