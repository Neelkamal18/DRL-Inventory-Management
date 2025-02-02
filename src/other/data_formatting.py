import os
import pandas as pd

# Define file paths
forecast_file = 'forecast.csv'
item_master_file = 'item_master.csv'
output_forecast_file = 'forecast_revised.csv'
output_directory = 'Data'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load forecast and item master data
df_forecast = pd.read_csv(forecast_file)
df_items = pd.read_csv(item_master_file)

# Extract unique items from item master
items_list = df_items.iloc[:, 0].unique().tolist()
print(f"Number of unique items: {len(items_list)}")

# Filter forecast data to include only items in item master
df_filtered = df_forecast[df_forecast['item'].isin(items_list)]

# Save the filtered forecast data
df_filtered.to_csv(output_forecast_file, index=False)

# Reload the filtered data
df_filtered = pd.read_csv(output_forecast_file)

# Group data by item and location
grouped_data = df_filtered.groupby(['item', 'location'])

# Iterate over each group to generate separate files
for (item, location), group in grouped_data:
    # Define file path for each item's location-specific data
    output_file = os.path.join(output_directory, f'{item}_{location}.csv')

    # Sort the group by week in ascending order
    group_sorted = group.sort_values(by=['week'])

    # Skip writing files where all unit values are zero
    if group_sorted['units'].sum() == 0:
        continue

    # Save the sorted group to a new CSV file
    group_sorted.to_csv(output_file, index=False)

print("Processing complete. Files have been saved in the 'Data' directory.")
