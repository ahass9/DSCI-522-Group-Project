import pandas as pd

# Load extracted data
hotel_data = pd.read_csv("hotel_data.csv")

# Filter only records from 2017
data_2017 = hotel_data[hotel_data['arrival_date_year'] == 2017]

# Now that we don't need the year column, we can drop it
hotel_data_cleaned = data_2017.drop(columns=['arrival_date_year'])

# Save cleaned dataframe to CSV
hotel_data_cleaned.to_csv("hotel_data_cleaned.csv", index=False)
