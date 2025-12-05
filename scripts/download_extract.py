import pandas as pd

# Download dataset
data = pd.read_csv( 'https://raw.githubusercontent.com/manthangandhi/hotel_cancellation_prediction/refs/heads/main/data/hotel_bookings.csv')

# columns of interest
cols_of_interest = [
    "hotel",
    "arrival_date_year",
    "is_canceled", 
    "lead_time",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "total_of_special_requests",
    "is_repeated_guest",
    "deposit_type",
    "reserved_room_type"
]

hotel_data = data[cols_of_interest]

# Save extracted dataframe to CSV
hotel_data.to_csv("hotel_data.csv", index=False)
