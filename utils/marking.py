import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/temperature_preprocessed.csv"  # Replace with your actual path
data = pd.read_csv(file_path)

# Convert datetime to pandas datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Calculate the first derivative (rate of change) for avg_leaf_temp
data['leaf_temp_rate'] = data['avg_leaf_temp'].diff()

# Filter data for a single day
single_day_data = data[(data['datetime'] >= '2024-12-29 00:00:00') & (data['datetime'] < '2024-12-30 00:00:00')]

# Interactive plot for marking phases
def interactive_marking(single_day_data):
    """
    Interactive plot for user to mark phases for a single day.
    """
    print("Use the interactive plot to mark points for each phase.")
    print("Click on points for 'Increasing', then press ENTER.")
    print("Click on points for 'Decreasing', then press ENTER.")
    print("Click on points for 'Holding', then press ENTER.")
    print("Click on points for 'Nothing Happens', then press ENTER.")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(single_day_data['datetime'], single_day_data['avg_leaf_temp'], label='Avg Leaf Temp', color='blue')
    ax.set_title('Interactive Marking of Phases')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Avg Leaf Temperature (°C)')
    ax.legend()
    ax.grid()

    # Collect points for each phase
    plt.tight_layout()
    points_increasing = plt.ginput(n=-1, timeout=0, show_clicks=True)
    print("Marked points for 'Increasing':", points_increasing)
    points_decreasing = plt.ginput(n=-1, timeout=0, show_clicks=True)
    print("Marked points for 'Decreasing':", points_decreasing)
    points_holding = plt.ginput(n=-1, timeout=0, show_clicks=True)
    print("Marked points for 'Holding':", points_holding)
    points_nothing = plt.ginput(n=-1, timeout=0, show_clicks=True)
    print("Marked points for 'Nothing Happens':", points_nothing)
    plt.close()

    return points_increasing, points_decreasing, points_holding, points_nothing

# Call the interactive marking function
points_increasing, points_decreasing, points_holding, points_nothing = interactive_marking(single_day_data)

# Save the marked points for further analysis
print("Now you can analyze the points and calculate thresholds.")