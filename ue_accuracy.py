import sys
import math
import matplotlib.pyplot as plt
import json
import numpy as np
import simplekml
import logging
import os
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.dates import DateFormatter, AutoDateLocator
import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)

def haversine_distance(coord1, coord2):
    """
    Calculate the distance between two GPS coordinates using the Haversine formula.
    
    Parameters:
    - coord1 (tuple): A tuple of (latitude, longitude) for the first coordinate.
    - coord2 (tuple): A tuple of (latitude, longitude) for the second coordinate.
    
    Returns:
    - float: The distance between the two coordinates in meters.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return distance

# Function to interpolate between two GPS points and return a list of interpolated points
def interpolate_points(point1, point2, num_points):
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1
    
    interpolated = [(lat1 + i * lat_diff / (num_points + 1), lon1 + i * lon_diff / (num_points + 1)) for i in range(1, num_points + 1)]
    return interpolated

class DualLogger:
    def get_original_stdout(self):
        return self.terminal
    def __init__(self, title):
        self.terminal = sys.stdout
        self.log = open(os.path.join(title, f"{title}_console_output.txt"), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        pass

# Updated function to plot the ground truth, interpolated, and estimated position GPS points using matplotlib
def plot_points_matplotlib(title, ground_truth, interpolated, estimated_position):
    # Extract latitudes and longitudes
    ground_truth_lon, ground_truth_lat = zip(*ground_truth)  # Swapped here
    interpolated_lon, interpolated_lat = zip(*interpolated)  # Swapped here
    estimated_position_lon, estimated_position_lat = zip(*estimated_position)  # Swapped here

    
    
    avg_latitude = np.mean(ground_truth_lat)
    meters_per_degree = 111132.92 - 559.82 * np.cos(2 * avg_latitude) + 1.175 * np.cos(4 * avg_latitude)

    # Calculate the data range
    data_range = (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * meters_per_degree

    # Adjust the grid spacing based on the data range
    if data_range > 100000:  # e.g., 100 km
        grid_spacing_meters = 10000  # 10 km
    elif data_range > 10000:  # e.g., 10 km
        grid_spacing_meters = 1000  # 1 km
    else:
        grid_spacing_meters = scalebar_length_meters

    grid_spacing = grid_spacing_meters / meters_per_degree
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(grid_spacing))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(grid_spacing))
    plt.grid(True)
    
    # Plot the ground truth, interpolated, and estimated position GPS points
    plt.scatter(ground_truth_lat, ground_truth_lon, color='blue', s=100, label='Ground Truth')
    plt.scatter(interpolated_lat, interpolated_lon, color='green', s=50, label='Interpolated Points')
    plt.scatter(estimated_position_lat, estimated_position_lon, color='red', s=75, label='Estimated Positions')
    
    # Plot a line connecting the sorted estimated GPS points
    plt.plot(estimated_position_lat, estimated_position_lon, color='black', linestyle='-', linewidth=1)
    plt.plot(ground_truth_lat, ground_truth_lon, color='blue', linestyle='-', linewidth=2)

    
    plt.xlabel('Longitude')  # Corrected label
    plt.ylabel('Latitude')  # Corrected label
    plt.title('GPS Points Plot')

    
    # Adjust the legend to be outside the plot area
    plt.legend(loc='upper center',fancybox=True, bbox_to_anchor=(0, 0))
    
    scalebar_length_fraction = 0.25
    scalebar_length_meters = scalebar_length_fraction * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * meters_per_degree
    
    grid_spacing_meters = scalebar_length_meters
    grid_spacing = grid_spacing_meters / meters_per_degree
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(grid_spacing))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(grid_spacing))
    plt.grid(True)
    
    scalebar = ScaleBar(meters_per_degree, location='lower right', scale_loc='bottom', units='m', length_fraction=scalebar_length_fraction, height_fraction=0.007, pad=0.5)
    plt.gca().add_artist(scalebar)
    plt.tight_layout()  # Adjusts the layout
    plt.savefig(os.path.join(title, f"{title}_gps_points_plot.png"), bbox_inches='tight')  # Adjusts the saved figure to include the legend
    plt.show()

# Updated function to generate a KML file for visualizing the GPS points on Google Earth
def generate_kml(title, ground_truth, interpolated, estimated_position):
    kml = simplekml.Kml()

    # Define styles for each type of point
    ground_truth_style = simplekml.Style()
    ground_truth_style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png"

    interpolated_style = simplekml.Style()
    interpolated_style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"

    estimated_position_style = simplekml.Style()
    estimated_position_style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"

    # Add ground truth points to KML without labels
    for point in ground_truth:
        pnt = kml.newpoint(coords=[(point[1], point[0])])
        pnt.style = ground_truth_style

    # Add interpolated points to KML without labels
    for point in interpolated:
        pnt = kml.newpoint(coords=[(point[1], point[0])])
        pnt.style = interpolated_style

    # Add estimated position points to KML without labels
    for point in estimated_position:
        pnt = kml.newpoint(coords=[(point[1], point[0])])
        pnt.style = estimated_position_style

    # Save the KML
    kml.save(os.path.join(title, "gps_points.kml"))

    print("KML file generated as 'gps_points.kml'. You can open this with Google Earth.")

# Updated function to extract estimated position GPS points and timestamps from the client response log
def extract_positions_from_client_response_log(log_content):
    estimated_position_points = []
    timestamps = []
    log_entries = log_content.split("\n")
    for entry in log_entries:
        if entry:
            try:
                data = json.loads(entry)
                lat = data["locationEstimate"]["point"]["lat"]
                lon = data["locationEstimate"]["point"]["lon"]
                timestamp = data.get("timestampOfLocationEstimate", "N/A")
                estimated_position_points.append((lat, lon))
                timestamps.append(timestamp)
            except (json.JSONDecodeError, KeyError):
                logging.warning(f"Unable to parse JSON or extract required keys for entry: {entry}")


    # Sort the estimated positions based on timestamps
    sorted_positions_timestamps = sorted(zip(estimated_position_points, timestamps), key=lambda x: x[1])
    estimated_position_sorted, timestamps_sorted = zip(*sorted_positions_timestamps)
    return estimated_position_sorted, timestamps_sorted

# Updated function to extract estimated position GPS points and timestamps from the nlg log
# Updated function to extract estimated position GPS points and timestamps from the nlg log
def extract_positions_from_nlg_log(log_content):
    estimated_position_points = []
    timestamps = []
    log_entries = log_content.split("\n")
    
    for entry in log_entries:
        if "[NF_REST][HttpServer::processData] received data :" in entry:
            try:
                json_str = entry.split("[NF_REST][HttpServer::processData] received data :")[1].strip()
                data = json.loads(json_str)
                lat = data["locationEstimate"]["point"]["lat"]
                lon = data["locationEstimate"]["point"]["lon"]
                timestamp = data.get("timestampOfLocationEstimate", "N/A")
                estimated_position_points.append((lat, lon))
                timestamps.append(timestamp)
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Unable to parse JSON or extract required keys for entry: {entry}. Error: {e}")

    # Sort the estimated positions based on timestamps
    sorted_positions_timestamps = sorted(zip(estimated_position_points, timestamps), key=lambda x: x[1])
    estimated_position_sorted, timestamps_sorted = zip(*sorted_positions_timestamps)
    return estimated_position_sorted, timestamps_sorted

def generate_cdf(data):
    """Generate CDF from data."""
    data_size = len(data)
    data_sorted = np.sort(data)
    data_y = np.arange(1, data_size + 1) / data_size
    return data_sorted, data_y

def plot_cdf(title, data):
    """Plot the CDF."""
    x, y = generate_cdf(data)
    plt.plot(x, y, marker='.', linestyle='-', color='blue', label='CDF')
    
    # Calculate 68th and 95th percentiles
    percentile_68 = np.percentile(data, 68)
    percentile_95 = np.percentile(data, 95)
    
    # Plot vertical lines for the percentiles
    plt.axvline(x=percentile_68, color='r', linestyle='--', label=f'68th: {percentile_68:.2f} meters')
    plt.axvline(x=percentile_95, color='g', linestyle='--', label=f'95th: {percentile_95:.2f} meters')
    
    plt.xlabel('Error Distance (meters)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function of Errors')
    plt.grid(True)
    plt.legend(loc='upper left')  # Display the legend in the upper left corner
    plt.tight_layout()  # Adjusts the layout
    plt.savefig(os.path.join(title, f"{title}_cdf_plot.png"))
    plt.show()




def parse_arguments(args):
    """
    Parse command-line arguments to extract necessary parameters.
    
    Parameters:
    - args (list): List of command-line arguments.
    
    Returns:
    - tuple: Extracted csv_path, log_path, plot_flag, kml_flag, cdf_flag.
    """
    # Check for required arguments
    if '--title' not in args:
        logging.error("Missing '--title' argument. Please provide a title for this test run.")
        sys.exit(1)
    if '--csv' not in args:
        logging.error("Missing '--csv' argument. Please provide the path to the CSV file containing ground truth points.")
        sys.exit(1)
    if '--log' not in args:
        logging.error("Missing '--log' argument. Please provide the path to the log file.")
        sys.exit(1)

    # Extract paths from arguments
    title = args[args.index('--title') + 1]
    csv_path = args[args.index('--csv') + 1]
    log_path = args[args.index('--log') + 1]
    plot_flag = args[args.index('--plot') + 1].lower() == 'true' if '--plot' in args else False
    kml_flag = args[args.index('--kml') + 1].lower() == 'true' if '--kml' in args else False
    cdf_flag = args[args.index('--cdf') + 1].lower() == 'true' if '--cdf' in args else False

    return title, csv_path, log_path, plot_flag, kml_flag, cdf_flag


def plot_timeseries(title, timestamps, errors):
    # Convert string timestamps to datetime objects
    datetime_timestamps = [datetime.datetime.strptime(ts.split('.')[0] + '.' + ts.split('.')[1][:6] + 'Z', "%Y-%m-%dT%H:%M:%S.%fZ") for ts in timestamps]


    plt.figure(figsize=(10, 6))
    plt.plot(datetime_timestamps, errors, marker='o', linestyle='-', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Error Distance (meters)')
    plt.title('Timeseries of Accuracy Values')

    # Format x-axis
    locator = AutoDateLocator()
    formatter = DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for better readability

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(title, f"{title}_timeseries_plot.png"))
    plt.show()

def plot_periodicity(title, periodicities):
    plt.figure(figsize=(10, 6))
    plt.plot(periodicities, marker='o', linestyle='-', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Periodicity (ms)')
    plt.title('Periodicity between Timestamps')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(title, f"{title}_periodicity_plot.png"))
    plt.show()

# ... [previous code]

def calculate_and_plot_timestamp_metrics(title, datetime_timestamps, periodicities):
    """Calculate and plot timestamp metrics."""
    # Test start time
    test_start_time = datetime_timestamps[0]
    
    # Times where periodicity is not 160ms ±20ms
    irregular_intervals = [(i, p) for i, p in enumerate(periodicities) if not (2000 <= p )]
    
    # Test duration
    test_duration = datetime_timestamps[-1] - test_start_time
    
    # End timestamp
    end_timestamp = datetime_timestamps[-1]
    
    # Median report time
    median_report_time = np.median(periodicities)
    
    # Mean report time
    mean_report_time = np.mean(periodicities)
    
    # Standard deviation report time
    std_report_time = np.std(periodicities)
    
    # Print the metrics
    print(f"Test Start Time: {test_start_time}")
    print(f"Times where periodicity is not 160ms ±20ms: {[datetime_timestamps[i] for i, _ in irregular_intervals]}")
    print(f"Test Duration: {test_duration}")
    print(f"End Timestamp: {end_timestamp}")
    print(f"Median Report Time: {median_report_time:.2f} ms")
    print(f"Mean Report Time: {mean_report_time:.2f} ms")
    print(f"Standard Deviation Report Time: {std_report_time:.2f} ms")
    
    # Plotting the irregular intervals
    plt.figure(figsize=(10, 6))
    plt.plot(datetime_timestamps[1:], periodicities, marker='o', linestyle='-', color='blue')
    for i, _ in irregular_intervals:
        plt.scatter(datetime_timestamps[i+1], periodicities[i], color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Periodicity (ms)')
    plt.title('Periodicities with Irregular Intervals Highlighted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(title, f"{title}_periodicity_irregular_intervals_plot.png"))
    plt.show()


def main():
    # Parse command-line arguments
    title, csv_path, log_path, plot_flag, kml_flag, cdf_flag = parse_arguments(sys.argv[1:])
  
    # Create a directory with the title's name
    if not os.path.exists(title):
        os.makedirs(title)

    # Redirect stdout to both console and file
    dual_logger = DualLogger(title)
    sys.stdout = dual_logger

    # Switch back to original stdout for user input
    sys.stdout = dual_logger.get_original_stdout()
    log_choice = input("Which log would you like to parse? (1. client_response_log, 2. nlg_log): ")
    # Switch back to DualLogger
    sys.stdout = dual_logger

    if log_choice == "1":
        extraction_function = extract_positions_from_client_response_log
    elif log_choice == "2":
        extraction_function = extract_positions_from_nlg_log
    else:
        print("Invalid choice. Please select either 1 or 2.")
        return

   # Extract ground truth from CSV
    with open(csv_path, 'r') as file:
        line = file.readline().strip()
        file.seek(0)  # Reset file pointer to the beginning after reading the first line
        delimiter = '\t' if '\t' in line else ','  # Determine the delimiter
        ground_truth = [tuple(map(float, line.strip().split(delimiter))) for line in file.readlines()]


    # Extract estimated position from log
    with open(log_path, 'r') as file:
        log_content = file.read()
        estimated_position, timestamps = extraction_function(log_content)

    # Interpolate ground truth points
    num_interpolated_points = 2
    interpolated = [ground_truth[0]]
    for i in range(len(ground_truth) - 1):
        interpolated.extend(interpolate_points(ground_truth[i], ground_truth[i+1], num_interpolated_points))
        interpolated.append(ground_truth[i+1])

     # Convert timestamps to datetime objects
    datetime_timestamps = []
    for ts in timestamps:
        try:
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            # Handle timestamps with extra precision
            dt = datetime.datetime.strptime(ts.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            dt += datetime.timedelta(microseconds=int(ts.split('.')[1][:6]))  # Add microseconds up to 2 decimal places
        datetime_timestamps.append(dt)
    
    # Calculate periodicity between each timestamp entry in milliseconds
    periodicities = [ (datetime_timestamps[i+1] - datetime_timestamps[i]).total_seconds() * 1000 for i in range(len(datetime_timestamps)-1)]
    

    # Calculate errors
    errors = []
    for i, position in enumerate(estimated_position):
        closest_point = min(interpolated, key=lambda x: haversine_distance(x, position))
        distance = haversine_distance(position, closest_point)
        errors.append(distance)
        if i < len(estimated_position) - 1:  # If not the last position
            print(f"Timestamp: {timestamps[i]}, Estimated Position Point: {position}, Closest Point on Ground Truth Line: {closest_point}, Accuracy: {distance:.2f} m, Periodicity: {periodicities[i]} ms")
        else:  # For the last estimated position
            print(f"Timestamp: {timestamps[i]}, Estimated Position Point: {position}, Closest Point on Ground Truth Line: {closest_point}, Accuracy: {distance:.2f} m")

    


    # Plotting
    if plot_flag:
        plot_points_matplotlib(title, ground_truth, interpolated, estimated_position)
    if cdf_flag:
        plot_cdf(title, errors)
    if kml_flag:
        generate_kml(title, ground_truth, interpolated, estimated_position)
    if plot_flag:
        plot_timeseries(title, timestamps, errors)
    plot_periodicity(title, periodicities)
    calculate_and_plot_timestamp_metrics(title, datetime_timestamps, periodicities)



if __name__ == "__main__":
    main()
