import json
import argparse
import os
from collections import defaultdict


def parse_animal_data(file_path, time_window=2.0):
    """
    Parse animal tracking data from a JSON file and create a time event map.

    Args:
        file_path (str): Path to the JSON file.
        time_window (float): Time window in minutes for averaging data points.

    Returns:
        dict: A time event map with rounded timestamps as keys and animal data as values.
    """
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a dictionary to store all data points by animal
    animal_data = defaultdict(list)

    # Create a set of all unique rounded timestamps
    all_rounded_timestamps = set()

    # Process each animal
    for animal_id, animal_data_points in data.items():
        timestamps = animal_data_points['timestamp']
        gps_coordinates = animal_data_points['gps_coordinates']
        vitals = animal_data_points['vitals']

        # Process each timestamp
        for i in range(len(timestamps)):
            timestamp = float(timestamps[i])
            gps = [float(coord) for coord in gps_coordinates[i]]
            vital = {k: float(v) for k, v in vitals[i].items()}

            # Round timestamp to the nearest whole number
            rounded_timestamp = round(timestamp)
            all_rounded_timestamps.add(rounded_timestamp)

            # Add data to the animal_data dictionary
            animal_data[animal_id].append({
                'exact_timestamp': timestamp,
                'rounded_timestamp': rounded_timestamp,
                'gps_coordinates': gps,
                'vitals': vital
            })

    # Create a time event map
    time_event_map = create_time_event_map(animal_data, all_rounded_timestamps, time_window)

    return time_event_map


def create_time_event_map(animal_data, all_rounded_timestamps, time_window):
    """
    Create a time event map from animal tracking data.

    Args:
        animal_data (dict): Dictionary of animal tracking data.
        all_rounded_timestamps (set): Set of all unique rounded timestamps.
        time_window (float): Time window in minutes for averaging data points.

    Returns:
        dict: A time event map with rounded timestamps as keys and animal data as values.
    """
    time_event_map = {}

    # Process each rounded timestamp
    for rounded_timestamp in sorted(all_rounded_timestamps):
        time_event_map[rounded_timestamp] = {}

        # Process each animal
        for animal_id, data_points in animal_data.items():
            # Find data points within the time window of this timestamp
            window_data_points = [
                point for point in data_points
                if abs(point['exact_timestamp'] - rounded_timestamp) <= time_window
            ]

            # If we have data points for this animal within the window
            if window_data_points:
                # Average the data points
                avg_gps = average_gps_coordinates(window_data_points)
                avg_vitals = average_vitals(window_data_points)

                # Add to the time event map
                time_event_map[rounded_timestamp][animal_id] = {
                    'gps_coordinates': avg_gps,
                    'vitals': avg_vitals
                }

    return time_event_map


def average_gps_coordinates(data_points):
    """
    Average GPS coordinates from multiple data points.

    Args:
        data_points (list): List of data points.

    Returns:
        list: Averaged GPS coordinates [latitude, longitude].
    """
    # Average latitude and longitude
    avg_lat = sum(point['gps_coordinates'][0] for point in data_points) / len(data_points)
    avg_lon = sum(point['gps_coordinates'][1] for point in data_points) / len(data_points)

    return [str(avg_lat), str(avg_lon)]


def average_vitals(data_points):
    """
    Average vital signs from multiple data points.

    Args:
        data_points (list): List of data points.

    Returns:
        dict: Averaged vital signs.
    """
    # Average oxygen saturation and heart rate
    avg_oxygen = sum(point['vitals']['oxygen_saturation'] for point in data_points) / len(data_points)
    avg_heart_rate = sum(point['vitals']['heart_rate'] for point in data_points) / len(data_points)

    return {
        'oxygen_saturation': str(avg_oxygen),
        'heart_rate': str(avg_heart_rate)
    }


def main():
    # Set up command-line arguments
    # parser = argparse.ArgumentParser(description='Parse animal tracking data into a time event map.')
    # parser.add_argument('input_file', help='Path to the input JSON file')
    # parser.add_argument('--output', '-o', default='time_event_map.json',
    #                     help='Path to the output JSON file (default: time_event_map.json)')
    # parser.add_argument('--window', '-w', type=float, default=2.0,
    #                     help='Time window in minutes for averaging data points (default: 2.0)')
    # args = parser.parse_args()

    # file = "simulation_2025_4_25_7_30_21"
    file = "temp"
    infile_path = f"out/{file}.json"
    outfile_path = f"time_data"
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)

    try:
        # Parse the data
        time_event_map = parse_animal_data(infile_path, 2)

        # Convert the time_event_map to a string-keyed dictionary for JSON serialization
        output_map = {str(k): v for k, v in time_event_map.items()}

        # Save the output to a file
        with open(f"{outfile_path}/{file}.json", 'w') as file:
            json.dump(output_map, file, indent=4)

        print(f"Time event map has been created successfully and saved to {outfile_path}.json")
    except FileNotFoundError:
        print(f"Error: The file {infile_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {infile_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()