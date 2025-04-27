import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math
from scipy import stats

if not os.path.exists("data_visualization"):
    os.mkdir("data_visualization")

# Load the JSON data
file_name = "example_data"  # Changed to match your file name
json_path = f"parsed_logs_time_data/{file_name}.json"

with open(json_path, 'r') as file:
    data = json.load(file)

# Create visualization directory for this dataset
vis_dir = f"data_visualization/{file_name}/enhanced"
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)


# Helper functions for data analysis
def extract_animal_species(animal_id):
    """Extract species from the animal ID."""
    return animal_id.split(':')[0]


def extract_trajectories(data):
    """
    Extract trajectories for each animal.
    Returns a dictionary with animal IDs as keys and lists of (timestamp, x, y) as values.
    """
    trajectories = {}
    # Sort timestamps numerically
    timestamps = sorted(data.keys(), key=int)

    # Initialize trajectories for all animals
    all_animals = set()
    for timestamp in timestamps:
        all_animals.update(data[timestamp].keys())

    for animal_id in all_animals:
        trajectory = []
        for timestamp in timestamps:
            if animal_id in data[timestamp]:
                x, y = map(float, data[timestamp][animal_id]["gps_coordinates"])
                trajectory.append((float(timestamp), x, y))
        trajectories[animal_id] = trajectory

    return trajectories


def calculate_speeds(trajectories):
    """
    Calculate movement speeds for each animal between consecutive timestamps.
    Returns a dictionary with animal IDs as keys and lists of (timestamp, speed) as values.
    """
    speeds = {}

    for animal_id, trajectory in trajectories.items():
        animal_speeds = []
        for i in range(1, len(trajectory)):
            prev_time, prev_x, prev_y = trajectory[i - 1]
            curr_time, curr_x, curr_y = trajectory[i]

            # Calculate distance using Euclidean distance
            distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

            # Calculate time difference
            time_diff = curr_time - prev_time

            # Calculate speed (distance per unit time)
            if time_diff > 0:
                speed = distance / time_diff
            else:
                speed = 0

            animal_speeds.append((curr_time, speed))

        speeds[animal_id] = animal_speeds

    return speeds


def calculate_distance_between_animals(data, timestamp, animal_id1, animal_id2):
    """Calculate distance between two animals at a specific timestamp."""
    if timestamp not in data or animal_id1 not in data[timestamp] or animal_id2 not in data[timestamp]:
        return None

    x1, y1 = map(float, data[timestamp][animal_id1]["gps_coordinates"])
    x2, y2 = map(float, data[timestamp][animal_id2]["gps_coordinates"])

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_consistent_pairs(data, species="Zebra", threshold=20.0, min_timestamps=2):
    """
    Find pairs of animals that consistently stay close to each other.
    Returns a dictionary of (animal_id1, animal_id2) -> list of timestamps.
    """
    close_pairs = {}
    timestamps = sorted(data.keys(), key=int)

    for timestamp in timestamps:
        # Get all animals of the specified species
        species_animals = [a for a in data[timestamp].keys() if a.startswith(f"{species}:")]

        # Check all pairs
        for i in range(len(species_animals)):
            for j in range(i + 1, len(species_animals)):
                animal_id1 = species_animals[i]
                animal_id2 = species_animals[j]

                distance = calculate_distance_between_animals(data, timestamp, animal_id1, animal_id2)

                if distance is not None and distance <= threshold:
                    pair = tuple(sorted([animal_id1, animal_id2]))

                    if pair not in close_pairs:
                        close_pairs[pair] = []

                    close_pairs[pair].append(timestamp)

    # Filter out pairs that aren't consistently close
    consistent_pairs = {pair: timestamps for pair, timestamps in close_pairs.items()
                        if len(timestamps) >= min_timestamps}

    return consistent_pairs


def create_location_heatmap(trajectories, grid_size=30):
    """
    Create a heatmap of animal locations.
    Returns a 2D array representing the heatmap and the bounds (min_x, max_x, min_y, max_y).
    """
    # Find bounding box
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for animal_id, trajectory in trajectories.items():
        for _, x, y in trajectory:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Create grid
    heatmap = np.zeros((grid_size, grid_size))

    # Count locations
    total_points = 0
    for animal_id, trajectory in trajectories.items():
        for _, x, y in trajectory:
            # Calculate grid indices
            x_idx = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * (grid_size - 1))))
            y_idx = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * (grid_size - 1))))

            heatmap[y_idx, x_idx] += 1
            total_points += 1

    # Normalize
    if total_points > 0:
        heatmap = heatmap / total_points

    return heatmap, (min_x, max_x, min_y, max_y)


def collect_vitals(data):
    """
    Collect vital statistics for each animal.
    Returns a dictionary with animal IDs as keys and vital statistics as values.
    """
    vitals = {}

    for timestamp in data:
        for animal_id, animal_data in data[timestamp].items():
            if animal_id not in vitals:
                vitals[animal_id] = {
                    "heart_rate": [],
                    "oxygen_saturation": []
                }

            heart_rate = float(animal_data["vitals"]["heart_rate"])
            oxygen = float(animal_data["vitals"]["oxygen_saturation"])

            vitals[animal_id]["heart_rate"].append(heart_rate)
            vitals[animal_id]["oxygen_saturation"].append(oxygen)

    return vitals


# Visualization functions
def plot_trajectories(trajectories, species=None, title=None, save_path=None):
    """Plot animal trajectories."""
    plt.figure(figsize=(12, 10))

    # Define colors for each species
    species_colors = {
        "Zebra": "black",
        "Lion": "orange",
        "Elephant": "gray"
    }

    # Track plotted species for legend
    plotted_species = set()

    for animal_id, trajectory in trajectories.items():
        animal_species = extract_animal_species(animal_id)

        if species and animal_species != species:
            continue

        if len(trajectory) < 2:
            continue

        _, x_coords, y_coords = zip(*trajectory)

        # Add to legend once per species
        label = animal_species if animal_species not in plotted_species else None
        if label:
            plotted_species.add(animal_species)

        plt.plot(x_coords, y_coords, color=species_colors.get(animal_species, "blue"),alpha=0.7, linewidth=1, label=label)
        # Mark start and end points
        plt.scatter(x_coords[0], y_coords[0], color=species_colors.get(animal_species, "blue"), marker="o", s=30, alpha=0.8)
        plt.scatter(x_coords[-1], y_coords[-1], color=species_colors.get(animal_species, "blue"), marker="x", s=30, alpha=0.8)

    plt.title(title or "Animal Movement Trajectories")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)

    if plotted_species:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_3d_density_map(trajectories, grid_size=30, title="3D Animal Activity Density Map", save_path=None):
    """Create a 3D surface plot showing animal location density."""
    # Find the bounding box for all animal locations
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for animal_id, trajectory in trajectories.items():
        for _, x, y in trajectory:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Create grid
    x_edges = np.linspace(min_x, max_x, grid_size + 1)
    y_edges = np.linspace(min_y, max_y, grid_size + 1)

    # Calculate the grid cell centers for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Initialize density grid
    density = np.zeros((grid_size, grid_size))

    # Count points in each grid cell
    for animal_id, trajectory in trajectories.items():
        for _, x, y in trajectory:
            # Find the grid cell indices
            x_idx = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * grid_size)))
            y_idx = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * grid_size)))

            density[y_idx, x_idx] += 1

    # Normalize density
    total_points = density.sum()
    if total_points > 0:
        density = density / total_points

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(X, Y, density, cmap='viridis',
                           linewidth=0, antialiased=True)

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Normalized Animal Density')

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Normalized Density')
    ax.set_title(title)

    # Adjust view angle
    ax.view_init(elev=30, azim=225)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return density, (min_x, max_x, min_y, max_y)


def plot_3d_trajectories(trajectories, species=None, title=None, save_path=None):
    """Plot 3D trajectories with time as the z-axis."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Define colors for each species
    species_colors = {
        "Zebra": "black",
        "Lion": "orange",
        "Elephant": "gray"
    }

    # Track plotted species for legend
    plotted_species = set()

    for animal_id, trajectory in trajectories.items():
        animal_species = extract_animal_species(animal_id)

        if species and animal_species != species:
            continue

        if len(trajectory) < 2:
            continue

        time_coords, x_coords, y_coords = zip(*trajectory)

        # Add to legend once per species
        label = animal_species if animal_species not in plotted_species else None
        if label:
            plotted_species.add(animal_species)

        ax.plot(x_coords, y_coords, time_coords, color=species_colors.get(animal_species, "blue"), alpha=0.7, linewidth=1, label=label)

    ax.set_title(title or "3D Animal Movement Trajectories (Z-axis: Time)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Time")

    if plotted_species:
        ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_speed_cdf(speeds, species=None, title=None, save_path=None):
    """Plot CDF of movement speeds."""
    plt.figure(figsize=(10, 6))

    # Collect speeds by species
    species_speeds = {}

    for animal_id, speed_data in speeds.items():
        animal_species = extract_animal_species(animal_id)

        if species and animal_species != species:
            continue

        if not speed_data:
            continue

        _, speed_values = zip(*speed_data)

        if animal_species not in species_speeds:
            species_speeds[animal_species] = []

        species_speeds[animal_species].extend(speed_values)

    # Plot CDF for each species
    for species_name, speed_values in species_speeds.items():
        sorted_speeds = np.sort(speed_values)
        cumulative_prob = np.arange(1, len(sorted_speeds) + 1) / len(sorted_speeds)

        plt.step(sorted_speeds, cumulative_prob, label=species_name, linewidth=2)

    plt.title(title or "CDF of Movement Speeds")
    plt.xlabel("Speed (distance per unit time)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)

    if species_speeds:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_vitals_boxplot(vitals, vital_type="heart_rate", title=None, save_path=None):
    """Plot boxplot of vital statistics by species."""
    plt.figure(figsize=(10, 6))

    # Organize data by species
    species_data = {}

    for animal_id, animal_vitals in vitals.items():
        species = extract_animal_species(animal_id)

        if species not in species_data:
            species_data[species] = []

        species_data[species].extend(animal_vitals[vital_type])

    # Prepare data for boxplot
    data = []
    labels = []

    for species, values in species_data.items():
        labels.append(species)
        data.append(values)

    plt.boxplot(data, labels=labels, patch_artist=True)

    plt.title(title or f"{vital_type.replace('_', ' ').title()} by Species")
    plt.ylabel(vital_type.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_location_heatmap(heatmap, bounds, title=None, save_path=None):
    """Plot heatmap of animal locations."""
    plt.figure(figsize=(10, 8))

    min_x, max_x, min_y, max_y = bounds

    plt.imshow(heatmap, cmap="hot", interpolation="nearest", extent=[min_x, max_x, min_y, max_y], origin="lower", aspect="auto")

    plt.colorbar(label="Normalized Frequency")

    plt.title(title or "Location Frequency Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_pair_distances(data, pairs, title=None, save_path=None):
    """Plot distances between pairs of animals over time."""
    plt.figure(figsize=(12, 6))

    timestamps = sorted(data.keys(), key=int)

    for pair in pairs:
        animal_id1, animal_id2 = pair
        distances = []
        valid_timestamps = []

        for timestamp in timestamps:
            distance = calculate_distance_between_animals(data, timestamp, animal_id1, animal_id2)

            if distance is not None:
                distances.append(distance)
                valid_timestamps.append(float(timestamp))

        if valid_timestamps:
            plt.plot(valid_timestamps, distances, label=f"{animal_id1} - {animal_id2}", marker="o", markersize=4, alpha=0.7)

    plt.title(title or "Distances Between Animal Pairs Over Time")
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.grid(True, alpha=0.3)

    if pairs:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_activity_patterns(speeds, low_threshold=0.5, high_threshold=2.0, title=None, save_path=None):
    """Analyze and plot animal activity patterns."""
    plt.figure(figsize=(12, 8))

    # Define activity levels based on speed thresholds
    def get_activity(speed):
        if speed < low_threshold:
            return "Resting"
        elif speed < high_threshold:
            return "Walking"
        else:
            return "Running"

    # Collect activity data by species
    species_activities = {}

    for animal_id, speed_data in speeds.items():
        if not speed_data:
            continue

        species = extract_animal_species(animal_id)

        if species not in species_activities:
            species_activities[species] = {"Resting": 0, "Walking": 0, "Running": 0, "Total": 0}

        for _, speed in speed_data:
            activity = get_activity(speed)
            species_activities[species][activity] += 1
            species_activities[species]["Total"] += 1

    # Convert to percentages
    for species, activities in species_activities.items():
        total = activities["Total"]
        if total > 0:
            for activity in ["Resting", "Walking", "Running"]:
                activities[activity] = (activities[activity] / total) * 100

    # Prepare data for plotting
    species_names = list(species_activities.keys())
    resting_percentages = [species_activities[s]["Resting"] for s in species_names]
    walking_percentages = [species_activities[s]["Walking"] for s in species_names]
    running_percentages = [species_activities[s]["Running"] for s in species_names]

    # Create stacked bar chart
    bar_width = 0.6
    indices = np.arange(len(species_names))

    plt.bar(indices, resting_percentages, bar_width, label="Resting", color="blue")
    plt.bar(indices, walking_percentages, bar_width, bottom=resting_percentages, label="Walking", color="green")

    bottom = np.array(resting_percentages) + np.array(walking_percentages)
    plt.bar(indices, running_percentages, bar_width, bottom=bottom, label="Running", color="red")

    plt.title(title or "Activity Patterns by Species")
    plt.xlabel("Species")
    plt.ylabel("Percentage of Time (%)")
    plt.xticks(indices, species_names)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_animal_density_over_time(data, species=None, save_path=None):
    """Plot the number of animals in different regions over time."""
    plt.figure(figsize=(12, 8))

    timestamps = sorted(data.keys(), key=int)

    # Define regions (quadrants)
    regions = {
        "Northeast": lambda x, y: x > 0 and y > 0,
        "Northwest": lambda x, y: x < 0 and y > 0,
        "Southeast": lambda x, y: x > 0 and y < 0,
        "Southwest": lambda x, y: x < 0 and y < 0
    }

    # Initialize counts
    region_counts = {region: [] for region in regions}

    for timestamp in timestamps:
        # Count animals in each region
        counts = {region: 0 for region in regions}

        for animal_id, animal_data in data[timestamp].items():
            if species and not animal_id.startswith(f"{species}:"):
                continue

            x, y = map(float, animal_data["gps_coordinates"])

            for region, condition in regions.items():
                if condition(x, y):
                    counts[region] += 1

        # Add counts to lists
        for region in regions:
            region_counts[region].append(counts[region])

    # Plot counts over time
    for region, counts in region_counts.items():
        plt.plot(timestamps, counts, label=region, marker="o", markersize=4, alpha=0.7)

    plt.title(f"Animal Distribution by Region Over Time" + (f" ({species})" if species else ""))
    plt.xlabel("Time")
    plt.ylabel("Number of Animals")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def analyze_predator_prey_encounters(data, predator_species="Lion", prey_species=["Zebra", "Elephant"],
                                     distance_threshold=20.0, title=None, save_path=None):
    """
    Analyze and visualize encounters between predators and prey animals.

    Args:
        data: The animal tracking data dictionary
        predator_species: The species considered as predators
        prey_species: List of species considered as prey
        distance_threshold: Maximum distance to consider as an encounter (spatial units)
        title: Plot title
        save_path: Where to save the visualization
    """
    # Sort timestamps to analyze encounters chronologically
    timestamps = sorted(data.keys(), key=int)

    # Store encounter data
    encounters = []

    # For each timestamp, find predator-prey pairs that are close to each other
    for timestamp in timestamps:
        # Get all predators and prey at this timestamp
        predators = [animal_id for animal_id in data[timestamp].keys()
                     if animal_id.startswith(f"{predator_species}:")]

        prey = [animal_id for animal_id in data[timestamp].keys()
                if any(animal_id.startswith(f"{species}:") for species in prey_species)]

        # Check each predator-prey pair
        for predator_id in predators:
            predator_x, predator_y = map(float, data[timestamp][predator_id]["gps_coordinates"])
            predator_vitals = data[timestamp][predator_id]["vitals"]

            for prey_id in prey:
                prey_x, prey_y = map(float, data[timestamp][prey_id]["gps_coordinates"])
                prey_vitals = data[timestamp][prey_id]["vitals"]

                # Calculate distance between predator and prey
                distance = math.sqrt((predator_x - prey_x) ** 2 + (predator_y - prey_y) ** 2)

                # If they're close enough, record the encounter
                if distance <= distance_threshold:
                    encounter = {
                        "timestamp": float(timestamp),
                        "predator_id": predator_id,
                        "prey_id": prey_id,
                        "distance": distance,
                        "predator_location": (predator_x, predator_y),
                        "prey_location": (prey_x, prey_y),
                        "encounter_location": ((predator_x + prey_x) / 2, (predator_y + prey_y) / 2),
                        "predator_vitals": {
                            "heart_rate": float(predator_vitals["heart_rate"]),
                            "oxygen_saturation": float(predator_vitals["oxygen_saturation"])
                        },
                        "prey_vitals": {
                            "heart_rate": float(prey_vitals["heart_rate"]),
                            "oxygen_saturation": float(prey_vitals["oxygen_saturation"])
                        }
                    }
                    encounters.append(encounter)

    # Now visualize the encounters
    if encounters:
        plt.figure(figsize=(14, 12))

        # Plot all animal trajectories for context (with lower alpha)
        trajectories = extract_trajectories(data)
        for animal_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue

            _, x_coords, y_coords = zip(*trajectory)

            # Plot with different colors based on species
            if animal_id.startswith("Lion:"):
                plt.plot(x_coords, y_coords, color='orange', alpha=0.2, linewidth=1)
            elif animal_id.startswith("Zebra:"):
                plt.plot(x_coords, y_coords, color='black', alpha=0.2, linewidth=1)
            elif animal_id.startswith("Elephant:"):
                plt.plot(x_coords, y_coords, color='gray', alpha=0.2, linewidth=1)

        # Extract encounter locations
        encounter_locs = [e["encounter_location"] for e in encounters]
        encounter_x, encounter_y = zip(*encounter_locs) if encounter_locs else ([], [])

        # Plot encounter points
        plt.scatter(encounter_x, encounter_y, c='red', s=100, alpha=0.7, edgecolors='white', label='Predator-Prey Encounters')

        # Add count of encounters to the title
        encounter_title = title or f"Predator-Prey Encounters (n={len(encounters)})"
        plt.title(encounter_title, fontsize=16)

        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Move legend to top right corner with proper placement
        plt.legend(loc='upper right', fontsize=12)

        # Add colorbar to show encounter density if there are many encounters
        if len(encounters) > 10:
            # Create a 2D histogram for encounter density
            heatmap, xedges, yedges = np.histogram2d(encounter_x, encounter_y, bins=20)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            # Plot density as a heatmap
            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', alpha=0.3, aspect='auto')
            plt.colorbar(label='Encounter Density')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

        # Create additional plot to show heart rate changes during encounters
        plt.figure(figsize=(12, 8))

        # Organize encounters by prey species
        prey_species_encounters = {}
        for encounter in encounters:
            prey_id = encounter["prey_id"]
            prey_species = prey_id.split(":")[0]

            if prey_species not in prey_species_encounters:
                prey_species_encounters[prey_species] = []

            prey_species_encounters[prey_species].append(encounter)

        # Plot average heart rate for each prey species during encounters
        for species, species_encounters in prey_species_encounters.items():
            heart_rates = [e["prey_vitals"]["heart_rate"] for e in species_encounters]
            distances = [e["distance"] for e in species_encounters]

            plt.scatter(distances, heart_rates, alpha=0.6, label=f"{species} (n={len(species_encounters)})")

        plt.title("Prey Heart Rate vs. Distance to Predator During Encounters", fontsize=16)
        plt.xlabel("Distance to Predator", fontsize=14)
        plt.ylabel("Prey Heart Rate", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Also move this legend to the upper right
        plt.legend(loc='upper right', fontsize=12)

        # Add trend line if there are enough points
        if len(encounters) > 5:
            # Combine all encounters for overall trend line
            all_distances = [e["distance"] for e in encounters]
            all_heart_rates = [e["prey_vitals"]["heart_rate"] for e in encounters]

            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_distances, all_heart_rates)
            line_x = np.array([min(all_distances), max(all_distances)])
            line_y = slope * line_x + intercept

            plt.plot(line_x, line_y, 'r--', label=f'Trend Line (r={r_value:.2f}, p={p_value:.3f})')
            plt.legend(loc='upper right', fontsize=12)

        if save_path:
            vitals_path = save_path.replace('.png', '_vitals.png')
            plt.savefig(vitals_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    return encounters


def analyze_potential_poacher_locations(data, animal_trajectories, animal_speeds,
                                        predator_species="Lion", distance_threshold=100.0,
                                        vitals_threshold=1.2, speed_threshold=2.5,
                                        title=None, save_path=None):
    """
    Analyze animal data to identify potential poacher locations based on abnormal behaviors
    when natural predators aren't present.

    Args:
        data: The main animal tracking data dictionary
        animal_trajectories: Dictionary of animal trajectories
        animal_speeds: Dictionary of animal speeds
        predator_species: Species considered as natural predators
        distance_threshold: Distance within which predator influence is considered (spatial units)
        vitals_threshold: Threshold multiplier for abnormal vital signs
        speed_threshold: Threshold multiplier for abnormal speed/movement
        title: Plot title
        save_path: Where to save the visualization
    """
    # Sort timestamps to analyze chronologically
    timestamps = sorted(data.keys(), key=int)

    # Store potential poacher indicators
    poacher_indicators = []

    # Calculate baseline heart rates and speeds for each species
    species_baselines = {}

    for animal_id in animal_trajectories.keys():
        species = extract_animal_species(animal_id)

        if species not in species_baselines:
            species_baselines[species] = {
                "heart_rates": [],
                "speeds": []
            }

        # Collect heart rate data
        for timestamp in timestamps:
            if animal_id in data[timestamp]:
                species_baselines[species]["heart_rates"].append(
                    float(data[timestamp][animal_id]["vitals"]["heart_rate"])
                )

        # Collect speed data
        if animal_id in animal_speeds and animal_speeds[animal_id]:
            _, speeds = zip(*animal_speeds[animal_id])
            species_baselines[species]["speeds"].extend(speeds)

    # Calculate median values as baselines (more robust than mean)
    for species, values in species_baselines.items():
        if values["heart_rates"]:
            values["heart_rate_baseline"] = np.median(values["heart_rates"])
        else:
            values["heart_rate_baseline"] = 0

        if values["speeds"]:
            values["speed_baseline"] = np.median(values["speeds"])
        else:
            values["speed_baseline"] = 0

    # For each timestamp, check for abnormal behavior not explained by predators
    for timestamp_idx in range(1, len(timestamps)):
        curr_timestamp = timestamps[timestamp_idx]
        prev_timestamp = timestamps[timestamp_idx - 1]

        # Get all predators at this timestamp
        predators = [animal_id for animal_id in data[curr_timestamp].keys()
                     if animal_id.startswith(f"{predator_species}:")]

        # For each animal, check if behavior is abnormal
        for animal_id in data[curr_timestamp].keys():
            # Skip predators in this analysis
            if animal_id.startswith(f"{predator_species}:"):
                continue

            species = extract_animal_species(animal_id)

            # Check if animal was also present in previous timestamp
            if animal_id not in data[prev_timestamp]:
                continue

            # Get current animal location and vitals
            curr_x, curr_y = map(float, data[curr_timestamp][animal_id]["gps_coordinates"])
            prev_x, prev_y = map(float, data[prev_timestamp][animal_id]["gps_coordinates"])

            heart_rate = float(data[curr_timestamp][animal_id]["vitals"]["heart_rate"])

            # Calculate distance traveled and speed
            distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            time_diff = float(curr_timestamp) - float(prev_timestamp)
            speed = distance / time_diff if time_diff > 0 else 0

            # Check if there's a predator nearby
            predator_nearby = False
            for predator_id in predators:
                predator_x, predator_y = map(float, data[curr_timestamp][predator_id]["gps_coordinates"])
                predator_distance = math.sqrt((predator_x - curr_x) ** 2 + (predator_y - curr_y) ** 2)

                if predator_distance <= distance_threshold:
                    predator_nearby = True
                    break

            # If no predator nearby, check for abnormal behavior
            if not predator_nearby:
                baseline_hr = species_baselines[species]["heart_rate_baseline"]
                baseline_speed = species_baselines[species]["speed_baseline"]

                # Check for significantly elevated heart rate
                elevated_hr = heart_rate > (baseline_hr * vitals_threshold)

                # Check for erratic movement (significantly higher speed)
                erratic_movement = speed > (baseline_speed * speed_threshold)

                # If either condition is met, flag as potential poacher indicator
                if elevated_hr or erratic_movement:
                    indicator = {
                        "timestamp": float(curr_timestamp),
                        "animal_id": animal_id,
                        "location": (curr_x, curr_y),
                        "heart_rate": heart_rate,
                        "baseline_heart_rate": baseline_hr,
                        "heart_rate_ratio": heart_rate / baseline_hr if baseline_hr > 0 else 0,
                        "speed": speed,
                        "baseline_speed": baseline_speed,
                        "speed_ratio": speed / baseline_speed if baseline_speed > 0 else 0,
                        "cause": []
                    }

                    if elevated_hr:
                        indicator["cause"].append("elevated_heart_rate")
                    if erratic_movement:
                        indicator["cause"].append("erratic_movement")

                    poacher_indicators.append(indicator)

    # Now visualize the potential poacher locations
    if poacher_indicators:
        plt.figure(figsize=(14, 12))

        # Plot all animal trajectories for context (with lower alpha)
        for animal_id, trajectory in animal_trajectories.items():
            if len(trajectory) < 2:
                continue

            _, x_coords, y_coords = zip(*trajectory)

            # Plot with different colors based on species
            if animal_id.startswith("Lion:"):
                plt.plot(x_coords, y_coords, color='orange', alpha=0.2, linewidth=1)
            elif animal_id.startswith("Zebra:"):
                plt.plot(x_coords, y_coords, color='black', alpha=0.2, linewidth=1)
            elif animal_id.startswith("Elephant:"):
                plt.plot(x_coords, y_coords, color='gray', alpha=0.2, linewidth=1)

        # Extract indicator locations
        indicator_locs = [i["location"] for i in poacher_indicators]
        indicator_x, indicator_y = zip(*indicator_locs) if indicator_locs else ([], [])

        # Color by cause
        colors = []
        for indicator in poacher_indicators:
            if "elevated_heart_rate" in indicator["cause"] and "erratic_movement" in indicator["cause"]:
                colors.append('red')  # Both causes - red
            elif "elevated_heart_rate" in indicator["cause"]:
                colors.append('orange')  # Heart rate only - orange
            elif "erratic_movement" in indicator["cause"]:
                colors.append('purple')  # Erratic movement only - purple
            else:
                colors.append('blue')  # Fallback

        # Plot indicator points
        plt.scatter(indicator_x, indicator_y, c=colors, s=100, alpha=0.7, edgecolors='white')

        # Create a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Elevated HR & Erratic Movement'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Elevated Heart Rate'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Erratic Movement')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

        # Add count of indicators to the title
        indicator_title = title or f"Potential Poacher Locations (n={len(poacher_indicators)})"
        plt.title(indicator_title, fontsize=16)

        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add kernel density estimation to identify likely poacher locations
        if len(poacher_indicators) >= 5:
            # Create a grid for 2D density estimation
            x_range = np.linspace(min(indicator_x), max(indicator_x), 100)
            y_range = np.linspace(min(indicator_y), max(indicator_y), 100)
            X, Y = np.meshgrid(x_range, y_range)
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([indicator_x, indicator_y])

            # Perform kernel density estimation
            kernel = stats.gaussian_kde(values)
            Z = np.reshape(kernel(positions), X.shape)

            # Plot density contours
            contour = plt.contourf(X, Y, Z, cmap='viridis', alpha=0.3, levels=10)
            plt.colorbar(contour, label='Poacher Probability Density')

            # Mark highest density points as most likely poacher locations
            max_idx = np.unravel_index(Z.argmax(), Z.shape)
            plt.scatter(X[max_idx], Y[max_idx], c='red', s=300, marker='X', edgecolors='white', linewidths=2, label='Most Likely Poacher Location')
            plt.annotate('LIKELY POACHER', (X[max_idx], Y[max_idx]),
                         xytext=(20, 20), textcoords='offset points',
                         color='red', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                         arrowprops=dict(arrowstyle="->", color='red'))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

        # Create additional plot to show temporal pattern of abnormal behaviors
        plt.figure(figsize=(12, 8))

        # Group indicators by timestamp
        timestamp_counts = {}
        for indicator in poacher_indicators:
            ts = indicator["timestamp"]
            if ts not in timestamp_counts:
                timestamp_counts[ts] = 0
            timestamp_counts[ts] += 1

        # Sort by timestamp
        sorted_timestamps = sorted(timestamp_counts.keys())
        counts = [timestamp_counts[ts] for ts in sorted_timestamps]

        plt.bar(sorted_timestamps, counts, alpha=0.7, color='darkred')
        plt.title("Temporal Pattern of Abnormal Animal Behavior", fontsize=16)
        plt.xlabel("Timestamp", fontsize=14)
        plt.ylabel("Number of Abnormal Behaviors", fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_path:
            temporal_path = save_path.replace('.png', '_temporal.png')
            plt.savefig(temporal_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    return poacher_indicators


def analyze_animal_avoidance_areas(trajectories, grid_size=50, avoidance_threshold=0.2,
                                   consider_species=True, title=None, save_path=None):
    """
    Analyze and visualize areas that animals tend to avoid.

    Args:
        trajectories: Dictionary of animal trajectories
        grid_size: Size of the grid for analysis (higher = more detailed)
        avoidance_threshold: Threshold below which a grid cell is considered an avoidance area
        consider_species: Whether to analyze avoidance patterns by species
        title: Plot title
        save_path: Where to save the visualization
    """
    # Find bounding box for all animal locations
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for animal_id, trajectory in trajectories.items():
        for _, x, y in trajectory:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Add a small buffer to the bounds
    buffer = max(max_x - min_x, max_y - min_y) * 0.05
    min_x -= buffer
    max_x += buffer
    min_y -= buffer
    max_y += buffer

    # Create analysis grid
    x_edges = np.linspace(min_x, max_x, grid_size + 1)
    y_edges = np.linspace(min_y, max_y, grid_size + 1)

    # Calculate the grid cell centers for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # If considering species separately, organize trajectories by species
    if consider_species:
        species_trajectories = {}
        for animal_id, trajectory in trajectories.items():
            species = extract_animal_species(animal_id)

            if species not in species_trajectories:
                species_trajectories[species] = []

            species_trajectories[species].extend([(x, y) for _, x, y in trajectory])

        # Create a figure for all species
        fig, axes = plt.subplots(1, len(species_trajectories) + 1, figsize=(20, 8))

        # Overall presence/absence map
        overall_presence = np.zeros((grid_size, grid_size))
        all_points = []

        for animal_id, trajectory in trajectories.items():
            all_points.extend([(x, y) for _, x, y in trajectory])

        # Count points in each grid cell for overall map
        for x, y in all_points:
            # Find the grid cell indices
            x_idx = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * (grid_size - 1))))
            y_idx = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * (grid_size - 1))))

            overall_presence[y_idx, x_idx] += 1

        # Normalize and identify avoidance areas (overall)
        total_presence = overall_presence.sum()
        if total_presence > 0:
            overall_presence = overall_presence / total_presence

        avg_presence = np.mean(overall_presence[overall_presence > 0])
        avoidance_mask = overall_presence < (avg_presence * avoidance_threshold)

        # Plot the overall avoidance area
        axes[0].imshow(avoidance_mask, cmap='binary', interpolation='nearest', extent=[min_x, max_x, min_y, max_y], origin='lower', aspect='auto')
        axes[0].set_title(f"Overall Animal Avoidance Areas\n{np.sum(avoidance_mask)} grid cells")
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")

        # Plot avoidance areas for each species
        for i, (species, points) in enumerate(species_trajectories.items(), 1):
            presence = np.zeros((grid_size, grid_size))

            # Count points in each grid cell
            for x, y in points:
                # Find the grid cell indices
                x_idx = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * (grid_size - 1))))
                y_idx = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * (grid_size - 1))))

                presence[y_idx, x_idx] += 1

            # Normalize and identify avoidance areas
            species_total = presence.sum()
            if species_total > 0:
                presence = presence / species_total

            species_avg = np.mean(presence[presence > 0])
            species_avoidance = presence < (species_avg * avoidance_threshold)

            # Plot the species avoidance area
            axes[i].imshow(species_avoidance, cmap='binary', interpolation='nearest', extent=[min_x, max_x, min_y, max_y], origin='lower', aspect='auto')
            axes[i].set_title(f"{species} Avoidance Areas\n{np.sum(species_avoidance)} grid cells")
            axes[i].set_xlabel("X Coordinate")

        plt.tight_layout()

        if save_path:
            species_path = save_path.replace('.png', '_by_species.png')
            plt.savefig(species_path, dpi=300, bbox_inches="tight")
            plt.close()

    # Create main avoidance plot with trajectories overlay
    plt.figure(figsize=(14, 12))

    # Initialize presence grid
    presence = np.zeros((grid_size, grid_size))

    # Count points in each grid cell
    all_points = []
    for animal_id, trajectory in trajectories.items():
        all_points.extend([(x, y) for _, x, y in trajectory])

        # Plot animal trajectories for context
        _, x_coords, y_coords = zip(*trajectory)

        # Color by species
        species = extract_animal_species(animal_id)
        if species == "Lion":
            plt.plot(x_coords, y_coords, color='orange', alpha=0.2, linewidth=1)
        elif species == "Zebra":
            plt.plot(x_coords, y_coords, color='black', alpha=0.2, linewidth=1)
        elif species == "Elephant":
            plt.plot(x_coords, y_coords, color='gray', alpha=0.2, linewidth=1)

    for x, y in all_points:
        # Find the grid cell indices
        x_idx = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * (grid_size - 1))))
        y_idx = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * (grid_size - 1))))

        presence[y_idx, x_idx] += 1

    # Calculate traversable area
    traversable = presence > 0

    # Calculate expected presence (average density across traversable areas)
    total_presence = presence.sum()
    traversable_cells = np.sum(traversable)

    if traversable_cells > 0:
        expected_presence = total_presence / traversable_cells
    else:
        expected_presence = 0

    # Identify significant avoidance areas
    # Only consider cells where the presence is:
    # 1. Non-zero (animals can reach it)
    # 2. Significantly lower than expected (avoidance area)
    avoidance_areas = (presence > 0) & (presence < (expected_presence * avoidance_threshold))

    # Create avoidance intensity (how much animals avoid each area)
    avoidance_intensity = np.zeros_like(presence)
    avoidance_intensity[avoidance_areas] = 1 - (presence[avoidance_areas] / (expected_presence * avoidance_threshold))

    # Visualize avoidance areas
    plt.imshow(avoidance_intensity, cmap='Reds', interpolation='nearest',
               extent=[min_x, max_x, min_y, max_y], origin='lower', aspect='auto',
               alpha=0.6, vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(label='Avoidance Intensity')

    # Add contour lines to highlight most avoided areas
    contour_levels = np.linspace(0.3, 1.0, 4)
    contour = plt.contour(X, Y, avoidance_intensity, levels=contour_levels,
                          colors='red', linewidths=1, alpha=0.7)
    plt.clabel(contour, inline=True, fontsize=10, fmt='%.1f')

    # Identify and mark the most avoided areas
    # Find the top 3 most avoided areas
    flat_indices = np.argsort(avoidance_intensity.flatten())[-3:]
    top_avoided_coords = []

    for flat_idx in flat_indices:
        y_idx, x_idx = np.unravel_index(flat_idx, avoidance_intensity.shape)

        # Only include areas with significant avoidance
        if avoidance_intensity[y_idx, x_idx] > 0.5:
            coord_x = x_centers[x_idx]
            coord_y = y_centers[y_idx]
            top_avoided_coords.append((coord_x, coord_y, avoidance_intensity[y_idx, x_idx]))

    # Mark the top avoided areas on the map
    for i, (x, y, intensity) in enumerate(top_avoided_coords):
        plt.scatter(x, y, c='white', s=200, marker='X',
                    edgecolors='black', linewidths=1.5, zorder=10)
        plt.annotate(f'Avoidance Area {i + 1}', (x, y),
                     xytext=(10, 10), textcoords='offset points',
                     color='black', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", color='black'))

    # Set title and labels
    avoidance_title = title or f"Animal Avoidance Areas (n={np.sum(avoidance_areas)} grid cells)"
    plt.title(avoidance_title, fontsize=16)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return avoidance_areas, avoidance_intensity


def plot_vitals_colored_trajectories(data, trajectories, vital_type="heart_rate",
                                     show_by_species=True, title=None, save_path=None):
    """
    Create trajectory plots with lines color-coded by vital signs and marked start/end positions.

    Args:
        data: The main animal tracking data dictionary
        trajectories: Dictionary of animal trajectories
        vital_type: Which vital sign to use for coloring ('heart_rate' or 'oxygen_saturation')
        show_by_species: Whether to create separate plots for each species
        title: Plot title
        save_path: Where to save the visualization
    """
    # Define distinct species present in the data
    species_set = set(extract_animal_species(animal_id) for animal_id in trajectories.keys())
    species_list = sorted(list(species_set))

    # Create the main plot with all animals
    plt.figure(figsize=(14, 12))

    # Track min/max vital values for consistent color scaling
    min_vital = float('inf')
    max_vital = float('-inf')

    # Prepare trajectory segments with color values
    animal_segments = {}
    animal_vitals = {}
    animal_starts = {}
    animal_ends = {}

    for animal_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        # Extract coordinates and timestamps
        times, x_coords, y_coords = zip(*trajectory)

        # Store start and end positions
        animal_starts[animal_id] = (x_coords[0], y_coords[0])
        animal_ends[animal_id] = (x_coords[-1], y_coords[-1])

        # Collect vital values for this animal
        vital_values = []
        for t in times:
            t_str = str(int(t))  # Convert to string to match data dictionary keys
            if t_str in data and animal_id in data[t_str]:
                vital_value = float(data[t_str][animal_id]["vitals"][vital_type])
                vital_values.append(vital_value)

                # Update min/max for color scaling
                min_vital = min(min_vital, vital_value)
                max_vital = max(max_vital, vital_value)
            else:
                # Use previous value or a default if no data
                if vital_values:
                    vital_values.append(vital_values[-1])
                else:
                    vital_values.append(0)

        # Create line segments for coloring
        segments = []
        for i in range(len(x_coords) - 1):
            segments.append([(x_coords[i], y_coords[i]), (x_coords[i + 1], y_coords[i + 1])])

        animal_segments[animal_id] = segments
        animal_vitals[animal_id] = vital_values[:-1]  # One less than points

    # Function to plot a single trajectory set
    def plot_trajectories(ax, animal_ids, plot_title):
        # Create color normalizer
        norm = plt.Normalize(min_vital, max_vital)

        # Plot each animal's trajectory
        for animal_id in animal_ids:
            if animal_id not in animal_segments:
                continue

            species = extract_animal_species(animal_id)
            segments = animal_segments[animal_id]
            vitals = animal_vitals[animal_id]

            # Create line collection with color mapping
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(np.array(vitals))
            lc.set_linewidth(2)
            line = ax.add_collection(lc)

            # Plot start position (green circle)
            start_x, start_y = animal_starts[animal_id]
            ax.scatter(start_x, start_y, c='green', s=100, edgecolor='white', linewidth=1, zorder=10, alpha=0.8)

            # Plot end position (red diamond)
            end_x, end_y = animal_ends[animal_id]
            ax.scatter(end_x, end_y, c='red', s=100, marker='D', edgecolor='white', linewidth=1, zorder=10, alpha=0.8)

            # Add animal ID label at end position
            ax.annotate(animal_id, (end_x, end_y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        # Set axis limits based on data
        ax.set_xlim(min(x for _, (x, _) in animal_starts.items()) - 10,
                    max(x for _, (x, _) in animal_ends.items()) + 10)
        ax.set_ylim(min(y for _, (_, y) in animal_starts.items()) - 10,
                    max(y for _, (_, y) in animal_ends.items()) + 10)

        # Add colorbar
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label(f'{vital_type.replace("_", " ").title()}')

        # Set title and labels
        ax.set_title(plot_title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)

        # Add legend for start/end markers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start Position'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='End Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # If plotting by species, create a grid of subplots
    if show_by_species:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Plot all animals
        plot_trajectories(axes[0], trajectories.keys(),
                          f"All Animal Trajectories - Colored by {vital_type.replace('_', ' ').title()}")

        # Plot each species separately
        for i, species in enumerate(species_list, 1):
            if i < len(axes):  # Ensure we don't exceed available subplots
                species_animals = [animal_id for animal_id in trajectories.keys()
                                   if extract_animal_species(animal_id) == species]
                plot_trajectories(axes[i], species_animals,
                                  f"{species} Trajectories - Colored by {vital_type.replace('_', ' ').title()}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    else:
        # Just plot the main figure with all animals
        plot_trajectories(plt.gca(), trajectories.keys(), title or f"Animal Trajectories - Colored by {vital_type.replace('_', ' ').title()}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


def plot_zebra_pair_trajectories(data, animal_trajectories, min_closeness_timestamps=3,
                                 distance_threshold=20.0, title=None, save_path=None):
    """
    Visualize trajectories of zebra pairs that consistently stay close to each other.

    Args:
        data: The main animal tracking data dictionary
        animal_trajectories: Dictionary of animal trajectories
        min_closeness_timestamps: Minimum number of timestamps where zebras must be close
        distance_threshold: Maximum distance to consider zebras as a pair
        title: Plot title
        save_path: Where to save the visualization
    """
    # Sort timestamps
    timestamps = sorted(data.keys(), key=int)

    # Collect all zebra IDs
    zebra_ids = [animal_id for animal_id in animal_trajectories.keys()
                 if animal_id.startswith("Zebra:")]

    # Find all possible zebra pairs and track when they're close
    pair_closeness = {}

    for i in range(len(zebra_ids)):
        for j in range(i + 1, len(zebra_ids)):
            zebra1_id = zebra_ids[i]
            zebra2_id = zebra_ids[j]

            # Initialize pair data
            pair = (zebra1_id, zebra2_id)
            pair_closeness[pair] = {
                "close_timestamps": [],
                "distances": []
            }

            # Check proximity at each timestamp
            for timestamp in timestamps:
                if (zebra1_id in data[timestamp] and
                        zebra2_id in data[timestamp]):

                    # Get coordinates
                    x1, y1 = map(float, data[timestamp][zebra1_id]["gps_coordinates"])
                    x2, y2 = map(float, data[timestamp][zebra2_id]["gps_coordinates"])

                    # Calculate distance
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    # Track all distances
                    pair_closeness[pair]["distances"].append((timestamp, distance))

                    # Record if they're close
                    if distance <= distance_threshold:
                        pair_closeness[pair]["close_timestamps"].append(timestamp)

    # Filter for consistent pairs (zebras that are close for enough timestamps)
    consistent_pairs = {pair: data for pair, data in pair_closeness.items()
                        if len(data["close_timestamps"]) >= min_closeness_timestamps}

    # Sort pairs by how often they're close (most consistent pairs first)
    sorted_pairs = sorted(consistent_pairs.items(), key=lambda x: len(x[1]["close_timestamps"]), reverse=True)

    # Plot the trajectories of consistent pairs
    if sorted_pairs:
        # Let's pick the top 5 most consistent pairs to avoid overcrowding
        top_pairs = sorted_pairs[:5]

        # Create figure
        plt.figure(figsize=(16, 12))

        # Define colors for the pairs - use named colors directly
        pair_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        # Plot each pair
        for i, ((zebra1_id, zebra2_id), pair_data) in enumerate(top_pairs):
            # Get pair color
            pair_color = pair_colors[i % len(pair_colors)]

            # Get trajectories
            trajectory1 = animal_trajectories[zebra1_id]
            trajectory2 = animal_trajectories[zebra2_id]

            # Extract coordinates
            times1, x1, y1 = zip(*trajectory1)
            times2, x2, y2 = zip(*trajectory2)

            # Plot individual trajectories with slightly different shades
            plt.plot(x1, y1, color=pair_color, linestyle='-', linewidth=2,
                     alpha=0.7, label=f"{zebra1_id}")
            plt.plot(x2, y2, color=pair_color, linestyle='--', linewidth=2,
                     alpha=0.7, label=f"{zebra2_id}")

            # Mark start positions
            plt.scatter(x1[0], y1[0], color=pair_color, marker='o', s=100, edgecolor='white', linewidth=1.5, zorder=10)
            plt.scatter(x2[0], y2[0], color=pair_color, marker='o', s=100, edgecolor='white', linewidth=1.5, zorder=10)

            # Mark end positions
            plt.scatter(x1[-1], y1[-1], color=pair_color, marker='X', s=100, edgecolor='white', linewidth=1.5, zorder=10)
            plt.scatter(x2[-1], y2[-1], color=pair_color, marker='X', s=100, edgecolor='white', linewidth=1.5, zorder=10)

            # Add connection lines when they're close
            for close_ts in pair_data["close_timestamps"]:
                # Find the closest timestamp in each trajectory
                close_ts_float = float(close_ts)
                ts_idx1 = min(range(len(times1)), key=lambda i: abs(times1[i] - close_ts_float))
                ts_idx2 = min(range(len(times2)), key=lambda i: abs(times2[i] - close_ts_float))

                # Draw a line connecting them at this timestamp
                plt.plot([x1[ts_idx1], x2[ts_idx2]], [y1[ts_idx1], y2[ts_idx2]], color=pair_color, linestyle=':', linewidth=1, alpha=0.5)

        # Add a legend for pair identification
        legend_elements = []
        for i, ((zebra1_id, zebra2_id), _) in enumerate(top_pairs):
            pair_color = pair_colors[i % len(pair_colors)]
            legend_elements.append(plt.Line2D([0], [0], color=pair_color, linewidth=2, label=f"Pair {i + 1}: {zebra1_id}-{zebra2_id}"))

        # Add legend elements for start/end markers
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Start Position'))
        legend_elements.append(plt.Line2D([0], [0], marker='X', color='w',markerfacecolor='black', markersize=10, label='End Position'))

        # Set plot title and labels
        plt.title(title or "Zebra Pair Trajectories", fontsize=16)
        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add data summary to the plot
        plt.figtext(0.02, 0.02,
                    f"Found {len(sorted_pairs)} consistent zebra pairs\n"
                    f"Showing top {min(len(sorted_pairs), 5)} pairs\n"
                    f"Proximity threshold: {distance_threshold} units",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        # Add the legend outside the plot to avoid overcrowding
        plt.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(1.15, 1), fontsize=10)

        # Create a separate table to show pair statistics
        fig_stats = plt.figure(figsize=(10, 2 + 0.5 * len(top_pairs)))
        ax_stats = fig_stats.add_subplot(111)
        ax_stats.axis('off')

        # Prepare table data
        table_data = []
        for i, ((zebra1_id, zebra2_id), pair_data) in enumerate(top_pairs):
            # Calculate statistics for this pair
            close_count = len(pair_data["close_timestamps"])
            distances = [d for _, d in pair_data["distances"]]
            avg_distance = np.mean(distances) if distances else 0
            min_distance = np.min(distances) if distances else 0
            max_distance = np.max(distances) if distances else 0

            table_data.append([
                f"Pair {i + 1}",
                f"{zebra1_id}-{zebra2_id}",
                f"{close_count}",
                f"{avg_distance:.2f}",
                f"{min_distance:.2f}",
                f"{max_distance:.2f}"
            ])

        # Create and format the table
        table = ax_stats.table(
            cellText=table_data,
            colLabels=["Pair", "IDs", "Times Close", "Avg Dist", "Min Dist", "Max Dist"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        fig_stats.tight_layout()

        # Save or show the plots
        if save_path:
            plt.figure(1)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            stats_path = save_path.replace('.png', '_stats.png')
            plt.figure(2)
            plt.savefig(stats_path, dpi=300, bbox_inches="tight")

            plt.close('all')
        else:
            plt.tight_layout()
            plt.show()

        return sorted_pairs
    else:
        print("No consistent zebra pairs found with the current thresholds.")
        return []


# Run enhanced visualizations
if __name__ == "__main__":
    print(f"Generating wildlife conservation visualizations for {file_name}...")

    # Extract trajectories and calculate speeds
    animal_trajectories = extract_trajectories(data)
    animal_speeds = calculate_speeds(animal_trajectories)
    animal_vitals = collect_vitals(data)

    # Density map
    plot_3d_density_map(animal_trajectories, title="3D Animal Activity Density Map", save_path=f"{vis_dir}/3d_animal_density_map.png")
    zebra_trajectories = {animal_id: trajectory for animal_id, trajectory in animal_trajectories.items() if animal_id.startswith("Zebra:")}
    plot_3d_density_map(zebra_trajectories, title="3D Zebra Activity Density Map", save_path=f"{vis_dir}/3d_zebra_density_map.png")
    lion_trajectories = {animal_id: trajectory for animal_id, trajectory in animal_trajectories.items() if animal_id.startswith("Lion:")}
    plot_3d_density_map(lion_trajectories, title="3D Lion Activity Density Map", save_path=f"{vis_dir}/3d_lion_density_map.png")
    elephant_trajectories = {animal_id: trajectory for animal_id, trajectory in animal_trajectories.items() if animal_id.startswith("Elephant:")}
    plot_3d_density_map(elephant_trajectories, title="3D Elephant Activity Density Map", save_path=f"{vis_dir}/3d_elephant_density_map.png")

    # 1. Is the Zebra population healthy?
    plot_vitals_boxplot(animal_vitals, "heart_rate", "Heart Rate Distribution by Species", f"{vis_dir}/heart_rate_boxplot.png")
    plot_vitals_boxplot(animal_vitals, "oxygen_saturation", "Oxygen Saturation Distribution by Species", f"{vis_dir}/oxygen_saturation_boxplot.png")
    # Create heart rate and oxygen saturation colored trajectory plots
    plot_vitals_colored_trajectories(data, animal_trajectories, vital_type="heart_rate", title="Animal Trajectories with Heart Rate Visualization", save_path=f"{vis_dir}/animal_trajectories_heart_rate.png")
    plot_vitals_colored_trajectories(data, animal_trajectories, vital_type="oxygen_saturation", title="Animal Trajectories with Oxygen Saturation Visualization", save_path=f"{vis_dir}/animal_trajectories_oxygen.png")

    # 2. Do the Zebras have enough room to move around in?
    plot_trajectories(animal_trajectories, "Zebra", "Zebra Movement Trajectories", f"{vis_dir}/zebra_trajectories.png")

    # 3. Do you see any signs of poachers?
    plot_trajectories(animal_trajectories, title="All Animal Movement Trajectories", save_path=f"{vis_dir}/all_animal_trajectories.png")

    # 4. Plot a CDF of the movement speed of Zebras
    plot_speed_cdf(animal_speeds, "Zebra", "CDF of Zebra Movement Speeds", f"{vis_dir}/zebra_speed_cdf.png")

    plot_speed_cdf(animal_speeds, title="CDF of Movement Speeds for All Species", save_path=f"{vis_dir}/all_speed_cdf.png")

    # 5. Do Zebras make friends? Find pairs that stay together
    zebra_pairs = find_consistent_pairs(data, "Zebra", threshold=20.0, min_timestamps=2)
    if zebra_pairs:
        # Get the top 5 most consistent pairs
        top_pairs = sorted(zebra_pairs.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        top_pair_ids = [pair for pair, _ in top_pairs]
        plot_pair_distances(data, top_pair_ids, "Distances Between Consistent Zebra Pairs", f"{vis_dir}/zebra_pair_distances.png")

    # Trajectory pairing, semi-redundant with above pairing data and less useful
    zebra_pairs = plot_zebra_pair_trajectories(
        data, animal_trajectories,
        min_closeness_timestamps=2,  # Require at least 2 timestamps of closeness
        distance_threshold=25.0,  # Consider zebras close if within 25 units
        title="Zebra Social Pairs - Movement Patterns",
        save_path=f"{vis_dir}/zebra_pair_trajectories.png"
    )

    # 6 & 7. What locations do Zebras tend to congregate at/avoid?
    zebra_trajectories = {animal_id: trajectory for animal_id, trajectory in animal_trajectories.items() if animal_id.startswith("Zebra:")}
    zebra_heatmap, zebra_bounds = create_location_heatmap(zebra_trajectories, grid_size=30)
    plot_location_heatmap(zebra_heatmap, zebra_bounds, "Zebra Location Frequency Heatmap", f"{vis_dir}/zebra_location_heatmap.png")

    # 8. Additional visualizations
    # 3D Trajectories plot, very hard to read with too much data
    plot_3d_trajectories(animal_trajectories, title="3D Animal Movement Trajectories Over Time", save_path=f"{vis_dir}/3d_animal_trajectories.png")
    # Activity patterns analysis
    plot_activity_patterns(animal_speeds, title="Activity Patterns by Species", save_path=f"{vis_dir}/activity_patterns.png")
    # Animal density over time
    plot_animal_density_over_time(data, "Zebra", f"{vis_dir}/zebra_density_over_time.png")

    # Predator/Prey encounters
    encounters = analyze_predator_prey_encounters(data, title="Lion Encounters with Zebras and Elephants", save_path=f"{vis_dir}/predator_prey_encounters.png")
    zebra_encounters = analyze_predator_prey_encounters(data, prey_species=["Zebra"], title="Lion-Zebra Encounters", save_path=f"{vis_dir}/lion_zebra_encounters.png")
    elephant_encounters = analyze_predator_prey_encounters(data, prey_species=["Elephant"], title="Lion-Elephant Encounters", save_path=f"{vis_dir}/lion_elephant_encounters.png")

    # Not super helpful visualizations
    poacher_indicators = analyze_potential_poacher_locations(
        data, animal_trajectories, animal_speeds,
        title="Potential Poacher Locations Based on Abnormal Animal Behavior",
        save_path=f"{vis_dir}/potential_poacher_locations.png"
    )
    avoidance_areas, avoidance_intensity = analyze_animal_avoidance_areas(
        animal_trajectories,
        title="Areas Consistently Avoided by Animals",
        save_path=f"{vis_dir}/animal_avoidance_areas.png"
    )
    zebra_avoidance_areas, _ = analyze_animal_avoidance_areas(
        zebra_trajectories,
        title="Areas Consistently Avoided by Zebras",
        save_path=f"{vis_dir}/zebra_avoidance_areas.png"
    )

    print(f"All visualizations have been saved to {vis_dir}/")
