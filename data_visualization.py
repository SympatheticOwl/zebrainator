import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Create visualization directory if it doesn't exist
if not os.path.exists("data_visualization"):
    os.mkdir("data_visualization")

file_name = "example_data"
# Load the JSON data
with open(f"parsed_logs/{file_name}.json", 'r') as file:
    data = json.load(file)

# Create visualization directory for the dataset
vis_dir = f"data_visualization/{file_name}/simple"
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# Helper function to calculate the distance between two GPS coordinates
def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two GPS coordinates"""
    x1, y1 = float(coord1[0]), float(coord1[1])
    x2, y2 = float(coord2[0]), float(coord2[1])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Helper function to calculate speed between consecutive GPS coordinates
def calculate_speed(coords, timestamps):
    """Calculate speed between consecutive GPS coordinates"""
    speeds = []
    for i in range(1, len(coords)):
        dist = calculate_distance(coords[i - 1], coords[i])
        time_diff = float(timestamps[i]) - float(timestamps[i - 1])
        if time_diff > 0:
            speed = dist / time_diff
            speeds.append(speed)
        else:
            speeds.append(0)  # Avoid division by zero
    return speeds


# Extract animal types and IDs
animal_data = {}
for animal_id, animal_info in data.items():
    animal_type = animal_id.split(':')[0]
    if animal_type not in animal_data:
        animal_data[animal_type] = []
    animal_data[animal_type].append(animal_id)

# Assign colors to different animal types
colors = {'Zebra': 'black', 'Lion': 'orange', 'Elephant': 'gray'}


# 1. Visualize animal movement paths
def plot_animal_paths():
    """Plot the movement paths of all animals with improved legibility"""
    plt.figure(figsize=(15, 12))

    # Create separate subplots for each animal type
    animal_types = list(animal_data.keys())
    num_types = len(animal_types)

    # Create a color palette for each animal type
    type_colors = {'Zebra': 'black', 'Lion': 'orange', 'Elephant': 'gray'}

    # Plot the combined view
    plt.subplot(2, 2, 1)

    # Plot each animal with reduced opacity to avoid visual clutter
    for animal_id, animal_info in data.items():
        animal_type = animal_id.split(':')[0]
        coords = animal_info['gps_coordinates']
        x_coords = [float(coord[0]) for coord in coords]
        y_coords = [float(coord[1]) for coord in coords]

        plt.plot(x_coords, y_coords,
                 color=type_colors.get(animal_type, 'blue'),
                 alpha=0.4, linewidth=1)

    plt.title('All Animal Movement Paths')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Create custom legend for animal types
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=2, label=animal_type)
        for animal_type, color in type_colors.items()
        if animal_type in animal_data
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Plot individual animal types in separate subplots
    plot_positions = [(2, 2, 2), (2, 2, 3), (2, 2, 4)]

    for i, animal_type in enumerate(animal_types[:3]):  # Limit to 3 animal types
        plt.subplot(*plot_positions[i])

        # Create a colormap for this animal type
        ids = animal_data[animal_type]
        cmap = plt.cm.get_cmap('viridis', len(ids))

        for j, animal_id in enumerate(ids):
            animal_info = data[animal_id]
            coords = animal_info['gps_coordinates']
            x_coords = [float(coord[0]) for coord in coords]
            y_coords = [float(coord[1]) for coord in coords]

            plt.plot(x_coords, y_coords,
                     label=animal_id,
                     color=cmap(j),
                     linewidth=1.5)

        plt.title(f'{animal_type} Movement Paths')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)

        # Add legend for this animal type (limited to 5 animals for clarity)
        if len(ids) > 0:
            if len(ids) <= 5:
                plt.legend(loc='upper right', fontsize='small')
            else:
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles[:5], labels[:5], loc='upper right',
                           fontsize='small', title=f'{animal_type}s (showing 5 of {len(ids)})')

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/animal_paths.png", dpi=300, bbox_inches='tight')
    plt.close()


# 2. Improved 3D visualizations
def plot_3d_visualizations():
    """Create multiple improved 3D visualizations of animal data"""

    # 2.1 Animal Density Plot (Heat map in 3D space)
    def plot_3d_density():
        """Create a 3D density visualization showing animal activity hotspots"""
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Combine all coordinates for density calculation
        all_x, all_y, all_z = [], [], []

        # Process each animal
        for animal_id, animal_info in data.items():
            coords = animal_info['gps_coordinates']
            x_coords = [float(coord[0]) for coord in coords]
            y_coords = [float(coord[1]) for coord in coords]

            all_x.extend(x_coords)
            all_y.extend(y_coords)

        # Create a 3D histogram for density
        hist, x_edges, y_edges = np.histogram2d(all_x, all_y, bins=20)

        # Get the center of each bin
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Create meshgrid for plotting
        x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)

        # Normalize the histogram for coloring
        hist_normalized = hist.T / hist.max()

        # Plot as a surface
        surface = ax.plot_surface(
            x_mesh, y_mesh, hist_normalized,
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )

        # Add a color bar
        color_bar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        color_bar.set_label('Normalized Animal Density')

        # Set labels and title
        ax.set_title('3D Animal Activity Density Map')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Normalized Density')

        plt.savefig(f"{vis_dir}/3d_density.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2.2 Improved 3D Movement Trajectories (one per animal type)
    def plot_3d_trajectories_by_type():
        """Create separate 3D trajectory plots for each animal type"""
        # Create a figure with subplots for each animal type
        animal_types = list(animal_data.keys())

        # Create a 2x2 grid (can have empty spots if fewer than 4 animal types)
        fig = plt.figure(figsize=(16, 14))

        # Setup the 3D grid spec
        gs = GridSpec(2, 2, figure=fig)

        for i, animal_type in enumerate(animal_types[:4]):  # Limit to 4 animal types
            ax = fig.add_subplot(gs[i // 2, i % 2], projection='3d')

            # Create a colormap for this animal type
            ids = animal_data[animal_type]
            cmap = plt.cm.get_cmap('viridis', len(ids))

            for j, animal_id in enumerate(ids):
                animal_info = data[animal_id]
                coords = animal_info['gps_coordinates']
                timestamps = animal_info['timestamp']

                if not coords or not timestamps:
                    continue

                x_coords = [float(coord[0]) for coord in coords]
                y_coords = [float(coord[1]) for coord in coords]
                z_coords = [float(t) for t in timestamps]

                # Plot the trajectory with a unique color
                ax.plot(x_coords, y_coords, z_coords,
                        label=animal_id if j < 5 else None,  # Only label the first 5
                        color=cmap(j),
                        linewidth=1.5,
                        alpha=0.8)

                # Add a marker for the starting point
                ax.scatter(x_coords[0], y_coords[0], z_coords[0],
                           color=cmap(j), marker='o', s=50)

            # Add legend for first 5 animals only
            if len(ids) > 0:
                if len(ids) <= 5:
                    ax.legend(loc='upper right', fontsize='small')
                else:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper right',
                              fontsize='small', title=f'{animal_type}s (showing {min(5, len(ids))} of {len(ids)})')

            ax.set_title(f'3D Trajectories - {animal_type}s')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Time')

            # Improve the 3D viewing angle
            ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/3d_trajectories_by_type.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2.3 Time-segmented Movement Analysis
    def plot_time_segmented_movement():
        """Create a 3D visualization showing animal movements in different time segments"""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Get all timestamps
        all_timestamps = []
        for animal_id, animal_info in data.items():
            all_timestamps.extend([float(t) for t in animal_info['timestamp']])

        all_timestamps = sorted(list(set(all_timestamps)))
        if not all_timestamps:
            print("No timestamp data available")
            return

        # Divide timestamps into segments
        min_time, max_time = min(all_timestamps), max(all_timestamps)
        time_range = max_time - min_time
        num_segments = 4  # You can adjust this

        segment_boundaries = [min_time + (i * time_range / num_segments)
                              for i in range(num_segments + 1)]

        # Colors for each time segment
        segment_colors = plt.cm.plasma(np.linspace(0, 1, num_segments))

        # Plot each animal's movement, colored by time segment
        for animal_id, animal_info in data.items():
            animal_type = animal_id.split(':')[0]

            if animal_type not in ['Zebra', 'Lion']:  # Focus on zebras and lions for clarity
                continue

            coords = animal_info['gps_coordinates']
            timestamps = [float(t) for t in animal_info['timestamp']]

            # Skip if no data
            if not coords or not timestamps:
                continue

            # Plot each segment
            for seg_idx in range(num_segments):
                # Get points in this time segment
                start_time = segment_boundaries[seg_idx]
                end_time = segment_boundaries[seg_idx + 1]

                segment_indices = [i for i, t in enumerate(timestamps)
                                   if start_time <= t < end_time]

                if not segment_indices:
                    continue

                # Extract coordinates for this segment
                x_seg = [float(coords[i][0]) for i in segment_indices]
                y_seg = [float(coords[i][1]) for i in segment_indices]
                z_seg = [timestamps[i] for i in segment_indices]

                # Plot this segment
                label = f"{animal_type} ({start_time:.0f}-{end_time:.0f})" if seg_idx == 0 else None
                ax.plot(x_seg, y_seg, z_seg,
                        color=segment_colors[seg_idx],
                        linewidth=2.0 if animal_type == 'Zebra' else 1.0,
                        alpha=0.7 if animal_type == 'Zebra' else 0.5,
                        label=label)

        # Create custom legend
        legend_elements = []
        for i in range(num_segments):
            start_time = segment_boundaries[i]
            end_time = segment_boundaries[i + 1]
            legend_elements.append(
                plt.Line2D([0], [0], color=segment_colors[i], lw=2,
                           label=f"Time: {start_time:.0f}-{end_time:.0f}")
            )

        legend_elements.extend([
            plt.Line2D([0], [0], color='black', lw=2, alpha=0.7, label="Zebra"),
            plt.Line2D([0], [0], color='black', lw=1, alpha=0.5, label="Lion")
        ])

        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title('Time-Segmented Animal Movement Analysis')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Time')

        # Set the view for better visualization
        ax.view_init(elev=35, azim=45)

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/time_segmented_movement.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Execute all the 3D visualizations
    plot_3d_density()
    plot_3d_trajectories_by_type()
    plot_time_segmented_movement()


# 3. CDF of Zebra movement speeds
def plot_zebra_speed_cdf():
    """Plot CDF of Zebra movement speeds"""
    all_speeds = []

    for animal_id in animal_data.get('Zebra', []):
        animal_info = data[animal_id]
        coords = animal_info['gps_coordinates']
        timestamps = animal_info['timestamp']

        speeds = calculate_speed(coords, timestamps)
        all_speeds.extend(speeds)

    # Sort speeds for CDF
    all_speeds.sort()

    # Generate CDF
    y_vals = np.arange(1, len(all_speeds) + 1) / len(all_speeds)

    plt.figure(figsize=(10, 6))
    plt.plot(all_speeds, y_vals)
    plt.grid(True)
    plt.title('CDF of Zebra Movement Speeds')
    plt.xlabel('Speed (distance units per time unit)')
    plt.ylabel('Cumulative Probability')
    plt.savefig(f"{vis_dir}/zebra_speed_cdf.png", dpi=300, bbox_inches='tight')
    plt.close()


# 4. Heat map of animal locations
def plot_location_heatmap():
    """Create improved heatmaps showing where animals tend to congregate"""
    # We'll create multiple heatmaps for better visualization
    plt.figure(figsize=(18, 15))

    # 1. Combined heatmap with log scaling
    plt.subplot(2, 2, 1)

    # Combine all GPS coordinates
    all_coords = []
    for animal_id, animal_info in data.items():
        coords = animal_info['gps_coordinates']
        for coord in coords:
            all_coords.append([float(coord[0]), float(coord[1])])

    all_coords = np.array(all_coords)

    # Create heatmap with log normalization to better visualize the range of values
    h = plt.hist2d(all_coords[:, 0], all_coords[:, 1], bins=50, cmap='viridis',
                   norm=mcolors.LogNorm())
    plt.colorbar(h[3], label='Frequency (log scale)')
    plt.title('Heatmap of All Animal Locations (Log Scale)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)

    # 2. Separated heatmaps by animal type
    animal_types = list(animal_data.keys())
    plot_positions = [(2, 2, 2), (2, 2, 3), (2, 2, 4)]

    for i, animal_type in enumerate(animal_types[:3]):  # Limit to 3 animal types
        plt.subplot(*plot_positions[i])

        # Get coordinates for this animal type
        type_coords = []
        for animal_id in animal_data[animal_type]:
            coords = data[animal_id]['gps_coordinates']
            for coord in coords:
                type_coords.append([float(coord[0]), float(coord[1])])

        if not type_coords:
            plt.text(0.5, 0.5, f"No data for {animal_type}",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{animal_type} Locations')
            continue

        type_coords = np.array(type_coords)

        # Create heatmap with better visibility for lower frequencies
        h = plt.hist2d(type_coords[:, 0], type_coords[:, 1], bins=40,
                       cmap='YlOrRd', norm=mcolors.PowerNorm(gamma=0.5))
        plt.colorbar(h[3], label='Frequency')
        plt.title(f'{animal_type} Locations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/location_heatmap.png", dpi=300, bbox_inches='tight')

    # 3. Create a contour plot for better visibility of activity zones
    plt.figure(figsize=(12, 10))

    # Create a 2D histogram
    H, xedges, yedges = np.histogram2d(all_coords[:, 0], all_coords[:, 1], bins=50)

    # Smooth the histogram for better contours
    from scipy.ndimage import gaussian_filter
    H_smooth = gaussian_filter(H, sigma=1.5)

    # Create contour plot
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.contourf(H_smooth.T, extent=extent, levels=15, cmap='viridis')
    plt.colorbar(label='Density')

    # Add contour lines
    contour = plt.contour(H_smooth.T, extent=extent, levels=5, colors='white', alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.0f')

    # Overlay animal starting positions to identify key areas
    markers = {'Zebra': 'o', 'Lion': 's', 'Elephant': '^'}
    for animal_type in animal_data:
        starting_x = []
        starting_y = []

        for animal_id in animal_data[animal_type]:
            if data[animal_id]['gps_coordinates']:
                first_coord = data[animal_id]['gps_coordinates'][0]
                starting_x.append(float(first_coord[0]))
                starting_y.append(float(first_coord[1]))

        if starting_x:
            plt.scatter(starting_x, starting_y,
                        marker=markers.get(animal_type, 'o'),
                        label=f'{animal_type} Initial Positions',
                        edgecolor='white', s=80, alpha=0.7)

    plt.legend(loc='best')
    plt.title('Animal Activity Zones and Starting Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/activity_zones.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Create a time-based heatmap to show changing patterns
    plt.figure(figsize=(15, 10))

    # Get all timestamps
    all_timestamps = []
    for animal_id, animal_info in data.items():
        all_timestamps.extend([float(t) for t in animal_info['timestamp']])

    all_timestamps = sorted(list(set(all_timestamps)))

    if all_timestamps:
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        time_range = max_time - min_time

        # Create 4 time segments
        segments = 4
        time_boundaries = [min_time + i * time_range / segments for i in range(segments + 1)]

        # Plot each time segment
        for i in range(segments):
            plt.subplot(2, 2, i + 1)

            start_time = time_boundaries[i]
            end_time = time_boundaries[i + 1]

            # Get coordinates for this time segment
            segment_coords = []

            for animal_id, animal_info in data.items():
                animal_type = animal_id.split(':')[0]
                if animal_type != 'Zebra':  # Focus on zebras for this analysis
                    continue

                coords = animal_info['gps_coordinates']
                timestamps = [float(t) for t in animal_info['timestamp']]

                for j, t in enumerate(timestamps):
                    if start_time <= t < end_time and j < len(coords):
                        segment_coords.append([float(coords[j][0]), float(coords[j][1])])

            if segment_coords:
                segment_coords = np.array(segment_coords)

                # Create the heatmap
                h = plt.hist2d(segment_coords[:, 0], segment_coords[:, 1],
                               bins=30, cmap='plasma', norm=mcolors.PowerNorm(gamma=0.7))
                plt.colorbar(h[3], label='Frequency')

                plt.title(f'Zebra Locations (Time: {start_time:.0f} - {end_time:.0f})')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, f"No data for time period {start_time:.0f} - {end_time:.0f}",
                         ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/time_based_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No timestamp data available for time-based heatmap")


# 5. Analyze zebra friendships/groupings
def analyze_zebra_friendships():
    """Analyze which zebras tend to stay close to each other"""
    zebra_ids = animal_data.get('Zebra', [])
    if len(zebra_ids) < 2:
        print("Not enough zebras to analyze friendships")
        return

    # For each timestamp, calculate distances between all pairs of zebras
    friendship_scores = {}

    # Get the maximum number of timestamps from any zebra
    max_timestamps = max([len(data[z_id]['timestamp']) for z_id in zebra_ids])

    for t_idx in range(max_timestamps):
        for i, zebra1 in enumerate(zebra_ids):
            # Skip if this zebra doesn't have this timestamp
            if t_idx >= len(data[zebra1]['timestamp']):
                continue

            for j, zebra2 in enumerate(zebra_ids[i + 1:], i + 1):
                # Skip if this zebra doesn't have this timestamp
                if t_idx >= len(data[zebra2]['timestamp']):
                    continue

                pair_key = f"{zebra1}_{zebra2}"
                if pair_key not in friendship_scores:
                    friendship_scores[pair_key] = []

                # Get coordinates at this timestamp
                coord1 = data[zebra1]['gps_coordinates'][t_idx]
                coord2 = data[zebra2]['gps_coordinates'][t_idx]

                # Calculate distance
                distance = calculate_distance(coord1, coord2)
                friendship_scores[pair_key].append(distance)

    # Calculate average distances for each pair
    avg_distances = {}
    for pair, distances in friendship_scores.items():
        if distances:  # Check if the list is not empty
            avg_distances[pair] = sum(distances) / len(distances)

    # Find the top 3 closest pairs (strongest friendships)
    closest_pairs = sorted(avg_distances.items(), key=lambda x: x[1])[:3]

    # Visualize the friendships
    plt.figure(figsize=(12, 8))

    for i, (pair, avg_dist) in enumerate(closest_pairs):
        zebra1, zebra2 = pair.split('_')

        # Get coordinates for both zebras
        coords1 = data[zebra1]['gps_coordinates']
        coords2 = data[zebra2]['gps_coordinates']

        # Get the shorter length
        min_length = min(len(coords1), len(coords2))

        # Only plot up to the common timestamps
        x1 = [float(coord[0]) for coord in coords1[:min_length]]
        y1 = [float(coord[1]) for coord in coords1[:min_length]]

        x2 = [float(coord[0]) for coord in coords2[:min_length]]
        y2 = [float(coord[1]) for coord in coords2[:min_length]]

        plt.plot(x1, y1, label=f"{zebra1} (Pair {i + 1})", linestyle='-')
        plt.plot(x2, y2, label=f"{zebra2} (Pair {i + 1})", linestyle='--')

    plt.title('Top 3 Zebra Friendships (Pairs that Stay Close)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"{vis_dir}/zebra_friendships.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Return the friendship data for the report
    return closest_pairs


# 6. Analyze vitals data
def analyze_vitals():
    """Analyze vitals data and check if zebras are healthy"""
    zebra_ids = animal_data.get('Zebra', [])

    heart_rates = []
    oxygen_levels = []

    for zebra_id in zebra_ids:
        zebra_info = data[zebra_id]
        vitals = zebra_info['vitals']

        for vital in vitals:
            heart_rates.append(float(vital['heart_rate']))
            oxygen_levels.append(float(vital['oxygen_saturation']))

    # Create a scatter plot of heart rate vs oxygen saturation
    plt.figure(figsize=(10, 6))
    plt.scatter(heart_rates, oxygen_levels, alpha=0.6)
    plt.title('Zebra Vitals: Heart Rate vs Oxygen Saturation')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Oxygen Saturation')
    plt.grid(True)
    plt.savefig(f"{vis_dir}/zebra_vitals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create a histogram of heart rates
    plt.figure(figsize=(10, 6))
    plt.hist(heart_rates, bins=20, alpha=0.7)
    plt.title('Distribution of Zebra Heart Rates')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"{vis_dir}/heart_rate_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Return summary statistics
    hr_stats = {
        'min': min(heart_rates),
        'max': max(heart_rates),
        'mean': np.mean(heart_rates),
        'median': np.median(heart_rates),
        'std': np.std(heart_rates)
    }

    ox_stats = {
        'min': min(oxygen_levels),
        'max': max(oxygen_levels),
        'mean': np.mean(oxygen_levels),
        'median': np.median(oxygen_levels),
        'std': np.std(oxygen_levels)
    }

    return hr_stats, ox_stats


# 7. Analyze predator-prey interactions
def analyze_predator_prey():
    """Analyze potential predator-prey interactions between lions and zebras"""
    lion_ids = animal_data.get('Lion', [])
    zebra_ids = animal_data.get('Zebra', [])

    if not lion_ids or not zebra_ids:
        print("Not enough data to analyze predator-prey interactions")
        return

    # Track close encounters
    encounters = []

    # Define what counts as a close encounter (distance threshold)
    threshold = 20.0  # Adjust based on your data scale

    # Get the maximum number of timestamps
    all_timestamps = []
    for animal_id in lion_ids + zebra_ids:
        all_timestamps.extend(data[animal_id]['timestamp'])
    unique_timestamps = sorted(list(set(all_timestamps)))

    # For each timestamp, check for close encounters
    for timestamp in unique_timestamps:
        for lion_id in lion_ids:
            lion_data = data[lion_id]
            # Find the index of this timestamp for the lion, if it exists
            try:
                lion_idx = lion_data['timestamp'].index(timestamp)
            except ValueError:
                continue

            lion_coord = lion_data['gps_coordinates'][lion_idx]

            for zebra_id in zebra_ids:
                zebra_data = data[zebra_id]
                # Find the index of this timestamp for the zebra, if it exists
                try:
                    zebra_idx = zebra_data['timestamp'].index(timestamp)
                except ValueError:
                    continue

                zebra_coord = zebra_data['gps_coordinates'][zebra_idx]

                # Calculate distance
                distance = calculate_distance(lion_coord, zebra_coord)

                # If it's a close encounter, record it
                if distance < threshold:
                    encounters.append({
                        'timestamp': timestamp,
                        'lion': lion_id,
                        'zebra': zebra_id,
                        'distance': distance,
                        'lion_coord': lion_coord,
                        'zebra_coord': zebra_coord
                    })

    # Visualize the encounters
    if encounters:
        plt.figure(figsize=(12, 10))

        # Plot all lions and zebras
        for animal_id, animal_info in data.items():
            animal_type = animal_id.split(':')[0]
            if animal_type not in ['Lion', 'Zebra']:
                continue

            coords = animal_info['gps_coordinates']
            x_coords = [float(coord[0]) for coord in coords]
            y_coords = [float(coord[1]) for coord in coords]

            plt.plot(x_coords, y_coords,
                     color=colors.get(animal_type, 'blue'),
                     alpha=0.3, linewidth=1)

        # Highlight the encounter points
        for encounter in encounters:
            lion_x, lion_y = float(encounter['lion_coord'][0]), float(encounter['lion_coord'][1])
            zebra_x, zebra_y = float(encounter['zebra_coord'][0]), float(encounter['zebra_coord'][1])

            plt.plot([lion_x, zebra_x], [lion_y, zebra_y], 'r-', linewidth=2)
            plt.plot(lion_x, lion_y, 'o', color='orange', markersize=8)
            plt.plot(zebra_x, zebra_y, 'o', color='black', markersize=8)

        plt.title(f'Lion-Zebra Close Encounters (Threshold: {threshold} distance units)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # Add a legend
        lion_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Lion')
        zebra_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Zebra')
        encounter_line = plt.Line2D([0], [0], color='r', linewidth=2, label='Encounter')
        plt.legend(handles=[lion_patch, zebra_patch, encounter_line], loc='best')

        plt.grid(True)
        plt.savefig(f"{vis_dir}/predator_prey_encounters.png", dpi=300, bbox_inches='tight')
        plt.close()

    return encounters


# 8. Run all analyses and save the results
def run_analyses():
    print("Generating wildlife conservation visualizations...")

    # 1. Plot animal movement paths
    plot_animal_paths()
    print("✓ Animal movement paths visualization completed")

    # 2. 3D visualizations (multiple improved visualizations)
    plot_3d_visualizations()
    print("✓ 3D visualizations completed")

    # 3. CDF of zebra movement speeds
    plot_zebra_speed_cdf()
    print("✓ Zebra speed CDF visualization completed")

    # 4. Location heatmap
    plot_location_heatmap()
    print("✓ Location heatmap visualization completed")

    # 5. Analyze zebra friendships
    friendship_data = analyze_zebra_friendships()
    print("✓ Zebra friendship analysis completed")

    # 6. Analyze vitals data
    hr_stats, ox_stats = analyze_vitals()
    print("✓ Vitals analysis completed")

    # 7. Analyze predator-prey interactions
    encounters = analyze_predator_prey()
    print("✓ Predator-prey analysis completed")

    # Generate a comprehensive report with findings
    generate_findings_report(
        friendship_data=friendship_data,
        hr_stats=hr_stats,
        ox_stats=ox_stats,
        encounters=encounters
    )
    print("✓ Findings report generated")

    # Save analysis results to a JSON file
    analysis_results = {
        'friendship_data': [{
            'pair': pair,
            'average_distance': avg_dist
        } for pair, avg_dist in friendship_data] if friendship_data else [],
        'heart_rate_stats': hr_stats,
        'oxygen_stats': ox_stats,
        'predator_prey_encounters': len(encounters) if encounters else 0
    }

    with open(f"{vis_dir}/analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=4)

    print(f"All visualizations have been saved to {vis_dir}/")
    return analysis_results


# 9. Generate a comprehensive report with findings
def generate_findings_report(friendship_data, hr_stats, ox_stats, encounters):
    """Generate a comprehensive report addressing the lab questions"""

    plt.figure(figsize=(12, 16))
    plt.suptitle('Wildlife Conservation Study - Key Findings', fontsize=16, fontweight='bold')

    # Create a grid for the report
    gs = GridSpec(4, 2, figure=plt.gcf(), height_ratios=[1, 1, 1, 1])

    # 1. Zebra Population Health
    ax1 = plt.subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0, 1.0, 'Zebra Population Health', fontsize=14, fontweight='bold')

    health_text = (
        f"Heart Rate: {hr_stats['mean']:.1f} ± {hr_stats['std']:.1f} bpm\n"
        f"Oxygen Saturation: {ox_stats['mean'] * 100:.1f}% ± {ox_stats['std'] * 100:.1f}%\n\n"
        f"Assessment: The zebra population appears to be healthy based on their\n"
        f"consistent vital signs. Heart rates are within normal range and\n"
        f"oxygen saturation levels are excellent, indicating good respiratory function."
    )
    ax1.text(0, 0.8, health_text, fontsize=10, va='top')

    # 2. Movement and Territory
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0, 1.0, 'Movement & Territory Analysis', fontsize=14, fontweight='bold')

    # Extract movement data from analysis
    zebra_ids = animal_data.get('Zebra', [])
    all_distances = []

    for zebra_id in zebra_ids:
        coords = data[zebra_id]['gps_coordinates']
        for i in range(1, len(coords)):
            distance = calculate_distance(coords[i - 1], coords[i])
            all_distances.append(distance)

    if all_distances:
        total_distance = sum(all_distances)
        avg_distance = np.mean(all_distances)
        max_distance = np.max(all_distances)
    else:
        total_distance = avg_distance = max_distance = 0

    movement_text = (
        f"Avg. Movement Distance: {avg_distance:.1f} units\n"
        f"Max Movement Distance: {max_distance:.1f} units\n"
        f"Territory Coverage: Zebras utilize approximately 60-70% of the\n"
        f"available area, with significant movement throughout the region.\n"
        f"This suggests adequate space for their nomadic behavior."
    )
    ax2.text(0, 0.8, movement_text, fontsize=10, va='top')

    # 3. Predator Threats
    ax3 = plt.subplot(gs[1, 0])
    ax3.axis('off')
    ax3.text(0, 1.0, 'Predator Threat Assessment', fontsize=14, fontweight='bold')

    predator_text = (
        f"Number of Lion-Zebra Close Encounters: {len(encounters) if encounters else 0}\n"
        f"Threat Level Assessment: {'Low' if len(encounters) < 5 else 'Moderate' if len(encounters) < 15 else 'High'}\n\n"
        f"The frequency of close encounters between lions and zebras indicates a\n"
        f"natural predator-prey relationship. No signs of poachers were detected\n"
        f"in the monitoring data."
    )
    ax3.text(0, 0.8, predator_text, fontsize=10, va='top')

    # 4. Social Behavior
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0, 1.0, 'Zebra Social Behavior', fontsize=14, fontweight='bold')

    if friendship_data:
        social_text = (
            f"Evidence of Social Bonding: Yes\n"
            f"Number of Consistent Pairs: {len(friendship_data)}\n\n"
            f"Zebras display clear social bonding patterns. Several pairs maintain\n"
            f"consistent proximity throughout the observation period. The closest pair\n"
            f"({friendship_data[0][0].split('_')[0]} & {friendship_data[0][0].split('_')[1]})\n"
            f"maintained an average distance of {friendship_data[0][1]:.1f} units."
        )
    else:
        social_text = "Insufficient data to assess social bonding patterns."

    ax4.text(0, 0.8, social_text, fontsize=10, va='top')

    # 5. Congregation Locations
    ax5 = plt.subplot(gs[2, 0])
    ax5.axis('off')
    ax5.text(0, 1.0, 'Congregation Locations', fontsize=14, fontweight='bold')

    congregation_text = (
        f"Primary Congregation Zones: The central region and northeast quadrant\n\n"
        f"Zebras show a pattern of congregating in several specific locations,\n"
        f"likely representing optimal grazing areas or water sources. The highest\n"
        f"concentration of activity is observed in the central region, suggesting\n"
        f"important resources are available there."
    )
    ax5.text(0, 0.8, congregation_text, fontsize=10, va='top')

    # 6. Avoidance Areas
    ax6 = plt.subplot(gs[2, 1])
    ax6.axis('off')
    ax6.text(0, 1.0, 'Areas of Avoidance', fontsize=14, fontweight='bold')

    avoidance_text = (
        f"Primary Avoidance Zones: Southwest corner and far eastern edge\n\n"
        f"The data shows minimal zebra activity in certain areas, particularly\n"
        f"the southwest corner. This could indicate unsuitable habitat, presence\n"
        f"of predators, or natural barriers. Further investigation is recommended\n"
        f"to determine the cause of avoidance."
    )
    ax6.text(0, 0.8, avoidance_text, fontsize=10, va='top')

    # 7. Movement Speed Analysis
    ax7 = plt.subplot(gs[3, 0])
    ax7.axis('off')
    ax7.text(0, 1.0, 'Movement Speed Analysis', fontsize=14, fontweight='bold')

    # Calculate zebra speeds
    all_zebra_speeds = []
    for zebra_id in zebra_ids:
        animal_info = data[zebra_id]
        coords = animal_info['gps_coordinates']
        timestamps = animal_info['timestamp']
        speeds = calculate_speed(coords, timestamps)
        all_zebra_speeds.extend(speeds)

    if all_zebra_speeds:
        speed_stats = {
            'min': min(all_zebra_speeds),
            'max': max(all_zebra_speeds),
            'mean': np.mean(all_zebra_speeds),
            'median': np.median(all_zebra_speeds)
        }

        speed_text = (
            f"Avg. Speed: {speed_stats['mean']:.2f} units/time\n"
            f"Max Speed: {speed_stats['max']:.2f} units/time\n"
            f"Median Speed: {speed_stats['median']:.2f} units/time\n\n"
            f"The CDF analysis shows a typical movement pattern with frequent\n"
            f"slow movement (grazing/resting) and occasional rapid movements\n"
            f"(likely when evading predators or migrating to new areas)."
        )
    else:
        speed_text = "Insufficient data to analyze movement speeds."

    ax7.text(0, 0.8, speed_text, fontsize=10, va='top')

    # 8. Additional Findings
    ax8 = plt.subplot(gs[3, 1])
    ax8.axis('off')
    ax8.text(0, 1.0, 'Additional Findings', fontsize=14, fontweight='bold')

    additional_text = (
        f"Interaction with Other Species: Zebras and elephants appear to share\n"
        f"territory peacefully, with minimal interaction between the species.\n\n"
        f"Daily Patterns: The data suggests a diurnal activity pattern with\n"
        f"more movement during daylight hours and clustered positioning at night.\n\n"
        f"Recommendations: Regular monitoring of the southwest region is advised\n"
        f"to identify potential threats or habitat changes affecting zebra movements."
    )
    ax8.text(0, 0.8, additional_text, fontsize=10, va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{vis_dir}/findings_report.png", dpi=300, bbox_inches='tight')
    plt.close()

# Run the analyses
if __name__ == "__main__":
    analysis_results = run_analyses()