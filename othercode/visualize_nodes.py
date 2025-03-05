import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load JBeam file
def load_jbeam(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


# Extract node positions from JBeam
def extract_nodes(jbeam_data):
    for key in jbeam_data.keys():  # Get vehicle name dynamically
        if "nodes" in jbeam_data[key]:
            nodes = jbeam_data[key]["nodes"]
            break
    print(nodes)
    node_df = pd.DataFrame(nodes[1:], columns=nodes[0])  # Skip header row
    node_df[["posX", "posY", "posZ"]] = node_df[["posX", "posY", "posZ"]].astype(
        float
    )  # Convert to float
    return node_df


# Compute deformation (Euclidean distance)
def compute_deformation(baseline_df, damaged_df):
    baseline_df = baseline_df.set_index("id")
    damaged_df = damaged_df.set_index("id")

    # Ensure both have the same nodes
    common_nodes = baseline_df.index.intersection(damaged_df.index)
    baseline_df = baseline_df.loc[common_nodes]
    damaged_df = damaged_df.loc[common_nodes]

    # Compute displacement per node
    displacement = np.linalg.norm(
        damaged_df[["posX", "posY", "posZ"]].values
        - baseline_df[["posX", "posY", "posZ"]].values,
        axis=1,
    )

    displacement_df = pd.DataFrame({"id": common_nodes, "displacement": displacement})
    return displacement_df


# Plot displacement
def plot_deformation(displacement_df):
    plt.figure(figsize=(10, 5))
    plt.hist(displacement_df["displacement"], bins=30, edgecolor="black")
    plt.xlabel("Displacement (m)")
    plt.ylabel("Node Count")
    plt.title("JBeam Deformation Analysis")
    plt.axvline(
        displacement_df["displacement"].mean(),
        color="r",
        linestyle="dashed",
        label="Mean Displacement",
    )
    plt.legend()
    plt.show()


# Main execution
baseline_file = "mesh_data_crashed.json"
damaged_file = "mesh_data_crashed.json"

baseline_data = load_jbeam(baseline_file)
damaged_data = load_jbeam(damaged_file)

damaged_df = pd.DataFrame(pd.json_normalize(damaged_data))
print(damaged_df.shape)

print(damaged_df)
# print(damaged_data)
baseline_nodes = extract_nodes(baseline_data)
damaged_nodes = extract_nodes(damaged_data)

displacement_df = compute_deformation(baseline_nodes, damaged_nodes)
plot_deformation(displacement_df)

# Display top deformed nodes
print(displacement_df.sort_values("displacement", ascending=False).head(10))
