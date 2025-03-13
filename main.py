import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Download the model (if not already downloaded)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Load image
image_path = "./wastewater.png"  # replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize SAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Generate masks
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Minimum area of a segment
)
masks = mask_generator.generate(image)

print(f"Number of segments: {len(masks)}")

# Display segmented image
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    m = mask["segmentation"]
    # Draw segment boundaries
    contours, _ = cv2.findContours(
        (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=1, color="red")
plt.axis("off")
plt.title("Image with Segmentations")
plt.savefig("segmented_image.png", bbox_inches="tight")
plt.show()

features = []
for mask in masks:
    m = mask["segmentation"]
    contours, _ = cv2.findContours(
        (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Extract shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Calculate convex hull and solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Extract color features - mean RGB in the mask area
        mask_3d = np.stack([m] * 3, axis=2)  # Create 3D mask for RGB
        mask_pixels = image[m]  # Get pixels where mask is True
        if len(mask_pixels) > 0:
            mean_color = np.mean(mask_pixels, axis=0)
            r_mean, g_mean, b_mean = mean_color

            # Calculate brightness/intensity (simple average of RGB)
            brightness = np.mean(mean_color)

            # Add shape and color features to list - applying weight reduction to color
            color_weight = 0.3  # Reduce weight of color features (adjust as needed)
            features.append(
                [
                    area,
                    perimeter,
                    circularity,
                    aspect_ratio,
                    solidity,
                    r_mean * color_weight,
                    g_mean * color_weight,
                    b_mean * color_weight,
                    brightness * color_weight,
                ]
            )

# Scale features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Calculate elbow curve
wcss = []
max_clusters = min(10, len(features))
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_curve.png")
plt.show()


# Automatically determine optimal number of clusters using the elbow method
def find_optimal_clusters(wcss):
    # Calculate the rate of change in inertia
    deltas = np.diff(wcss)
    # Calculate the rate of change of the rate of change
    delta_deltas = np.diff(deltas)

    # The elbow is typically where the second derivative is maximum
    elbow_index = np.argmax(delta_deltas) + 2  # +2 because of two diff operations

    # Ensure we don't return a value that's too small or too large
    if elbow_index < 2:
        elbow_index = 2
    elif elbow_index > max_clusters - 1:
        elbow_index = max_clusters - 1

    return elbow_index


n_clusters = 5
print(f"Optimal number of clusters detected: {n_clusters}")

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized_features)

# Visualize clusters
plt.figure(figsize=(12, 10))
plt.imshow(image)
colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "white", "orange"]

for i, mask in enumerate(masks):
    if i < len(clusters):
        m = mask["segmentation"]
        contours, _ = cv2.findContours(
            (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            color = colors[clusters[i] % len(colors)]
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=1.5, color=color)

# Add legend
handles = [
    plt.Line2D([0], [0], color=colors[i], lw=4, label=f"Cluster {i + 1}")
    for i in range(min(n_clusters, len(colors)))
]
plt.legend(handles=handles, loc="upper right")

plt.axis("off")
plt.title("Image with Clustered Segments (Shape + Color)")
plt.savefig("clustered_segments_with_color.png", bbox_inches="tight")
plt.show()

# Display cluster characteristics
plt.figure(figsize=(15, 6))
cluster_data = []

for cluster_id in range(n_clusters):
    cluster_members = [i for i, c in enumerate(clusters) if c == cluster_id]
    if cluster_members:
        avg_brightness = np.mean(
            [features[i][8] for i in cluster_members]
        )  # brightness is at index 8
        avg_area = np.mean(
            [features[i][0] for i in cluster_members]
        )  # area is at index 0
        cluster_data.append(
            (cluster_id, len(cluster_members), avg_brightness, avg_area)
        )

# Plot cluster statistics
cluster_ids = [d[0] + 1 for d in cluster_data]  # Adding 1 for display purposes
member_counts = [d[1] for d in cluster_data]
brightnesses = [d[2] for d in cluster_data]
areas = [d[3] for d in cluster_data]

plt.subplot(1, 2, 1)
plt.bar(cluster_ids, brightnesses, color=[colors[i - 1] for i in cluster_ids])
plt.title("Average Brightness by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Brightness Value (weighted)")

plt.subplot(1, 2, 2)
plt.bar(cluster_ids, areas, color=[colors[i - 1] for i in cluster_ids])
plt.title("Average Area by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Area (pixels)")
plt.yscale("log")  # Log scale for better visualization if areas vary greatly

plt.tight_layout()
plt.savefig("cluster_statistics.png")
plt.show()
