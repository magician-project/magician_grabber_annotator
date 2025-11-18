import csv
import matplotlib.pyplot as plt

model_names = []
steps = []
framerates = []

with open("perf.csv", "r") as f:
    reader = csv.reader(f)

    # Read header and strip whitespace
    header = next(reader)
    header = [h.strip() for h in header]

    # Build index lookup
    idx_model = header.index("model_name")
    idx_step = header.index("step")
    idx_framerate = header.index("framerate_hz")

    # Read rows
    for row in reader:
        row = [v.strip() for v in row]

        # Clean model name
        name = row[idx_model]
        if name.endswith(".pth"):
            name = name[:-4]        # remove .pth
        name = name.replace("_", " ")  # remove underscores

        model_names.append(name)
        steps.append(int(row[idx_step]))
        framerates.append(float(row[idx_framerate]))

# Plot
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    model_names,
    framerates,
    c=steps,
    s=60,
    alpha=0.8
)

plt.xlabel("Classification Model")
plt.ylabel("Framerate (Hz)")
plt.title("Performance by Model / Step / Tile Size 42 (colored by step size)")

plt.xticks(rotation=45, ha="right")

cbar = plt.colorbar(scatter)
cbar.set_label("Classifier Tile Step")

# Add red horizontal line at 23 Hz
plt.axhline(y=23, color='red', linestyle='--', linewidth=2, label="Camera Acquisition Framerate")

# Add legend (must include at least one labeled plotted object)
plt.legend()

plt.tight_layout()
plt.savefig("perf_plot.png", dpi=300)

print("Saved perf_plot.png")

