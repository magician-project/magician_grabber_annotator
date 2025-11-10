import csv
import cv2
import numpy as np
import os

# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
def calculateRelativeValue(y, h, value, minVal, maxVal):
    if maxVal == minVal:
        return int(y + h / 2)
    normalized = (value - minVal) / (maxVal - minVal)
    return int(y + h - normalized * h)
# -------------------------------------------------------
def getColor(index):
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (200, 200, 200), (100, 255, 100), (255, 150, 0)
    ]
    return colors[index % len(colors)]
# -------------------------------------------------------
def drawSinglePlot(history, plotNumber, itemName, image, x, y, w, h, minimumValue, maximumValue):
    color = getColor(plotNumber)
    if minimumValue == maximumValue:
        color = (40, 40, 40)

    cv2.line(image, (x, y + h), (x + w, y + h), color, 1)
    cv2.line(image, (x, y), (x, y + h), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    thickness = 1
    
    shift = 10 * plotNumber
    cv2.putText(image, f'{itemName}', (x,shift + y + 0), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, f'Max {maximumValue:.2f}', (x,shift + y + 10), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, f'Min {minimumValue:.2f}', (x,shift + y + h + 20), font, fontScale, color, thickness, cv2.LINE_AA)

    for frameID in range(1, len(history)):
        prevVal = calculateRelativeValue(0, h, history[frameID - 1][itemName], minimumValue, maximumValue)
        nextVal = calculateRelativeValue(0, h, history[frameID][itemName], minimumValue, maximumValue)
        cv2.line(image, (x + frameID - 1, prevVal), (x + frameID, nextVal), color, 1)

    # Label the last value
    org = (x + len(history), calculateRelativeValue(0, h, history[-1][itemName], minimumValue, maximumValue))
    cv2.putText(image, f'{history[-1][itemName]:.2f}', org, font, fontScale, color, thickness, cv2.LINE_AA)
# -------------------------------------------------------
# CSV Loaders
# -------------------------------------------------------
def load_csv_with_headers(path):
    """Load a CSV that includes a header row."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = {col: [] for col in header}
        for row in reader:
            if len(row) < len(header):
                continue
            for i, col in enumerate(header):
                data[col].append(float(row[i]))
    for k in data:
        data[k] = np.array(data[k], dtype=float)
    return data
# -------------------------------------------------------
def load_csv_without_headers(path, x_label="x", y_label="y"):
    """Load a simple two-column CSV without headers."""
    x_vals, y_vals = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                x_vals.append(float(row[0]))
                y_vals.append(float(row[1]))
    return {x_label: np.array(x_vals, dtype=float), y_label: np.array(y_vals, dtype=float)}
# -------------------------------------------------------
# Main Visualization Class
# -------------------------------------------------------
class SensorVisualizer:
    def __init__(self):
        self.data = {}
        self.images = {}  # per-plot CV mats

    def add_dataset(self, name, dataset):
        self.data[name] = dataset

    def drop_column(self, dataset_name, column_name):
        """
        Removes a specific column from one of the loaded datasets.

        Args:
            dataset_name (str): Name of the dataset (e.g. 'force', 'accelerometer')
            column_name (str): Name of the column/key to drop (e.g. 'fZ')

        Example:
            visualizer.drop_column('force', 'fZ')
        """
        if dataset_name not in self.data:
            print(f"[WARN] Dataset '{dataset_name}' not found.")
            return

        dataset = self.data[dataset_name]
        if column_name not in dataset:
            print(f"[WARN] Column '{column_name}' not found in dataset '{dataset_name}'.")
            return

        del dataset[column_name]
        print(f"[INFO] Dropped column '{column_name}' from dataset '{dataset_name}'.")

    def plot_window(self, sample_number, window_size=10, width=100, height=100):
        """Create a separate image (cv::Mat) for each dataset."""
        self.images.clear()


        labelMarginX = 100
        labelMarginY = 10

        for idx, (name, dataset) in enumerate(self.data.items()):
            #img = np.zeros((height, width, 3), dtype=np.uint8)
            img = np.full((height, width, 3), (10,10,10), dtype=np.uint8)

            x0 = 2
            w = width  - labelMarginX
            h = height - labelMarginY  # margin for labels

            # Determine timestamps
            if "timestamp" in dataset:
                timestamps = dataset["timestamp"]
            else:
                timestamps = list(dataset.values())[0]

            N = len(timestamps)
            start_idx = max(0, sample_number - window_size)
            end_idx   = min(N, sample_number)

            # Build history
            history = []
            for i in range(start_idx, end_idx):
                entry = {f"{name}:{col}": dataset[col][i] for col in dataset}
                history.append(entry)

            for col_idx, col in enumerate(dataset):
                if col in ("timestamp", "dev_timestamp"):
                    continue
                item_key = f"{name}:{col}"
                values = [h[item_key] for h in history]
                if not values:
                    continue

                drawSinglePlot(
                    history,
                    col_idx,
                    item_key,
                    img,
                    x0,
                    20,   # top offset
                    w,
                    h,
                    min(values),
                    max(values),
                )

            self.images[name] = img

        return self.images

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    directory = "/media/ammar/games2/Datasets/Magician/PDB_3_A_T1/tactile"  # Change if needed

    visualizer = SensorVisualizer()

    # --- Load labeled (2-column) CSVs ---
    visualizer.add_dataset("acceleration_psd",
        load_csv_without_headers(os.path.join(directory, "acceleration_psd.csv"), "freq", "power"))
    visualizer.add_dataset("acceleration_spikeness",
        load_csv_without_headers(os.path.join(directory, "acceleration_spikeness.csv"), "time", "spike"))
    visualizer.add_dataset("force_psd",
        load_csv_without_headers(os.path.join(directory, "force_psd.csv"), "freq", "power"))
    visualizer.add_dataset("friction",
        load_csv_without_headers(os.path.join(directory, "friction.csv"), "time", "value"))

    # --- Load headered CSVs ---
    visualizer.add_dataset("accelerometer",
        load_csv_with_headers(os.path.join(directory, "accelerometer.csv")))
    visualizer.add_dataset("force",
        load_csv_with_headers(os.path.join(directory, "force.csv")))

    # --- Plot ---
    images = visualizer.plot_window(sample_number=100, window_size=100)

    # --- Show each plot in its own window ---
    for name, img in images.items():
        cv2.imshow(f"{name}", img)

    print("Press any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

