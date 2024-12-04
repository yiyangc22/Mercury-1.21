import os
import pandas as pd
import matplotlib.pyplot as plt
from mercury_01 import pyplot_create_region

# name of the experiment
EXPERM = "_latest"

# subgroup with tif images
FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", EXPERM, "Subgroup 1")

# .csv coordinates
COORDS = os.path.join(os.path.expanduser("~"), "Desktop", EXPERM, "_coordinates.csv")

# read folder, save image paths in list, and print
images = []
print("Images:")
with os.scandir(FOLDER) as folder:
    for image in folder:
        images.append(os.path.abspath(image))
print(images)
print("==============================================")

# read .csv, save xy coordinates in list, and print
csv = pd.read_csv(COORDS).values.tolist()
coords = []
print("Coordinates:")
for row in csv:
    # invert x (row[1]) or y (row[2]) axis here
    coords.append([-row[1], -row[2]])
print(coords)
print("==============================================")

# create regions in matplotlib based on the coordinates
for i, coord in enumerate(coords):
    pyplot_create_region(coord[0], coord[1], 300, 300, j=images[i], a=0.5)
plt.gca().set_aspect('equal')
plt.gcf().set_figheight(10)
plt.gcf().set_figwidth(10)
plt.show()
