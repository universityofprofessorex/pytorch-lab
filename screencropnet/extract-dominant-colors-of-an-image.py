#!/usr/bin/env python

# SOURCE: https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/

import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd

batman_image = img.imread('autocropped/cropped-ObjLocModelV1-2021-10-20_12-44-46_000.png')

r = []
g = []
b = []
for row in batman_image:
	for temp_r, temp_g, temp_b in row:
		r.append(temp_r)
		g.append(temp_g)
		b.append(temp_b)

batman_df = pd.DataFrame({'red' : r,
						'green' : g,
						'blue' : b})

batman_df['scaled_color_red'] = whiten(batman_df['red'])
batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
batman_df['scaled_color_green'] = whiten(batman_df['green'])

cluster_centers, _ = kmeans(batman_df[['scaled_color_red',
									'scaled_color_blue',
									'scaled_color_green']], 3)

dominant_colors = []

red_std, green_std, blue_std = batman_df[['red',
										'green',
										'blue']].std()

for cluster_center in cluster_centers:
	red_scaled, green_scaled, blue_scaled = cluster_center
	dominant_colors.append((
		red_scaled * red_std / 255,
		green_scaled * green_std / 255,
		blue_scaled * blue_std / 255
	))

plt.imshow([dominant_colors])
plt.show()
