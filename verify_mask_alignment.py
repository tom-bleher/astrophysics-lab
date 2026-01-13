#!/usr/bin/env python3
"""
Verify that the transformed star mask aligns with bright stars in the mosaic.
"""

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load mosaic
print("Loading mosaic...")
with fits.open("fits_official/f450_mosaic_v2.fits") as hdul:
    mosaic = hdul[0].data.copy()

# Load mask
print("Loading mask...")
with fits.open("data/star_mask_mosaic.fits") as hdul:
    mask = hdul[0].data.astype(bool)

# Find mask regions
from scipy import ndimage
labeled, num_features = ndimage.label(mask)
print(f"Found {num_features} mask regions")

# Create figure with zoomed views of each region
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

vmin, vmax = np.percentile(mosaic, [1, 99.5])

for i in range(num_features):
    region_y, region_x = np.where(labeled == i+1)
    cx, cy = np.mean(region_x), np.mean(region_y)

    # Zoom to region with padding
    pad = 100
    x1 = max(0, int(cx - pad))
    x2 = min(mosaic.shape[1], int(cx + pad))
    y1 = max(0, int(cy - pad))
    y2 = min(mosaic.shape[0], int(cy + pad))

    ax = axes[i]
    zoom_data = mosaic[y1:y2, x1:x2]
    zoom_mask = mask[y1:y2, x1:x2]

    ax.imshow(zoom_data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
              extent=[x1, x2, y1, y2])
    ax.contour(zoom_mask, colors='cyan', linewidths=2,
               extent=[x1, x2, y1, y2], origin='lower')
    ax.set_title(f'Region {i+1}: center ({cx:.0f}, {cy:.0f})')
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')

plt.tight_layout()
plt.savefig('output/verify_mask_regions.png', dpi=150, bbox_inches='tight')
print("Saved to output/verify_mask_regions.png")
plt.close()

# Also create full mosaic view with mask
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(mosaic, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.contour(mask, colors='cyan', linewidths=1)

# Add circles at region centers
for i in range(num_features):
    region_y, region_x = np.where(labeled == i+1)
    cx, cy = np.mean(region_x), np.mean(region_y)
    ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
    ax.annotate(f'{i+1}', (cx+30, cy+30), color='red', fontsize=12, fontweight='bold')

ax.set_title('Full Mosaic with Star Mask Regions')
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')

plt.tight_layout()
plt.savefig('output/verify_mask_full.png', dpi=100, bbox_inches='tight')
print("Saved to output/verify_mask_full.png")
plt.close()

print("Done!")
