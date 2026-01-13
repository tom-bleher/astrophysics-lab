#!/usr/bin/env python3
"""
Transform the star mask from Chip 3 coordinates to the full 4096x4096 mosaic.
Use direct source matching to find the offset.
"""

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

print("Loading data...")
with fits.open("data/planet_mask2.fits") as hdul:
    mask_data = hdul[0].data.copy()
print(f"Mask shape: {mask_data.shape}")

with fits.open("fits_official/f450_3_v2.fits") as hdul:
    chip3_data = hdul[0].data.copy()
print(f"Chip 3 shape: {chip3_data.shape}")

with fits.open("fits_official/f450_mosaic_v2.fits") as hdul:
    mosaic_data = hdul[0].data.copy()
print(f"Mosaic shape: {mosaic_data.shape}")

# The user said chip 3 needs H+V flip (180 rotation) to match mosaic
chip3_flipped = np.flipud(np.fliplr(chip3_data))
mask_flipped = np.flipud(np.fliplr(mask_data))

# Find THE brightest source in each image (using local max)
def find_brightest_peak(data, n_peaks=5):
    """Find n brightest peaks in data using local maximum."""
    from scipy.ndimage import maximum_filter, label
    # Find local maxima
    local_max = maximum_filter(data, size=20)
    peaks = (data == local_max) & (data > np.percentile(data, 99.9))

    # Get peak positions and values
    py, px = np.where(peaks)
    values = data[py, px]

    # Sort by value and return top n
    order = np.argsort(values)[::-1][:n_peaks]
    return [(px[i], py[i], values[i]) for i in order]

chip3_peaks = find_brightest_peak(chip3_flipped, n_peaks=10)
mosaic_peaks = find_brightest_peak(mosaic_data, n_peaks=20)

print(f"\nBrightest peaks in flipped chip 3:")
for i, (x, y, v) in enumerate(chip3_peaks):
    print(f"  {i+1}: ({x}, {y}), value={v:.6f}")

print(f"\nBrightest peaks in mosaic:")
for i, (x, y, v) in enumerate(mosaic_peaks):
    print(f"  {i+1}: ({x}, {y}), value={v:.6f}")

# Match the brightest peak from chip3 to mosaic
# The brightest peak in flipped chip3 is at chip3_peaks[0]
# We need to find where this appears in the mosaic

# Use template matching on a small region around the brightest peak
c3x, c3y, _ = chip3_peaks[0]
template_size = 50
y1 = max(0, c3y - template_size)
y2 = min(chip3_flipped.shape[0], c3y + template_size)
x1 = max(0, c3x - template_size)
x2 = min(chip3_flipped.shape[1], c3x + template_size)

template = chip3_flipped[y1:y2, x1:x2]
template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)

print(f"\nTemplate matching using region around brightest peak ({c3x}, {c3y})...")

# Search in mosaic (try multiple candidate positions based on brightest mosaic peaks)
best_corr = -1
best_pos = None

for mx, my, _ in mosaic_peaks[:10]:
    # Test if chip3 brightest matches this mosaic peak
    # Offset would be: (mx - c3x, my - c3y)
    offset_x = mx - c3x
    offset_y = my - c3y

    # Check if the template would fit at this offset
    if offset_y + y1 < 0 or offset_y + y2 > mosaic_data.shape[0]:
        continue
    if offset_x + x1 < 0 or offset_x + x2 > mosaic_data.shape[1]:
        continue

    # Extract corresponding region from mosaic
    my1 = y1 + int(offset_y)
    my2 = y2 + int(offset_y)
    mx1 = x1 + int(offset_x)
    mx2 = x2 + int(offset_x)

    mosaic_region = mosaic_data[my1:my2, mx1:mx2]
    if mosaic_region.shape != template.shape:
        continue

    mosaic_norm = (mosaic_region - np.mean(mosaic_region)) / (np.std(mosaic_region) + 1e-10)

    corr = np.mean(template_norm * mosaic_norm)
    print(f"  Testing offset ({offset_x}, {offset_y}) -> corr={corr:.4f}")

    if corr > best_corr:
        best_corr = corr
        best_pos = (offset_x, offset_y)

print(f"\nBest match: offset=({best_pos[0]}, {best_pos[1]}), corr={best_corr:.4f}")

offset_x, offset_y = int(best_pos[0]), int(best_pos[1])

# Create the transformed mask
mosaic_mask = np.zeros(mosaic_data.shape, dtype=np.uint8)

# Place the flipped mask at the correct offset
y_start = max(0, offset_y)
y_end = min(mosaic_data.shape[0], offset_y + mask_flipped.shape[0])
x_start = max(0, offset_x)
x_end = min(mosaic_data.shape[1], offset_x + mask_flipped.shape[1])

mask_y_start = max(0, -offset_y)
mask_y_end = mask_y_start + (y_end - y_start)
mask_x_start = max(0, -offset_x)
mask_x_end = mask_x_start + (x_end - x_start)

mosaic_mask[y_start:y_end, x_start:x_end] = mask_flipped[mask_y_start:mask_y_end, mask_x_start:mask_x_end].astype(np.uint8)

# Dilate slightly
mosaic_mask = ndimage.binary_dilation(mosaic_mask.astype(bool), iterations=3).astype(np.uint8)

# Count regions
labeled, num_features = ndimage.label(mosaic_mask)
print(f"\nNumber of regions in transformed mask: {num_features}")

# Get region info
region_info = []
for i in range(1, num_features + 1):
    region_y, region_x = np.where(labeled == i)
    center_y, center_x = np.mean(region_y), np.mean(region_x)
    region_info.append((i, center_x, center_y, len(region_y)))
    print(f"  Region {i}: center=({center_x:.0f}, {center_y:.0f}), pixels={len(region_y)}")

# Save the transformed mask
print("\nSaving transformed mask...")
hdu = fits.PrimaryHDU(mosaic_mask)
hdu.header['COMMENT'] = 'Star mask transformed from Chip 3 to mosaic via template matching'
hdu.header['OFFSET_X'] = offset_x
hdu.header['OFFSET_Y'] = offset_y
hdu.writeto('data/star_mask_mosaic.fits', overwrite=True)
print("Saved to data/star_mask_mosaic.fits")

# Create verification visualization
print("\nCreating verification visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

vmin, vmax = np.percentile(mosaic_data, [1, 99.5])

for i in range(min(6, num_features)):
    region_y, region_x = np.where(labeled == i+1)
    cx, cy = np.mean(region_x), np.mean(region_y)

    pad = 100
    x1 = max(0, int(cx - pad))
    x2 = min(mosaic_data.shape[1], int(cx + pad))
    y1 = max(0, int(cy - pad))
    y2 = min(mosaic_data.shape[0], int(cy + pad))

    ax = axes[i]
    zoom_data = mosaic_data[y1:y2, x1:x2]
    zoom_mask = mosaic_mask[y1:y2, x1:x2]

    ax.imshow(zoom_data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
              extent=[x1, x2, y1, y2])
    ax.contour(zoom_mask > 0, colors='cyan', linewidths=2,
               extent=[x1, x2, y1, y2], origin='lower')
    ax.set_title(f'Region {i+1}: center ({cx:.0f}, {cy:.0f})')

plt.tight_layout()
plt.savefig('output/verify_mask_regions.png', dpi=150, bbox_inches='tight')
print("Saved to output/verify_mask_regions.png")
plt.close()

# Also show full mosaic with mask
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(mosaic_data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.contour(mosaic_mask, colors='cyan', linewidths=1)
for i, cx, cy, _ in region_info:
    ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
    ax.annotate(f'{i}', (cx+30, cy+30), color='red', fontsize=12, fontweight='bold')
ax.set_title(f'Full Mosaic with Star Mask (offset: {offset_x}, {offset_y})')
plt.tight_layout()
plt.savefig('output/verify_mask_full.png', dpi=100, bbox_inches='tight')
print("Saved to output/verify_mask_full.png")
plt.close()

print("\nDone!")
