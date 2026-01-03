"""Debug script to inspect FITS files"""

import numpy as np
from astropy.io import fits

files = {
    "b": "./fits/b.fits",
    "i": "./fits/i.fits",
    "u": "./fits/u.fits",
    "v": "./fits/v.fits",
}

for band, filepath in files.items():
    print(f"\n{'='*60}")
    print(f"Band: {band}")
    print(f"{'='*60}")

    try:
        hdul = fits.open(filepath)

        # Show all HDUs
        print(f"\nNumber of HDUs: {len(hdul)}")
        hdul.info()

        # Inspect each HDU
        for i, hdu in enumerate(hdul):
            print(f"\n--- HDU {i} ---")
            print(f"Type: {type(hdu).__name__}")
            print(
                f"Header shape: {hdu.data.shape if hdu.data is not None else 'No data'}"
            )

            if hdu.data is not None:
                print(f"Data dtype: {hdu.data.dtype}")
                print(f"Data min: {np.min(hdu.data)}")
                print(f"Data max: {np.max(hdu.data)}")
                print(f"Data mean: {np.mean(hdu.data)}")
                print(f"Data std: {np.std(hdu.data)}")
                print(f"Non-zero elements: {np.count_nonzero(hdu.data)}")
                print(f"Total elements: {hdu.data.size}")

        hdul.close()

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

print(f"\n{'='*60}")
print("Diagnosis complete")
