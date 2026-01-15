"""Download EAZY template spectra for comparison and validation.

EAZY (Easy and Accurate Zphot from Yale) provides high-quality galaxy
templates that can be used to benchmark our SED fitting.

Templates available:
- CWW+KIN: Coleman-Wu-Weedman extended with Kinney starburst templates
- EAZY_v1.0: Optimized non-negative linear combination templates
- FSPS: Flexible Stellar Population Synthesis templates

Usage:
    python -m download.download_eazy_templates
"""

import urllib.error
import urllib.request
from pathlib import Path

EAZY_TEMPLATE_BASE = (
    "https://raw.githubusercontent.com/gbrammer/eazy-photoz/master/templates"
)

# Template sets available in EAZY
TEMPLATE_SETS: dict[str, list[str]] = {
    "cww_kin": [
        "CWW+KIN/cww_E_ext.sed",
        "CWW+KIN/cww_Sbc_ext.sed",
        "CWW+KIN/cww_Scd_ext.sed",
        "CWW+KIN/cww_Im_ext.sed",
        "CWW+KIN/kinney_starb1.sed",
        "CWW+KIN/kinney_starb2.sed",
    ],
    "eazy_v1": [
        "EAZY_v1.0/eazy_v1.0_sed1.dat",
        "EAZY_v1.0/eazy_v1.0_sed2.dat",
        "EAZY_v1.0/eazy_v1.0_sed3.dat",
        "EAZY_v1.0/eazy_v1.0_sed4.dat",
        "EAZY_v1.0/eazy_v1.0_sed5.dat",
    ],
    "eazy_v1.1": [
        "EAZY_v1.1_lines/eazy_v1.1_sed1.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed2.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed3.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed4.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed5.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed6.dat",
        "EAZY_v1.1_lines/eazy_v1.1_sed7.dat",
    ],
}

# Filter response curves (useful for photometric calibration)
FILTER_CURVES = [
    "FILTER.RES.latest",
]


def download_template(
    template_path: str,
    output_dir: Path,
    base_url: str = EAZY_TEMPLATE_BASE,
) -> bool:
    """Download a single template file.

    Parameters
    ----------
    template_path : str
        Relative path to template (e.g., "CWW+KIN/cww_E_ext.sed")
    output_dir : Path
        Base output directory
    base_url : str
        Base URL for EAZY templates

    Returns
    -------
    bool
        True if download succeeded
    """
    url = f"{base_url}/{template_path}"
    filename = Path(template_path).name
    set_name = Path(template_path).parent.name

    local_path = output_dir / set_name / filename

    if local_path.exists():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, local_path)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"    [ERROR] Failed to download {filename}: {e}")
        return False


def download_templates(
    output_dir: str | Path = "./spectra/eazy",
    template_sets: list[str] | None = None,
) -> dict[str, bool]:
    """Download EAZY template spectra.

    Parameters
    ----------
    output_dir : str or Path
        Output directory for templates
    template_sets : list of str, optional
        Which template sets to download. If None, downloads all.
        Options: 'cww_kin', 'eazy_v1', 'eazy_v1.1'

    Returns
    -------
    dict
        Mapping of template name to download success
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if template_sets is None:
        template_sets = list(TEMPLATE_SETS.keys())

    print("Downloading EAZY templates...")
    print(f"Output directory: {output_path}")

    results = {}

    for set_name in template_sets:
        if set_name not in TEMPLATE_SETS:
            print(f"  [WARN] Unknown template set: {set_name}")
            continue

        print(f"\n  Template set: {set_name}")
        templates = TEMPLATE_SETS[set_name]

        for template in templates:
            filename = Path(template).name
            success = download_template(template, output_path)
            results[f"{set_name}/{filename}"] = success

            status = "[OK]" if success else "[FAIL]"
            print(f"    {status} {filename}")

    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\nDownloaded {success_count}/{total_count} templates")

    return results


def convert_eazy_to_cww_format(
    input_path: Path,
    output_path: Path,
) -> bool:
    """Convert EAZY template format to CWW format used by classify.py.

    EAZY format: wavelength [Angstrom], flux [arbitrary units]
    CWW format: wavelength [Angstrom], flux [arbitrary units]

    Both use the same format, but EAZY templates may have different
    wavelength coverage and sampling.

    Parameters
    ----------
    input_path : Path
        Path to EAZY template file
    output_path : Path
        Path to save converted template

    Returns
    -------
    bool
        True if conversion succeeded
    """
    try:
        import numpy as np

        # Load EAZY template
        data = np.loadtxt(input_path)

        if data.ndim == 1:
            print(f"    [WARN] Unexpected format in {input_path}")
            return False

        wavelength = data[:, 0]
        flux = data[:, 1]

        # Normalize flux
        flux = flux / np.median(flux[flux > 0])

        # Save in same format (compatible with classify.py)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_path,
            np.column_stack([wavelength, flux]),
            fmt="%.6e",
            header="wavelength[A]  flux[normalized]",
        )

        return True

    except Exception as e:
        print(f"    [ERROR] Conversion failed: {e}")
        return False


def setup_eazy_templates_for_classify(
    eazy_dir: str | Path = "./spectra/eazy",
    output_dir: str | Path = "./spectra/eazy_converted",
) -> dict[str, Path]:
    """Set up EAZY templates for use with classify.py.

    Downloads templates if needed and converts them to the format
    expected by classify.py.

    Parameters
    ----------
    eazy_dir : str or Path
        Directory containing EAZY templates
    output_dir : str or Path
        Directory for converted templates

    Returns
    -------
    dict
        Mapping of template type to file path
    """
    eazy_path = Path(eazy_dir)
    output_path = Path(output_dir)

    # Download if needed
    if not eazy_path.exists():
        download_templates(eazy_path)

    # Convert CWW+KIN templates (most similar to our existing templates)
    template_mapping = {
        "elliptical_eazy": "cww_kin/cww_E_ext.sed",
        "Sbc_eazy": "cww_kin/cww_Sbc_ext.sed",
        "Scd_eazy": "cww_kin/cww_Scd_ext.sed",
        "Im_eazy": "cww_kin/cww_Im_ext.sed",
        "starburst1_eazy": "cww_kin/kinney_starb1.sed",
        "starburst2_eazy": "cww_kin/kinney_starb2.sed",
    }

    converted_paths = {}

    print("Converting EAZY templates for classify.py...")

    for output_name, input_rel in template_mapping.items():
        input_file = eazy_path / input_rel
        output_file = output_path / f"{output_name}.dat"

        if not input_file.exists():
            print(f"  [SKIP] Source not found: {input_file}")
            continue

        if output_file.exists():
            print(f"  [SKIP] Already converted: {output_name}")
            converted_paths[output_name] = output_file
            continue

        success = convert_eazy_to_cww_format(input_file, output_file)
        if success:
            print(f"  [OK] {output_name}")
            converted_paths[output_name] = output_file

    return converted_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download EAZY templates")
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=list(TEMPLATE_SETS.keys()),
        help="Template sets to download",
    )
    parser.add_argument(
        "--output",
        default="./spectra/eazy",
        help="Output directory",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Also convert templates for use with classify.py",
    )
    args = parser.parse_args()

    download_templates(args.output, args.sets)

    if args.convert:
        setup_eazy_templates_for_classify(args.output)
