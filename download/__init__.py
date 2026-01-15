"""Data acquisition scripts for external catalogs and templates.

This module provides tools for downloading:
- Hubble Legacy Fields (HLF) catalogs
- EAZY template spectra
- MUSE spectroscopic redshift catalogs
- VizieR catalog data

Example usage:
    from download import fetch_all_data, download_eazy_templates

    # Download all external data
    fetch_all_data.download_all()

    # Or download specific items
    download_eazy_templates.download_templates()
"""

from download.download_eazy_templates import TEMPLATE_SETS, download_templates
from download.fetch_all_data import DATA_SOURCES, download_all, download_file
from download.query_mast import download_hdf_products, search_hdf_observations

__all__ = [
    "DATA_SOURCES",
    "TEMPLATE_SETS",
    "download_all",
    "download_file",
    "download_hdf_products",
    "download_templates",
    "search_hdf_observations",
]
