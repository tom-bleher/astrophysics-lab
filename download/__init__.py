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

from download.fetch_all_data import download_all, download_file, DATA_SOURCES
from download.download_eazy_templates import download_templates, TEMPLATE_SETS
from download.query_mast import search_hdf_observations, download_hdf_products

__all__ = [
    "download_all",
    "download_file",
    "download_templates",
    "search_hdf_observations",
    "download_hdf_products",
    "DATA_SOURCES",
    "TEMPLATE_SETS",
]
