"""Benchmarking module for comparing photo-z methods.

This module provides tools for:
- Running EAZY-py on the same photometry
- Comparing our photo-z against EAZY results
- Template set comparison
- Performance metrics and visualization

Example usage:
    from benchmark import run_eazy_photoz, compare_photoz_methods
    from benchmark.template_comparison import compare_template_sets

    # Run EAZY and compare
    eazy_results = run_eazy_photoz(catalog, output_dir="./eazy_output")
    comparison = compare_photoz_methods(our_results, eazy_results, spec_z)
"""

from benchmark.eazy_comparison import (
    EazyBenchmarkResult,
    compare_with_eazy,
    run_eazy_photoz,
)
from benchmark.template_comparison import (
    compare_template_sets,
    plot_template_comparison,
)

__all__ = [
    "EazyBenchmarkResult",
    "compare_template_sets",
    "compare_with_eazy",
    "plot_template_comparison",
    "run_eazy_photoz",
]
