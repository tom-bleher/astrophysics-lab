"""Resource configuration for machines with limited resources.

This module provides centralized resource management with automatic detection
and configurable profiles for different hardware capabilities.

Usage:
    # In run_analysis.py or classify.py:
    from resource_config import get_config, ResourceProfile

    # Use auto-detected settings (recommended)
    config = get_config()

    # Or specify a profile explicitly
    config = get_config(ResourceProfile.LOW)

    # Access settings
    n_threads = config.n_threads
    z_step = config.z_step
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil


class ResourceProfile(Enum):
    """Resource usage profiles for different hardware capabilities."""
    LOW = "low"           # 2-4 GB RAM, 2 cores - constrained machines
    MEDIUM = "medium"     # 4-8 GB RAM, 4 cores - typical laptops
    HIGH = "high"         # 8+ GB RAM, 8+ cores - workstations
    AUTO = "auto"         # Auto-detect based on system resources


@dataclass
class ResourceConfig:
    """Configuration settings for resource-constrained operation.

    Attributes:
        n_threads: Number of threads for parallel operations
        n_template_workers: Number of parallel workers for template fitting
        z_step: Redshift grid step (larger = faster, less precise)
        z_step_coarse: Coarse grid step for initial search
        use_float32: Use float32 instead of float64 for arrays (saves 50% memory)
        enable_template_cache: Cache template grids to disk
        detection_sigma: Source detection threshold (higher = fewer sources, faster)
        batch_size: Batch size for processing (smaller = less memory)
        gc_aggressive: Run garbage collection more frequently
        parallel_detection: Enable parallel band detection (requires more memory)
        profile: The resource profile being used
    """
    n_threads: int
    n_template_workers: int
    z_step: float
    z_step_coarse: float
    use_float32: bool
    enable_template_cache: bool
    detection_sigma: float
    batch_size: int
    gc_aggressive: bool
    parallel_detection: bool
    profile: ResourceProfile

    def apply_environment(self) -> None:
        """Apply thread settings to environment variables.

        Must be called BEFORE importing numpy/numba for full effect.
        """
        n_str = str(self.n_threads)
        os.environ["NUMBA_NUM_THREADS"] = n_str
        os.environ["OMP_NUM_THREADS"] = n_str
        os.environ["MKL_NUM_THREADS"] = n_str
        os.environ["OPENBLAS_NUM_THREADS"] = n_str

    def __str__(self) -> str:
        return (
            f"ResourceConfig(profile={self.profile.value}, "
            f"threads={self.n_threads}, workers={self.n_template_workers}, "
            f"z_step={self.z_step}, float32={self.use_float32})"
        )


# Profile presets (base values - HIGH profile is dynamically adjusted)
# Note: z_step controls photo-z precision vs speed tradeoff
#   0.01 = high precision (~600 z points, ~0.5s/galaxy with vectorized code)
#   0.02 = good precision (~300 z points, ~0.25s/galaxy)
#   0.05 = fast (~120 z points, ~0.1s/galaxy)
_PROFILES = {
    ResourceProfile.LOW: ResourceConfig(
        n_threads=2,
        n_template_workers=2,
        z_step=0.05,           # Fast: ~120 z points (vectorized: ~0.1s/galaxy)
        z_step_coarse=0.10,    # Very coarse initial search
        use_float32=True,      # Save 50% memory on arrays
        enable_template_cache=True,
        detection_sigma=2.0,   # Higher threshold = fewer detections
        batch_size=500,        # Smaller batches
        gc_aggressive=True,    # Frequent garbage collection
        parallel_detection=False,  # Sequential to save memory
        profile=ResourceProfile.LOW,
    ),
    ResourceProfile.MEDIUM: ResourceConfig(
        n_threads=4,
        n_template_workers=4,
        z_step=0.02,           # Good precision: ~300 z points
        z_step_coarse=0.05,
        use_float32=True,
        enable_template_cache=True,
        detection_sigma=1.5,
        batch_size=1000,
        gc_aggressive=False,
        parallel_detection=True,   # Parallel detection enabled
        profile=ResourceProfile.MEDIUM,
    ),
    ResourceProfile.HIGH: ResourceConfig(
        n_threads=8,           # Will be dynamically adjusted
        n_template_workers=7,  # Will be dynamically adjusted
        z_step=0.02,           # Good balance: ~300 z points (vectorized: ~0.25s/galaxy)
        z_step_coarse=0.05,
        use_float32=False,     # Use full float64 precision
        enable_template_cache=True,
        detection_sigma=1.5,
        batch_size=3000,
        gc_aggressive=False,
        parallel_detection=True,   # Parallel detection enabled
        profile=ResourceProfile.HIGH,
    ),
}


def _create_dynamic_high_profile() -> ResourceConfig:
    """Create HIGH profile with dynamic CPU count detection.

    Uses all available CPU cores for maximum parallelization.
    """
    cpu_count = os.cpu_count() or 8
    # Use all cores for threads, leave one core free for workers
    # to avoid overwhelming the system
    n_threads = max(4, cpu_count)
    n_workers = max(3, cpu_count - 1)

    return ResourceConfig(
        n_threads=n_threads,
        n_template_workers=n_workers,
        z_step=0.02,           # Good balance with vectorized code
        z_step_coarse=0.05,
        use_float32=False,
        enable_template_cache=True,
        detection_sigma=1.5,
        batch_size=3000,
        gc_aggressive=False,
        parallel_detection=True,   # Parallel detection enabled
        profile=ResourceProfile.HIGH,
    )


def _detect_system_resources() -> ResourceProfile:
    """Auto-detect appropriate resource profile based on system capabilities."""
    # Get CPU count
    cpu_count = os.cpu_count() or 2

    # Try to get available memory
    available_memory_gb = _get_available_memory_gb()

    # Decision logic
    if available_memory_gb is not None:
        if available_memory_gb < 4 or cpu_count <= 2:
            return ResourceProfile.LOW
        elif available_memory_gb < 8 or cpu_count <= 4:
            return ResourceProfile.MEDIUM
        else:
            return ResourceProfile.HIGH
    else:
        # Fallback to CPU-based detection
        if cpu_count <= 2:
            return ResourceProfile.LOW
        elif cpu_count <= 4:
            return ResourceProfile.MEDIUM
        else:
            return ResourceProfile.HIGH


def _get_available_memory_gb() -> Optional[float]:
    """Get available system memory in GB."""
    try:
        # Try psutil first (most reliable)
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        pass

    # Linux fallback
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except (OSError, ValueError):
            pass

    return None


def _get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)


# Global cached config
_cached_config: Optional[ResourceConfig] = None


def get_config(profile: Optional[ResourceProfile] = None) -> ResourceConfig:
    """Get resource configuration, optionally specifying a profile.

    Args:
        profile: Resource profile to use. If None or AUTO, auto-detects.

    Returns:
        ResourceConfig with appropriate settings for the system.
    """
    global _cached_config

    if profile is None or profile == ResourceProfile.AUTO:
        # Check environment variable override
        env_profile = os.environ.get("ASTRO_RESOURCE_PROFILE", "").lower()
        if env_profile in ("low", "medium", "high"):
            profile = ResourceProfile(env_profile)
        else:
            profile = _detect_system_resources()

    # Return cached if same profile
    if _cached_config is not None and _cached_config.profile == profile:
        return _cached_config

    # Use dynamic configuration for HIGH profile to use all available cores
    if profile == ResourceProfile.HIGH:
        _cached_config = _create_dynamic_high_profile()
    else:
        _cached_config = _PROFILES[profile]

    return _cached_config


def print_system_info() -> None:
    """Print detected system information and recommended profile."""
    cpu_count = os.cpu_count() or 2
    total_mem = _get_total_memory_gb()
    avail_mem = _get_available_memory_gb()
    detected_profile = _detect_system_resources()

    print("System Resource Detection:")
    print(f"  CPU cores: {cpu_count}")
    if total_mem is not None:
        print(f"  Total memory: {total_mem:.1f} GB")
    if avail_mem is not None:
        print(f"  Available memory: {avail_mem:.1f} GB")
    print(f"  Recommended profile: {detected_profile.value}")
    print()


if __name__ == "__main__":
    # Print system info when run directly
    print_system_info()
    config = get_config()
    print(f"Selected config: {config}")
