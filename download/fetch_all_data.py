"""Download all external data for the HDF analysis project.

This script downloads:
- EAZY templates and example data

Usage:
    python -m download.fetch_all_data
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

# Data sources with metadata
DATA_SOURCES: dict[str, dict] = {}


def download_file(
    url: str,
    local_path: str | Path,
    description: str = "",
    timeout: int = 60,
) -> bool:
    """Download a file with progress indication.

    Parameters
    ----------
    url : str
        URL to download from
    local_path : str or Path
        Local path to save the file
    description : str
        Human-readable description of the file
    timeout : int
        Timeout in seconds

    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    local_path = Path(local_path)

    if local_path.exists():
        print(f"  [SKIP] Already exists: {local_path}")
        return True

    # Create parent directories
    local_path.parent.mkdir(parents=True, exist_ok=True)

    desc = description or url
    print(f"  Downloading: {desc}")
    print(f"    URL: {url}")
    print(f"    -> {local_path}")

    try:
        # Download with progress callback
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                sys.stdout.write(f"\r    Progress: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, local_path, reporthook=progress_hook)
        print()  # Newline after progress
        print(f"    [OK] Saved to: {local_path}")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n    [ERROR] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n    [ERROR] URL Error: {e.reason}")
        return False
    except TimeoutError:
        print(f"\n    [ERROR] Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"\n    [ERROR] {type(e).__name__}: {e}")
        return False


def download_all() -> dict[str, bool]:
    """Download all external data sources.

    Returns
    -------
    dict
        Mapping of data source name to download success status
    """
    print("=" * 60)
    print("HDF Analysis: External Data Download")
    print("=" * 60)

    results = {}

    # Download direct URL sources
    print("\nDownloading from direct URLs...")
    print("-" * 40)

    for name, info in DATA_SOURCES.items():
        success = download_file(
            info["url"],
            info["local"],
            info["description"],
        )
        results[name] = success

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {name}")

    if total_count > 0:
        print(f"\nTotal: {success_count}/{total_count} successful")
    else:
        print("\nNo data sources configured.")

    return results


if __name__ == "__main__":
    download_all()
