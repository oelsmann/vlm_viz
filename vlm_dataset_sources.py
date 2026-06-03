from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


MIDAS_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/midas/midas.IGS.txt"
README_URL = "https://geodesy.unr.edu/velocities/midas.readme.txt"
GIA_URL = "https://vesl.jpl.nasa.gov/solid-earth/gia/downloads/GIA_maps_Caron_Ivins_2019"
INSAR_URL = "https://zenodo.org/records/15015923/files/gridVLM.zip?download=1"
GNS_METADATA_URL = "https://data.gns.cri.nz/metadata/srv/eng/catalog.search#/metadata/fdbb8847-c882-4324-ae48-ca7ed9b7433b"
GNS_RECORD_XML_URL = "https://data.gns.cri.nz/metadata/srv/api/records/fdbb8847-c882-4324-ae48-ca7ed9b7433b/formatters/xml"
GNS_ATTACHMENT_URL = "https://data.gns.cri.nz/metadata/srv/api/records/fdbb8847-c882-4324-ae48-ca7ed9b7433b/attachments/NZ_InSAR_GPS%20(6).zip"
GHSL_POP_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_WUP_POP_GLOBE_R2025A/GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000/V1-0/GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000_V1_0.zip"
TIDE_GAUGE_RECORD_URL = "https://zenodo.org/records/18777050"
TIDE_GAUGE_REQUESTED_URL = "https://zenodo.org/records/18777050/files/CSL-TG?download=1"
TIDE_GAUGE_MAT_URL = "https://zenodo.org/records/18777050/files/CSL-TG.mat?download=1"
OELSMANN_HYBRID_RECORD_URL = "https://zenodo.org/records/19830370"
OELSMANN_HYBRID_NC_URL = "https://zenodo.org/records/19830370/files/Global_VLM_data_Oelsmann_2025_data_supplement.nc?download=1"

DATASETS_DIR = Path("datasets")
EXTERNAL_DATASETS_DIR = Path("external_datasets")
GNSS_DATASET_DIR = DATASETS_DIR / "gnss_blewitt_2018"
GIA_DATASET_DIR = DATASETS_DIR / "gia_caron_2020"
INSAR_DATASET_DIR = DATASETS_DIR / "insar_ohenhen_2025"
GNS_DATASET_DIR = DATASETS_DIR / "insar_gnss_hamling_2022"
TIDE_GAUGE_DATASET_DIR = DATASETS_DIR / "tide_gauge_dangendorf_2026"
OELSMANN_HYBRID_DATASET_DIR = DATASETS_DIR / "hybrid_oelsmann_2026"
GHSL_POP_DATASET_DIR = EXTERNAL_DATASETS_DIR / "ghsl_schiavina_2025"

CACHE_FILE = GNSS_DATASET_DIR / "midas.IGS.cache.txt"
GIA_CACHE_FILE = GIA_DATASET_DIR / "GIA_maps_Caron_Ivins_2019"
INSAR_ZIP_FILE = INSAR_DATASET_DIR / "gridVLM.zip"
INSAR_DIR = INSAR_DATASET_DIR / "gridVLM"
GNS_ZIP_FILE = GNS_DATASET_DIR / "gns_hamling_2022.zip"
GNS_DIR = GNS_DATASET_DIR / "gns_hamling_2022"
GNS_DATA_DIR = GNS_DIR / "NZ_InSAR_GPS"
GNS_COAST_FILE = GNS_DATA_DIR / "NZ_coast_1km_GRL.txt"
GNS_METADATA_FILE = GNS_DATASET_DIR / "gns_record.xml"
TIDE_GAUGE_MAT_FILE = TIDE_GAUGE_DATASET_DIR / "CSL-TG.mat"
TIDE_GAUGE_RAW_FILE = TIDE_GAUGE_DATASET_DIR / "CSL-TG"
OELSMANN_HYBRID_NC_FILE = OELSMANN_HYBRID_DATASET_DIR / "Global_VLM_data_Oelsmann_2025_data_supplement.nc"
GHSL_POP_ZIP_FILE = GHSL_POP_DATASET_DIR / "GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000_V1_0.zip"
GHSL_POP_DIR = GHSL_POP_DATASET_DIR / "GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000_V1_0"


def ensure_dataset_dirs() -> None:
    for path in (
        GNSS_DATASET_DIR,
        GIA_DATASET_DIR,
        INSAR_DATASET_DIR,
        GNS_DATASET_DIR,
        TIDE_GAUGE_DATASET_DIR,
        OELSMANN_HYBRID_DATASET_DIR,
        GHSL_POP_DATASET_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def migrate_legacy_dataset_paths() -> None:
    ensure_dataset_dirs()
    moves = [
        (Path("midas.IGS.cache.txt"), CACHE_FILE),
        (Path("GIA_maps_Caron_Ivins_2019"), GIA_CACHE_FILE),
        (Path("gridVLM.zip"), INSAR_ZIP_FILE),
        (Path("gridVLM"), INSAR_DIR),
        (Path("gns_hamling_2022.zip"), GNS_ZIP_FILE),
        (Path("gns_record.xml"), GNS_METADATA_FILE),
        (Path("gns_hamling_2022"), GNS_DIR),
        (Path("CSL-TG.mat"), TIDE_GAUGE_MAT_FILE),
        (Path("CSL-TG"), TIDE_GAUGE_RAW_FILE),
        (Path("Global_VLM_data_Oelsmann_2025_data_supplement.nc"), OELSMANN_HYBRID_NC_FILE),
        (Path("GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000_V1_0.zip"), GHSL_POP_ZIP_FILE),
        (Path("GHS_WUP_POP_E2025_GLOBE_R2025A_54009_1000_V1_0"), GHSL_POP_DIR),
    ]

    for source, target in moves:
        if not source.exists() or target.exists():
            continue
        shutil.move(str(source), str(target))


def fetch_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vlm-viz-midas-globe/1.0 (+static generator)",
            "Accept": "text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def fetch_bytes(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vlm-viz-midas-globe/1.0 (+static generator)",
            "Accept": "application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_raw_midas(force_refresh: bool = False) -> tuple[str, str]:
    migrate_legacy_dataset_paths()
    if CACHE_FILE.exists() and not force_refresh:
        return CACHE_FILE.read_text(encoding="utf-8"), "cache"

    try:
        text = fetch_text(MIDAS_URL)
    except (urllib.error.URLError, TimeoutError) as exc:
        if CACHE_FILE.exists():
            print(
                f"Warning: live fetch failed ({exc}); using existing cache.",
                file=sys.stderr,
            )
            return CACHE_FILE.read_text(encoding="utf-8"), "cache-after-fetch-error"
        raise RuntimeError(
            f"Could not fetch {MIDAS_URL} and no cache exists. "
            "Check the network connection, then rerun this script."
        ) from exc

    CACHE_FILE.write_text(text, encoding="utf-8")
    return text, "live"


def load_raw_gia(force_refresh: bool = False) -> tuple[str, str]:
    migrate_legacy_dataset_paths()
    if GIA_CACHE_FILE.exists() and not force_refresh:
        return GIA_CACHE_FILE.read_text(encoding="utf-8"), "cache"

    try:
        text = fetch_text(GIA_URL, timeout=180)
    except (urllib.error.URLError, TimeoutError) as exc:
        if GIA_CACHE_FILE.exists():
            print(
                f"Warning: GIA live fetch failed ({exc}); using existing cache.",
                file=sys.stderr,
            )
            return GIA_CACHE_FILE.read_text(encoding="utf-8"), "cache-after-fetch-error"
        raise RuntimeError(
            f"Could not fetch {GIA_URL} and no GIA cache exists. "
            "Check the network connection, then rerun this script."
        ) from exc

    GIA_CACHE_FILE.write_text(text, encoding="utf-8")
    return text, "live"


def ensure_insar_dataset(force_refresh: bool = False) -> str:
    migrate_legacy_dataset_paths()
    tif_files = list(INSAR_DIR.glob("*_vlm.tif")) if INSAR_DIR.exists() else []
    if tif_files and not force_refresh:
        return "cache"

    if force_refresh or not INSAR_ZIP_FILE.exists():
        try:
            data = fetch_bytes(INSAR_URL, timeout=180)
        except (urllib.error.URLError, TimeoutError) as exc:
            if INSAR_ZIP_FILE.exists():
                print(
                    f"Warning: InSAR live fetch failed ({exc}); using existing zip.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch {INSAR_URL} and no InSAR zip exists. "
                    "Check the network connection, then rerun this script."
                ) from exc
        else:
            INSAR_ZIP_FILE.write_bytes(data)

    with zipfile.ZipFile(INSAR_ZIP_FILE) as archive:
        archive.extractall(INSAR_DATASET_DIR)

    return "live" if force_refresh else "cache"


def ensure_gns_dataset(force_refresh: bool = False) -> str:
    migrate_legacy_dataset_paths()
    if force_refresh or not GNS_METADATA_FILE.exists():
        try:
            GNS_METADATA_FILE.write_text(
                fetch_text(GNS_RECORD_XML_URL, timeout=60), encoding="utf-8"
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            if not GNS_METADATA_FILE.exists():
                print(
                    f"Warning: GNS metadata fetch failed ({exc}); continuing without XML cache.",
                    file=sys.stderr,
                )

    if GNS_COAST_FILE.exists() and not force_refresh:
        return "cache"

    if force_refresh or not GNS_ZIP_FILE.exists() or GNS_ZIP_FILE.stat().st_size == 0:
        try:
            data = fetch_bytes(GNS_ATTACHMENT_URL, timeout=300)
        except (urllib.error.URLError, TimeoutError) as exc:
            if GNS_ZIP_FILE.exists() and GNS_ZIP_FILE.stat().st_size > 0:
                print(
                    f"Warning: GNS live fetch failed ({exc}); using existing zip.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch {GNS_ATTACHMENT_URL} and no GNS zip exists. "
                    "Check the network connection, then rerun this script."
                ) from exc
        else:
            GNS_ZIP_FILE.write_bytes(data)

    GNS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GNS_ZIP_FILE) as archive:
        archive.extractall(GNS_DIR)

    return "live" if force_refresh else "cache"


def ensure_ghsl_population_dataset(force_refresh: bool = False) -> str:
    migrate_legacy_dataset_paths()
    tif_files = list(GHSL_POP_DIR.rglob("*.tif")) if GHSL_POP_DIR.exists() else []
    if tif_files and not force_refresh:
        return "cache"

    if force_refresh or not GHSL_POP_ZIP_FILE.exists() or GHSL_POP_ZIP_FILE.stat().st_size == 0:
        try:
            data = fetch_bytes(GHSL_POP_URL, timeout=900)
        except (urllib.error.URLError, TimeoutError) as exc:
            if GHSL_POP_ZIP_FILE.exists() and GHSL_POP_ZIP_FILE.stat().st_size > 0:
                print(
                    f"Warning: GHSL population live fetch failed ({exc}); using existing zip.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch {GHSL_POP_URL} and no GHSL population zip exists. "
                    "Check the network connection, then rerun this script."
                ) from exc
        else:
            GHSL_POP_ZIP_FILE.write_bytes(data)

    GHSL_POP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GHSL_POP_ZIP_FILE) as archive:
        archive.extractall(GHSL_POP_DIR)

    return "live" if force_refresh else "cache"


def ensure_tide_gauge_dataset(force_refresh: bool = False) -> str:
    migrate_legacy_dataset_paths()
    if TIDE_GAUGE_MAT_FILE.exists() and TIDE_GAUGE_MAT_FILE.stat().st_size > 100_000 and not force_refresh:
        return "cache"

    if force_refresh or not TIDE_GAUGE_MAT_FILE.exists() or TIDE_GAUGE_MAT_FILE.stat().st_size <= 100_000:
        try:
            data = fetch_bytes(TIDE_GAUGE_MAT_URL, timeout=300)
        except (urllib.error.URLError, TimeoutError) as exc:
            if TIDE_GAUGE_MAT_FILE.exists() and TIDE_GAUGE_MAT_FILE.stat().st_size > 100_000:
                print(
                    f"Warning: CSL-TG live fetch failed ({exc}); using existing MAT cache.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch {TIDE_GAUGE_MAT_URL} and no CSL-TG MAT cache exists. "
                    "Check the network connection, then rerun this script."
                ) from exc
        else:
            TIDE_GAUGE_MAT_FILE.write_bytes(data)

    return "live" if force_refresh else "cache"


def ensure_oelsmann_hybrid_dataset(force_refresh: bool = False) -> str:
    migrate_legacy_dataset_paths()
    if OELSMANN_HYBRID_NC_FILE.exists() and OELSMANN_HYBRID_NC_FILE.stat().st_size > 100_000 and not force_refresh:
        return "cache"

    if force_refresh or not OELSMANN_HYBRID_NC_FILE.exists() or OELSMANN_HYBRID_NC_FILE.stat().st_size <= 100_000:
        try:
            data = fetch_bytes(OELSMANN_HYBRID_NC_URL, timeout=300)
        except (urllib.error.URLError, TimeoutError) as exc:
            if OELSMANN_HYBRID_NC_FILE.exists() and OELSMANN_HYBRID_NC_FILE.stat().st_size > 100_000:
                print(
                    f"Warning: Oelsmann hybrid VLM live fetch failed ({exc}); using existing NetCDF cache.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"Could not fetch {OELSMANN_HYBRID_NC_URL} and no Oelsmann hybrid NetCDF cache exists. "
                    "Check the network connection, then rerun this script."
                ) from exc
        else:
            OELSMANN_HYBRID_NC_FILE.write_bytes(data)

    return "live" if force_refresh else "cache"
