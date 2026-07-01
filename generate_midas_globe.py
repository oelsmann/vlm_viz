#!/usr/bin/env python3
"""
Generate a static MapLibre globe visualization of UNR/NGL MIDAS GPS station
UP velocities with deck.gl VLM overlay layers.

The script fetches the live MIDAS velocity file, caches the raw text locally,
parses and precomputes compact render fields, then writes a standalone HTML file
that can be hosted on GitHub Pages or any static file host.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vlm_dataset_sources import (
    GHSL_POP_DIR,
    GHSL_POP_URL,
    GIA_URL,
    GNS_ATTACHMENT_URL,
    GNS_COAST_FILE,
    GNS_METADATA_URL,
    INSAR_DIR,
    INSAR_URL,
    MIDAS_URL,
    NGL_IMAGED_VLM_URL,
    OELSMANN_HYBRID_NC_FILE,
    OELSMANN_HYBRID_NC_URL,
    OELSMANN_HYBRID_RECORD_URL,
    README_URL,
    TIDE_GAUGE_MAT_FILE,
    TIDE_GAUGE_MAT_URL,
    TIDE_GAUGE_RECORD_URL,
    ensure_ghsl_population_dataset,
    ensure_gns_dataset,
    ensure_insar_dataset,
    ensure_oelsmann_hybrid_dataset,
    ensure_tide_gauge_dataset,
    load_raw_gia,
    load_raw_midas,
    load_raw_ngl_imaged_vlm,
)

ATTRIBUTE_DIR = Path("dataset_attributes")
EXTERNAL_ATTRIBUTE_DIR = Path("external_dataset_attributes")
OUTPUT_HTML = Path("index.html")
CATALOG_HTML = Path("catalogue.html")
ABOUT_HTML = Path("about.html")
COMPARE_HTML = Path("compare.html")
UNCERTAINTY_PAYLOADS = {
    "gnss_blewitt_2018": Path("datasets/gnss_blewitt_2018/render_payloads/uncertainty.json"),
    "gnss_imaged_hammond_2021": Path("datasets/gnss_imaged_hammond_2021/render_payloads/uncertainty.json"),
    "gia_caron_2020": Path("datasets/gia_caron_2020/render_payloads/uncertainty.json"),
    "insar_gnss_hamling_2022": Path("datasets/insar_gnss_hamling_2022/render_payloads/uncertainty.json"),
    "hybrid_oelsmann_2026": Path("datasets/hybrid_oelsmann_2026/render_payloads/uncertainty.json"),
    "tide_gauge_dangendorf_2026": Path("datasets/tide_gauge_dangendorf_2026/render_payloads/uncertainty.json"),
}
RENDER_PAYLOADS = {
    "gnss_blewitt_2018": Path("datasets/gnss_blewitt_2018/render_payloads/trends.json"),
    "gnss_imaged_hammond_2021": Path("datasets/gnss_imaged_hammond_2021/render_payloads/trends.json"),
    "gia_caron_2020": Path("datasets/gia_caron_2020/render_payloads/trends.json"),
    "insar_ohenhen_2025": Path("datasets/insar_ohenhen_2025/render_payloads/trends.json"),
    "insar_gnss_hamling_2022": Path("datasets/insar_gnss_hamling_2022/render_payloads/trends.json"),
    "hybrid_oelsmann_2026": Path("datasets/hybrid_oelsmann_2026/render_payloads/trends.json"),
    "tide_gauge_dangendorf_2026": Path("datasets/tide_gauge_dangendorf_2026/render_payloads/trends.json"),
}
TIDE_GAUGE_PROCESSED_CACHE = Path("datasets/tide_gauge_dangendorf_2026/csl_tg_processed.json")
TIDE_GAUGE_NEARBY_GNSS_CACHE = Path("datasets/tide_gauge_dangendorf_2026/tg_nearby_gnss.json")
POPULATION_PAYLOAD_JS = Path("external_datasets/ghsl_schiavina_2025/ghsl_population_payload.js")
POPULATION_METADATA_CACHE = Path("external_datasets/ghsl_schiavina_2025/ghsl_population_metadata.json")

NEUTRAL_COLOR = (246, 246, 242)
NEGATIVE_COLOR = (34, 104, 209)
POSITIVE_COLOR = (207, 50, 45)


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def normalize_longitude(lon: float) -> float:
    return ((lon + 180.0) % 360.0 + 360.0) % 360.0 - 180.0


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def diverging_color(value: float, color_limit: float) -> list[int]:
    if color_limit <= 0:
        return [*NEUTRAL_COLOR, 220]

    t = min(abs(value) / color_limit, 1.0)
    target = POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
    color = [lerp(NEUTRAL_COLOR[i], target[i], t) for i in range(3)]
    return [*color, 220]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * pct
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_values[low]
    fraction = idx - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def compact_record(tokens: list[str]) -> dict | None:
    if len(tokens) < 27:
        return None

    latitude = parse_float(tokens[24])
    longitude = parse_float(tokens[25])
    height = parse_float(tokens[26])
    first_epoch = parse_float(tokens[2])
    last_epoch = parse_float(tokens[3])
    duration = parse_float(tokens[4])
    up_velocity_m_yr = parse_float(tokens[10])
    up_uncertainty_m_yr = parse_float(tokens[13])
    steps = parse_int(tokens[23])

    required = [
        latitude,
        longitude,
        height,
        first_epoch,
        last_epoch,
        duration,
        up_velocity_m_yr,
    ]
    if any(value is None for value in required):
        return None

    lon_norm = normalize_longitude(longitude)
    up_mm_yr = up_velocity_m_yr * 1000.0
    up_sigma_mm_yr = (
        up_uncertainty_m_yr * 1000.0 if up_uncertainty_m_yr is not None else None
    )
    abs_up = abs(up_mm_yr)
    first_epoch_filter_year = first_epoch if first_epoch > 1900 else last_epoch - duration

    if abs_up < 0.05:
        sign_class = "near_zero"
    elif up_mm_yr > 0:
        sign_class = "positive"
    else:
        sign_class = "negative"

    return {
        "station": tokens[0],
        "version": tokens[1],
        "first_epoch": tokens[2],
        "last_epoch": tokens[3],
        "first_epoch_year": round(first_epoch_filter_year, 4),
        "last_epoch_year": round(last_epoch, 4),
        "duration": round(duration, 3),
        "up_mm_yr": round(up_mm_yr, 3),
        "up_sigma_mm_yr": round(up_sigma_mm_yr, 3)
        if up_sigma_mm_yr is not None
        else None,
        "latitude": round(latitude, 6),
        "longitude": round(lon_norm, 6),
        "source_longitude": round(longitude, 6),
        "height_m": round(height, 2),
        "steps": steps if steps is not None else 0,
        "sign": sign_class,
        "abs_up": round(abs_up, 3),
    }


def parse_midas(text: str) -> tuple[list[dict], dict]:
    records: list[dict] = []
    malformed = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        record = compact_record(stripped.split())
        if record is None:
            malformed += 1
            continue
        records.append(record)

    if not records:
        raise RuntimeError("No valid MIDAS records were parsed.")

    color_limit = max(5.0, percentile([r["abs_up"] for r in records], 0.95))
    color_limit = round(color_limit, 1)

    for record in records:
        abs_up = record["abs_up"]
        display_abs_up = min(abs_up, color_limit * 2.0)
        record["color"] = diverging_color(record["up_mm_yr"], color_limit)
        record["point_radius_m"] = round(26000 + min(abs_up, color_limit) * 2200)
        record["bar_elevation"] = round(max(0.3, display_abs_up) * 26000)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "source_url": MIDAS_URL,
        "readme_url": README_URL,
        "generated_at_utc": generated_at,
        "station_count": len(records),
        "malformed_rows_skipped": malformed,
        "color_limit_mm_yr": color_limit,
        "positive_count": sum(1 for r in records if r["up_mm_yr"] > 0),
        "negative_count": sum(1 for r in records if r["up_mm_yr"] < 0),
        "near_zero_count": sum(1 for r in records if abs(r["up_mm_yr"]) <= 0.05),
        "max_abs_up_mm_yr": round(max(r["abs_up"] for r in records), 3),
        "first_epoch_min": math.floor(min(r["first_epoch_year"] for r in records)),
        "first_epoch_max": math.ceil(max(r["first_epoch_year"] for r in records)),
        "last_epoch_min": math.floor(min(r["last_epoch_year"] for r in records)),
        "last_epoch_max": math.ceil(max(r["last_epoch_year"] for r in records)),
    }
    return records, metadata


def parse_gia_grid(text: str) -> tuple[list[float | None], list[float | None], dict]:
    width = 360
    height = 180
    values: list[float | None] = [None] * (width * height)
    uncertainties: list[float | None] = [None] * (width * height)
    parsed_values: list[float] = []
    parsed_uncertainties: list[float] = []
    malformed = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue

        tokens = stripped.split()
        if len(tokens) < 3:
            malformed += 1
            continue

        colatitude = parse_float(tokens[0])
        longitude = parse_float(tokens[1])
        vertical_land_motion = parse_float(tokens[2])
        vertical_land_motion_uncertainty = parse_float(tokens[3]) if len(tokens) > 3 else None
        if colatitude is None or longitude is None or vertical_land_motion is None:
            malformed += 1
            continue

        y = int(round(colatitude))
        lon_norm = normalize_longitude(longitude)
        x = int(round(lon_norm + 180.0))
        if x == width:
            x = 0

        if 0 <= x < width and 0 <= y < height:
            value = round(vertical_land_motion, 3)
            values[y * width + x] = value
            parsed_values.append(value)
            if vertical_land_motion_uncertainty is not None:
                uncertainty = round(vertical_land_motion_uncertainty, 3)
                uncertainties[y * width + x] = uncertainty
                parsed_uncertainties.append(uncertainty)
        else:
            malformed += 1

    if not parsed_values:
        raise RuntimeError("No valid GIA grid records were parsed.")

    metadata = {
        "source_url": GIA_URL,
        "publication": "Caron and Ivins (2019)",
        "component": "Tdur",
        "description": "Expected vertical land motion from total GIA sources",
        "units": "mm/yr",
        "width": width,
        "height": height,
        "bounds": [-180, -90, 180, 90],
        "cell_size_degrees": 1,
        "value_count": len(parsed_values),
        "malformed_rows_skipped": malformed,
        "min_mm_yr": round(min(parsed_values), 3),
        "max_mm_yr": round(max(parsed_values), 3),
        "max_abs_mm_yr": round(max(abs(v) for v in parsed_values), 3),
        "uncertainty_component": "Tsdur",
        "uncertainty_count": len(parsed_uncertainties),
        "min_sigma_mm_yr": round(min(parsed_uncertainties), 3) if parsed_uncertainties else None,
        "max_sigma_mm_yr": round(max(parsed_uncertainties), 3) if parsed_uncertainties else None,
    }
    return values, uncertainties, metadata


def parse_ngl_imaged_vlm_grid(text: str) -> tuple[list[float | None], list[float | None], dict]:
    width = 360
    min_lat = -90
    max_lat = 84
    height = max_lat - min_lat
    cell_count = width * height
    trend_sums = [0.0] * cell_count
    uncertainty_sums = [0.0] * cell_count
    zeta_sums = [0.0] * cell_count
    counts = [0] * cell_count
    uncertainty_counts = [0] * cell_count
    zeta_counts = [0] * cell_count
    source_lons: set[float] = set()
    source_lats: set[float] = set()
    parsed_values: list[float] = []
    parsed_uncertainties: list[float] = []
    parsed_zetas: list[float] = []
    malformed = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue

        tokens = stripped.split()
        if len(tokens) < 5:
            malformed += 1
            continue

        lon = parse_float(tokens[0])
        lat = parse_float(tokens[1])
        trend = parse_float(tokens[2])
        uncertainty = parse_float(tokens[3])
        zeta = parse_float(tokens[4])
        if lon is None or lat is None or trend is None:
            malformed += 1
            continue

        source_lons.add(lon)
        source_lats.add(lat)
        lon_norm = normalize_longitude(lon)
        x = int(math.floor(lon_norm + 180.0))
        if x == width:
            x = 0
        y = int(math.floor(lat - min_lat))
        if x < 0 or x >= width or y < 0 or y >= height:
            malformed += 1
            continue

        index = y * width + x
        trend_sums[index] += trend
        counts[index] += 1
        parsed_values.append(trend)

        if uncertainty is not None:
            uncertainty_sums[index] += uncertainty
            uncertainty_counts[index] += 1
            parsed_uncertainties.append(uncertainty)
        if zeta is not None:
            zeta_sums[index] += zeta
            zeta_counts[index] += 1
            parsed_zetas.append(zeta)

    values: list[float | None] = [None] * cell_count
    uncertainties: list[float | None] = [None] * cell_count
    zetas: list[float | None] = [None] * cell_count
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        values[index] = round(trend_sums[index] / count, 4)
        if uncertainty_counts[index] > 0:
            uncertainties[index] = round(uncertainty_sums[index] / uncertainty_counts[index], 4)
        if zeta_counts[index] > 0:
            zetas[index] = round(zeta_sums[index] / zeta_counts[index], 4)

    rendered_values = [value for value in values if value is not None]
    rendered_uncertainties = [value for value in uncertainties if value is not None]
    if not rendered_values:
        raise RuntimeError("No valid NGL GPS Imaging VLM grid records were parsed.")

    metadata = {
        "source_url": NGL_IMAGED_VLM_URL,
        "publication": "Hammond et al. (2021)",
        "component": "Vu",
        "uncertainty_component": "Vu uncertainty",
        "zeta_component": "nearest-neighbor spatial variability in Vu",
        "description": "GPS Imaging interpolated vertical land motion on Earth's land masses",
        "units": "mm/yr",
        "source_cell_size_degrees": 0.25,
        "render_cell_size_degrees": 1,
        "render_aggregation": "mean of source quarter-degree samples within each one-degree cell",
        "width": width,
        "height": height,
        "bounds": [-180, min_lat, 180, max_lat],
        "value_count": len(rendered_values),
        "source_record_count": len(parsed_values),
        "malformed_rows_skipped": malformed,
        "source_lon_count": len(source_lons),
        "source_lat_count": len(source_lats),
        "source_bounds": [
            round(min(source_lons), 4) if source_lons else None,
            round(min(source_lats), 4) if source_lats else None,
            round(max(source_lons), 4) if source_lons else None,
            round(max(source_lats), 4) if source_lats else None,
        ],
        "min_mm_yr": round(min(rendered_values), 4),
        "max_mm_yr": round(max(rendered_values), 4),
        "max_abs_mm_yr": round(max(abs(v) for v in rendered_values), 4),
        "median_mm_yr": round(percentile(rendered_values, 0.5), 4),
        "uncertainty_count": len(rendered_uncertainties),
        "min_sigma_mm_yr": round(min(rendered_uncertainties), 4) if rendered_uncertainties else None,
        "max_sigma_mm_yr": round(max(rendered_uncertainties), 4) if rendered_uncertainties else None,
        "min_zeta_mm_yr": round(min(parsed_zetas), 4) if parsed_zetas else None,
        "max_zeta_mm_yr": round(max(parsed_zetas), 4) if parsed_zetas else None,
        "zeta_values": zetas,
    }
    return values, uncertainties, metadata


TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}


def packbits_decode(data: bytes, expected_length: int | None = None) -> bytes:
    output = bytearray()
    idx = 0
    data_len = len(data)

    while idx < data_len:
        control = data[idx]
        idx += 1

        if control <= 127:
            count = control + 1
            output.extend(data[idx : idx + count])
            idx += count
        elif control >= 129:
            count = 257 - control
            if idx < data_len:
                output.extend(data[idx : idx + 1] * count)
                idx += 1
        else:
            continue

    if expected_length is not None:
        return bytes(output[:expected_length])
    return bytes(output)


def read_tiff_tag_value(
    data: bytes,
    endian: str,
    tag_type: int,
    count: int,
    value_offset: int,
    inline_value: bytes,
) -> object:
    size = TIFF_TYPE_SIZES[tag_type] * count
    raw = inline_value if size <= 4 else data[value_offset : value_offset + size]

    if tag_type == 2:
        return raw[:size].rstrip(b"\x00").decode("ascii", errors="replace")
    if tag_type == 3:
        return struct.unpack(endian + ("H" * count), raw[:size])
    if tag_type == 4:
        return struct.unpack(endian + ("I" * count), raw[:size])
    if tag_type == 12:
        return struct.unpack(endian + ("d" * count), raw[:size])
    return raw[:size]


def read_simple_geotiff(path: Path) -> dict:
    data = path.read_bytes()
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise RuntimeError(f"{path} is not a TIFF file.")

    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        raise RuntimeError(f"{path} has unsupported TIFF magic value {magic}.")

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[int, object] = {}
    cursor = ifd_offset + 2

    for _ in range(entry_count):
        entry = data[cursor : cursor + 12]
        cursor += 12
        tag, tag_type, count, value_offset = struct.unpack(endian + "HHII", entry)
        tags[tag] = read_tiff_tag_value(
            data, endian, tag_type, count, value_offset, entry[8:12]
        )

    width = int(tags[256][0])
    height = int(tags[257][0])
    bits_per_sample = int(tags[258][0])
    compression = int(tags[259][0])
    sample_format = int(tags.get(339, (3,))[0])

    if bits_per_sample != 64 or sample_format != 3:
        raise RuntimeError(f"{path} is not a 64-bit floating-point GeoTIFF.")
    if compression != 32773:
        raise RuntimeError(f"{path} uses unsupported TIFF compression {compression}.")

    values = [math.nan] * (width * height)

    if 324 in tags:
        tile_width = int(tags[322][0])
        tile_height = int(tags[323][0])
        tile_offsets = tags[324]
        tile_byte_counts = tags[325]
        tile_columns = math.ceil(width / tile_width)

        for tile_idx, (offset, byte_count) in enumerate(
            zip(tile_offsets, tile_byte_counts)
        ):
            tile_x = tile_idx % tile_columns
            tile_y = tile_idx // tile_columns
            raw = packbits_decode(
                data[offset : offset + byte_count], tile_width * tile_height * 8
            )
            tile_values = struct.unpack(endian + ("d" * (len(raw) // 8)), raw)

            for yy in range(tile_height):
                target_y = tile_y * tile_height + yy
                if target_y >= height:
                    break
                for xx in range(tile_width):
                    target_x = tile_x * tile_width + xx
                    if target_x >= width:
                        break
                    values[target_y * width + target_x] = tile_values[
                        yy * tile_width + xx
                    ]
    else:
        strip_offsets = tags[273]
        strip_byte_counts = tags[279]
        rows_per_strip = int(tags.get(278, (height,))[0])

        for strip_idx, (offset, byte_count) in enumerate(
            zip(strip_offsets, strip_byte_counts)
        ):
            start_y = strip_idx * rows_per_strip
            row_count = min(rows_per_strip, height - start_y)
            raw = packbits_decode(data[offset : offset + byte_count], row_count * width * 8)
            strip_values = struct.unpack(endian + ("d" * (len(raw) // 8)), raw)

            for yy in range(row_count):
                for xx in range(width):
                    values[(start_y + yy) * width + xx] = strip_values[yy * width + xx]

    scale = tags[33550]
    tiepoint = tags[33922]
    west = float(tiepoint[3])
    north = float(tiepoint[4])
    east = west + width * float(scale[0])
    south = north - height * float(scale[1])

    return {
        "width": width,
        "height": height,
        "bounds": [west, south, east, north],
        "values": values,
    }


def lzw_decode(data: bytes) -> bytes:
    clear_code = 256
    eoi_code = 257
    code_size = 9
    next_code = 258
    dictionary = {i: bytes([i]) for i in range(256)}
    output = bytearray()
    previous: bytes | None = None
    bit_pos = 0
    total_bits = len(data) * 8

    def read_code(size: int) -> int | None:
        nonlocal bit_pos
        if bit_pos + size > total_bits:
            return None
        code = 0
        for _ in range(size):
            byte = data[bit_pos // 8]
            bit = 7 - (bit_pos % 8)
            code = (code << 1) | ((byte >> bit) & 1)
            bit_pos += 1
        return code

    while True:
        code = read_code(code_size)
        if code is None:
            break
        if code == clear_code:
            dictionary = {i: bytes([i]) for i in range(256)}
            code_size = 9
            next_code = 258
            previous = None
            continue
        if code == eoi_code:
            break

        if code in dictionary:
            entry = dictionary[code]
        elif previous is not None and code == next_code:
            entry = previous + previous[:1]
        else:
            raise RuntimeError(f"Invalid LZW code {code}.")

        output.extend(entry)
        if previous is not None:
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
            if next_code == (1 << code_size) - 1 and code_size < 12:
                code_size += 1
        previous = entry

    return bytes(output)


def read_geotiff_ifd(path: Path) -> dict:
    data = path.read_bytes()
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise RuntimeError(f"{path} is not a TIFF file.")

    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        raise RuntimeError(f"{path} has unsupported TIFF magic value {magic}.")

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[int, object] = {}
    cursor = ifd_offset + 2

    for _ in range(entry_count):
        entry = data[cursor : cursor + 12]
        cursor += 12
        tag, tag_type, count, value_offset = struct.unpack(endian + "HHII", entry)
        tags[tag] = read_tiff_tag_value(
            data, endian, tag_type, count, value_offset, entry[8:12]
        )

    return {"data": data, "endian": endian, "tags": tags}


def mollweide_forward(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    radius = 6378137.0
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(max(min(lat_deg, 89.999999), -89.999999))
    theta = lat_rad
    target = math.pi * math.sin(lat_rad)

    for _ in range(12):
        denominator = 2.0 + 2.0 * math.cos(2.0 * theta)
        if abs(denominator) < 1e-12:
            break
        delta = (2.0 * theta + math.sin(2.0 * theta) - target) / denominator
        theta -= delta
        if abs(delta) < 1e-12:
            break

    x = (2.0 * math.sqrt(2.0) * radius / math.pi) * lon_rad * math.cos(theta)
    y = math.sqrt(2.0) * radius * math.sin(theta)
    return x, y


def parse_ghsl_population_grid() -> tuple[list[float | None], dict]:
    tif_paths = sorted(GHSL_POP_DIR.rglob("GHS_WUP_POP_E2025*.tif"))
    tif_paths = [path for path in tif_paths if not path.name.endswith(".ovr")]
    if not tif_paths:
        raise RuntimeError(f"No GHSL population GeoTIFF found in {GHSL_POP_DIR}.")

    main_path = tif_paths[0]
    main_raster = read_geotiff_ifd(main_path)
    main_tags = main_raster["tags"]
    main_scale = main_tags[33550]
    main_tiepoint = main_tags[33922]
    main_width = int(main_tags[256][0])
    main_height = int(main_tags[257][0])
    main_west = float(main_tiepoint[3])
    main_north = float(main_tiepoint[4])
    main_east = main_west + main_width * float(main_scale[0])
    main_south = main_north - main_height * float(main_scale[1])

    overview_path = main_path.with_suffix(main_path.suffix + ".ovr")
    path = overview_path if overview_path.exists() else main_path
    raster = read_geotiff_ifd(path)
    data = raster["data"]
    endian = raster["endian"]
    tags = raster["tags"]

    source_width = int(tags[256][0])
    source_height = int(tags[257][0])
    bits_per_sample = int(tags[258][0])
    compression = int(tags[259][0])
    sample_format = int(tags.get(339, (1,))[0])
    no_data_raw = tags.get(42113, "-200")
    no_data = float(no_data_raw) if isinstance(no_data_raw, str) else -200.0

    if bits_per_sample != 64 or sample_format != 3:
        raise RuntimeError(f"{path} is not a 64-bit floating-point population GeoTIFF.")
    if compression not in {5, 8, 32946}:
        raise RuntimeError(f"{path} uses unsupported GHSL TIFF compression {compression}.")

    west = main_west
    east = main_east
    south = main_south
    north = main_north
    pixel_width = (east - west) / source_width
    pixel_height = (north - south) / source_height

    tile_width = int(tags[322][0])
    tile_height = int(tags[323][0])
    tile_offsets = tags[324]
    tile_byte_counts = tags[325]
    tile_columns = math.ceil(source_width / tile_width)
    decoded_tiles: dict[int, bytes] = {}
    decode_order: list[int] = []
    max_cached_tiles = 48

    def get_tile(tile_index: int) -> bytes:
        if tile_index in decoded_tiles:
            return decoded_tiles[tile_index]

        offset = tile_offsets[tile_index]
        byte_count = tile_byte_counts[tile_index]
        compressed = data[offset : offset + byte_count]
        if compression == 5:
            decoded = lzw_decode(compressed)
        else:
            decoded = zlib.decompress(compressed)
        decoded_tiles[tile_index] = decoded
        decode_order.append(tile_index)
        if len(decode_order) > max_cached_tiles:
            old_index = decode_order.pop(0)
            decoded_tiles.pop(old_index, None)
        return decoded

    def sample_population(lon: float, lat: float) -> float | None:
        x, y = mollweide_forward(lon, lat)
        if x < west or x >= east or y <= south or y > north:
            return None

        col = int((x - west) / pixel_width)
        row = int((north - y) / pixel_height)
        if col < 0 or row < 0 or col >= source_width or row >= source_height:
            return None

        tile_x = col // tile_width
        tile_y = row // tile_height
        tile_index = tile_y * tile_columns + tile_x
        tile = get_tile(tile_index)
        local_x = col - tile_x * tile_width
        local_y = row - tile_y * tile_height
        byte_offset = (local_y * tile_width + local_x) * 8
        if byte_offset + 8 > len(tile):
            return None

        value = struct.unpack_from(endian + "d", tile, byte_offset)[0]
        if not math.isfinite(value) or value == no_data or value <= 0:
            return None
        return value

    output_width = 9216
    output_height = 4608
    sub_samples = 1
    values: list[list[float]] = []
    valid_values: list[float] = []
    lon_step = 360.0 / output_width
    lat_step = 180.0 / output_height

    for y_index in range(output_height):
        for x_index in range(output_width):
            sampled_values: list[float] = []
            for sy in range(sub_samples):
                lat = 90.0 - (y_index + (sy + 0.5) / sub_samples) * lat_step
                for sx in range(sub_samples):
                    lon = -180.0 + (x_index + (sx + 0.5) / sub_samples) * lon_step
                    value = sample_population(lon, lat)
                    if value is not None:
                        sampled_values.append(value)

            if not sampled_values:
                continue

            rounded = int(round(max(sampled_values)))
            center_lon = round(-180.0 + (x_index + 0.5) * lon_step, 5)
            center_lat = round(90.0 - (y_index + 0.5) * lat_step, 5)
            log_value = math.log10(rounded + 1.0)
            radius_m = round(3500 + min(log_value, 5.0) * 1400)
            values.append([center_lon, center_lat, rounded, radius_m])
            valid_values.append(rounded)

    if not valid_values:
        raise RuntimeError("No valid GHSL population cells were sampled.")

    metadata = {
        "source_url": GHSL_POP_URL,
        "dataset_title": "GHS-WUP population spatial raster dataset R2025A",
        "component": "population",
        "description": "Projected 2025 population counts sampled from the GHSL GHS-WUP-POP R2025A 1 km Mollweide raster.",
        "units": "people per source grid cell",
        "sampling_strategy": f"{sub_samples}x{sub_samples} maximum subsampling per rendered lon/lat cell",
        "source_projection": "World Mollweide / WGS 84",
        "source_width": source_width,
        "source_height": source_height,
        "source_file": str(path),
        "source_bounds_projected_m": [round(west, 3), round(south, 3), round(east, 3), round(north, 3)],
        "width": output_width,
        "height": output_height,
        "bounds": [-180, -90, 180, 90],
        "render_record_format": ["longitude", "latitude", "population", "radius_m"],
        "valid_pixel_count": len(valid_values),
        "min_population": int(round(min(valid_values))),
        "max_population": int(round(max(valid_values))),
        "p95_population": int(round(percentile(valid_values, 0.95))),
        "p99_population": int(round(percentile(valid_values, 0.99))),
        "max_log10_population": round(math.log10(percentile(valid_values, 0.99) + 1.0), 4),
    }
    return values, metadata


def build_ghsl_population_sidecar(force_refresh: bool = False) -> dict:
    if (
        POPULATION_PAYLOAD_JS.exists()
        and POPULATION_METADATA_CACHE.exists()
        and not force_refresh
    ):
        return json.loads(POPULATION_METADATA_CACHE.read_text(encoding="utf-8"))

    values, metadata = parse_ghsl_population_grid()
    POPULATION_PAYLOAD_JS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "values": values,
    }
    POPULATION_PAYLOAD_JS.write_text(
        "window.GHSL_POPULATION_PAYLOAD="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    POPULATION_METADATA_CACHE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def parse_insar_grids() -> tuple[list[dict], dict]:
    tif_paths = sorted(INSAR_DIR.glob("*_vlm.tif"))
    if not tif_paths:
        raise RuntimeError(f"No *_vlm.tif files found in {INSAR_DIR}.")

    grids: list[dict] = []
    all_values: list[float] = []
    raw_values: list[float] = []
    scale_factor_to_mm = 10.0

    for path in tif_paths:
        raster = read_simple_geotiff(path)
        values: list[float | None] = []
        valid_count = 0

        for raw_value in raster["values"]:
            if not math.isfinite(raw_value) or abs(raw_value) > 1e20:
                values.append(None)
                continue

            raw_values.append(raw_value)
            value_mm_yr = round(raw_value * scale_factor_to_mm, 3)
            values.append(value_mm_yr)
            all_values.append(value_mm_yr)
            valid_count += 1

        grids.append(
            {
                "id": path.stem.replace("_vlm", ""),
                "name": path.stem.replace("_vlm", "").replace("_", " ").title(),
                "width": raster["width"],
                "height": raster["height"],
                "bounds": [round(v, 8) for v in raster["bounds"]],
                "values": values,
                "valid_count": valid_count,
            }
        )

    if not all_values:
        raise RuntimeError("No valid InSAR VLM pixels were parsed.")

    metadata = {
        "source_url": INSAR_URL,
        "publication_url": "https://www.nature.com/articles/s41586-025-09928-6",
        "dataset_title": "The Global Threat of Sinking Deltas",
        "component": "VLM",
        "description": "InSAR vertical land motion grids for 40 global deltas",
        "units": "mm/yr",
        "source_units_inferred": "cm/yr",
        "scale_factor_to_mm_yr": scale_factor_to_mm,
        "grid_count": len(grids),
        "valid_pixel_count": len(all_values),
        "min_mm_yr": round(min(all_values), 3),
        "max_mm_yr": round(max(all_values), 3),
        "max_abs_mm_yr": round(max(abs(v) for v in all_values), 3),
        "raw_min": round(min(raw_values), 6),
        "raw_max": round(max(raw_values), 6),
    }
    return grids, metadata


def parse_gns_coastal_vlm() -> tuple[list[dict], dict]:
    if not GNS_COAST_FILE.exists():
        raise RuntimeError(f"GNS coastal VLM file not found: {GNS_COAST_FILE}")

    records: list[dict] = []
    malformed = 0
    values: list[float] = []
    sigmas: list[float] = []

    with GNS_COAST_FILE.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("lon"):
                continue

            tokens = stripped.replace(",", " ").split()
            if len(tokens) < 7:
                malformed += 1
                continue

            lon = parse_float(tokens[0])
            lat = parse_float(tokens[1])
            up_mm_yr = parse_float(tokens[2])
            sigma_mm_yr = parse_float(tokens[3])
            observations = parse_int(tokens[4])
            quality_factor = parse_float(tokens[5])
            average_radius_km = parse_float(tokens[6])

            if lon is None or lat is None or up_mm_yr is None:
                malformed += 1
                continue

            lon_norm = normalize_longitude(lon)
            abs_up = abs(up_mm_yr)
            values.append(up_mm_yr)
            if sigma_mm_yr is not None:
                sigmas.append(sigma_mm_yr)

            records.append(
                {
                    "id": f"gns-coast-{line_number:05d}",
                    "dataset": "GNS New Zealand coastal VLM",
                    "longitude": round(lon_norm, 6),
                    "source_longitude": round(lon, 6),
                    "latitude": round(lat, 6),
                    "up_mm_yr": round(up_mm_yr, 4),
                    "up_sigma_mm_yr": round(sigma_mm_yr, 4)
                    if sigma_mm_yr is not None
                    else None,
                    "observations": observations if observations is not None else None,
                    "quality_factor": round(quality_factor, 4)
                    if quality_factor is not None
                    else None,
                    "average_radius_km": round(average_radius_km, 4)
                    if average_radius_km is not None
                    else None,
                    "abs_up": round(abs_up, 4),
                    "point_radius_m": round(10500 + min(abs_up, 12.0) * 950),
                }
            )

    if not records:
        raise RuntimeError("No valid GNS coastal VLM records were parsed.")

    metadata = {
        "source_url": GNS_ATTACHMENT_URL,
        "metadata_url": GNS_METADATA_URL,
        "article_url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL096465",
        "dataset_doi": "10.21420/E1C1-MQ19",
        "article_doi": "10.1029/2021GL096465",
        "dataset_title": "A snapshot of New Zealand's dynamic deformation field from Envisat InSAR and GNSS observations between 2003 and 2011",
        "component": "coastal vertical land movement",
        "rendered_file": str(GNS_COAST_FILE),
        "units": "mm/yr",
        "record_count": len(records),
        "malformed_rows_skipped": malformed,
        "min_mm_yr": round(min(values), 4),
        "max_mm_yr": round(max(values), 4),
        "max_abs_mm_yr": round(max(abs(value) for value in values), 4),
        "median_mm_yr": round(percentile(values, 0.5), 4),
        "min_sigma_mm_yr": round(min(sigmas), 4) if sigmas else None,
        "max_sigma_mm_yr": round(max(sigmas), 4) if sigmas else None,
        "archive_files": [
            "NZ_InSAR_2003-2011_GRL.txt",
            "NZ_Vertical_GPS_2003-2011_GRL.txt",
            "NZ_coast_1km_GRL.txt",
            "NZ_coast_1km_GRL_BOP-corrected.txt",
            "Cull.dat",
            "NZ_2km_cut.dat",
        ],
    }
    return records, metadata


def decode_netcdf_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def parse_oelsmann_hybrid_vlm() -> tuple[list[dict], dict]:
    if not OELSMANN_HYBRID_NC_FILE.exists():
        raise RuntimeError(f"Oelsmann hybrid VLM NetCDF file not found: {OELSMANN_HYBRID_NC_FILE}")

    import h5py

    records: list[dict] = []
    malformed = 0
    values: list[float] = []
    sigmas: list[float] = []

    with h5py.File(OELSMANN_HYBRID_NC_FILE, "r") as hdf:
        lon_values = hdf["lon"][:]
        lat_values = hdf["lat"][:]
        names = hdf["name"][:]
        trend_values = hdf["OE24_GPS_InSAR_GIA"][:]
        uncertainty_values = hdf["OE24_GPS_InSAR_GIA_un"][:]
        datatype_values = hdf["datatype"][:] if "datatype" in hdf else [math.nan] * len(trend_values)

        for index, (lon, lat, name, trend, uncertainty, datatype) in enumerate(
            zip(lon_values, lat_values, names, trend_values, uncertainty_values, datatype_values),
            start=1,
        ):
            lon = float(lon)
            lat = float(lat)
            trend = float(trend)
            uncertainty = float(uncertainty)
            datatype = float(datatype)

            if not math.isfinite(lon) or not math.isfinite(lat) or not math.isfinite(trend):
                malformed += 1
                continue

            lon_norm = normalize_longitude(lon)
            abs_up = abs(trend)
            values.append(trend)
            if math.isfinite(uncertainty):
                sigmas.append(uncertainty)

            records.append(
                {
                    "id": f"oelsmann-hybrid-{index:05d}",
                    "name": decode_netcdf_string(name),
                    "dataset": "Oelsmann hybrid coastal VLM",
                    "dataset_id": "hybrid_oelsmann_2026",
                    "longitude": round(lon_norm, 6),
                    "source_longitude": round(lon, 6),
                    "latitude": round(lat, 6),
                    "up_mm_yr": round(trend, 4),
                    "up_sigma_mm_yr": round(uncertainty, 4) if math.isfinite(uncertainty) else None,
                    "datatype": round(datatype, 4) if math.isfinite(datatype) else None,
                    "abs_up": round(abs_up, 4),
                    "point_radius_m": round(52000 + min(abs_up, 12.0) * 1600),
                }
            )

    if not records:
        raise RuntimeError("No valid Oelsmann hybrid coastal VLM records were parsed.")

    metadata = {
        "source_url": OELSMANN_HYBRID_NC_URL,
        "record_url": OELSMANN_HYBRID_RECORD_URL,
        "article_url": "https://www.nature.com/articles/s41467-026-72293-z",
        "article_doi": "10.1038/s41467-026-72293-z",
        "dataset_title": "Global hybrid vertical land motion estimates along global coastlines",
        "component": "OE24_GPS_InSAR_GIA",
        "uncertainty_component": "OE24_GPS_InSAR_GIA_un",
        "rendered_file": str(OELSMANN_HYBRID_NC_FILE),
        "units": "mm/yr",
        "record_count": len(records),
        "malformed_rows_skipped": malformed,
        "min_mm_yr": round(min(values), 4),
        "max_mm_yr": round(max(values), 4),
        "max_abs_mm_yr": round(max(abs(value) for value in values), 4),
        "median_mm_yr": round(percentile(values, 0.5), 4),
        "min_sigma_mm_yr": round(min(sigmas), 4) if sigmas else None,
        "max_sigma_mm_yr": round(max(sigmas), 4) if sigmas else None,
    }
    return records, metadata


def decode_matlab_string(hdf_file, ref) -> str:
    values = hdf_file[ref][()]
    chars: list[str] = []
    for value in values.ravel():
        code = int(value)
        if code:
            chars.append(chr(code))
    return "".join(chars).strip()


def linear_fit(series: list[tuple[float, float]]) -> tuple[float, float, float | None]:
    n = len(series)
    if n < 2:
        raise ValueError("At least two samples are required for a linear fit.")

    sum_x = sum(x for x, _ in series)
    sum_y = sum(y for _, y in series)
    mean_x = sum_x / n
    mean_y = sum_y / n
    sxx = sum((x - mean_x) ** 2 for x, _ in series)
    if sxx <= 0:
        raise ValueError("Cannot fit a trend with zero time variance.")

    sxy = sum((x - mean_x) * (y - mean_y) for x, y in series)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    slope_sigma = None
    if n > 2:
        rss = sum((y - (slope * x + intercept)) ** 2 for x, y in series)
        residual_variance = rss / (n - 2)
        slope_sigma = math.sqrt(residual_variance / sxx)

    return slope, intercept, slope_sigma


def haversine_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    radius_km = 6371.0088
    lat1 = math.radians(lat_a)
    lat2 = math.radians(lat_b)
    d_lat = math.radians(lat_b - lat_a)
    d_lon = math.radians(lon_b - lon_a)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def attach_nearby_gnss_to_tide_gauges(
    tide_gauge_records: list[dict], gnss_records: list[dict], max_distance_km: float = 100.0
) -> dict:
    counts: list[int] = []
    for tide_gauge in tide_gauge_records:
        candidates: list[dict] = []
        for station in gnss_records:
            distance = haversine_km(
                tide_gauge["longitude"],
                tide_gauge["latitude"],
                station["longitude"],
                station["latitude"],
            )
            if distance > max_distance_km:
                continue
            candidates.append(
                {
                    "station": station["station"],
                    "longitude": station["longitude"],
                    "latitude": station["latitude"],
                    "distance_km": round(distance, 2),
                    "up_mm_yr": station["up_mm_yr"],
                    "up_sigma_mm_yr": station["up_sigma_mm_yr"],
                    "first_year": station["first_epoch_year"],
                    "last_year": station["last_epoch_year"],
                    "duration": station["duration"],
                }
            )

        nearby = sorted(candidates, key=lambda item: item["distance_km"])[:10]
        tide_gauge["nearby_gnss"] = nearby
        counts.append(len(nearby))

    TIDE_GAUGE_NEARBY_GNSS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TIDE_GAUGE_NEARBY_GNSS_CACHE.write_text(
        json.dumps(
            {
                "max_distance_km": max_distance_km,
                "max_station_count": 10,
                "tide_gauges": [
                    {
                        "id": tide_gauge["id"],
                        "name": tide_gauge["name"],
                        "psmsl_id": tide_gauge["psmsl_id"],
                        "nearby_gnss": tide_gauge["nearby_gnss"],
                    }
                    for tide_gauge in tide_gauge_records
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    TIDE_GAUGE_PROCESSED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if TIDE_GAUGE_PROCESSED_CACHE.exists():
        payload = json.loads(TIDE_GAUGE_PROCESSED_CACHE.read_text(encoding="utf-8"))
        payload["records"] = tide_gauge_records
        payload.setdefault("metadata", {})["nearby_gnss_max_distance_km"] = max_distance_km
        payload["metadata"]["nearby_gnss_max_count"] = 10
        payload["metadata"]["nearby_gnss_station_count_min"] = min(counts) if counts else 0
        payload["metadata"]["nearby_gnss_station_count_max"] = max(counts) if counts else 0
        TIDE_GAUGE_PROCESSED_CACHE.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "nearby_gnss_max_distance_km": max_distance_km,
        "nearby_gnss_max_count": 10,
        "nearby_gnss_station_count_min": min(counts) if counts else 0,
        "nearby_gnss_station_count_max": max(counts) if counts else 0,
    }


def load_processed_tide_gauge_cache() -> tuple[list[dict], dict] | None:
    if not TIDE_GAUGE_PROCESSED_CACHE.exists():
        return None
    payload = json.loads(TIDE_GAUGE_PROCESSED_CACHE.read_text(encoding="utf-8"))
    return payload["records"], payload["metadata"]


def parse_tide_gauge_vlm() -> tuple[list[dict], dict]:
    if not TIDE_GAUGE_MAT_FILE.exists():
        raise RuntimeError(f"CSL-TG MAT file not found: {TIDE_GAUGE_MAT_FILE}")

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        cached = load_processed_tide_gauge_cache()
        if cached is not None:
            return cached
        raise RuntimeError(
            "Parsing CSL-TG.mat requires h5py and numpy because the file is MATLAB v7.3/HDF5. "
            "Install them or run this generator in the conda environment used for data processing."
        ) from exc

    records: list[dict] = []
    trends: list[float] = []
    sigmas: list[float] = []
    sample_counts: list[int] = []
    first_years: list[float] = []
    last_years: list[float] = []

    with h5py.File(TIDE_GAUGE_MAT_FILE, "r") as hdf:
        time_values = [float(value) for value in hdf["t"][()].ravel()]
        lat_values = [float(value) for value in hdf["lat"][()].ravel()]
        lon_values = [float(value) for value in hdf["lon"][()].ravel()]
        ids = [int(round(float(value))) for value in hdf["MID"][()].ravel()]
        names = [decode_matlab_string(hdf, ref) for ref in hdf["N"][0]]
        values = hdf["MDadj3"][()]

        for index, name in enumerate(names):
            if index >= values.shape[0]:
                break

            row = values[index, :]
            series_m = [
                (year, float(value))
                for year, value in zip(time_values, row)
                if math.isfinite(year) and np.isfinite(value)
            ]
            if len(series_m) < 10:
                continue

            slope_m_yr, intercept_m, sigma_m_yr = linear_fit(series_m)
            trend_mm_yr = slope_m_yr * 1000.0
            intercept_mm = intercept_m * 1000.0
            sigma_mm_yr = sigma_m_yr * 1000.0 if sigma_m_yr is not None else None
            series_mm = [[round(year, 3), round(value * 1000.0, 3)] for year, value in series_m]
            lon_norm = normalize_longitude(lon_values[index])
            first_year = series_mm[0][0]
            last_year = series_mm[-1][0]
            abs_trend = abs(trend_mm_yr)

            trends.append(trend_mm_yr)
            if sigma_mm_yr is not None:
                sigmas.append(sigma_mm_yr)
            sample_counts.append(len(series_mm))
            first_years.append(first_year)
            last_years.append(last_year)

            records.append(
                {
                    "id": f"tg-{ids[index] if index < len(ids) else index + 1}",
                    "psmsl_id": ids[index] if index < len(ids) else None,
                    "dataset": "CSL tide-gauge VLM",
                    "dataset_id": "tide_gauge_dangendorf_2026",
                    "name": name or f"Tide gauge {index + 1}",
                    "longitude": round(lon_norm, 6),
                    "source_longitude": round(lon_values[index], 6),
                    "latitude": round(lat_values[index], 6),
                    "up_mm_yr": round(trend_mm_yr, 4),
                    "up_sigma_mm_yr": round(sigma_mm_yr, 4) if sigma_mm_yr is not None else None,
                    "trend_intercept_mm": round(intercept_mm, 4),
                    "first_year": round(first_year, 3),
                    "last_year": round(last_year, 3),
                    "sample_count": len(series_mm),
                    "abs_up": round(abs_trend, 4),
                    "point_radius_m": round(26000 + min(abs_trend, 14.0) * 1500),
                    "series": series_mm,
                }
            )

    if not records:
        raise RuntimeError("No valid CSL-TG tide-gauge VLM records were parsed.")

    metadata = {
        "source_url": TIDE_GAUGE_MAT_URL,
        "record_url": TIDE_GAUGE_RECORD_URL,
        "dataset_title": "Variable Vertical Land Motion Inferred from Multi-Decadal to Centennial Tide Gauge Records",
        "component": "MDadj3 adjusted CSL-minus-tide-gauge residual trend",
        "units": "mm/yr",
        "source_units": "m",
        "source_variable": "MDadj3",
        "record_count": len(records),
        "min_mm_yr": round(min(trends), 4),
        "max_mm_yr": round(max(trends), 4),
        "max_abs_mm_yr": round(max(abs(value) for value in trends), 4),
        "median_mm_yr": round(percentile(trends, 0.5), 4),
        "min_sigma_mm_yr": round(min(sigmas), 4) if sigmas else None,
        "max_sigma_mm_yr": round(max(sigmas), 4) if sigmas else None,
        "time_period": {"min": round(min(first_years), 3), "max": round(max(last_years), 3)},
        "min_sample_count": min(sample_counts),
        "max_sample_count": max(sample_counts),
        "mat_file": str(TIDE_GAUGE_MAT_FILE),
    }

    TIDE_GAUGE_PROCESSED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TIDE_GAUGE_PROCESSED_CACHE.write_text(
        json.dumps({"records": records, "metadata": metadata}, ensure_ascii=False),
        encoding="utf-8",
    )
    return records, metadata


def bounds_from_records(records: list[dict]) -> dict:
    return {
        "min_lon": round(min(r["longitude"] for r in records), 6),
        "max_lon": round(max(r["longitude"] for r in records), 6),
        "min_lat": round(min(r["latitude"] for r in records), 6),
        "max_lat": round(max(r["latitude"] for r in records), 6),
    }


def bounds_from_grid_bounds(bounds_list: Iterable[list[float]]) -> dict:
    bounds = list(bounds_list)
    return {
        "min_lon": round(min(b[0] for b in bounds), 6),
        "max_lon": round(max(b[2] for b in bounds), 6),
        "min_lat": round(min(b[1] for b in bounds), 6),
        "max_lat": round(max(b[3] for b in bounds), 6),
    }


PROCESS_FIELDS = [
    "influenced_by_vertical_accretion_erosion_sedimentation",
    "influenced_by_mineral_sedimentation",
    "influenced_by_biogenic_organic_accretion",
    "influenced_by_coastal_erosion_shoreline_retreat",
    "influenced_by_storm_event_deposition_overwash",
    "influenced_by_autocompaction_sediment_self_weight",
    "influenced_by_shallow_compaction_drainage",
    "influenced_by_peat_oxidation_organic_soil_loss",
    "influenced_by_wetting_drying_shrink_swell",
    "influenced_by_permafrost_thaw_frost_heave_thermokarst",
    "influenced_by_local_surface_loading",
    "influenced_by_anthropogenic_fill_construction_loading",
    "influenced_by_reservoir_impoundment_dam_loading",
    "influenced_by_hydrologic_loading",
    "influenced_by_ocean_atmosphere_loading",
    "influenced_by_groundwater_withdrawal_recharge",
    "influenced_by_hydrocarbon_extraction_injection",
    "influenced_by_geothermal_extraction_injection",
    "influenced_by_subsurface_fluid_extraction_general",
    "influenced_by_mining_tunneling_cavity_collapse",
    "influenced_by_karst_dissolution_sinkholes",
    "influenced_by_drainage_reclamation_polder_management",
    "influenced_by_river_delta_channel_migration_overbank_processes",
    "influenced_by_landslide_mass_wasting",
    "influenced_by_tectonic_deformation_general",
    "influenced_by_earthquake_cycle_coseismic_postseismic",
    "influenced_by_fault_creep",
    "influenced_by_volcanic_activity",
    "influenced_by_salt_tectonics_halokinesis",
    "influenced_by_mantle_dynamics",
    "influenced_by_grd_contemporary_mass_redistribution",
    "influenced_by_gia",
    "influenced_by_sediment_loading_unloading_isostatic_response",
    "influenced_by_residual_vlm",
]


def process_flags(**overrides: str) -> dict:
    flags = {field: "uncertain" for field in PROCESS_FIELDS}
    flags.update(overrides)
    return flags


GNSS_CLASSIFICATION = {
    "technique": "GNSS",
    "vlm_note": "Direct geodetic point measurement of land motion, typically in a terrestrial/geocentric reference frame.",
    "anchor_depth_note": "Approximately monument anchor depth.",
    "reference_frame": "terrestrial_reference_frame",
    "vlm_measurement_type": "direct",
    "benchmark_dependence": "moderate",
    "observes_land_surface_directly": "false",
    "observes_infrastructure_motion": "true",
    "observes_geocentric_land_motion": "true",
    "sensitive_to_shallow_vlm": "true",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "false",
    **process_flags(
        influenced_by_vertical_accretion_erosion_sedimentation="false",
        influenced_by_mineral_sedimentation="false",
        influenced_by_biogenic_organic_accretion="false",
        influenced_by_coastal_erosion_shoreline_retreat="false",
        influenced_by_storm_event_deposition_overwash="false",
        influenced_by_autocompaction_sediment_self_weight="true",
        influenced_by_shallow_compaction_drainage="true",
        influenced_by_peat_oxidation_organic_soil_loss="true",
        influenced_by_wetting_drying_shrink_swell="true",
        influenced_by_permafrost_thaw_frost_heave_thermokarst="true",
        influenced_by_local_surface_loading="true",
        influenced_by_anthropogenic_fill_construction_loading="true",
        influenced_by_reservoir_impoundment_dam_loading="true",
        influenced_by_hydrologic_loading="true",
        influenced_by_ocean_atmosphere_loading="true",
        influenced_by_groundwater_withdrawal_recharge="true",
        influenced_by_hydrocarbon_extraction_injection="true",
        influenced_by_geothermal_extraction_injection="true",
        influenced_by_subsurface_fluid_extraction_general="true",
        influenced_by_mining_tunneling_cavity_collapse="true",
        influenced_by_karst_dissolution_sinkholes="true",
        influenced_by_drainage_reclamation_polder_management="true",
        influenced_by_river_delta_channel_migration_overbank_processes="uncertain",
        influenced_by_landslide_mass_wasting="true",
        influenced_by_tectonic_deformation_general="true",
        influenced_by_earthquake_cycle_coseismic_postseismic="true",
        influenced_by_fault_creep="true",
        influenced_by_volcanic_activity="true",
        influenced_by_salt_tectonics_halokinesis="true",
        influenced_by_mantle_dynamics="true",
        influenced_by_grd_contemporary_mass_redistribution="true",
        influenced_by_gia="true",
        influenced_by_sediment_loading_unloading_isostatic_response="true",
        influenced_by_residual_vlm="uncertain",
    ),
    "process_influence_note": "This is a direct geodetic point measurement referenced to a monument in a terrestrial reference frame. It can contain shallow and deep VLM signals, but it is not sensitive to vertical accretion at the land surface.",
    "classification_confidence": "high",
}


GIA_CLASSIFICATION = {
    "technique": "GIA model",
    "vlm_note": "Model of glacial isostatic adjustment and related deep solid-Earth response.",
    "anchor_depth_note": "No physical anchor depth; modelled deep solid-Earth response.",
    "reference_frame": "model_reference_frame",
    "vlm_measurement_type": "modelled",
    "benchmark_dependence": "low",
    "observes_land_surface_directly": "false",
    "observes_infrastructure_motion": "false",
    "observes_geocentric_land_motion": "uncertain",
    "sensitive_to_shallow_vlm": "false",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "false",
    **process_flags(
        influenced_by_vertical_accretion_erosion_sedimentation="false",
        influenced_by_mineral_sedimentation="false",
        influenced_by_biogenic_organic_accretion="false",
        influenced_by_coastal_erosion_shoreline_retreat="false",
        influenced_by_storm_event_deposition_overwash="false",
        influenced_by_autocompaction_sediment_self_weight="false",
        influenced_by_shallow_compaction_drainage="false",
        influenced_by_peat_oxidation_organic_soil_loss="false",
        influenced_by_wetting_drying_shrink_swell="false",
        influenced_by_permafrost_thaw_frost_heave_thermokarst="false",
        influenced_by_local_surface_loading="uncertain",
        influenced_by_anthropogenic_fill_construction_loading="false",
        influenced_by_reservoir_impoundment_dam_loading="uncertain",
        influenced_by_hydrologic_loading="uncertain",
        influenced_by_ocean_atmosphere_loading="false",
        influenced_by_groundwater_withdrawal_recharge="false",
        influenced_by_hydrocarbon_extraction_injection="false",
        influenced_by_geothermal_extraction_injection="false",
        influenced_by_subsurface_fluid_extraction_general="false",
        influenced_by_mining_tunneling_cavity_collapse="false",
        influenced_by_karst_dissolution_sinkholes="false",
        influenced_by_drainage_reclamation_polder_management="false",
        influenced_by_river_delta_channel_migration_overbank_processes="false",
        influenced_by_landslide_mass_wasting="false",
        influenced_by_tectonic_deformation_general="false",
        influenced_by_earthquake_cycle_coseismic_postseismic="false",
        influenced_by_fault_creep="false",
        influenced_by_volcanic_activity="false",
        influenced_by_salt_tectonics_halokinesis="false",
        influenced_by_mantle_dynamics="true",
        influenced_by_grd_contemporary_mass_redistribution="uncertain",
        influenced_by_gia="true",
        influenced_by_sediment_loading_unloading_isostatic_response="true",
        influenced_by_residual_vlm="false",
    ),
    "process_influence_note": "This is a modelled deep solid-Earth VLM product rather than a direct observation. It represents GIA-related deformation and is generally not sensitive to shallow sediment, infrastructure, or vertical accretion processes.",
    "classification_confidence": "high",
}


INSAR_CLASSIFICATION = {
    "technique": "InSAR",
    "vlm_note": "Direct remote-sensing observation of surface deformation, usually in line-of-sight or relative form and spatially dense.",
    "anchor_depth_note": "Approximately anchor depth of dominant reflector / phase center; remote-sensing signal not tied to a deep physical monument.",
    "reference_frame": "line_of_sight_or_relative_remote_sensing",
    "vlm_measurement_type": "direct",
    "benchmark_dependence": "low",
    "observes_land_surface_directly": "true",
    "observes_infrastructure_motion": "uncertain",
    "observes_geocentric_land_motion": "false",
    "sensitive_to_shallow_vlm": "true",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "uncertain",
    **process_flags(
        influenced_by_vertical_accretion_erosion_sedimentation="uncertain",
        influenced_by_mineral_sedimentation="uncertain",
        influenced_by_biogenic_organic_accretion="uncertain",
        influenced_by_coastal_erosion_shoreline_retreat="uncertain",
        influenced_by_storm_event_deposition_overwash="uncertain",
        influenced_by_autocompaction_sediment_self_weight="true",
        influenced_by_shallow_compaction_drainage="true",
        influenced_by_peat_oxidation_organic_soil_loss="true",
        influenced_by_wetting_drying_shrink_swell="true",
        influenced_by_permafrost_thaw_frost_heave_thermokarst="true",
        influenced_by_local_surface_loading="true",
        influenced_by_anthropogenic_fill_construction_loading="true",
        influenced_by_reservoir_impoundment_dam_loading="true",
        influenced_by_hydrologic_loading="true",
        influenced_by_ocean_atmosphere_loading="uncertain",
        influenced_by_groundwater_withdrawal_recharge="true",
        influenced_by_hydrocarbon_extraction_injection="true",
        influenced_by_geothermal_extraction_injection="true",
        influenced_by_subsurface_fluid_extraction_general="true",
        influenced_by_mining_tunneling_cavity_collapse="true",
        influenced_by_karst_dissolution_sinkholes="true",
        influenced_by_drainage_reclamation_polder_management="true",
        influenced_by_river_delta_channel_migration_overbank_processes="uncertain",
        influenced_by_landslide_mass_wasting="true",
        influenced_by_tectonic_deformation_general="true",
        influenced_by_earthquake_cycle_coseismic_postseismic="true",
        influenced_by_fault_creep="true",
        influenced_by_volcanic_activity="true",
        influenced_by_salt_tectonics_halokinesis="true",
        influenced_by_mantle_dynamics="true",
        influenced_by_grd_contemporary_mass_redistribution="uncertain",
        influenced_by_gia="true",
        influenced_by_sediment_loading_unloading_isostatic_response="uncertain",
        influenced_by_residual_vlm="uncertain",
    ),
    "process_influence_note": "This is a direct remote-sensing observation of surface deformation in a relative/line-of-sight frame. It is especially sensitive to shallow deltaic and anthropogenic subsidence processes, while deeper tectonic or GIA signals may also be present but are not isolated by the product.",
    "classification_confidence": "high",
}


INSAR_GNSS_HYBRID_CLASSIFICATION = {
    "technique": "InSAR + GNSS",
    "vlm_note": "Hybrid InSAR and GNSS velocity-field product estimating vertical land motion over New Zealand.",
    "anchor_depth_note": "Mixed remote-sensing phase-center and GNSS monument control; no single physical anchor depth.",
    "reference_frame": "mixed_or_unclear",
    "vlm_measurement_type": "hybrid",
    "benchmark_dependence": "moderate",
    "observes_land_surface_directly": "true",
    "observes_infrastructure_motion": "uncertain",
    "observes_geocentric_land_motion": "uncertain",
    "sensitive_to_shallow_vlm": "true",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "uncertain",
    **process_flags(
        influenced_by_vertical_accretion_erosion_sedimentation="uncertain",
        influenced_by_mineral_sedimentation="uncertain",
        influenced_by_biogenic_organic_accretion="uncertain",
        influenced_by_coastal_erosion_shoreline_retreat="uncertain",
        influenced_by_storm_event_deposition_overwash="uncertain",
        influenced_by_autocompaction_sediment_self_weight="true",
        influenced_by_shallow_compaction_drainage="true",
        influenced_by_peat_oxidation_organic_soil_loss="true",
        influenced_by_wetting_drying_shrink_swell="true",
        influenced_by_permafrost_thaw_frost_heave_thermokarst="true",
        influenced_by_local_surface_loading="true",
        influenced_by_anthropogenic_fill_construction_loading="true",
        influenced_by_reservoir_impoundment_dam_loading="true",
        influenced_by_hydrologic_loading="true",
        influenced_by_ocean_atmosphere_loading="uncertain",
        influenced_by_groundwater_withdrawal_recharge="true",
        influenced_by_hydrocarbon_extraction_injection="true",
        influenced_by_geothermal_extraction_injection="true",
        influenced_by_subsurface_fluid_extraction_general="true",
        influenced_by_mining_tunneling_cavity_collapse="true",
        influenced_by_karst_dissolution_sinkholes="true",
        influenced_by_drainage_reclamation_polder_management="true",
        influenced_by_river_delta_channel_migration_overbank_processes="uncertain",
        influenced_by_landslide_mass_wasting="true",
        influenced_by_tectonic_deformation_general="true",
        influenced_by_earthquake_cycle_coseismic_postseismic="true",
        influenced_by_fault_creep="true",
        influenced_by_volcanic_activity="true",
        influenced_by_salt_tectonics_halokinesis="true",
        influenced_by_mantle_dynamics="true",
        influenced_by_grd_contemporary_mass_redistribution="uncertain",
        influenced_by_gia="true",
        influenced_by_sediment_loading_unloading_isostatic_response="uncertain",
        influenced_by_residual_vlm="uncertain",
    ),
    "process_influence_note": "This is a hybrid InSAR and GNSS velocity product, with the rendered layer showing coastal vertical rates. It can include shallow surface deformation and deeper tectonic or volcanic signals, but the mixed reference and sensing depths mean process attribution should be treated conservatively.",
    "classification_confidence": "medium",
}

HYBRID_ESTIMATES_CLASSIFICATION = {
    "technique": "Hybrid estimates",
    "vlm_note": "Hybrid coastal VLM estimate combining reconstructed VLM, GNSS, InSAR, and GIA information.",
    "anchor_depth_note": "Mixed input methods; no single physical anchor depth.",
    "reference_frame": "mixed_or_unclear",
    "vlm_measurement_type": "hybrid",
    "benchmark_dependence": "moderate",
    "observes_land_surface_directly": "uncertain",
    "observes_infrastructure_motion": "uncertain",
    "observes_geocentric_land_motion": "uncertain",
    "sensitive_to_shallow_vlm": "true",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "uncertain",
    "influenced_by_vertical_accretion_erosion_sedimentation": "uncertain",
    "influenced_by_mineral_sedimentation": "uncertain",
    "influenced_by_biogenic_organic_accretion": "uncertain",
    "influenced_by_coastal_erosion_shoreline_retreat": "uncertain",
    "influenced_by_storm_event_deposition_overwash": "uncertain",
    "influenced_by_autocompaction_sediment_self_weight": "true",
    "influenced_by_shallow_compaction_drainage": "true",
    "influenced_by_peat_oxidation_organic_soil_loss": "true",
    "influenced_by_wetting_drying_shrink_swell": "true",
    "influenced_by_permafrost_thaw_frost_heave_thermokarst": "true",
    "influenced_by_local_surface_loading": "true",
    "influenced_by_anthropogenic_fill_construction_loading": "true",
    "influenced_by_reservoir_impoundment_dam_loading": "true",
    "influenced_by_hydrologic_loading": "true",
    "influenced_by_ocean_atmosphere_loading": "uncertain",
    "influenced_by_groundwater_withdrawal_recharge": "true",
    "influenced_by_hydrocarbon_extraction_injection": "true",
    "influenced_by_geothermal_extraction_injection": "true",
    "influenced_by_subsurface_fluid_extraction_general": "true",
    "influenced_by_mining_tunneling_cavity_collapse": "true",
    "influenced_by_karst_dissolution_sinkholes": "true",
    "influenced_by_drainage_reclamation_polder_management": "true",
    "influenced_by_river_delta_channel_migration_overbank_processes": "uncertain",
    "influenced_by_landslide_mass_wasting": "true",
    "influenced_by_tectonic_deformation_general": "true",
    "influenced_by_earthquake_cycle_coseismic_postseismic": "true",
    "influenced_by_fault_creep": "true",
    "influenced_by_volcanic_activity": "true",
    "influenced_by_salt_tectonics_halokinesis": "true",
    "influenced_by_mantle_dynamics": "true",
    "influenced_by_grd_contemporary_mass_redistribution": "uncertain",
    "influenced_by_gia": "true",
    "influenced_by_sediment_loading_unloading_isostatic_response": "uncertain",
    "influenced_by_residual_vlm": "uncertain",
    "process_influence_note": "This is a hybrid coastal VLM estimate rather than a single-technique observation. It can include shallow subsidence, deeper geodynamic or tectonic motion, loading, and GIA influences, but process attribution depends on the contributing input data at each location.",
    "classification_confidence": "medium",
}


TIDE_GAUGE_CLASSIFICATION = {
    "technique": "Tide gauge",
    "vlm_note": "Indirect VLM estimate inferred from differences between coastal sea-level reconstructions and tide-gauge records.",
    "anchor_depth_note": "Indirect benchmark/gauge infrastructure control; no single sensing depth.",
    "reference_frame": "sea_level_relative",
    "vlm_measurement_type": "indirect",
    "benchmark_dependence": "high",
    "observes_land_surface_directly": "false",
    "observes_infrastructure_motion": "true",
    "observes_geocentric_land_motion": "false",
    "sensitive_to_shallow_vlm": "uncertain",
    "sensitive_to_deep_vlm": "true",
    "sensitive_to_vertical_accretion": "false",
    **process_flags(
        influenced_by_vertical_accretion_erosion_sedimentation="false",
        influenced_by_mineral_sedimentation="false",
        influenced_by_biogenic_organic_accretion="false",
        influenced_by_coastal_erosion_shoreline_retreat="false",
        influenced_by_storm_event_deposition_overwash="false",
        influenced_by_autocompaction_sediment_self_weight="uncertain",
        influenced_by_shallow_compaction_drainage="uncertain",
        influenced_by_peat_oxidation_organic_soil_loss="uncertain",
        influenced_by_wetting_drying_shrink_swell="uncertain",
        influenced_by_permafrost_thaw_frost_heave_thermokarst="uncertain",
        influenced_by_local_surface_loading="true",
        influenced_by_anthropogenic_fill_construction_loading="true",
        influenced_by_reservoir_impoundment_dam_loading="true",
        influenced_by_hydrologic_loading="true",
        influenced_by_ocean_atmosphere_loading="true",
        influenced_by_groundwater_withdrawal_recharge="true",
        influenced_by_hydrocarbon_extraction_injection="true",
        influenced_by_geothermal_extraction_injection="true",
        influenced_by_subsurface_fluid_extraction_general="true",
        influenced_by_mining_tunneling_cavity_collapse="uncertain",
        influenced_by_karst_dissolution_sinkholes="uncertain",
        influenced_by_drainage_reclamation_polder_management="uncertain",
        influenced_by_river_delta_channel_migration_overbank_processes="false",
        influenced_by_landslide_mass_wasting="uncertain",
        influenced_by_tectonic_deformation_general="true",
        influenced_by_earthquake_cycle_coseismic_postseismic="true",
        influenced_by_fault_creep="true",
        influenced_by_volcanic_activity="true",
        influenced_by_salt_tectonics_halokinesis="true",
        influenced_by_mantle_dynamics="true",
        influenced_by_grd_contemporary_mass_redistribution="true",
        influenced_by_gia="true",
        influenced_by_sediment_loading_unloading_isostatic_response="true",
        influenced_by_residual_vlm="uncertain",
    ),
    "process_influence_note": "This is an indirect tide-gauge-based VLM estimate rather than a direct geocentric point measurement. It may include shallow, deep, loading, tectonic, and infrastructure-dependent influences, and the result depends on the gauge/benchmark system and sea-level reconstruction.",
    "classification_confidence": "medium",
}


def build_dataset_attributes(
    records: list[dict],
    ngl_imaged_metadata: dict,
    gia_metadata: dict,
    insar_grids: list[dict],
    gns_records: list[dict],
    gns_metadata: dict,
    tide_gauge_records: list[dict],
    tide_gauge_metadata: dict,
    oelsmann_hybrid_records: list[dict],
    oelsmann_hybrid_metadata: dict,
) -> dict[str, dict]:
    return {
        "gnss_blewitt_2018": {
            "id": "gnss_blewitt_2018",
            "label": "GPS",
            "technique": "GNSS",
            "display_name": "GNSS MIDAS - Blewitt et al. 2018",
            "authors": "Blewitt, G., Hammond, W. C., and Kreemer, C.",
            "citation": "Blewitt, G., W. C. Hammond, and C. Kreemer (2018), Harnessing the GPS data explosion for interdisciplinary science, Eos, 99.",
            "doi": "10.1029/2018EO104623",
            "original_file_url": MIDAS_URL,
            "publication_year": 2018,
            "abstract": "The associated Eos article describes how expanding geodetic GPS station networks, faster data delivery, and automated processing enable broad Earth-science applications. The NGL MIDAS velocity file used here is one of the station velocity products exposed by that operational geodetic processing system.",
            "time_period_covered": {
                "min": round(min(r["first_epoch_year"] for r in records), 4),
                "max": round(max(r["last_epoch_year"] for r in records), 4),
            },
            "short_description": [
                "NGL MIDAS provides station-based GNSS velocity estimates derived from position time series.",
                "This visualization uses station latitude, longitude, duration, and vertical UP velocity converted to mm/yr.",
                "Station uncertainty is available from the MIDAS file and is shown in station hover cards on the globe.",
            ],
            "coverage": bounds_from_records(records),
            "uncertainty_provided": True,
            **GNSS_CLASSIFICATION,
        },
        "gnss_imaged_hammond_2021": {
            "id": "gnss_imaged_hammond_2021",
            "label": "NGL imaged VLM",
            "technique": "GNSS",
            "display_name": "NGL GPS Imaging interpolated VLM - Hammond et al. 2021",
            "authors": "Hammond, W. C., Blewitt, G., Kreemer, C., and Nerem, R. S.",
            "citation": "Hammond, W. C., G. Blewitt, C. Kreemer, and S. Nerem (2021), Global vertical land motion for studies of sea level rise, Journal of Geophysical Research: Solid Earth, 126(7), e2021JB022355.",
            "doi": "10.1029/2021JB022355",
            "original_file_url": NGL_IMAGED_VLM_URL,
            "metadata_url": "https://geodesy.unr.edu/vlm.php",
            "publication_year": 2021,
            "abstract": "The associated JGR Solid Earth paper uses GPS Imaging to estimate rates and patterns of vertical land motion on Earth's land surface from globally processed GNSS station velocities. The product provides gridded vertical rate estimates, formal uncertainty, and nearest-neighbor spatial variability for sea-level applications.",
            "time_period_covered": None,
            "short_description": [
                "This layer renders the NGL GPS Imaging interpolated vertical land motion product.",
                "The source text file provides longitude, latitude, vertical rate, formal uncertainty, and nearest-neighbor spatial variability in mm/yr.",
                "For browser performance the quarter-degree source samples are aggregated into one-degree render polygons.",
            ],
            "coverage": bounds_from_grid_bounds([ngl_imaged_metadata["bounds"]]),
            "rendered_file": "datasets/gnss_imaged_hammond_2021/VLM_Global_Imaged.txt",
            "source_variable": ngl_imaged_metadata["component"],
            "uncertainty_variable": ngl_imaged_metadata["uncertainty_component"],
            "spatial_variability_variable": ngl_imaged_metadata["zeta_component"],
            "source_record_count": ngl_imaged_metadata["source_record_count"],
            "record_count_rendered": ngl_imaged_metadata["value_count"],
            "source_cell_size_degrees": ngl_imaged_metadata["source_cell_size_degrees"],
            "render_cell_size_degrees": ngl_imaged_metadata["render_cell_size_degrees"],
            "value_range_mm_yr": {
                "min": ngl_imaged_metadata["min_mm_yr"],
                "max": ngl_imaged_metadata["max_mm_yr"],
                "median": ngl_imaged_metadata["median_mm_yr"],
            },
            "uncertainty_provided": True,
            **GNSS_CLASSIFICATION,
            "vlm_note": "Interpolated GNSS Imaging estimate of vertical land motion derived from globally processed GNSS station velocities.",
            "vlm_measurement_type": "interpolated",
            "anchor_depth_note": "Derived from GNSS monument velocities; no single physical anchor depth for interpolated grid cells.",
            "process_influence_note": "This is a gridded GPS Imaging product derived from GNSS station velocities rather than a direct measurement at every grid cell. It can contain shallow and deep VLM signals represented in the contributing GNSS network, with uncertainty and nearest-neighbor spatial variability supplied by the source product.",
            "classification_confidence": "medium",
        },
        "gia_caron_2020": {
            "id": "gia_caron_2020",
            "label": "GIA",
            "technique": "GIA model",
            "display_name": "GIA model - Caron and Ivins 2020",
            "authors": "Caron, L., and Ivins, E. R.",
            "citation": "Caron, L., and Ivins, E. R. (2020), A baseline Antarctic GIA correction for space gravimetry, Earth and Planetary Science Letters, 531, 115957.",
            "doi": "10.1016/j.epsl.2019.115957",
            "original_file_url": GIA_URL,
            "publication_year": 2020,
            "abstract": "The associated paper presents a baseline Antarctic GIA correction intended for space-gravimetry studies of ice-mass balance. It uses GPS and geochronological constraints to support improved estimates of present-day solid-Earth motion and associated mass-change corrections.",
            "time_period_covered": None,
            "short_description": [
                "The GIA grid represents modeled vertical land motion associated with glacial isostatic adjustment.",
                "This visualization renders the Tdur component, the expected vertical land motion from total GIA sources in mm/yr.",
                "The grid is global and should be interpreted as a model correction rather than a direct observation time series.",
            ],
            "coverage": bounds_from_grid_bounds([gia_metadata["bounds"]]),
            "uncertainty_provided": True,
            **GIA_CLASSIFICATION,
        },
        "insar_ohenhen_2025": {
            "id": "insar_ohenhen_2025",
            "label": "InSAR",
            "technique": "InSAR",
            "display_name": "InSAR delta VLM - Ohenhen et al. 2025",
            "authors": "Ohenhen, L. O. et al.",
            "citation": "Ohenhen, L. O., Shirzaei, M., Davis, J. L. et al. (2026), Global subsidence of river deltas, Nature, 649, 894-901. Dataset: Ohenhen, L. (2025), The global threat of sinking deltas, Zenodo.",
            "doi": "10.5281/zenodo.15015923",
            "associated_article_doi": "10.1038/s41586-025-09928-6",
            "associated_article_publication_year": 2026,
            "original_file_url": INSAR_URL,
            "publication_year": 2025,
            "abstract": "The Zenodo dataset description presents spatially variable surface-elevation changes across 40 global deltas derived from InSAR. It supports analysis of contemporary delta subsidence, including the role of groundwater extraction, sediment-supply reduction, urban expansion, and relative sea-level-rise impacts.",
            "associated_article_abstract": "The associated Nature paper estimates vertical land motion across major river deltas using Sentinel-1 InSAR and evaluates how subsidence interacts with relative sea-level rise and delta exposure. It emphasizes that many deltas are sinking faster than global mean sea-level rise and that human activities such as groundwater extraction, sediment trapping, and urban loading contribute to the hazard.",
            "time_period_covered": {"min": 2014, "max": 2023},
            "short_description": [
                "The InSAR dataset contains regional GeoTIFF grids of vertical land motion for 40 global river deltas.",
                "The associated Nature study measured surface-elevation change using Sentinel-1 InSAR over 2014-2023.",
                "The rendered values are converted to mm/yr and share the same diverging color scale as the GPS and GIA layers.",
            ],
            "coverage": bounds_from_grid_bounds([grid["bounds"] for grid in insar_grids]),
            "uncertainty_provided": False,
            **INSAR_CLASSIFICATION,
        },
        "insar_gnss_hamling_2022": {
            "id": "insar_gnss_hamling_2022",
            "label": "NZ VLM",
            "technique": "InSAR + GNSS",
            "display_name": "InSAR + GNSS New Zealand VLM - Hamling et al. 2022",
            "authors": "Hamling, I. J., Wright, T. J., Hreinsdottir, S., and Wallace, L. M.",
            "citation": "Hamling, I. J., Wright, T. J., Hreinsdottir, S., and Wallace, L. M. (2022), A snapshot of New Zealand's dynamic deformation field from Envisat InSAR and GNSS observations between 2003 and 2011, Geophysical Research Letters, 49, e2021GL096465.",
            "doi": "10.21420/E1C1-MQ19",
            "associated_article_doi": "10.1029/2021GL096465",
            "original_file_url": GNS_ATTACHMENT_URL,
            "metadata_url": GNS_METADATA_URL,
            "publication_year": 2022,
            "dataset_publication_year": 2021,
            "license": "Creative Commons Attribution-ShareAlike 4.0 International License",
            "abstract": "The dataset contains New Zealand velocities derived from Envisat InSAR and GNSS observations for 2003-2011. It includes full InSAR velocities, stand-alone vertical GPS velocities over the same period, and coastal vertical land movement estimates used in the manuscript.",
            "associated_article_abstract": "The associated Geophysical Research Letters article combines approximately a decade of Envisat InSAR observations with interseismic campaign and continuous GNSS velocities to build a high-resolution New Zealand deformation field. The study estimates vertical deformation and coastal vertical land motion for sea-level and tectonic applications.",
            "time_period_covered": {"min": 2003, "max": 2011},
            "short_description": [
                "The rendered layer uses the 1 km coastal vertical land movement file from the GNS archive.",
                "Values are provided in mm/yr with 1-sigma uncertainty, observation count, quality factor, and average radius from the coastal location.",
                "The archive also contains the full InSAR velocity point cloud, vertical GPS velocities, and outlier/quality support files.",
            ],
            "coverage": bounds_from_records(gns_records),
            "source_coverage_lon_0_360": {
                "min_lon": 162.75146484,
                "max_lon": 191.75537109,
                "min_lat": -54.71323549,
                "max_lat": -30.35586818,
            },
            "rendered_file": gns_metadata["rendered_file"],
            "archive_files": gns_metadata["archive_files"],
            "record_count_rendered": gns_metadata["record_count"],
            "value_range_mm_yr": {
                "min": gns_metadata["min_mm_yr"],
                "max": gns_metadata["max_mm_yr"],
                "median": gns_metadata["median_mm_yr"],
            },
            "uncertainty_provided": True,
            **INSAR_GNSS_HYBRID_CLASSIFICATION,
        },
        "tide_gauge_dangendorf_2026": {
            "id": "tide_gauge_dangendorf_2026",
            "label": "Tide gauges",
            "technique": "Tide gauge",
            "display_name": "Tide-gauge VLM - Dangendorf et al. 2026",
            "authors": "Dangendorf, S. et al.",
            "citation": "Dangendorf, S. et al. (2026), Variable Vertical Land Motion Inferred from Multi-Decadal to Centennial Tide Gauge Records, Zenodo.",
            "doi": "10.5281/zenodo.18777050",
            "original_file_url": TIDE_GAUGE_MAT_URL,
            "metadata_url": TIDE_GAUGE_RECORD_URL,
            "publication_year": 2026,
            "license": "Creative Commons Attribution 4.0 International",
            "abstract": "The Zenodo record provides code and data for nonlinear vertical land motion estimates inferred from long tide-gauge records. This layer uses the adjusted CSL-minus-tide-gauge residual variable in CSL-TG.mat and computes a linear trend for each available tide-gauge time series.",
            "time_period_covered": tide_gauge_metadata["time_period"],
            "short_description": [
                "The dataset estimates VLM indirectly from differences between coastal sea-level reconstructions and tide-gauge records.",
                "This visualization renders linear trends from the MDadj3 adjusted residual time series in mm/yr.",
                "Clicking a square marker opens the annual residual series and its fitted linear trend.",
            ],
            "coverage": bounds_from_records(tide_gauge_records),
            "rendered_file": tide_gauge_metadata["mat_file"],
            "nearby_gnss_helper_file": str(TIDE_GAUGE_NEARBY_GNSS_CACHE),
            "source_variable": tide_gauge_metadata["source_variable"],
            "record_count_rendered": tide_gauge_metadata["record_count"],
            "value_range_mm_yr": {
                "min": tide_gauge_metadata["min_mm_yr"],
                "max": tide_gauge_metadata["max_mm_yr"],
                "median": tide_gauge_metadata["median_mm_yr"],
            },
            "uncertainty_provided": True,
            **TIDE_GAUGE_CLASSIFICATION,
        },
        "hybrid_oelsmann_2026": {
            "id": "hybrid_oelsmann_2026",
            "label": "Hybrid VLM",
            "technique": "Hybrid estimates",
            "display_name": "Hybrid coastal VLM - Oelsmann et al. 2026",
            "authors": "Oelsmann, J., Nicholls, R. J., Lincke, D. et al.",
            "citation": "Oelsmann, J., Nicholls, R. J., Lincke, D. et al. (2026), Subsidence more than doubles sea-level rise today along densely populated coasts, Nature Communications, 17, 4382.",
            "doi": "10.1038/s41467-026-72293-z",
            "original_file_url": OELSMANN_HYBRID_NC_URL,
            "metadata_url": OELSMANN_HYBRID_RECORD_URL,
            "publication_year": 2026,
            "abstract": "The associated Nature Communications paper combines high-resolution vertical land motion observations and hybrid estimates to show that subsidence substantially increases relative sea-level rise along densely populated coasts. The study uses reconstructed VLM, GNSS, InSAR, and GIA information to estimate contemporary coastal vertical motion and its implications for exposure.",
            "time_period_covered": {"min": 1995, "max": 2020},
            "short_description": [
                "The rendered layer is a coastal profile of hybrid vertical land motion estimates.",
                "This visualization uses the OE24_GPS_InSAR_GIA variable and its OE24_GPS_InSAR_GIA_un uncertainty in mm/yr.",
                "The product combines multiple observation and model sources and should be interpreted as a hybrid estimate rather than a single-sensor measurement.",
            ],
            "coverage": bounds_from_records(oelsmann_hybrid_records),
            "rendered_file": oelsmann_hybrid_metadata["rendered_file"],
            "source_variable": oelsmann_hybrid_metadata["component"],
            "uncertainty_variable": oelsmann_hybrid_metadata["uncertainty_component"],
            "record_count_rendered": oelsmann_hybrid_metadata["record_count"],
            "value_range_mm_yr": {
                "min": oelsmann_hybrid_metadata["min_mm_yr"],
                "max": oelsmann_hybrid_metadata["max_mm_yr"],
                "median": oelsmann_hybrid_metadata["median_mm_yr"],
            },
            "uncertainty_provided": True,
            **HYBRID_ESTIMATES_CLASSIFICATION,
        },
    }


def write_dataset_attribute_files(attributes: dict[str, dict]) -> None:
    ATTRIBUTE_DIR.mkdir(exist_ok=True)
    for stale_file in ATTRIBUTE_DIR.glob("*.json"):
        stale_file.unlink()
    for dataset_id, payload in attributes.items():
        path = ATTRIBUTE_DIR / f"{dataset_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_external_dataset_attributes(population_metadata: dict) -> dict[str, dict]:
    return {
        "ghsl_schiavina_2025": {
            "id": "ghsl_schiavina_2025",
            "label": "Population",
            "theme": "population",
            "main_author": "Schiavina",
            "year": 2025,
            "doi": "10.2905/adba95af-db56-4569-acd3-9513201eba30",
            "website_link": "https://human-settlement.emergency.copernicus.eu/ghs_wup_pop_r2025a.php",
            "original_file_url": GHSL_POP_URL,
            "sampled_layer": {
                "width": population_metadata["width"],
                "height": population_metadata["height"],
                "units": population_metadata["units"],
                "source_projection": population_metadata["source_projection"],
                "valid_pixel_count": population_metadata["valid_pixel_count"],
                "max_population": population_metadata["max_population"],
                "p99_population": population_metadata["p99_population"],
            },
        }
    }


def write_external_dataset_attribute_files(attributes: dict[str, dict]) -> None:
    EXTERNAL_ATTRIBUTE_DIR.mkdir(exist_ok=True)
    for stale_file in EXTERNAL_ATTRIBUTE_DIR.glob("*.json"):
        stale_file.unlink()
    for dataset_id, payload in attributes.items():
        path = EXTERNAL_ATTRIBUTE_DIR / f"{dataset_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_records(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    positive = []
    negative = []

    for record in records:
        if record["up_mm_yr"] >= 0:
            positive.append(record)
        else:
            negative.append(record)

    return positive, negative


def html_escape_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def strip_uncertainty_fields(records: list[dict]) -> list[dict]:
    return [{key: value for key, value in record.items() if key != "up_sigma_mm_yr"} for record in records]


def write_json_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_escape_json(payload), encoding="utf-8")


def point_uncertainty_payload(records: list[dict], id_field: str) -> dict:
    values = [
        [record.get(id_field), record.get("up_sigma_mm_yr")]
        for record in records
        if record.get(id_field) is not None
        and record.get("up_sigma_mm_yr") is not None
        and math.isfinite(float(record.get("up_sigma_mm_yr")))
    ]
    sigmas = [float(value[1]) for value in values]
    return {
        "type": "point_uncertainty",
        "id_field": id_field,
        "units": "mm/yr",
        "values": values,
        "min": round(min(sigmas), 4) if sigmas else None,
        "max": round(percentile(sigmas, 0.99), 4) if sigmas else None,
        "raw_max": round(max(sigmas), 4) if sigmas else None,
    }


def write_uncertainty_payloads(
    records: list[dict],
    ngl_imaged_uncertainties: list[float | None],
    ngl_imaged_metadata: dict,
    gia_uncertainties: list[float | None],
    gia_metadata: dict,
    gns_records: list[dict],
    tide_gauge_records: list[dict],
    oelsmann_hybrid_records: list[dict],
) -> dict[str, str]:
    payloads = {
        "gnss_blewitt_2018": point_uncertainty_payload(records, "station"),
        "gnss_imaged_hammond_2021": {
            "type": "grid_uncertainty",
            "units": "mm/yr",
            "component": ngl_imaged_metadata.get("uncertainty_component", "Vu uncertainty"),
            "width": ngl_imaged_metadata["width"],
            "height": ngl_imaged_metadata["height"],
            "bounds": ngl_imaged_metadata["bounds"],
            "values": ngl_imaged_uncertainties,
            "min": ngl_imaged_metadata.get("min_sigma_mm_yr"),
            "max": ngl_imaged_metadata.get("max_sigma_mm_yr"),
        },
        "gia_caron_2020": {
            "type": "grid_uncertainty",
            "units": "mm/yr",
            "component": gia_metadata.get("uncertainty_component", "Tsdur"),
            "width": gia_metadata["width"],
            "height": gia_metadata["height"],
            "bounds": gia_metadata["bounds"],
            "values": gia_uncertainties,
            "min": gia_metadata.get("min_sigma_mm_yr"),
            "max": gia_metadata.get("max_sigma_mm_yr"),
        },
        "insar_gnss_hamling_2022": point_uncertainty_payload(gns_records, "id"),
        "hybrid_oelsmann_2026": point_uncertainty_payload(oelsmann_hybrid_records, "id"),
        "tide_gauge_dangendorf_2026": point_uncertainty_payload(tide_gauge_records, "id"),
    }
    for dataset_id, payload in payloads.items():
        write_json_payload(UNCERTAINTY_PAYLOADS[dataset_id], payload)
    return {dataset_id: path.as_posix() for dataset_id, path in UNCERTAINTY_PAYLOADS.items()}


def write_render_payloads(
    records: list[dict],
    ngl_imaged_values: list[float | None],
    gia_values: list[float | None],
    insar_grids: list[dict],
    gns_records: list[dict],
    tide_gauge_records: list[dict],
    oelsmann_hybrid_records: list[dict],
) -> dict[str, str]:
    positive_records, negative_records = split_records(records)
    payloads = {
        "gnss_blewitt_2018": {
            "type": "station_split",
            "positive": strip_uncertainty_fields(positive_records),
            "negative": strip_uncertainty_fields(negative_records),
        },
        "gnss_imaged_hammond_2021": {
            "type": "grid_trend",
            "values": ngl_imaged_values,
        },
        "gia_caron_2020": {
            "type": "grid_trend",
            "values": gia_values,
        },
        "insar_ohenhen_2025": {
            "type": "insar_grids",
            "grids": insar_grids,
        },
        "insar_gnss_hamling_2022": {
            "type": "point_trend",
            "records": strip_uncertainty_fields(gns_records),
        },
        "hybrid_oelsmann_2026": {
            "type": "point_trend",
            "records": strip_uncertainty_fields(oelsmann_hybrid_records),
        },
        "tide_gauge_dangendorf_2026": {
            "type": "point_trend",
            "records": strip_uncertainty_fields(tide_gauge_records),
        },
    }
    for dataset_id, payload in payloads.items():
        write_json_payload(RENDER_PAYLOADS[dataset_id], payload)
    return {dataset_id: path.as_posix() for dataset_id, path in RENDER_PAYLOADS.items()}


def build_html(
    records: list[dict],
    metadata: dict,
    ngl_imaged_values: list[float | None],
    ngl_imaged_metadata: dict,
    gia_values: list[float | None],
    gia_metadata: dict,
    insar_grids: list[dict],
    insar_metadata: dict,
    gns_records: list[dict],
    gns_metadata: dict,
    tide_gauge_records: list[dict],
    tide_gauge_metadata: dict,
    oelsmann_hybrid_records: list[dict],
    oelsmann_hybrid_metadata: dict,
    dataset_attributes: dict[str, dict],
    population_metadata: dict,
    external_dataset_attributes: dict[str, dict],
    render_urls: dict[str, str],
    uncertainty_urls: dict[str, str],
) -> str:
    metadata_json = html_escape_json(metadata)
    ngl_imaged_metadata_json = html_escape_json(
        {key: value for key, value in ngl_imaged_metadata.items() if key != "zeta_values"}
    )
    gia_metadata_json = html_escape_json(gia_metadata)
    insar_metadata_json = html_escape_json(insar_metadata)
    gns_metadata_json = html_escape_json(gns_metadata)
    tide_gauge_metadata_json = html_escape_json(tide_gauge_metadata)
    oelsmann_hybrid_metadata_json = html_escape_json(oelsmann_hybrid_metadata)
    dataset_attributes_json = html_escape_json(dataset_attributes)
    population_metadata_json = html_escape_json(population_metadata)
    external_dataset_attributes_json = html_escape_json(external_dataset_attributes)
    render_urls_json = html_escape_json(render_urls)
    uncertainty_urls_json = html_escape_json(uncertainty_urls)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Global Vertical Land Motion</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg">

  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css" />
  <script src="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root {{
      --nav-height: 58px;
      --panel-border: #d8dee8;
      --panel-muted: #657487;
    }}

    html,
    body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #34383d;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .app-navbar {{
      height: var(--nav-height);
      background: rgba(255, 255, 255, 0.97);
      border-bottom: 1px solid #d7dde5;
      box-shadow: 0 2px 12px rgba(21, 31, 43, 0.12);
      z-index: 1100;
    }}

    .brand-stack {{
      min-width: 0;
      line-height: 1;
    }}

    .brand-title {{
      color: #1d2733;
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0;
      white-space: nowrap;
    }}

    .brand-title a {{
      color: inherit;
      text-decoration: none;
    }}

    .brand-title a:hover {{
      color: #0f5ca8;
      text-decoration: underline;
    }}

    .brand-short {{
      display: none;
    }}

    .navbar-stat {{
      display: none;
      color: #46576a;
      font-size: 12px;
      white-space: nowrap;
    }}

    .nav-link {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #344559;
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
      line-height: 1;
    }}

    .nav-link:hover,
    .nav-link:focus {{
      color: #0f5ca8;
      text-decoration: none;
    }}

    .showcase-disclaimer {{
      position: fixed;
      top: calc(var(--nav-height) + 10px);
      left: 50%;
      transform: translateX(-50%);
      z-index: 1180;
      width: min(940px, calc(100vw - 32px));
      padding: 8px 12px;
      border: 1px solid #d8c88e;
      border-radius: 8px;
      background: rgba(255, 248, 218, 0.96);
      color: #5a4a16;
      box-shadow: 0 8px 22px rgba(20, 32, 46, 0.14);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.35;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      pointer-events: auto;
    }}

    .showcase-disclaimer-close {{
      width: 22px;
      height: 22px;
      border: 0;
      border-radius: 999px;
      color: #5a4a16;
      background: rgba(90, 74, 22, 0.1);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      padding: 0;
    }}

    .showcase-disclaimer-close:hover,
    .showcase-disclaimer-close:focus {{
      background: rgba(90, 74, 22, 0.18);
    }}

    @media (min-width: 860px) {{
      .navbar-stat {{
        display: inline-flex;
      }}
    }}

    #deck-container {{
      position: absolute;
      inset: var(--nav-height) 0 0 0;
      width: 100vw;
      height: calc(100vh - var(--nav-height));
    }}

    .offcanvas-gis {{
      top: var(--nav-height);
      bottom: 0;
      height: auto;
      width: min(430px, calc(100vw - 18px));
      border-right: 1px solid var(--panel-border);
      box-shadow: 10px 0 28px rgba(20, 32, 46, 0.18);
    }}

    .offcanvas-gis .offcanvas-header {{
      border-bottom: 1px solid var(--panel-border);
      background: #f8fafc;
    }}

    .panel-section {{
      border-bottom: 1px solid #e5e9ef;
      padding: 15px 16px;
    }}

    .section-title {{
      color: #263241;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}

    #techniqueLayerGroups .accordion-button {{
      color: #263241;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
    }}

    #techniqueLayerGroups .accordion-button::after {{
      width: 0.8rem;
      height: 0.8rem;
      background-size: 0.8rem;
    }}

    #techniqueLayerGroups .accordion-button:not(.collapsed) {{
      color: #1e2937;
    }}

    .dataset-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid #dfe5ec;
      border-radius: 8px;
      background: #fff;
      padding: 10px 11px;
    }}

    .dataset-row.unavailable {{
      opacity: 0.58;
    }}

    .dataset-row.unavailable .dataset-title-link {{
      color: #8b98a8;
    }}

    .dataset-row.loading .dataset-icon i {{
      display: none;
    }}

    .dataset-row.loading .dataset-icon::after {{
      content: "";
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.45);
      border-top-color: #fff;
      border-radius: 999px;
      animation: dataset-spin 0.75s linear infinite;
    }}

    @keyframes dataset-spin {{
      to {{ transform: rotate(360deg); }}
    }}

    .dataset-icon {{
      width: 32px;
      height: 32px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      background: #2f669d;
      flex: 0 0 auto;
    }}

    .dataset-name {{
      color: #1e2937;
      font-size: 14px;
      font-weight: 700;
    }}

    .dataset-title-link {{
      appearance: none;
      border: 0;
      background: transparent;
      color: inherit;
      display: inline;
      font: inherit;
      font-weight: inherit;
      padding: 0;
      text-align: left;
    }}

    .dataset-title-link:hover,
    .dataset-title-link:focus {{
      color: #24476f;
      text-decoration: underline;
    }}

    .dataset-detail {{
      color: var(--panel-muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .dataset-actions {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex: 0 0 auto;
    }}

    .dataset-download-btn {{
      width: 22px;
      height: 22px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 0;
      background: transparent;
      color: #6d7d90;
      padding: 0;
      font-size: 13px;
    }}

    .dataset-download-btn:hover,
    .dataset-download-btn:focus {{
      color: #24476f;
      background: #edf2f7;
    }}

    .selector-status {{
      color: #596b80;
      font-size: 12px;
      font-weight: 650;
      line-height: 1.35;
      min-height: 32px;
    }}

    .selector-tool-popover {{
      width: 38px;
      padding: 8px 0 9px;
      border-top: 1px solid #d6dee8;
      background: rgba(255, 255, 255, 0.98);
      display: grid;
      justify-items: center;
      gap: 9px;
      overflow: hidden;
    }}

    .selector-tool-popover[hidden] {{
      display: none;
    }}

    .selector-tool-hint {{
      position: absolute;
      top: 3px;
      right: 46px;
      height: 30px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 8px 5px 10px;
      border: 1px solid #d6dee8;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.96);
      color: #263241;
      box-shadow: 0 4px 14px rgba(20, 32, 46, 0.14);
      font-size: 12px;
      font-weight: 800;
      line-height: 1;
      white-space: nowrap;
      pointer-events: none;
    }}

    .selector-tool-hint[hidden] {{
      display: none;
    }}

    .selector-tool-hint i {{
      color: #0d6efd;
      animation: selectorHintNudge 1.25s ease-in-out infinite;
    }}

    @keyframes selectorHintNudge {{
      0%, 100% {{
        transform: translateX(-3px);
      }}
      50% {{
        transform: translateX(3px);
      }}
    }}

    .selector-tool-button.active {{
      color: #0d6efd;
      background: #e8f1ff;
    }}

    .selector-control {{
      position: absolute;
      right: 14px;
      top: 158px;
      z-index: 930;
      display: grid;
      border: 1px solid #d6dee8;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 4px 18px rgba(20, 32, 46, 0.18);
    }}

    .selector-tool-button {{
      width: 38px;
      height: 36px;
      border: 0;
      background: transparent;
      color: #263241;
      display: grid;
      place-items: center;
      font-size: 17px;
      line-height: 0;
      padding: 0;
      border-radius: 8px;
    }}

    .selector-tool-button > i {{
      display: block;
      line-height: 1;
    }}

    .selector-tool-button.active {{
      border-bottom: 1px solid #d6dee8;
      border-radius: 8px 8px 0 0;
    }}

    .selector-tool-button:hover,
    .selector-tool-button:focus-visible {{
      background: #eef3f8;
    }}

    .selector-radius-readout {{
      color: #44546a;
      font-size: 10px;
      font-weight: 800;
      line-height: 1.05;
      text-align: center;
      letter-spacing: 0;
    }}

    .selector-radius-readout span {{
      display: block;
    }}

    .selector-vertical-range {{
      width: 30px;
      height: 118px;
      writing-mode: vertical-lr;
      direction: rtl;
      accent-color: #0d6efd;
      margin: 0;
    }}

    .selector-histogram-btn {{
      width: 30px;
      height: 30px;
      border: 0;
      border-radius: 7px;
      display: grid;
      place-items: center;
      color: #263241;
      background: transparent;
      line-height: 0;
      padding: 0;
    }}

    .selector-histogram-btn > i {{
      display: block;
      line-height: 1;
    }}

    .selector-histogram-btn:disabled {{
      color: #a7b1bd;
      cursor: not-allowed;
    }}

    .selector-histogram-btn:not(:disabled):hover,
    .selector-histogram-btn:not(:disabled):focus-visible {{
      color: #0d6efd;
      background: #eef3f8;
    }}

    .selection-histogram-body {{
      position: relative;
      overflow: hidden;
    }}

    .selection-histogram-tooltip {{
      position: absolute;
      z-index: 3;
      display: none;
      min-width: 150px;
      max-width: 230px;
      padding: 7px 9px;
      border: 1px solid #cfd8e3;
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.97);
      box-shadow: 0 6px 18px rgba(20, 32, 46, 0.16);
      color: #1e293b;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
    }}

    #selectionHistogramCanvas {{
      display: block;
      width: 100%;
      height: 286px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #fff;
    }}

    .selection-dataset-toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-bottom: 10px;
    }}

    .selection-dataset-toggles .form-check {{
      min-width: 150px;
    }}

    .metadata-modal .modal-dialog {{
      max-width: min(760px, calc(100vw - 28px));
    }}

    .metadata-modal .modal-content {{
      border-radius: 8px;
      border: 1px solid #cfd8e3;
    }}

    .station-modal .modal-dialog {{
      max-width: min(680px, calc(100vw - 24px));
      margin: calc(var(--nav-height) + 12px) auto 12px;
      max-height: calc(100vh - var(--nav-height) - 24px);
    }}

    .station-modal .modal-dialog-scrollable .modal-content {{
      max-height: calc(100vh - var(--nav-height) - 24px);
    }}

    .station-plot-meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}

    .station-plot-stat {{
      border: 1px solid #dfe5ec;
      border-radius: 8px;
      background: #f8fafc;
      padding: 8px 10px;
    }}

    .station-plot-stat-label {{
      color: #64748b;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}

    .station-plot-stat-value {{
      color: #1f2937;
      font-size: 13px;
      font-weight: 750;
      margin-top: 2px;
    }}

    .station-plot-wrap {{
      width: 100%;
      min-height: 300px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #fff;
      overflow: auto;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    #stationTimeSeriesImage {{
      display: block;
      width: 100%;
      max-width: 100%;
      height: auto;
      min-height: 260px;
      object-fit: contain;
    }}

    #tideGaugePlotCanvas {{
      display: block;
      width: 100%;
      height: 360px;
    }}

    .tide-gauge-histogram {{
      margin-top: 10px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}

    .tide-gauge-histogram-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 10px;
      border-bottom: 1px solid #e2e8f0;
      background: #f8fafc;
      color: #334155;
      font-size: 12px;
      font-weight: 800;
    }}

    .tide-gauge-histogram-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #64748b;
      font-size: 11px;
      font-weight: 650;
    }}

    #tideGaugeHistogramCanvas {{
      display: block;
      width: 100%;
      height: 170px;
      cursor: grab;
    }}

    #tideGaugeHistogramCanvas:active {{
      cursor: grabbing;
    }}

    .plot-hover-card {{
      position: fixed;
      z-index: 2500;
      display: none;
      min-width: 190px;
      max-width: 260px;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.97);
      color: #1f2937;
      font-size: 12px;
      line-height: 1.35;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
      pointer-events: none;
    }}

    .nearby-gnss-list {{
      margin-top: 10px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}

    .nearby-gnss-list-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      color: #334155;
      font-size: 12px;
      font-weight: 800;
    }}

    .nearby-gnss-row {{
      display: grid;
      grid-template-columns: 10px minmax(72px, .8fr) minmax(0, 1.4fr) minmax(82px, .8fr);
      gap: 8px;
      align-items: center;
      padding: 7px 10px;
      border-bottom: 1px solid #edf2f7;
      color: #334155;
      font-size: 12px;
    }}

    .nearby-gnss-row:last-child {{
      border-bottom: 0;
    }}

    .nearby-gnss-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, .18);
    }}

    .station-plot-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      align-items: center;
      margin-top: 10px;
      font-size: 12px;
    }}

    .station-citation-note {{
      margin-top: 8px;
      padding: 8px 10px;
      border: 1px solid #e3e8ef;
      border-radius: 8px;
      background: #f8fafc;
      color: #475569;
      font-size: 12px;
      line-height: 1.35;
    }}

    @media (max-width: 760px) {{
      .station-plot-meta {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    .metadata-row {{
      display: grid;
      grid-template-columns: 132px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      padding: 8px 0;
      border-bottom: 1px solid #edf1f5;
      font-size: 13px;
    }}

    .metadata-row:last-child {{
      border-bottom: 0;
    }}

    .metadata-label {{
      color: #68788c;
      font-weight: 700;
    }}

    .metadata-value {{
      color: #243244;
      word-break: break-word;
    }}

    .copy-metadata-btn {{
      width: 30px;
      height: 30px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0;
    }}

    .form-check-input,
    .form-range,
    .btn {{
      cursor: pointer;
    }}

    .form-switch .form-check-input {{
      width: 2.4em;
      height: 1.25em;
    }}

    .segmented {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid #cfd7e2;
      border-radius: 8px;
      overflow: hidden;
    }}

    .segmented input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}

    .segmented label {{
      margin: 0;
      padding: 8px 10px;
      text-align: center;
      color: #3e4d60;
      font-size: 13px;
      font-weight: 700;
      background: #fff;
      border-right: 1px solid #cfd7e2;
      user-select: none;
      cursor: pointer;
    }}

    .segmented label:last-child {{
      border-right: 0;
    }}

    .segmented input:checked + label {{
      color: #fff;
      background: #24476f;
    }}

    .control-label {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: #334258;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
    }}

    .control-value {{
      color: #657487;
      font-weight: 600;
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}

    .stat-tile {{
      border: 1px solid #dfe5ec;
      border-radius: 8px;
      background: #fff;
      padding: 9px 10px;
      min-height: 62px;
    }}

    .stat-value {{
      color: #1e2937;
      font-size: 19px;
      line-height: 1.1;
      font-weight: 800;
    }}

    .stat-label {{
      margin-top: 4px;
      color: #6c7b8d;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .02em;
      text-transform: uppercase;
    }}

    .legend-stack {{
      position: absolute;
      right: 14px;
      bottom: 38px;
      z-index: 900;
      width: min(330px, calc(100vw - 28px));
      display: grid;
      gap: 8px;
      user-select: none;
    }}

    .zoom-control {{
      position: absolute;
      right: 14px;
      top: 74px;
      z-index: 930;
      display: grid;
      border: 1px solid #d6dee8;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 4px 18px rgba(20, 32, 46, 0.18);
    }}

    .zoom-control > button {{
      width: 38px;
      height: 36px;
      border: 0;
      border-bottom: 1px solid #d6dee8;
      background: transparent;
      color: #263241;
      display: grid;
      place-items: center;
      font-size: 17px;
      line-height: 0;
      padding: 0;
    }}

    .zoom-control > button > i {{
      display: block;
      line-height: 1;
    }}

    .zoom-control > button:last-of-type {{
      border-bottom: 0;
    }}

    .zoom-control > button:hover,
    .zoom-control > button:focus-visible {{
      background: #eef3f8;
    }}

    .legend {{
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid #d6dee8;
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 4px 18px rgba(20, 32, 46, 0.18);
      position: relative;
    }}

    .legend[hidden] {{
      display: none;
    }}

    .legend-toggle {{
      display: none;
      border: 0;
      border-radius: 7px;
      background: #eef3f8;
      color: #263241;
      width: 34px;
      height: 32px;
      place-items: center;
      padding: 0;
    }}

    .legend-toggle:hover,
    .legend-toggle:focus-visible {{
      background: #e2eaf3;
    }}

    .legend-title {{
      color: #263241;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 7px;
    }}

    .legend-ramp {{
      height: 12px;
      border-radius: 999px;
      border: 1px solid #c6ced9;
      background: linear-gradient(90deg, rgb(34,104,209), rgb(246,246,242), rgb(207,50,45));
    }}

    .legend-ramp.population-ramp {{
      background: linear-gradient(90deg, rgb(255,252,214), rgb(253,179,75), rgb(214,74,64), rgb(84,39,143));
    }}

    .legend-ramp.uncertainty-ramp {{
      background: linear-gradient(90deg, rgb(255,247,188), rgb(254,196,79), rgb(217,95,14), rgb(127,0,0));
    }}

    .legend-labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 5px;
      color: #5d6d80;
      font-size: 11px;
      font-weight: 700;
    }}

    .legend-scale-control {{
      margin-top: 9px;
      padding-top: 8px;
      border-top: 1px solid #e1e7ef;
    }}

    .legend-scale-control .control-label {{
      margin-bottom: 3px;
    }}

    .legend-scale-control .form-range {{
      margin: 0;
    }}

    .render-variable-control {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      margin-bottom: 8px;
    }}

    .render-variable-control input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}

    .render-variable-control label {{
      border: 1px solid #ccd6e2;
      border-radius: 6px;
      background: #fff;
      color: #344559;
      cursor: pointer;
      font-size: 11px;
      font-weight: 800;
      line-height: 1;
      padding: 6px 8px;
      text-align: center;
    }}

    .render-variable-control input:checked + label {{
      background: #24476f;
      border-color: #24476f;
      color: #fff;
    }}

    @media (max-width: 640px) {{
      html {{
        font-size: 13px;
      }}

      .brand-long {{
        display: inline;
      }}

      .brand-short {{
        display: none;
      }}

      .app-navbar {{
        padding-left: 9px !important;
        padding-right: 9px !important;
      }}

      .nav-link {{
        padding-left: 8px;
        padding-right: 8px;
      }}

      .nav-link span {{
        display: none !important;
      }}

      .navbar-stat {{
        display: none;
      }}

      .legend {{
        width: 100%;
      }}

      .vlm-legend .legend-toggle {{
        display: grid;
        position: absolute;
        top: 8px;
        right: 8px;
      }}

      .vlm-legend .legend-body {{
        padding-right: 38px;
      }}

      .vlm-legend.collapsed {{
        width: auto;
        justify-self: end;
        padding: 6px;
      }}

      .vlm-legend.collapsed .legend-body {{
        display: none;
      }}

      .vlm-legend.collapsed .legend-toggle {{
        position: static;
        width: 38px;
        height: 36px;
      }}

      .legend-stack {{
        right: 10px;
        bottom: 46px;
        width: min(330px, calc(100vw - 20px));
      }}

      .zoom-control {{
        right: 10px;
      }}

      .selector-control {{
        right: 10px;
      }}

      .selector-tool-popover {{
        position: absolute;
        top: 0;
        right: 46px;
        width: min(190px, calc(100vw - 66px));
        min-height: 36px;
        padding: 5px 7px;
        border: 1px solid #d6dee8;
        border-radius: 8px;
        grid-template-columns: 28px minmax(0, 1fr) 30px;
        align-items: center;
        gap: 7px;
        box-shadow: 0 4px 18px rgba(20, 32, 46, 0.18);
      }}

      .selector-tool-button.active {{
        border-bottom: 0;
        border-radius: 8px;
      }}

      .selector-vertical-range {{
        width: 100%;
        height: auto;
        writing-mode: horizontal-tb;
        direction: ltr;
      }}
    }}
  </style>
</head>
<body>
  <nav class="navbar app-navbar fixed-top px-3">
    <div class="container-fluid px-0">
      <button class="btn btn-outline-secondary btn-sm me-2" type="button" data-bs-toggle="offcanvas" data-bs-target="#layerPanel" aria-controls="layerPanel" aria-label="Open layers">
        <i class="bi bi-layers"></i>
      </button>
      <div class="brand-stack me-auto">
        <div class="brand-title"><a href="index.html"><span class="brand-long">Global Vertical Land Motion</span></a></div>
      </div>
      <div class="navbar-stat align-items-center gap-2">
        <i class="bi bi-broadcast-pin"></i>
        <span id="navbar-summary">Loading stations...</span>
      </div>
      <a class="nav-link ms-3" href="catalogue.html" aria-label="Open dataset catalogue">
        <i class="bi bi-table"></i>
        <span class="d-none d-sm-inline">Catalogue</span>
      </a>
      <a class="nav-link ms-3" href="compare.html" aria-label="Open comparison view">
        <i class="bi bi-columns-gap"></i>
        <span class="d-none d-sm-inline">Compare</span>
      </a>
      <a class="nav-link ms-3" href="about.html" aria-label="Open about page">
        <i class="bi bi-info-circle"></i>
        <span class="d-none d-sm-inline">About</span>
      </a>
    </div>
  </nav>

  <div class="showcase-disclaimer" id="showcaseDisclaimer">
    <span>Preliminary showcase: this site is largely AI-generated and intended to motivate real community development, review, and shared stewardship of VLM data. Please cite original data sources, DOIs, and associated papers when using any dataset.</span>
    <button class="showcase-disclaimer-close" type="button" id="showcaseDisclaimerClose" aria-label="Dismiss preliminary showcase notice">
      <i class="bi bi-x-lg"></i>
    </button>
  </div>

  <div id="deck-container"></div>

  <div class="zoom-control" aria-label="Globe zoom controls">
    <button type="button" id="zoom-in" title="Zoom in" aria-label="Zoom in">
      <i class="bi bi-plus-lg"></i>
    </button>
    <button type="button" id="zoom-out" title="Zoom out" aria-label="Zoom out">
      <i class="bi bi-dash-lg"></i>
    </button>
  </div>

  <div class="selector-control" aria-label="Spatial selection controls">
    <div class="selector-tool-hint" id="selector-tool-hint">
      <span>Select data</span>
      <i class="bi bi-arrow-right"></i>
    </div>
    <button type="button" class="selector-tool-button" id="selection-toggle" title="Circle selector" aria-label="Circle selector" aria-pressed="false">
      <i class="bi bi-circle"></i>
    </button>
    <div class="selector-tool-popover" id="selector-tool-popover" hidden>
      <div class="selector-radius-readout" aria-hidden="true">
        <span id="selection-radius-value">200</span>
        <span>km</span>
      </div>
      <input class="selector-vertical-range" type="range" min="1" max="200" step="1" value="200" id="selection-radius-slider" aria-label="Selection radius in kilometers">
      <button class="selector-histogram-btn" type="button" id="selection-open-histogram" title="Selection histogram" aria-label="Selection histogram" disabled>
        <i class="bi bi-bar-chart-fill"></i>
      </button>
      <div class="visually-hidden" id="selection-status" aria-live="polite">Enable selector, then click the globe.</div>
    </div>
  </div>

  <aside class="offcanvas offcanvas-start offcanvas-gis" data-bs-scroll="true" data-bs-backdrop="false" tabindex="-1" id="layerPanel" aria-labelledby="layerPanelLabel">
    <div class="offcanvas-header py-3">
      <h5 class="offcanvas-title fs-6 fw-bold" id="layerPanelLabel">Layers</h5>
      <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
    </div>
    <div class="offcanvas-body p-0">
      <section class="panel-section">
        <div class="section-title">Technique Groups</div>

        <div class="accordion accordion-flush" id="techniqueLayerGroups">
          <div class="accordion-item bg-transparent border-0">
            <h2 class="accordion-header" id="heading-gnss">
              <button class="accordion-button py-2 px-0 bg-transparent shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#group-gnss" aria-expanded="true" aria-controls="group-gnss">
                <i class="bi bi-crosshair me-2"></i>GNSS
              </button>
            </h2>
            <div id="group-gnss" class="accordion-collapse collapse show" aria-labelledby="heading-gnss">
              <div class="accordion-body px-0 py-1">
                <div class="dataset-row" data-layer-row="gnss_blewitt_2018">
                  <div class="dataset-icon"><i class="bi bi-crosshair"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="gnss_blewitt_2018">GPS</button></div>
                    <div class="dataset-detail" title="NGL MIDAS station velocities (1994-2026)">NGL MIDAS station velo...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="gnss_blewitt_2018" aria-label="GPS original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-gps" data-dataset-toggle="gnss_blewitt_2018" checked>
                    </div>
                  </div>
                </div>

                <div class="dataset-row mt-2" data-layer-row="gnss_imaged_hammond_2021">
                  <div class="dataset-icon"><i class="bi bi-grid-3x3-gap"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="gnss_imaged_hammond_2021">NGL GPS Imaging</button></div>
                    <div class="dataset-detail" title="Hammond et al. interpolated GNSS VLM">Hammond interpolated G...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="gnss_imaged_hammond_2021" aria-label="NGL GPS Imaging original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-ngl-imaged" data-dataset-toggle="gnss_imaged_hammond_2021">
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="accordion-item bg-transparent border-0">
            <h2 class="accordion-header" id="heading-gia">
              <button class="accordion-button py-2 px-0 bg-transparent shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#group-gia" aria-expanded="true" aria-controls="group-gia">
                <i class="bi bi-grid-3x3 me-2"></i>GIA Models
              </button>
            </h2>
            <div id="group-gia" class="accordion-collapse collapse show" aria-labelledby="heading-gia">
              <div class="accordion-body px-0 py-1">
                <div class="dataset-row" data-layer-row="gia_caron_2020">
                  <div class="dataset-icon"><i class="bi bi-grid-3x3"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="gia_caron_2020">GIA</button></div>
                    <div class="dataset-detail" title="Caron and Ivins 2019 gridded VLM">Caron and Ivins 2019 g...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="gia_caron_2020" aria-label="GIA original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-gia" data-dataset-toggle="gia_caron_2020" checked>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="accordion-item bg-transparent border-0">
            <h2 class="accordion-header" id="heading-insar">
              <button class="accordion-button py-2 px-0 bg-transparent shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#group-insar" aria-expanded="true" aria-controls="group-insar">
                <i class="bi bi-bounding-box-circles me-2"></i>InSAR
              </button>
            </h2>
            <div id="group-insar" class="accordion-collapse collapse show" aria-labelledby="heading-insar">
              <div class="accordion-body px-0 py-1">
                <div class="dataset-row" data-layer-row="insar_ohenhen_2025">
                  <div class="dataset-icon"><i class="bi bi-bounding-box-circles"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="insar_ohenhen_2025">InSAR Deltas</button></div>
                    <div class="dataset-detail" title="Global delta VLM GeoTIFF grids (2014-2023)">Global delta VLM GeoTI...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="insar_ohenhen_2025" aria-label="InSAR original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-insar" data-dataset-toggle="insar_ohenhen_2025" checked>
                    </div>
                  </div>
                </div>

                <div class="dataset-row mt-2" data-layer-row="insar_gnss_hamling_2022">
                  <div class="dataset-icon"><i class="bi bi-geo-alt"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="insar_gnss_hamling_2022">New Zealand VLM</button></div>
                    <div class="dataset-detail" title="Hamling InSAR + GNSS coastal VLM (2003-2011)">Hamling InSAR + GNSS c...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="insar_gnss_hamling_2022" aria-label="New Zealand VLM original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-gns" data-dataset-toggle="insar_gnss_hamling_2022" checked>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="accordion-item bg-transparent border-0">
            <h2 class="accordion-header" id="heading-hybrid-estimates">
              <button class="accordion-button py-2 px-0 bg-transparent shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#group-hybrid-estimates" aria-expanded="true" aria-controls="group-hybrid-estimates">
                <i class="bi bi-intersect me-2"></i>Hybrid estimates
              </button>
            </h2>
            <div id="group-hybrid-estimates" class="accordion-collapse collapse show" aria-labelledby="heading-hybrid-estimates">
              <div class="accordion-body px-0 py-1">
                <div class="dataset-row" data-layer-row="hybrid_oelsmann_2026">
                  <div class="dataset-icon"><i class="bi bi-intersect"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="hybrid_oelsmann_2026">Global coastal VLM</button></div>
                    <div class="dataset-detail" title="Oelsmann hybrid OE24 + GPS + InSAR + GIA (1995-2020)">Oelsmann hybrid OE24 +...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="hybrid_oelsmann_2026" aria-label="Hybrid coastal VLM original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-oelsmann-hybrid" data-dataset-toggle="hybrid_oelsmann_2026">
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="accordion-item bg-transparent border-0">
            <h2 class="accordion-header" id="heading-tide-gauge">
              <button class="accordion-button py-2 px-0 bg-transparent shadow-none" type="button" data-bs-toggle="collapse" data-bs-target="#group-tide-gauge" aria-expanded="true" aria-controls="group-tide-gauge">
                <i class="bi bi-water me-2"></i>Tide Gauges
              </button>
            </h2>
            <div id="group-tide-gauge" class="accordion-collapse collapse show" aria-labelledby="heading-tide-gauge">
              <div class="accordion-body px-0 py-1">
                <div class="dataset-row" data-layer-row="tide_gauge_dangendorf_2026">
                  <div class="dataset-icon"><i class="bi bi-water"></i></div>
                  <div class="flex-grow-1 min-w-0">
                    <div class="dataset-name"><button class="dataset-title-link" type="button" data-dataset-info="tide_gauge_dangendorf_2026">CSL-TG VLM</button></div>
                    <div class="dataset-detail" title="Dangendorf tide-gauge residual trends (1900-2021)">Dangendorf tide-gauge ...</div>
                  </div>
                  <div class="dataset-actions">
                    <button class="dataset-download-btn" type="button" data-dataset-download="tide_gauge_dangendorf_2026" aria-label="Tide gauge original data download">
                      <i class="bi bi-download"></i>
                    </button>
                    <div class="form-check form-switch m-0">
                      <input class="form-check-input" type="checkbox" role="switch" id="toggle-tide-gauge" data-dataset-toggle="tide_gauge_dangendorf_2026" checked>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel-section">
        <div class="section-title">External Context</div>
        <div class="dataset-row" data-layer-row="ghsl_schiavina_2025">
          <div class="dataset-icon"><i class="bi bi-people"></i></div>
          <div class="flex-grow-1 min-w-0">
            <div class="dataset-name"><button class="dataset-title-link" type="button" data-external-info="ghsl_schiavina_2025">Population</button></div>
            <div class="dataset-detail" title="GHSL GHS-WUP-POP 2025 raster">GHSL GHS-WUP-POP 2025 ...</div>
          </div>
          <div class="dataset-actions">
            <button class="dataset-download-btn" type="button" data-external-download="ghsl_schiavina_2025" aria-label="Population original data download">
              <i class="bi bi-download"></i>
            </button>
            <div class="form-check form-switch m-0">
              <input class="form-check-input" type="checkbox" role="switch" id="toggle-population">
            </div>
          </div>
        </div>
      </section>

      <section class="panel-section">
        <div class="section-title">Render Mode</div>
        <div class="segmented" role="radiogroup" aria-label="Render mode">
          <input type="radio" name="render-mode" id="mode-bars" value="bars">
          <label for="mode-bars"><i class="bi bi-bar-chart-fill me-1"></i>Bars</label>
          <input type="radio" name="render-mode" id="mode-points" value="points" checked>
          <label for="mode-points"><i class="bi bi-record-circle-fill me-1"></i>Points</label>
        </div>
      </section>

      <section class="panel-section">
        <div class="section-title">Filters</div>
        <label class="control-label" for="gia-opacity-slider">
          <span>GIA opacity</span>
          <span class="control-value"><span id="gia-opacity-value">35</span>%</span>
        </label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="35" id="gia-opacity-slider">

        <label class="control-label mt-3" for="insar-opacity-slider">
          <span>InSAR opacity</span>
          <span class="control-value"><span id="insar-opacity-value">85</span>%</span>
        </label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="85" id="insar-opacity-slider">

        <label class="control-label mt-3" for="population-opacity-slider">
          <span>Population opacity</span>
          <span class="control-value"><span id="population-opacity-value">70</span>%</span>
        </label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="70" id="population-opacity-slider">

        <label class="control-label mt-3" for="duration-slider">
          <span>Minimum duration</span>
          <span class="control-value"><span id="duration-value">3.0</span> years</span>
        </label>
        <input type="range" class="form-range" min="0" max="25" step="0.5" value="3" id="duration-slider">

        <label class="control-label mt-3" for="first-epoch-slider">
          <span>Minimum first epoch</span>
          <span class="control-value"><span id="first-epoch-value">0.0</span></span>
        </label>
        <input type="range" class="form-range" min="0" max="1" step="0.1" value="0" id="first-epoch-slider">

        <label class="control-label mt-3" for="last-epoch-slider">
          <span>Maximum last epoch</span>
          <span class="control-value"><span id="last-epoch-value">0.0</span></span>
        </label>
        <input type="range" class="form-range" min="0" max="1" step="0.1" value="1" id="last-epoch-slider">

        <label class="control-label mt-3" for="station-search">
          <span>Station ID</span>
          <span class="control-value" id="search-state">All</span>
        </label>
        <input type="search" class="form-control form-control-sm" id="station-search" placeholder="Search e.g. P123">
      </section>

      <section class="panel-section">
        <div class="section-title">Live Stats</div>
        <div class="stats-grid">
          <div class="stat-tile">
            <div class="stat-value" id="stat-total">0</div>
            <div class="stat-label">Loaded</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-positive">0</div>
            <div class="stat-label">Positive Shown</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-negative">0</div>
            <div class="stat-label">Negative Shown</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-mode">Bars</div>
            <div class="stat-label">Mode</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-color-scale">0.0</div>
            <div class="stat-label">Color +/-</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-duration">0.0</div>
            <div class="stat-label">Years</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-first-epoch">0.0</div>
            <div class="stat-label">Min First</div>
          </div>
          <div class="stat-tile">
            <div class="stat-value" id="stat-last-epoch">0.0</div>
            <div class="stat-label">Max Last</div>
          </div>
        </div>
      </section>
    </div>
  </aside>

  <div class="legend-stack">
    <div class="legend" id="populationLegend" aria-label="Population color scale legend" hidden>
      <div class="legend-title">Population context (people/cell)</div>
      <div class="legend-ramp population-ramp"></div>
      <div class="legend-labels">
        <span id="population-legend-min">1</span>
        <span id="population-legend-mid">p95</span>
        <span id="population-legend-max">p99+</span>
      </div>
    </div>

    <div class="legend vlm-legend" id="vlmLegend" aria-label="VLM color scale legend">
      <button class="legend-toggle" type="button" id="vlmLegendToggle" aria-label="Toggle VLM color controls" aria-expanded="true">
        <i class="bi bi-chevron-down"></i>
      </button>
      <div class="legend-body">
        <div class="render-variable-control" role="radiogroup" aria-label="Render variable">
          <input type="radio" name="render-variable" id="render-variable-trend" value="trend" checked>
          <label for="render-variable-trend">Trend</label>
          <input type="radio" name="render-variable" id="render-variable-uncertainty" value="uncertainty">
          <label for="render-variable-uncertainty">Uncertainty</label>
        </div>
        <div class="legend-title" id="legend-title">Shared VLM color scale (mm/yr)</div>
        <div class="legend-ramp" id="vlm-legend-ramp"></div>
        <div class="legend-labels">
          <span id="legend-min">negative</span>
          <span id="legend-mid">0</span>
          <span id="legend-max">positive</span>
        </div>
        <div class="legend-scale-control">
          <label class="control-label" for="color-scale-slider">
            <span id="color-scale-label-text">Trend color range</span>
            <span class="control-value"><span id="color-scale-prefix">+/-</span> <span id="color-scale-value">0.0</span> mm/yr</span>
          </label>
          <input type="range" class="form-range" min="1" max="100" step="0.5" value="5" id="color-scale-slider">
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade metadata-modal" id="datasetInfoModal" tabindex="-1" aria-labelledby="datasetInfoTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title fs-6 fw-bold" id="datasetInfoTitle">Dataset information</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" id="datasetInfoBody"></div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade metadata-modal" id="datasetDownloadModal" tabindex="-1" aria-labelledby="datasetDownloadTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title fs-6 fw-bold" id="datasetDownloadTitle">Download original dataset</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" id="datasetDownloadBody"></div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Cancel</button>
          <a class="btn btn-primary btn-sm" id="datasetDownloadLink" href="#" target="_blank" rel="noreferrer">
            <i class="bi bi-download"></i>
            Download from source
          </a>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade metadata-modal" id="startupDisclaimerModal" tabindex="-1" aria-labelledby="startupDisclaimerTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title fs-6 fw-bold" id="startupDisclaimerTitle">Preliminary showcase</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <p class="mb-2">
            This site is largely AI-generated and intended to motivate real community development, review, and shared stewardship of VLM data.
          </p>
          <div class="alert alert-warning py-2 small mb-0">
            Please cite the original data sources, DOIs, and associated papers when using any dataset shown here. This website is only a visualization and metadata companion.
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-primary btn-sm" data-bs-dismiss="modal">Continue</button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade metadata-modal" id="selectionHistogramModal" tabindex="-1" aria-labelledby="selectionHistogramTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <div>
            <h5 class="modal-title fs-6 fw-bold" id="selectionHistogramTitle">Selection histogram</h5>
            <div class="text-secondary small" id="selectionHistogramSubtitle">No selection yet</div>
          </div>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body selection-histogram-body">
          <div class="selection-dataset-toggles" id="selectionDatasetToggles"></div>
          <canvas id="selectionHistogramCanvas" aria-label="Histogram of selected VLM trends"></canvas>
          <div class="selection-histogram-tooltip" id="selectionHistogramTooltip"></div>
          <div class="small text-secondary mt-2" id="selectionHistogramStatus">
            Circle selection includes VLM datasets only; external context layers are excluded.
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-primary btn-sm" id="selectionDownloadCsv" disabled>
            <i class="bi bi-download"></i>
            Download CSV
          </button>
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade station-modal" id="stationTimeSeriesModal" tabindex="-1" aria-labelledby="stationTimeSeriesTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
          <div>
            <h5 class="modal-title fs-6 fw-bold" id="stationTimeSeriesTitle">GPS station plot</h5>
            <div class="text-secondary small" id="stationTimeSeriesSubtitle">Loading...</div>
          </div>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="station-plot-meta">
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">MIDAS trend</div>
              <div class="station-plot-stat-value" id="stationTrendStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">MIDAS uncertainty</div>
              <div class="station-plot-stat-value" id="stationSigmaStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Duration</div>
              <div class="station-plot-stat-value" id="stationDurationStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Source</div>
              <div class="station-plot-stat-value" id="stationSampleStat">N/A</div>
            </div>
          </div>
          <div class="station-plot-wrap">
            <img id="stationTimeSeriesImage" alt="UNR GPS station time series plot" loading="lazy" />
          </div>
          <div class="station-plot-links">
            <a id="stationPageLink" href="#" target="_blank" rel="noreferrer">
              <i class="bi bi-box-arrow-up-right"></i>
              Open NGL station page
            </a>
            <a id="stationInlinePlotLink" href="#" target="_blank" rel="noreferrer">
              <i class="bi bi-image"></i>
              Open plot image
            </a>
          </div>
          <div class="station-citation-note">
            Please cite the Nevada Geodetic Laboratory station products and Blewitt et al. (2018) when using these plots or MIDAS velocities.
          </div>
          <div class="small text-secondary mt-2" id="stationTimeSeriesStatus">
            Click a GPS station to load its UNR time-series plot.
          </div>
        </div>
        <div class="modal-footer">
          <a class="btn btn-outline-primary btn-sm" id="stationPlotLink" href="#" target="_blank" rel="noreferrer">
            <i class="bi bi-box-arrow-up-right"></i>
            Open plot
          </a>
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade station-modal" id="tideGaugeModal" tabindex="-1" aria-labelledby="tideGaugeTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
          <div>
            <h5 class="modal-title fs-6 fw-bold" id="tideGaugeTitle">Tide gauge VLM</h5>
            <div class="text-secondary small" id="tideGaugeSubtitle">Loading...</div>
          </div>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="station-plot-meta">
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Trend</div>
              <div class="station-plot-stat-value" id="tideGaugeTrendStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Uncertainty</div>
              <div class="station-plot-stat-value" id="tideGaugeSigmaStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Period</div>
              <div class="station-plot-stat-value" id="tideGaugePeriodStat">N/A</div>
            </div>
            <div class="station-plot-stat">
              <div class="station-plot-stat-label">Samples</div>
              <div class="station-plot-stat-value" id="tideGaugeSampleStat">N/A</div>
            </div>
          </div>
          <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-2">
            <div class="form-check form-switch m-0">
              <input class="form-check-input" type="checkbox" role="switch" id="tideGaugeGnssToggle" checked>
              <label class="form-check-label small fw-semibold" for="tideGaugeGnssToggle">Nearest GNSS trends</label>
            </div>
            <div class="small text-secondary" id="tideGaugeGnssSummary">Nearest GNSS within 100 km</div>
          </div>
          <div class="station-plot-wrap">
            <canvas id="tideGaugePlotCanvas" aria-label="Tide gauge VLM time series plot"></canvas>
          </div>
          <div class="tide-gauge-histogram">
            <div class="tide-gauge-histogram-header">
              <span>Trend distribution</span>
              <div class="tide-gauge-histogram-actions">
                <span id="tideGaugeHistogramRange">Wheel to zoom, drag to pan</span>
                <button class="btn btn-outline-secondary btn-sm py-0 px-2" type="button" id="tideGaugeHistogramReset">Reset</button>
              </div>
            </div>
            <canvas id="tideGaugeHistogramCanvas" aria-label="Histogram of tide gauge and nearby GPS trends"></canvas>
          </div>
          <div class="plot-hover-card" id="tideGaugePlotTooltip"></div>
          <div class="nearby-gnss-list" id="tideGaugeGnssList"></div>
          <div class="station-plot-links">
            <a id="tideGaugeSourceLink" href="{TIDE_GAUGE_RECORD_URL}" target="_blank" rel="noreferrer">
              <i class="bi bi-box-arrow-up-right"></i>
              Open Zenodo record
            </a>
          </div>
          <div class="station-citation-note">
            Please cite Dangendorf et al. and the Zenodo dataset DOI when using the CSL-TG tide-gauge VLM product.
          </div>
          <div class="small text-secondary mt-2" id="tideGaugeStatus">
            Linear trend is fitted from the embedded MDadj3 annual residual series.
          </div>
        </div>
        <div class="modal-footer">
          <a class="btn btn-outline-primary btn-sm" href="{TIDE_GAUGE_RECORD_URL}" target="_blank" rel="noreferrer">
            <i class="bi bi-box-arrow-up-right"></i>
            Source
          </a>
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const {{
      MapboxOverlay,
      BitmapLayer,
      ColumnLayer,
      PolygonLayer,
      ScatterplotLayer,
      COORDINATE_SYSTEM
    }} = deck;

    let POSITIVE_DATA = [];
    let NEGATIVE_DATA = [];
    const METADATA = {metadata_json};
    let NGL_IMAGED_GRID_VALUES = null;
    const NGL_IMAGED_METADATA = {ngl_imaged_metadata_json};
    let GIA_GRID_VALUES = null;
    const GIA_METADATA = {gia_metadata_json};
    let INSAR_GRIDS = [];
    const INSAR_METADATA = {insar_metadata_json};
    let GNS_DATA = [];
    const GNS_METADATA = {gns_metadata_json};
    let TIDE_GAUGE_DATA = [];
    const TIDE_GAUGE_METADATA = {tide_gauge_metadata_json};
    let OELSMANN_HYBRID_DATA = [];
    const OELSMANN_HYBRID_METADATA = {oelsmann_hybrid_metadata_json};
    const DATASET_ATTRIBUTES = {dataset_attributes_json};
    const POPULATION_METADATA = {population_metadata_json};
    const POPULATION_DATA_URL = "{POPULATION_PAYLOAD_JS.as_posix()}";
    const EXTERNAL_DATASET_ATTRIBUTES = {external_dataset_attributes_json};
    const RENDER_PAYLOAD_URLS = {render_urls_json};
    const UNCERTAINTY_PAYLOAD_URLS = {uncertainty_urls_json};
    let ALL_DATA = [];

    const state = {{
      showGPS: true,
      showNglImaged: false,
      showGIA: true,
      showInSAR: true,
      showGNS: true,
      showTideGauge: true,
      showOelsmannHybrid: false,
      showPopulation: false,
      renderVariable: "trend",
      renderMode: "points",
      colorLimit: 5,
      uncertaintyLimit: 4,
      giaOpacity: 0.35,
      insarOpacity: 0.85,
      populationOpacity: 0.7,
      minDuration: 3,
      minFirstEpoch: METADATA.first_epoch_min,
      maxLastEpoch: METADATA.last_epoch_max,
      search: "",
      zoomBucket: 1.25,
      markerScale: markerScaleForZoom(1.25),
      tideGaugeScale: tideGaugeScaleForZoom(1.25),
      gpsAltitudeOffset: gpsAltitudeOffsetForZoom(1.25)
    }};

    const SELECTION_DATASETS = [
      {{id: "gnss", datasetId: "gnss_blewitt_2018", label: "GNSS", color: [36, 105, 180], kind: "point", axis: "left"}},
      {{id: "gia", datasetId: "gia_caron_2020", label: "GIA", color: [126, 87, 194], kind: "giaGrid", axis: "left"}},
      {{id: "insar", datasetId: "insar_ohenhen_2025", label: "InSAR deltas", color: [230, 126, 34], kind: "insarGrid", axis: "right"}},
      {{id: "gns", datasetId: "insar_gnss_hamling_2022", label: "NZ InSAR + GNSS", color: [34, 150, 94], kind: "point", axis: "right"}},
      {{id: "hybrid", datasetId: "hybrid_oelsmann_2026", label: "Hybrid estimates", color: [211, 84, 0], kind: "point", axis: "left"}},
      {{id: "tideGauge", datasetId: "tide_gauge_dangendorf_2026", label: "Tide gauges", color: [143, 86, 59], kind: "point", axis: "left"}}
    ];
    const RENDER_DATASET_IDS = {{
      gps: "gnss_blewitt_2018",
      nglImaged: "gnss_imaged_hammond_2021",
      gia: "gia_caron_2020",
      insar: "insar_ohenhen_2025",
      gns: "insar_gnss_hamling_2022",
      hybrid: "hybrid_oelsmann_2026",
      tideGauge: "tide_gauge_dangendorf_2026"
    }};
    const UNCERTAINTY_DATASET_IDS = new Set(Object.keys(UNCERTAINTY_PAYLOAD_URLS));
    const uncertaintyState = {{
      loaded: false,
      loading: null,
      payloads: {{}},
      nglImagedValues: null,
      giaValues: null,
      max: 5
    }};
    const selectionState = {{
      enabled: false,
      center: null,
      radiusKm: 200,
      records: [],
      visibleDatasets: Object.fromEntries(SELECTION_DATASETS.map(dataset => [dataset.id, true]))
    }};
    let selectionHistogramBars = [];

    function formatNumber(value, digits = 1) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) {{
        return "N/A";
      }}
      return Number(value).toFixed(digits);
    }}

    function markerScaleForZoom(zoom) {{
      const scale = Math.pow(0.72, zoom - 1.32);
      return Math.max(0.001, Math.min(1.35, scale));
    }}

    function tideGaugeScaleForZoom(zoom) {{
      const scale = Math.pow(0.64, zoom - 1.32);
      return Math.max(0.001, Math.min(1.35, scale));
    }}

    function gpsAltitudeOffsetForZoom(zoom) {{
      const scale = Math.pow(0.62, zoom - 1.32);
      return Math.max(50, Math.min(3500, 2500 * scale));
    }}

    function zoomBucketFor(zoom) {{
      return Math.round(Number(zoom || 0) * 4) / 4;
    }}

    function lerp(a, b, t) {{
      return Math.round(a + (b - a) * t);
    }}

    function colorForUp(value) {{
      return colorForValue(value, state.colorLimit, 220);
    }}

    function colorForValue(value, colorLimit, alpha) {{
      const neutral = [246, 246, 242, 220];
      const negative = [34, 104, 209, 220];
      const positive = [207, 50, 45, 220];
      const limit = Math.max(Number(colorLimit) || 1, 0.1);
      const t = Math.min(Math.abs(value) / limit, 1);
      const target = value >= 0 ? positive : negative;
      return [
        lerp(neutral[0], target[0], t),
        lerp(neutral[1], target[1], t),
        lerp(neutral[2], target[2], t),
        alpha
      ];
    }}

    function colorForUncertainty(value, alpha = 230) {{
      const safeMax = Math.max(0.5, Number(state.uncertaintyLimit) || 4);
      const t = Math.max(0, Math.min(1, Number(value) / safeMax));
      const stops = [
        [255, 247, 188],
        [254, 196, 79],
        [217, 95, 14],
        [127, 0, 0]
      ];
      const scaled = t * (stops.length - 1);
      const index = Math.min(Math.floor(scaled), stops.length - 2);
      const k = scaled - index;
      const a = stops[index];
      const b = stops[index + 1];
      return [
        lerp(a[0], b[0], k),
        lerp(a[1], b[1], k),
        lerp(a[2], b[2], k),
        alpha
      ];
    }}

    function renderValue(record) {{
      return state.renderVariable === "uncertainty" ? Number(record.up_sigma_mm_yr) : Number(record.up_mm_yr);
    }}

    function colorForRecord(record, alpha = 220) {{
      const value = renderValue(record);
      if (!Number.isFinite(value)) return [180, 188, 198, Math.round(alpha * 0.45)];
      return state.renderVariable === "uncertainty"
        ? colorForUncertainty(value, alpha)
        : colorForValue(value, state.colorLimit, alpha);
    }}

    function datasetHasActiveVariable(datasetId) {{
      return state.renderVariable !== "uncertainty" || UNCERTAINTY_DATASET_IDS.has(datasetId);
    }}

    function loadJsonPayload(url) {{
      return fetch(url, {{cache: "force-cache"}}).then(response => {{
        if (!response.ok) throw new Error(`Could not load ${{url}}`);
        return response.json();
      }});
    }}

    function hydratePointUncertainty(records, idField, payload) {{
      if (!payload || !Array.isArray(payload.values)) return;
      const lookup = new Map(payload.values.map(item => [String(item[0]), Number(item[1])]));
      for (const record of records) {{
        const value = lookup.get(String(record[idField]));
        if (Number.isFinite(value)) record.up_sigma_mm_yr = value;
      }}
    }}

    const renderPayloadState = {{
      payloads: {{}},
      loading: {{}},
      errors: {{}}
    }};

    function setDatasetLoading(datasetId, loading) {{
      document.querySelectorAll(`[data-layer-row="${{datasetId}}"]`).forEach(row => {{
        row.classList.toggle("loading", loading);
        row.setAttribute("aria-busy", loading ? "true" : "false");
      }});
    }}

    function hydrateLoadedUncertaintyForDataset(datasetId) {{
      if (!uncertaintyState.loaded) return;
      if (datasetId === RENDER_DATASET_IDS.gps) {{
        hydratePointUncertainty(ALL_DATA, "station", uncertaintyState.payloads.gnss_blewitt_2018);
      }} else if (datasetId === RENDER_DATASET_IDS.gns) {{
        hydratePointUncertainty(GNS_DATA, "id", uncertaintyState.payloads.insar_gnss_hamling_2022);
      }} else if (datasetId === RENDER_DATASET_IDS.hybrid) {{
        hydratePointUncertainty(OELSMANN_HYBRID_DATA, "id", uncertaintyState.payloads.hybrid_oelsmann_2026);
      }} else if (datasetId === RENDER_DATASET_IDS.tideGauge) {{
        hydratePointUncertainty(TIDE_GAUGE_DATA, "id", uncertaintyState.payloads.tide_gauge_dangendorf_2026);
      }}
    }}

    function assignRenderPayload(datasetId, payload) {{
      if (datasetId === RENDER_DATASET_IDS.gps) {{
        POSITIVE_DATA = Array.isArray(payload.positive) ? payload.positive : [];
        NEGATIVE_DATA = Array.isArray(payload.negative) ? payload.negative : [];
        ALL_DATA = POSITIVE_DATA.concat(NEGATIVE_DATA);
      }} else if (datasetId === RENDER_DATASET_IDS.nglImaged) {{
        NGL_IMAGED_GRID_VALUES = Array.isArray(payload.values) ? payload.values : null;
        nglImagedCellCache = null;
        nglImagedCellCacheVariable = null;
      }} else if (datasetId === RENDER_DATASET_IDS.gia) {{
        GIA_GRID_VALUES = Array.isArray(payload.values) ? payload.values : null;
        giaCellCache = null;
        giaCellCacheVariable = null;
        giaBitmapCanvasCache.clear();
      }} else if (datasetId === RENDER_DATASET_IDS.insar) {{
        INSAR_GRIDS = Array.isArray(payload.grids) ? payload.grids : [];
      }} else if (datasetId === RENDER_DATASET_IDS.gns) {{
        GNS_DATA = Array.isArray(payload.records) ? payload.records : [];
      }} else if (datasetId === RENDER_DATASET_IDS.hybrid) {{
        OELSMANN_HYBRID_DATA = Array.isArray(payload.records) ? payload.records : [];
      }} else if (datasetId === RENDER_DATASET_IDS.tideGauge) {{
        TIDE_GAUGE_DATA = Array.isArray(payload.records) ? payload.records : [];
      }}
      renderPayloadState.payloads[datasetId] = payload;
      hydrateLoadedUncertaintyForDataset(datasetId);
    }}

    function loadRenderPayload(datasetId) {{
      if (renderPayloadState.payloads[datasetId]) return Promise.resolve(renderPayloadState.payloads[datasetId]);
      if (renderPayloadState.loading[datasetId]) return renderPayloadState.loading[datasetId];
      const url = RENDER_PAYLOAD_URLS[datasetId];
      if (!url) return Promise.resolve(null);

      setDatasetLoading(datasetId, true);
      renderPayloadState.loading[datasetId] = loadJsonPayload(url).then(payload => {{
        assignRenderPayload(datasetId, payload || {{}});
        delete renderPayloadState.errors[datasetId];
        return payload;
      }}).catch(error => {{
        renderPayloadState.errors[datasetId] = error;
        console.warn(error);
        return null;
      }}).finally(() => {{
        delete renderPayloadState.loading[datasetId];
        setDatasetLoading(datasetId, false);
        computeSelectionRecords();
        updateSelectionStatus();
        updateLayers();
      }});
      return renderPayloadState.loading[datasetId];
    }}

    function loadActiveRenderPayloads() {{
      const activeDatasetIds = [];
      if (state.showGPS) activeDatasetIds.push(RENDER_DATASET_IDS.gps);
      if (state.showNglImaged) activeDatasetIds.push(RENDER_DATASET_IDS.nglImaged);
      if (state.showGIA) activeDatasetIds.push(RENDER_DATASET_IDS.gia);
      if (state.showInSAR) activeDatasetIds.push(RENDER_DATASET_IDS.insar);
      if (state.showGNS) activeDatasetIds.push(RENDER_DATASET_IDS.gns);
      if (state.showOelsmannHybrid) activeDatasetIds.push(RENDER_DATASET_IDS.hybrid);
      if (state.showTideGauge) activeDatasetIds.push(RENDER_DATASET_IDS.tideGauge);
      activeDatasetIds.forEach(datasetId => {{
        if (datasetHasActiveVariable(datasetId)) loadRenderPayload(datasetId);
      }});
      if (state.showPopulation && !makePopulationPointData()) {{
        loadPopulationDataset().then(() => {{
          if (state.showPopulation) updateLayers();
        }});
      }}
    }}

    function updateUncertaintyMax() {{
      const maxValues = Object.values(uncertaintyState.payloads)
        .map(payload => Number(payload && payload.max))
        .filter(Number.isFinite);
      uncertaintyState.max = maxValues.length ? Math.max(0.5, ...maxValues) : 5;
    }}

    function loadUncertaintyPayloads() {{
      if (uncertaintyState.loaded) return Promise.resolve(uncertaintyState.payloads);
      if (uncertaintyState.loading) return uncertaintyState.loading;

      uncertaintyState.loading = Promise.all(Object.entries(UNCERTAINTY_PAYLOAD_URLS).map(([datasetId, url]) =>
        loadJsonPayload(url).then(payload => [datasetId, payload])
      )).then(entries => {{
        uncertaintyState.payloads = Object.fromEntries(entries);
        hydratePointUncertainty(ALL_DATA, "station", uncertaintyState.payloads.gnss_blewitt_2018);
        hydratePointUncertainty(GNS_DATA, "id", uncertaintyState.payloads.insar_gnss_hamling_2022);
        hydratePointUncertainty(OELSMANN_HYBRID_DATA, "id", uncertaintyState.payloads.hybrid_oelsmann_2026);
        hydratePointUncertainty(TIDE_GAUGE_DATA, "id", uncertaintyState.payloads.tide_gauge_dangendorf_2026);
        uncertaintyState.nglImagedValues = uncertaintyState.payloads.gnss_imaged_hammond_2021?.values || null;
        uncertaintyState.giaValues = uncertaintyState.payloads.gia_caron_2020?.values || null;
        updateUncertaintyMax();
        uncertaintyState.loaded = true;
        uncertaintyState.loading = null;
        return uncertaintyState.payloads;
      }}).catch(error => {{
        uncertaintyState.loading = null;
        console.warn(error);
        return uncertaintyState.payloads;
      }});

      return uncertaintyState.loading;
    }}

    function colorForPopulation(value, alpha) {{
      const logValue = Math.log10(Math.max(Number(value) || 0, 0) + 1);
      const limit = Math.max(POPULATION_METADATA.max_log10_population || 1, 1);
      const t = Math.min(logValue / limit, 1);
      const stops = [
        [255, 252, 214],
        [253, 179, 75],
        [214, 74, 64],
        [84, 39, 143]
      ];
      const scaled = t * (stops.length - 1);
      const index = Math.min(Math.floor(scaled), stops.length - 2);
      const localT = scaled - index;
      const a = stops[index];
      const b = stops[index + 1];
      return [
        lerp(a[0], b[0], localT),
        lerp(a[1], b[1], localT),
        lerp(a[2], b[2], localT),
        alpha
      ];
    }}

    let populationPointCache = null;
    let populationLoadPromise = null;

    function makePopulationPointData() {{
      if (populationPointCache) return populationPointCache;
      const payload = window.GHSL_POPULATION_PAYLOAD;
      if (!payload || !Array.isArray(payload.values)) return null;

      if (payload.metadata) {{
        Object.assign(POPULATION_METADATA, payload.metadata);
      }}

      const points = [];

      for (let i = 0; i < payload.values.length; i += 1) {{
        const record = payload.values[i];
        const value = record[2];

        points.push({{
          longitude: record[0],
          latitude: record[1],
          population: value,
          radius_m: record[3]
        }});
      }}

      populationPointCache = points;
      return populationPointCache;
    }}

    function loadPopulationDataset() {{
      if (makePopulationPointData()) return Promise.resolve(populationPointCache);
      if (populationLoadPromise) return populationLoadPromise;

      setDatasetLoading("ghsl_schiavina_2025", true);
      populationLoadPromise = new Promise((resolve, reject) => {{
        const script = document.createElement("script");
        script.src = POPULATION_DATA_URL;
        script.async = true;
        script.onload = () => {{
          const points = makePopulationPointData();
          if (points) resolve(points);
          else reject(new Error("Population payload did not define values."));
        }};
        script.onerror = () => reject(new Error(`Could not load ${{POPULATION_DATA_URL}}`));
        document.head.appendChild(script);
      }}).catch(error => {{
        populationLoadPromise = null;
        console.warn(error);
        return null;
      }}).finally(() => {{
        setDatasetLoading("ghsl_schiavina_2025", false);
      }});

      return populationLoadPromise;
    }}

    function lonDeltaDegrees(a, b) {{
      return ((((Number(a) - Number(b)) + 540) % 360) - 180);
    }}

    function haversineKm(lonA, latA, lonB, latB) {{
      const radiusKm = 6371.0088;
      const phi1 = Number(latA) * Math.PI / 180;
      const phi2 = Number(latB) * Math.PI / 180;
      const dPhi = (Number(latB) - Number(latA)) * Math.PI / 180;
      const dLambda = lonDeltaDegrees(lonB, lonA) * Math.PI / 180;
      const a = Math.sin(dPhi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
      return radiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
    }}

    function pointWithinSelection(lon, lat, center = selectionState.center, radiusKm = selectionState.radiusKm) {{
      if (!center) return false;
      const latPad = radiusKm / 111.32;
      const lonPad = radiusKm / Math.max(20, 111.32 * Math.cos(center.latitude * Math.PI / 180));
      if (Math.abs(Number(lat) - center.latitude) > latPad) return false;
      if (Math.abs(lonDeltaDegrees(Number(lon), center.longitude)) > lonPad) return false;
      return haversineKm(center.longitude, center.latitude, lon, lat) <= radiusKm;
    }}

    function finiteOrBlank(value) {{
      const number = Number(value);
      return Number.isFinite(number) ? number : "";
    }}

    function selectionCsvFields(datasetId, datasetLabel, source, name) {{
      const fields = {{
        yearFrom: "",
        yearTo: "",
        info1: name || source.name || source.station || source.id || "",
        info2: ""
      }};

      if (datasetId === "gnss") {{
        fields.yearFrom = finiteOrBlank(source.first_epoch_year);
        fields.yearTo = finiteOrBlank(source.last_epoch_year);
        fields.info1 = source.station || fields.info1;
        fields.info2 = [
          source.version ? `version=${{source.version}}` : "",
          Number.isFinite(Number(source.duration)) ? `duration_years=${{source.duration}}` : "",
          Number.isFinite(Number(source.steps)) ? `steps=${{source.steps}}` : ""
        ].filter(Boolean).join("; ");
      }} else if (datasetId === "gns") {{
        fields.yearFrom = 2003;
        fields.yearTo = 2011;
        fields.info1 = source.id || "GNS coastal point";
        fields.info2 = [
          Number.isFinite(Number(source.up_sigma_mm_yr)) ? `uncertainty_mm_yr=${{source.up_sigma_mm_yr}}` : "",
          Number.isFinite(Number(source.observations)) ? `observations=${{source.observations}}` : "",
          Number.isFinite(Number(source.quality_factor)) ? `quality_factor=${{source.quality_factor}}` : "",
          Number.isFinite(Number(source.average_radius_km)) ? `average_radius_km=${{source.average_radius_km}}` : ""
        ].filter(Boolean).join("; ");
      }} else if (datasetId === "tideGauge") {{
        fields.yearFrom = finiteOrBlank(source.first_year);
        fields.yearTo = finiteOrBlank(source.last_year);
        fields.info1 = source.name || fields.info1;
        fields.info2 = [
          source.psmsl_id ? `psmsl_id=${{source.psmsl_id}}` : "",
          Number.isFinite(Number(source.up_sigma_mm_yr)) ? `uncertainty_mm_yr=${{source.up_sigma_mm_yr}}` : "",
          Number.isFinite(Number(source.sample_count)) ? `samples=${{source.sample_count}}` : ""
        ].filter(Boolean).join("; ");
      }} else if (datasetId === "hybrid") {{
        fields.yearFrom = 1995;
        fields.yearTo = 2020;
        fields.info1 = source.name || source.id || fields.info1;
        fields.info2 = [
          Number.isFinite(Number(source.up_sigma_mm_yr)) ? `uncertainty_mm_yr=${{source.up_sigma_mm_yr}}` : "",
          Number.isFinite(Number(source.datatype)) ? `datatype=${{source.datatype}}` : ""
        ].filter(Boolean).join("; ");
      }} else if (datasetId === "insar") {{
        fields.yearFrom = 2014;
        fields.yearTo = 2023;
        fields.info1 = source.deltaName || source.name || fields.info1;
        fields.info2 = "Ohenhen et al. 2025";
      }} else if (datasetId === "gia") {{
        fields.info1 = "Tdur vertical land motion";
        fields.info2 = "Caron and Ivins 2020 GIA model";
      }}

      return fields;
    }}

    function selectionRecord(datasetId, datasetLabel, color, source, value, sigma, lon, lat, name) {{
      source = source || {{}};
      const csvFields = selectionCsvFields(datasetId, datasetLabel, source, name);
      return {{
        datasetId,
        datasetLabel,
        color,
        value: Number(value),
        sigma: sigma === null || sigma === undefined || Number.isNaN(Number(sigma)) ? null : Number(sigma),
        longitude: Number(lon),
        latitude: Number(lat),
        name: name || source.name || source.station || source.id || datasetLabel,
        yearFrom: csvFields.yearFrom,
        yearTo: csvFields.yearTo,
        info1: csvFields.info1,
        info2: csvFields.info2
      }};
    }}

    function selectedPointRecords(datasetId, datasetLabel, color, data) {{
      const output = [];
      for (const record of data) {{
        const value = renderValue(record);
        if (!Number.isFinite(value)) continue;
        if (!pointWithinSelection(record.longitude, record.latitude)) continue;
        output.push(selectionRecord(
          datasetId,
          datasetLabel,
          color,
          record,
          value,
          record.up_sigma_mm_yr,
          record.longitude,
          record.latitude
        ));
      }}
      return output;
    }}

    function selectedGIARecords(datasetId, datasetLabel, color) {{
      const output = [];
      const width = GIA_METADATA.width;
      const height = GIA_METADATA.height;
      const values = state.renderVariable === "uncertainty" ? uncertaintyState.giaValues : GIA_GRID_VALUES;
      if (!values) return output;
      for (let y = 0; y < height; y += 1) {{
        const latitude = 90 - y - 0.5;
        for (let x = 0; x < width; x += 1) {{
          const value = values[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          const longitude = -180 + x + 0.5;
          if (!pointWithinSelection(longitude, latitude)) continue;
          output.push(selectionRecord(datasetId, datasetLabel, color, {{}}, value, null, longitude, latitude, `GIA ${{formatNumber(longitude, 1)}}, ${{formatNumber(latitude, 1)}}`));
        }}
      }}
      return output;
    }}

    function selectedInSARRecords(datasetId, datasetLabel, color) {{
      const output = [];
      if (state.renderVariable === "uncertainty") return output;
      for (const grid of INSAR_GRIDS) {{
        const [west, south, east, north] = grid.bounds;
        const lonStep = (east - west) / grid.width;
        const latStep = (north - south) / grid.height;
        for (let row = 0; row < grid.height; row += 1) {{
          const latitude = north - (row + 0.5) * latStep;
          for (let col = 0; col < grid.width; col += 1) {{
            const value = grid.values[row * grid.width + col];
            if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
            const longitude = west + (col + 0.5) * lonStep;
            if (!pointWithinSelection(longitude, latitude)) continue;
            output.push(selectionRecord(datasetId, datasetLabel, color, {{deltaName: grid.name}}, value, null, longitude, latitude, grid.name));
          }}
        }}
      }}
      return output;
    }}

    function computeSelectionRecords() {{
      if (!selectionState.center) {{
        selectionState.records = [];
        return selectionState.records;
      }}

      const gnssData = POSITIVE_DATA.concat(NEGATIVE_DATA).filter(stationMatches);
      const records = [
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.gps) ? selectedPointRecords("gnss", "GNSS", SELECTION_DATASETS[0].color, gnssData) : []),
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.gia) ? selectedGIARecords("gia", "GIA", SELECTION_DATASETS[1].color) : []),
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.insar) ? selectedInSARRecords("insar", "InSAR deltas", SELECTION_DATASETS[2].color) : []),
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.gns) ? selectedPointRecords("gns", "NZ InSAR + GNSS", SELECTION_DATASETS[3].color, GNS_DATA) : []),
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.hybrid) ? selectedPointRecords("hybrid", "Hybrid estimates", SELECTION_DATASETS[4].color, OELSMANN_HYBRID_DATA) : []),
        ...(datasetHasActiveVariable(RENDER_DATASET_IDS.tideGauge) ? selectedPointRecords("tideGauge", "Tide gauges", SELECTION_DATASETS[5].color, TIDE_GAUGE_DATA) : [])
      ];

      selectionState.records = records;
      return records;
    }}

    function selectionCountsByDataset() {{
      return Object.fromEntries(SELECTION_DATASETS.map(dataset => [
        dataset.id,
        selectionState.records.filter(record => record.datasetId === dataset.id).length
      ]));
    }}

    function updateSelectionStatus() {{
      document.getElementById("selection-radius-value").textContent = selectionState.radiusKm.toString();
      const status = document.getElementById("selection-status");
      const openButton = document.getElementById("selection-open-histogram");
      if (!selectionState.center) {{
        status.textContent = selectionState.enabled ? "Click the globe to place the selector circle." : "Enable selector, then click the globe.";
        openButton.disabled = true;
        return;
      }}
      const total = selectionState.records.length;
      status.textContent = `${{total.toLocaleString()}} VLM trends selected around ${{formatNumber(selectionState.center.longitude, 2)}}, ${{formatNumber(selectionState.center.latitude, 2)}}.`;
      openButton.disabled = total === 0;
    }}

    function renderSelectionDatasetToggles() {{
      const counts = selectionCountsByDataset();
      const container = document.getElementById("selectionDatasetToggles");
      container.innerHTML = SELECTION_DATASETS.map(dataset => {{
        const color = `rgb(${{dataset.color[0]}}, ${{dataset.color[1]}}, ${{dataset.color[2]}})`;
        const count = counts[dataset.id] || 0;
        return `
          <div class="form-check">
            <input class="form-check-input" type="checkbox" id="selection-toggle-${{dataset.id}}" data-selection-dataset="${{dataset.id}}" ${{selectionState.visibleDatasets[dataset.id] ? "checked" : ""}} ${{count ? "" : "disabled"}}>
            <label class="form-check-label small fw-semibold" for="selection-toggle-${{dataset.id}}">
              <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${{color}};margin-right:5px;"></span>
              ${{dataset.label}} (${{count.toLocaleString()}})
            </label>
          </div>
        `;
      }}).join("");

      container.querySelectorAll("[data-selection-dataset]").forEach(input => {{
        input.addEventListener("change", event => {{
          selectionState.visibleDatasets[event.target.getAttribute("data-selection-dataset")] = event.target.checked;
          drawSelectionHistogram();
        }});
      }});
    }}

    function activeSelectionRecords() {{
      return selectionState.records.filter(record => selectionState.visibleDatasets[record.datasetId]);
    }}

    function csvCell(value) {{
      if (value === null || value === undefined) return "";
      const text = String(value);
      return /[",\n\r]/.test(text) ? `"${{text.replace(/"/g, '""')}}"` : text;
    }}

    function downloadSelectionCsv() {{
      const records = activeSelectionRecords();
      if (!records.length) return;
      const modeLabel = state.renderVariable === "uncertainty" ? "uncertainty" : "trend";
      const rows = [[
        "lon",
        "lat",
        "value_mm_year",
        "type_or_reference",
        "year_from",
        "year_to",
        "info_col_1",
        "info_col_2"
      ]];
      records.forEach(record => {{
        rows.push([
          Number.isFinite(record.longitude) ? record.longitude.toFixed(6) : "",
          Number.isFinite(record.latitude) ? record.latitude.toFixed(6) : "",
          Number.isFinite(record.value) ? record.value.toFixed(4) : "",
          `${{record.datasetLabel}} ${{modeLabel}}`,
          record.yearFrom,
          record.yearTo,
          record.info1,
          record.info2
        ]);
      }});
      const csv = rows.map(row => row.map(csvCell).join(",")).join("\n") + "\n";
      const blob = new Blob([csv], {{type: "text/csv;charset=utf-8"}});
      const link = document.createElement("a");
      const center = selectionState.center || {{longitude: 0, latitude: 0}};
      link.href = URL.createObjectURL(blob);
      link.download = `vlm_selection_${{formatNumber(center.longitude, 2)}}_${{formatNumber(center.latitude, 2)}}_${{selectionState.radiusKm}}km_${{modeLabel}}.csv`.replace(/[^a-zA-Z0-9_.-]+/g, "_");
      document.body.appendChild(link);
      link.click();
      URL.revokeObjectURL(link.href);
      link.remove();
    }}

    function drawSelectionHistogram() {{
      const canvas = document.getElementById("selectionHistogramCanvas");
      if (!canvas) return;
      const width = Math.max(320, Math.floor(canvas.getBoundingClientRect().width || canvas.parentElement.clientWidth || 720));
      const height = 286;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = "100%";
      canvas.style.height = `${{height}}px`;
      selectionHistogramBars = [];
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const activeRecords = activeSelectionRecords();
      const downloadButton = document.getElementById("selectionDownloadCsv");
      if (downloadButton) downloadButton.disabled = activeRecords.length === 0;
      const variableLabel = state.renderVariable === "uncertainty" ? "uncertainties" : "trends";
      document.getElementById("selectionHistogramSubtitle").textContent = selectionState.center
        ? `${{activeRecords.length.toLocaleString()}} active of ${{selectionState.records.length.toLocaleString()}} selected ${{variableLabel}} | radius ${{selectionState.radiusKm}} km`
        : "No selection yet";
      if (!activeRecords.length) {{
        ctx.fillStyle = "#64748b";
        ctx.font = "13px Segoe UI, Arial, sans-serif";
        ctx.fillText("No selected records are enabled.", 24, 42);
        return;
      }}

      const values = activeRecords.map(record => record.value).filter(Number.isFinite);
      let xMin = Math.min(...values);
      let xMax = Math.max(...values);
      if (xMin === xMax) {{
        xMin -= 1;
        xMax += 1;
      }}
      const pad = Math.max((xMax - xMin) * 0.08, 0.5);
      xMin -= pad;
      xMax += pad;

      const margin = {{left: 52, right: 54, top: 22, bottom: 48}};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const binCount = Math.max(8, Math.min(24, Math.round(plotWidth / 42)));
      const binWidth = (xMax - xMin) / binCount;
      const binData = Object.fromEntries(SELECTION_DATASETS.map(dataset => [
        dataset.id,
        Array.from({{length: binCount}}, () => 0)
      ]));

      activeRecords.forEach(record => {{
        const index = Math.min(binCount - 1, Math.max(0, Math.floor((record.value - xMin) / Math.max(binWidth, 1e-9))));
        binData[record.datasetId][index] += 1;
      }});

      const enabledDatasets = SELECTION_DATASETS.filter(dataset => selectionState.visibleDatasets[dataset.id]);
      const leftDatasetIds = enabledDatasets.filter(dataset => dataset.axis !== "right").map(dataset => dataset.id);
      const rightDatasetIds = enabledDatasets.filter(dataset => dataset.axis === "right").map(dataset => dataset.id);
      const yMaxLeft = Math.max(1, ...leftDatasetIds.flatMap(id => binData[id] || [0]));
      const yMaxRight = Math.max(1, ...rightDatasetIds.flatMap(id => binData[id] || [0]));
      const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
      const yScale = (count, axis) => {{
        const yMax = axis === "right" ? yMaxRight : yMaxLeft;
        return margin.top + (1 - count / yMax) * plotHeight;
      }};

      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 4; i += 1) {{
        const y = margin.top + plotHeight * i / 4;
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
      }}
      ctx.stroke();

      const groupWidth = plotWidth / binCount;
      for (let bin = 0; bin < binCount; bin += 1) {{
        enabledDatasets.forEach((dataset, datasetIndex) => {{
          const count = binData[dataset.id][bin] || 0;
          if (!count) return;
          const offsetStep = Math.min(3, Math.max(1, groupWidth * 0.045));
          const barWidth = Math.max(4, groupWidth * 0.72);
          const x = margin.left + bin * groupWidth + (groupWidth - barWidth) / 2 + (datasetIndex - (enabledDatasets.length - 1) / 2) * offsetStep;
          const y = yScale(count, dataset.axis);
          const h = margin.top + plotHeight - y;
          ctx.fillStyle = `rgba(${{dataset.color[0]}}, ${{dataset.color[1]}}, ${{dataset.color[2]}}, 0.52)`;
          ctx.fillRect(x, y, barWidth, margin.top + plotHeight - y);
          selectionHistogramBars.push({{
            x,
            y,
            width: barWidth,
            height: h,
            dataset: dataset.label,
            axis: dataset.axis === "right" ? "InSAR count axis" : "count axis",
            count,
            xMin: xMin + bin * binWidth,
            xMax: xMin + (bin + 1) * binWidth,
            color: dataset.color
          }});
        }});
      }}

      ctx.strokeStyle = "#475569";
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, margin.top + plotHeight);
      ctx.lineTo(width - margin.right, margin.top + plotHeight);
      ctx.lineTo(width - margin.right, margin.top);
      ctx.stroke();

      ctx.fillStyle = "#334155";
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      ctx.textAlign = "center";
      for (let i = 0; i <= 5; i += 1) {{
        const value = xMin + (xMax - xMin) * i / 5;
        ctx.fillText(formatNumber(value, 1), xScale(value), height - 22);
      }}
      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i += 1) {{
        const value = Math.round(yMaxLeft * (4 - i) / 4);
        const y = margin.top + plotHeight * i / 4;
        ctx.fillText(value.toString(), margin.left - 8, y + 4);
      }}
      if (rightDatasetIds.length) {{
        ctx.textAlign = "left";
        ctx.fillStyle = "#8a4b13";
        for (let i = 0; i <= 4; i += 1) {{
          const value = Math.round(yMaxRight * (4 - i) / 4);
          const y = margin.top + plotHeight * i / 4;
          ctx.fillText(value.toString(), width - margin.right + 8, y + 4);
        }}
      }}
      ctx.textAlign = "right";
      ctx.fillText("mm/yr", width - margin.right, height - 7);
      ctx.save();
      ctx.translate(16, margin.top + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("count", 0, 0);
      ctx.restore();
      if (rightDatasetIds.length) {{
        ctx.save();
        ctx.translate(width - 14, margin.top + plotHeight / 2);
        ctx.rotate(Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillStyle = "#8a4b13";
        ctx.fillText("InSAR count", 0, 0);
        ctx.restore();
      }}
    }}

    function openSelectionHistogram() {{
      if (!selectionState.center) return;
      computeSelectionRecords();
      renderSelectionDatasetToggles();
      drawSelectionHistogram();
      bootstrap.Modal.getOrCreateInstance(document.getElementById("selectionHistogramModal")).show();
      updateSelectionStatus();
    }}

    function makeSelectionLayer() {{
      if (!selectionState.center) return null;
      return new ScatterplotLayer({{
        id: "circle-selection-overlay",
        data: [selectionState.center],
        getPosition: d => [d.longitude, d.latitude, 2600],
        getRadius: () => selectionState.radiusKm * 1000,
        radiusUnits: "meters",
        stroked: true,
        filled: true,
        lineWidthMinPixels: 2,
        getLineColor: [47, 111, 237, 230],
        getFillColor: [47, 111, 237, 32],
        pickable: false,
        parameters: {{
          depthTest: false,
          depthMask: false
        }}
      }});
    }}

    function makePopulationLayer() {{
      if (!state.showPopulation) return null;
      const populationData = makePopulationPointData();
      if (!populationData) return null;
      return new ScatterplotLayer({{
        id: "external-ghsl-population",
        data: populationData,
        getPosition: d => [d.longitude, d.latitude, 1800],
        getRadius: d => d.radius_m,
        radiusUnits: "meters",
        radiusMinPixels: 1,
        radiusMaxPixels: 5,
        stroked: false,
        filled: true,
        getFillColor: d => colorForPopulation(d.population, Math.round(255 * state.populationOpacity)),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 180],
        parameters: {{
          depthTest: false,
          depthMask: false
        }},
        updateTriggers: {{
          getFillColor: [state.populationOpacity]
        }}
      }});
    }}

    let giaPointCache = null;

    function makeGIAPointData() {{
      if (giaPointCache) return giaPointCache;

      const points = [];
      const width = GIA_METADATA.width;
      const height = GIA_METADATA.height;
      for (let y = 0; y < height; y += 1) {{
        const latitude = 90 - y - 0.5;
        for (let x = 0; x < width; x += 1) {{
          const value = GIA_GRID_VALUES[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          points.push({{
            longitude: -180 + x + 0.5,
            latitude,
            up_mm_yr: value
          }});
        }}
      }}

      giaPointCache = points;
      return giaPointCache;
    }}

    let giaCellCache = null;
    let giaCellCacheVariable = null;
    let nglImagedCellCache = null;
    let nglImagedCellCacheVariable = null;

    function makeGridCellData(metadata, values, variable) {{
      const cells = [];
      if (!values) return cells;
      const width = metadata.width;
      const height = metadata.height;
      const bounds = metadata.bounds;
      const lonStep = (bounds[2] - bounds[0]) / width;
      const latStep = (bounds[3] - bounds[1]) / height;
      for (let y = 0; y < height; y += 1) {{
        const south = Math.max(-89.999, bounds[1] + y * latStep);
        const north = Math.min(89.999, south + latStep);
        for (let x = 0; x < width; x += 1) {{
          const value = values[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          const west = bounds[0] + x * lonStep;
          const east = west + lonStep;
          cells.push({{
            up_mm_yr: variable === "uncertainty" ? null : value,
            up_sigma_mm_yr: variable === "uncertainty" ? value : null,
            polygon: [[west, south], [east, south], [east, north], [west, north]]
          }});
        }}
      }}
      return cells;
    }}

    function makeGIACellData() {{
      if (giaCellCache && giaCellCacheVariable === state.renderVariable) return giaCellCache;

      const cells = [];
      const width = GIA_METADATA.width;
      const height = GIA_METADATA.height;
      const values = state.renderVariable === "uncertainty" ? uncertaintyState.giaValues : GIA_GRID_VALUES;
      if (!values) return cells;
      for (let y = 0; y < height; y += 1) {{
        const north = Math.min(89.999, 90 - y);
        const south = Math.max(-89.999, 90 - y - 1);
        for (let x = 0; x < width; x += 1) {{
          const value = values[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          const west = -180 + x;
          const east = west + 1;
          cells.push({{
            up_mm_yr: state.renderVariable === "uncertainty" ? null : value,
            up_sigma_mm_yr: state.renderVariable === "uncertainty" ? value : null,
            polygon: [[west, south], [east, south], [east, north], [west, north]]
          }});
        }}
      }}

      giaCellCache = cells;
      giaCellCacheVariable = state.renderVariable;
      return giaCellCache;
    }}

    function makeNglImagedCellData() {{
      if (nglImagedCellCache && nglImagedCellCacheVariable === state.renderVariable) return nglImagedCellCache;
      const values = state.renderVariable === "uncertainty" ? uncertaintyState.nglImagedValues : NGL_IMAGED_GRID_VALUES;
      nglImagedCellCache = makeGridCellData(NGL_IMAGED_METADATA, values, state.renderVariable);
      nglImagedCellCacheVariable = state.renderVariable;
      return nglImagedCellCache;
    }}

    const giaBitmapCanvasCache = new Map();

    function makeGIABitmapCanvas(xStart, yStart, xCount, yCount, alpha, colorLimit) {{
      const cacheKey = `${{xStart}}:${{yStart}}:${{xCount}}:${{yCount}}:${{alpha}}:${{colorLimit}}`;
      if (giaBitmapCanvasCache.has(cacheKey)) return giaBitmapCanvasCache.get(cacheKey);

      const canvas = document.createElement("canvas");
      canvas.width = xCount;
      canvas.height = yCount;

      const context = canvas.getContext("2d");
      const image = context.createImageData(canvas.width, canvas.height);

      for (let y = 0; y < yCount; y += 1) {{
        for (let x = 0; x < xCount; x += 1) {{
          const value = GIA_GRID_VALUES[(yStart + y) * GIA_METADATA.width + xStart + x];
          const offset = (y * xCount + x) * 4;
          if (value === null || value === undefined || Number.isNaN(Number(value))) {{
            image.data[offset + 3] = 0;
            continue;
          }}

          const color = colorForValue(value, colorLimit, alpha);
          image.data[offset] = color[0];
          image.data[offset + 1] = color[1];
          image.data[offset + 2] = color[2];
          image.data[offset + 3] = color[3];
        }}
      }}

      context.putImageData(image, 0, 0);
      giaBitmapCanvasCache.set(cacheKey, canvas);
      return canvas;
    }}

    function makeGIALayers() {{
      if (!state.showGIA) return [];
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.gia)) return [];

      if (false) {{
        return [
          new ScatterplotLayer({{
            id: "gia-vlm-grid",
            data: makeGIAPointData(),
            getPosition: d => [d.longitude, d.latitude, 900],
            getRadius: 72000,
            radiusUnits: "meters",
            radiusMinPixels: 1,
            radiusMaxPixels: 8,
            stroked: false,
            filled: true,
            getFillColor: d => colorForRecord(d, Math.round(255 * state.giaOpacity)),
            pickable: false,
            parameters: {{
              depthTest: false,
              depthMask: false
            }},
            updateTriggers: {{
              getFillColor: [state.colorLimit, state.uncertaintyLimit, state.giaOpacity, state.renderVariable]
            }}
          }})
        ];
      }}

      if (false) {{
        const alpha = Math.round(255 * state.giaOpacity);
        const layers = [];
        for (let band = 0; band < 20; band += 1) {{
          const yStart = band + 1;
          const north = 89 - band;
          const south = north - 1;
          layers.push(new BitmapLayer({{
            id: `gia-vlm-bitmap-ring-test-${{band}}`,
            image: makeGIABitmapCanvas(0, yStart, GIA_METADATA.width, 1, alpha, state.colorLimit),
            bounds: [-180, south, 180, north],
            _imageCoordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
            pickable: false,
            parameters: {{
              depthTest: false,
              depthMask: false
            }}
          }}));
        }}
        return layers;
      }}

      return [
        new PolygonLayer({{
          id: "gia-vlm-cell-mesh",
          data: makeGIACellData(),
          getPolygon: d => d.polygon,
          getFillColor: d => colorForRecord(d, Math.round(255 * state.giaOpacity)),
          stroked: false,
          filled: true,
          pickable: false,
          parameters: {{
            depthTest: false,
            depthMask: false
          }},
          updateTriggers: {{
            getFillColor: [state.colorLimit, state.uncertaintyLimit, state.giaOpacity, state.renderVariable]
          }}
        }})
      ];
    }}

    function makeNglImagedLayer() {{
      if (!state.showNglImaged) return null;
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.nglImaged)) return null;

      return new PolygonLayer({{
        id: "ngl-gps-imaging-vlm-grid",
        data: makeNglImagedCellData(),
        getPolygon: d => d.polygon,
        getFillColor: d => colorForRecord(d, 215),
        stroked: false,
        filled: true,
        pickable: false,
        parameters: {{
          depthTest: false,
          depthMask: false
        }},
        updateTriggers: {{
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable]
        }}
      }});
    }}

    function makeGridCanvas(grid, opacity) {{
      const canvas = document.createElement("canvas");
      canvas.width = grid.width;
      canvas.height = grid.height;

      const context = canvas.getContext("2d");
      const image = context.createImageData(canvas.width, canvas.height);
      const alpha = Math.round(255 * opacity);

      for (let i = 0; i < grid.values.length; i += 1) {{
        const value = grid.values[i];
        const offset = i * 4;
        if (value === null || value === undefined || Number.isNaN(Number(value))) {{
          image.data[offset + 3] = 0;
          continue;
        }}

        const color = colorForValue(value, state.colorLimit, alpha);
        image.data[offset] = color[0];
        image.data[offset + 1] = color[1];
        image.data[offset + 2] = color[2];
        image.data[offset + 3] = color[3];
      }}

      context.putImageData(image, 0, 0);
      return canvas;
    }}

    function makeInSARLayers() {{
      if (!state.showInSAR) return [];
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.insar)) return [];

      const rasterLayers = INSAR_GRIDS.map(grid => new BitmapLayer({{
        id: `insar-vlm-${{grid.id}}`,
        image: makeGridCanvas(grid, state.insarOpacity),
        bounds: grid.bounds,
        _imageCoordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: false,
        parameters: {{
          depthTest: false,
          depthMask: false
        }}
      }}));

      const footprintData = INSAR_GRIDS.map(grid => {{
        const [west, south, east, north] = grid.bounds;
        return {{
          id: grid.id,
          type: "insar_delta_extent",
          delta_name: grid.name,
          author_year: "Ohenhen et al. 2025",
          observation_period: "2014-2023",
          valid_pixel_count: grid.valid_count,
          polygon: [[west, south], [east, south], [east, north], [west, north]]
        }};
      }});

      return [
        ...rasterLayers,
        new PolygonLayer({{
          id: "insar-delta-footprints",
          data: footprintData,
          getPolygon: d => d.polygon,
          stroked: true,
          filled: true,
          getFillColor: [255, 255, 255, 1],
          getLineColor: [28, 38, 52, 225],
          lineWidthMinPixels: 1.4,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 55],
          parameters: {{
            depthTest: false,
            depthMask: false
          }}
        }})
      ];
    }}

    function makeGNSLayer() {{
      if (!state.showGNS) return null;
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.gns)) return null;

      return new ScatterplotLayer({{
        id: "gns-hamling-coastal-vlm",
        data: GNS_DATA,
        getPosition: d => [d.longitude, d.latitude, 1200],
        getRadius: d => d.point_radius_m * state.markerScale,
        radiusUnits: "meters",
        radiusMinPixels: 3,
        radiusMaxPixels: Math.max(5, 16 * state.markerScale),
        stroked: true,
        filled: true,
        lineWidthMinPixels: 0.7,
        getLineColor: [255, 255, 255, 190],
        getFillColor: d => colorForRecord(d, 255),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 170],
        parameters: {{
          depthTest: false,
          depthMask: false
        }},
        updateTriggers: {{
          getRadius: [state.markerScale],
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable]
        }}
      }});
    }}

    function makeOelsmannHybridLayer() {{
      if (!state.showOelsmannHybrid) return null;
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.hybrid)) return null;

      return new ScatterplotLayer({{
        id: "oelsmann-hybrid-coastal-vlm",
        data: OELSMANN_HYBRID_DATA,
        getPosition: d => [d.longitude, d.latitude, 1450],
        getRadius: d => d.point_radius_m * state.markerScale,
        radiusUnits: "meters",
        radiusMinPixels: 4,
        radiusMaxPixels: Math.max(7, 24 * state.markerScale),
        stroked: true,
        filled: true,
        lineWidthMinPixels: 0.8,
        getLineColor: [80, 80, 80, 210],
        getFillColor: d => colorForRecord(d, 255),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 180],
        parameters: {{
          depthTest: false,
          depthMask: false
        }},
        updateTriggers: {{
          getRadius: [state.markerScale],
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable]
        }}
      }});
    }}

    function makeTideGaugeLayer() {{
      if (!state.showTideGauge) return null;
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.tideGauge)) return null;

      return new ColumnLayer({{
        id: "tide-gauge-dangendorf-vlm",
        data: TIDE_GAUGE_DATA,
        getPosition: d => [d.longitude, d.latitude],
        getElevation: 1,
        elevationScale: 0,
        radius: Math.round(120000 * state.tideGaugeScale),
        diskResolution: 4,
        extruded: false,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 1,
        getLineColor: [28, 38, 52, 210],
        getFillColor: d => colorForRecord(d, 255),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 190],
        parameters: {{
          depthTest: false,
          depthMask: false
        }},
        updateTriggers: {{
          radius: [state.tideGaugeScale],
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable]
        }}
      }});
    }}

    function stationMatches(record) {{
      if (record.duration < state.minDuration) return false;
      if (record.first_epoch_year < state.minFirstEpoch) return false;
      if (record.last_epoch_year > state.maxLastEpoch) return false;
      if (state.search && !record.station.toLowerCase().includes(state.search)) return false;
      return true;
    }}

    function filteredPositiveData() {{
      if (!state.showGPS) return [];
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.gps)) return [];
      return POSITIVE_DATA.filter(stationMatches);
    }}

    function filteredNegativeData() {{
      if (!state.showGPS) return [];
      if (!datasetHasActiveVariable(RENDER_DATASET_IDS.gps)) return [];
      return NEGATIVE_DATA.filter(stationMatches);
    }}

    function makeBarLayer(id, data) {{
      return new ColumnLayer({{
        id,
        data,
        getPosition: d => [d.longitude, d.latitude],
        getElevation: d => d.bar_elevation,
        radius: Math.round(18000 * state.markerScale),
        elevationScale: 1,
        diskResolution: 8,
        getFillColor: d => colorForRecord(d, 220),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 170],
        updateTriggers: {{
          getPosition: [state.minDuration, state.minFirstEpoch, state.maxLastEpoch, state.search],
          getElevation: [state.minDuration, state.minFirstEpoch, state.maxLastEpoch, state.search],
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable],
          radius: [state.markerScale]
        }}
      }});
    }}

    function makePointLayer(id, data) {{
      return new ScatterplotLayer({{
        id,
        data,
        getPosition: d => [d.longitude, d.latitude, state.gpsAltitudeOffset],
        getRadius: d => d.point_radius_m * state.markerScale,
        radiusUnits: "meters",
        radiusMinPixels: 2,
        radiusMaxPixels: 13,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 1,
        getLineColor: [120, 130, 142, 220],
        getFillColor: d => colorForRecord(d, 220),
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 170],
        parameters: {{
          depthTest: true,
          depthMask: false
        }},
        updateTriggers: {{
          getPosition: [state.gpsAltitudeOffset],
          getRadius: [state.minDuration, state.minFirstEpoch, state.maxLastEpoch, state.search, state.markerScale],
          getFillColor: [state.colorLimit, state.uncertaintyLimit, state.renderVariable]
        }}
      }});
    }}

    function makeStationLayers(positiveData, negativeData) {{
      if (state.renderMode === "points") {{
        return [
          makePointLayer("gps-positive-points", positiveData),
          makePointLayer("gps-negative-points", negativeData)
        ];
      }}

      return [
        makeBarLayer("gps-positive-bars", positiveData),
        makeBarLayer("gps-negative-bars", negativeData)
      ];
    }}

    function makeLayers() {{
      const positiveData = filteredPositiveData();
      const negativeData = filteredNegativeData();
      return [
        ...makeGIALayers(),
        makeNglImagedLayer(),
        ...makeInSARLayers(),
        makePopulationLayer(),
        makeGNSLayer(),
        makeOelsmannHybridLayer(),
        makeTideGaugeLayer(),
        ...makeStationLayers(positiveData, negativeData),
        makeSelectionLayer()
      ].filter(Boolean);
    }}

    let pendingZoomFrame = null;
    const INITIAL_VIEW_STATE = {{
      longitude: -115,
      latitude: 28,
      zoom: 1.32,
      minZoom: 0.75,
      maxZoom: 20
    }};
    let currentViewState = {{...INITIAL_VIEW_STATE}};
    let map = null;
    let deckgl = null;

    function scheduleZoomScaleUpdate(viewState) {{
      const nextBucket = zoomBucketFor(viewState && viewState.zoom);
      if (nextBucket === state.zoomBucket) return;

      state.zoomBucket = nextBucket;
      state.markerScale = markerScaleForZoom(nextBucket);
      state.tideGaugeScale = tideGaugeScaleForZoom(nextBucket);
      state.gpsAltitudeOffset = gpsAltitudeOffsetForZoom(nextBucket);

      if (pendingZoomFrame !== null) return;
      pendingZoomFrame = window.requestAnimationFrame(() => {{
        pendingZoomFrame = null;
        updateLayers();
      }});
    }}

    function setGlobeViewState(nextViewState) {{
      currentViewState = {{
        ...currentViewState,
        ...nextViewState,
        zoom: Math.max(INITIAL_VIEW_STATE.minZoom, Math.min(INITIAL_VIEW_STATE.maxZoom, Number(nextViewState.zoom ?? currentViewState.zoom)))
      }};
      if (map) {{
        map.easeTo({{
          center: [currentViewState.longitude, currentViewState.latitude],
          zoom: currentViewState.zoom,
          duration: 120
        }});
      }}
      scheduleZoomScaleUpdate(currentViewState);
    }}

    function changeZoom(delta) {{
      setGlobeViewState({{
        zoom: Number(currentViewState.zoom || INITIAL_VIEW_STATE.zoom) + delta
      }});
    }}

    const deckOverlayProps = {{
      interleaved: false,
      layers: makeLayers(),
      onClick: (info) => {{
        if (selectionState.enabled) {{
          return false;
        }}
        if (info && info.object && info.object.station) {{
          openStationTimeSeries(info.object);
          return true;
        }}
        if (info && info.object && info.object.dataset_id === "tide_gauge_dangendorf_2026") {{
          openTideGaugeTimeSeries(info.object);
          return true;
        }}
        return false;
      }},
      parameters: {{
        cull: true
      }},
      getTooltip: ({{object}}) => object && object.station ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">${{object.station}}</div>
          <div><b>Version:</b> ${{object.version}}</div>
          <div><b>First epoch:</b> ${{object.first_epoch}}</div>
          <div><b>Last epoch:</b> ${{object.last_epoch}}</div>
          <div><b>Duration:</b> ${{formatNumber(object.duration, 2)}} yr</div>
          <div><b>UP velocity:</b> ${{formatNumber(object.up_mm_yr, 2)}} mm/yr</div>
          <div><b>UP uncertainty:</b> ${{formatNumber(object.up_sigma_mm_yr, 2)}} mm/yr</div>
          <div><b>Latitude:</b> ${{formatNumber(object.latitude, 5)}}</div>
          <div><b>Longitude:</b> ${{formatNumber(object.longitude, 5)}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : object && object.population ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">GHSL population</div>
          <div><b>Population:</b> ${{formatNumber(object.population, 1)}}</div>
          <div><b>Latitude:</b> ${{formatNumber(object.latitude, 3)}}</div>
          <div><b>Longitude:</b> ${{formatNumber(object.longitude, 3)}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : object && object.dataset_id === "tide_gauge_dangendorf_2026" ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">${{object.name}}</div>
          <div><b>Dataset:</b> CSL-TG tide-gauge VLM</div>
          <div><b>Linear trend:</b> ${{formatNumber(object.up_mm_yr, 2)}} mm/yr</div>
          <div><b>Trend uncertainty:</b> ${{formatNumber(object.up_sigma_mm_yr, 2)}} mm/yr</div>
          <div><b>Period:</b> ${{formatNumber(object.first_year, 0)}}-${{formatNumber(object.last_year, 0)}}</div>
          <div><b>Samples:</b> ${{object.sample_count ?? "N/A"}}</div>
          <div><b>Latitude:</b> ${{formatNumber(object.latitude, 5)}}</div>
          <div><b>Longitude:</b> ${{formatNumber(object.longitude, 5)}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : object && object.type === "insar_delta_extent" ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">${{object.delta_name}} Delta</div>
          <div><b>Dataset:</b> ${{object.author_year}}</div>
          <div><b>Observation period:</b> ${{object.observation_period}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : object && object.dataset_id === "hybrid_oelsmann_2026" ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">${{object.name || "Hybrid coastal VLM"}}</div>
          <div><b>Dataset:</b> Oelsmann hybrid coastal VLM</div>
          <div><b>Trend:</b> ${{formatNumber(object.up_mm_yr, 2)}} mm/yr</div>
          <div><b>Trend uncertainty:</b> ${{formatNumber(object.up_sigma_mm_yr, 2)}} mm/yr</div>
          <div><b>Variable:</b> OE24_GPS_InSAR_GIA</div>
          <div><b>Latitude:</b> ${{formatNumber(object.latitude, 5)}}</div>
          <div><b>Longitude:</b> ${{formatNumber(object.longitude, 5)}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : object && object.dataset ? {{
        html: `
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">${{object.dataset}}</div>
          <div><b>UP velocity:</b> ${{formatNumber(object.up_mm_yr, 2)}} mm/yr</div>
          <div><b>UP uncertainty:</b> ${{formatNumber(object.up_sigma_mm_yr, 2)}} mm/yr</div>
          <div><b>Observations:</b> ${{object.observations ?? "N/A"}}</div>
          <div><b>Quality factor:</b> ${{formatNumber(object.quality_factor, 2)}}</div>
          <div><b>Average radius:</b> ${{formatNumber(object.average_radius_km, 1)}} km</div>
          <div><b>Latitude:</b> ${{formatNumber(object.latitude, 5)}}</div>
          <div><b>Longitude:</b> ${{formatNumber(object.longitude, 5)}}</div>
        `,
        style: {{
          backgroundColor: "rgba(255,255,255,0.97)",
          color: "#1f2933",
          fontSize: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)"
        }}
      }} : null
    }};

    map = new maplibregl.Map({{
      container: "deck-container",
      style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
      center: [INITIAL_VIEW_STATE.longitude, INITIAL_VIEW_STATE.latitude],
      zoom: INITIAL_VIEW_STATE.zoom,
      minZoom: INITIAL_VIEW_STATE.minZoom,
      maxZoom: INITIAL_VIEW_STATE.maxZoom,
      bearing: 0,
      pitch: 0
    }});

    map.on("style.load", () => {{
      map.setProjection({{type: "globe"}});
      map.setFog({{
        color: "rgba(255,255,255,0.03)",
        "high-color": "#3a3a3a",
        "horizon-blend": 0.03,
        "space-color": "#2f2f2f",
        "star-intensity": 0
      }});
    }});

    map.on("click", event => {{
      if (!selectionState.enabled) return;
      selectionState.center = {{
        longitude: event.lngLat.lng,
        latitude: event.lngLat.lat
      }};
      computeSelectionRecords();
      updateSelectionStatus();
      updateLayers();
      openSelectionHistogram();
    }});

    map.on("move", () => {{
      const center = map.getCenter();
      currentViewState = {{
        ...currentViewState,
        longitude: center.lng,
        latitude: center.lat,
        zoom: map.getZoom()
      }};
      scheduleZoomScaleUpdate(currentViewState);
    }});

    map.once("load", () => {{
      deckgl = new MapboxOverlay(deckOverlayProps);
      map.addControl(deckgl);
      updateLayers();
    }});

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function formatTimePeriod(period) {{
      if (!period) return "Not applicable";
      if (typeof period === "string") return period;
      return `${{period.min}} to ${{period.max}}`;
    }}

    function stationPlotImageUrl(stationId) {{
      const safeStation = encodeURIComponent(String(stationId || "").trim().toUpperCase());
      return `https://geodesy.unr.edu/gps_timeseries/IGS20/tsplots/IGS20/TimeSeries/${{safeStation}}.png`;
    }}

    function stationPageUrl(stationId) {{
      const safeStation = encodeURIComponent(String(stationId || "").trim().toUpperCase());
      return `https://geodesy.unr.edu/NGLStationPages/stations/${{safeStation}}.sta`;
    }}

    function openStationTimeSeries(station) {{
      const modalElement = document.getElementById("stationTimeSeriesModal");
      const modal = bootstrap.Modal.getOrCreateInstance(modalElement);
      const title = document.getElementById("stationTimeSeriesTitle");
      const subtitle = document.getElementById("stationTimeSeriesSubtitle");
      const status = document.getElementById("stationTimeSeriesStatus");
      const image = document.getElementById("stationTimeSeriesImage");
      const sourceLink = document.getElementById("stationPlotLink");
      const stationPageLink = document.getElementById("stationPageLink");
      const inlinePlotLink = document.getElementById("stationInlinePlotLink");
      const url = stationPlotImageUrl(station.station);
      const pageUrl = stationPageUrl(station.station);

      title.textContent = `${{station.station}} GPS time-series plot`;
      subtitle.textContent = "UNR IGS20 station plot";
      document.getElementById("stationTrendStat").textContent = `${{formatNumber(station.up_mm_yr, 2)}} mm/yr`;
      document.getElementById("stationSigmaStat").textContent = `${{formatNumber(station.up_sigma_mm_yr, 2)}} mm/yr`;
      document.getElementById("stationDurationStat").textContent = `${{formatNumber(station.duration, 2)}} yr`;
      document.getElementById("stationSampleStat").textContent = "PNG";
      sourceLink.href = url;
      stationPageLink.href = pageUrl;
      inlinePlotLink.href = url;
      status.innerHTML = `Loading UNR-rendered station plot. The image is loaded only after clicking this station; no tenv3 table is fetched by this page.`;
      image.removeAttribute("src");
      image.alt = `${{station.station}} UNR GPS time-series plot`;
      image.onload = () => {{
        status.innerHTML = `Showing UNR station plot. MIDAS UP trend and uncertainty above come from the embedded velocity table.`;
      }};
      image.onerror = () => {{
        status.innerHTML = `Could not load the UNR plot image for this station. Try the station page or plot-image links below.`;
        image.removeAttribute("src");
      }};
      modal.show();
      image.src = url;
    }}

    let activeTideGauge = null;
    let tideGaugePlotElements = [];
    let tideGaugeHistogramElements = [];
    let tideGaugeHistogramState = {{
      xMin: null,
      xMax: null,
      dataMin: null,
      dataMax: null,
      dragging: false,
      dragStartX: 0,
      dragStartRange: null
    }};

    function rgba(color, alpha) {{
      return `rgba(${{color[0]}}, ${{color[1]}}, ${{color[2]}}, ${{alpha}})`;
    }}

    function colorForDistance(distanceKm) {{
      const stops = [
        [30, 125, 96],
        [237, 177, 65],
        [169, 74, 141]
      ];
      const t = Math.min(Math.max((Number(distanceKm) || 0) / 100, 0), 1);
      const scaled = t * (stops.length - 1);
      const index = Math.min(Math.floor(scaled), stops.length - 2);
      const localT = scaled - index;
      const a = stops[index];
      const b = stops[index + 1];
      return [
        lerp(a[0], b[0], localT),
        lerp(a[1], b[1], localT),
        lerp(a[2], b[2], localT)
      ];
    }}

    function drawTrendBand(ctx, xScale, yScale, xStart, xEnd, anchorYear, anchorValue, slope, sigma, color, alpha) {{
      if (!Number.isFinite(sigma) || sigma <= 0) return;
      const upperStart = anchorValue + (slope + sigma) * (xStart - anchorYear);
      const upperEnd = anchorValue + (slope + sigma) * (xEnd - anchorYear);
      const lowerEnd = anchorValue + (slope - sigma) * (xEnd - anchorYear);
      const lowerStart = anchorValue + (slope - sigma) * (xStart - anchorYear);
      ctx.fillStyle = rgba(color, alpha);
      ctx.beginPath();
      ctx.moveTo(xScale(xStart), yScale(upperStart));
      ctx.lineTo(xScale(xEnd), yScale(upperEnd));
      ctx.lineTo(xScale(xEnd), yScale(lowerEnd));
      ctx.lineTo(xScale(xStart), yScale(lowerStart));
      ctx.closePath();
      ctx.fill();
    }}

    function drawTrendSegment(ctx, xScale, yScale, xStart, xEnd, anchorYear, anchorValue, slope, color, alpha, dashed = false, width = 2) {{
      if (xEnd <= xStart) return null;
      const yStart = anchorValue + slope * (xStart - anchorYear);
      const yEnd = anchorValue + slope * (xEnd - anchorYear);
      ctx.strokeStyle = rgba(color, alpha);
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(xScale(xStart), yScale(yStart));
      ctx.lineTo(xScale(xEnd), yScale(yEnd));
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);
      return {{
        x1: xScale(xStart),
        y1: yScale(yStart),
        x2: xScale(xEnd),
        y2: yScale(yEnd),
        xStart,
        xEnd
      }};
    }}

    function distanceToSegment(px, py, element) {{
      const dx = element.x2 - element.x1;
      const dy = element.y2 - element.y1;
      if (dx === 0 && dy === 0) {{
        return Math.hypot(px - element.x1, py - element.y1);
      }}
      const t = Math.max(0, Math.min(1, ((px - element.x1) * dx + (py - element.y1) * dy) / (dx * dx + dy * dy)));
      const x = element.x1 + t * dx;
      const y = element.y1 + t * dy;
      return Math.hypot(px - x, py - y);
    }}

    function drawTideGaugePlot(gauge) {{
      const canvas = document.getElementById("tideGaugePlotCanvas");
      const wrapper = canvas.parentElement;
      const width = Math.max(520, wrapper.clientWidth || 620);
      const height = 360;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${{width}}px`;
      canvas.style.height = `${{height}}px`;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      tideGaugePlotElements = [];

      const series = (gauge.series || []).filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1]));
      if (series.length < 2) {{
        ctx.fillStyle = "#64748b";
        ctx.font = "13px Segoe UI, Arial, sans-serif";
        ctx.fillText("Not enough annual samples to plot this tide gauge.", 24, 40);
        return;
      }}

      const xValues = series.map(point => point[0]);
      const yValues = series.map(point => point[1]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const showGnss = document.getElementById("tideGaugeGnssToggle").checked;
      const nearbyGnss = showGnss ? (gauge.nearby_gnss || []) : [];
      const tideMeanYear = xValues.reduce((sum, value) => sum + value, 0) / xValues.length;
      const tideMeanTrendValue = gauge.trend_intercept_mm + gauge.up_mm_yr * tideMeanYear;
      const trendStart = gauge.trend_intercept_mm + gauge.up_mm_yr * xMin;
      const trendEnd = gauge.trend_intercept_mm + gauge.up_mm_yr * xMax;
      const plotValues = [...yValues, trendStart, trendEnd];
      if (Number.isFinite(gauge.up_sigma_mm_yr)) {{
        plotValues.push(tideMeanTrendValue + (gauge.up_mm_yr + gauge.up_sigma_mm_yr) * (xMin - tideMeanYear));
        plotValues.push(tideMeanTrendValue + (gauge.up_mm_yr - gauge.up_sigma_mm_yr) * (xMin - tideMeanYear));
        plotValues.push(tideMeanTrendValue + (gauge.up_mm_yr + gauge.up_sigma_mm_yr) * (xMax - tideMeanYear));
        plotValues.push(tideMeanTrendValue + (gauge.up_mm_yr - gauge.up_sigma_mm_yr) * (xMax - tideMeanYear));
      }}
      nearbyGnss.forEach(station => {{
        const anchorYear = Math.max(Math.min(station.first_year, xMax), xMin);
        const anchorValue = gauge.trend_intercept_mm + gauge.up_mm_yr * anchorYear;
        const yA = anchorValue + station.up_mm_yr * (xMin - anchorYear);
        const yB = anchorValue + station.up_mm_yr * (xMax - anchorYear);
        plotValues.push(yA, yB);
        if (Number.isFinite(station.up_sigma_mm_yr)) {{
          plotValues.push(anchorValue + (station.up_mm_yr + station.up_sigma_mm_yr) * (xMin - anchorYear));
          plotValues.push(anchorValue + (station.up_mm_yr - station.up_sigma_mm_yr) * (xMin - anchorYear));
          plotValues.push(anchorValue + (station.up_mm_yr + station.up_sigma_mm_yr) * (xMax - anchorYear));
          plotValues.push(anchorValue + (station.up_mm_yr - station.up_sigma_mm_yr) * (xMax - anchorYear));
        }}
      }});
      let yMin = Math.min(...plotValues);
      let yMax = Math.max(...plotValues);
      if (yMin === yMax) {{
        yMin -= 1;
        yMax += 1;
      }}
      const yPad = Math.max((yMax - yMin) * 0.08, 10);
      yMin -= yPad;
      yMax += yPad;

      const margin = {{left: 58, right: 18, top: 24, bottom: 46}};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotWidth;
      const yScale = value => margin.top + (1 - ((value - yMin) / Math.max(yMax - yMin, 1))) * plotHeight;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 4; i += 1) {{
        const y = margin.top + (plotHeight * i / 4);
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
      }}
      ctx.stroke();

      ctx.strokeStyle = "#475569";
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, height - margin.bottom);
      ctx.lineTo(width - margin.right, height - margin.bottom);
      ctx.stroke();

      const tideColor = [207, 50, 45];
      drawTrendBand(ctx, xScale, yScale, xMin, xMax, tideMeanYear, tideMeanTrendValue, gauge.up_mm_yr, gauge.up_sigma_mm_yr, tideColor, 0.14);

      nearbyGnss.forEach(station => {{
        const color = colorForDistance(station.distance_km);
        const stationStart = Math.max(station.first_year, xMin);
        const stationEnd = Math.min(station.last_year, xMax);
        const anchorYear = stationStart <= stationEnd ? stationStart : Math.max(Math.min(station.first_year, xMax), xMin);
        const anchorValue = gauge.trend_intercept_mm + gauge.up_mm_yr * anchorYear;
        drawTrendBand(ctx, xScale, yScale, xMin, xMax, anchorYear, anchorValue, station.up_mm_yr, station.up_sigma_mm_yr, color, 0.08);

        const leftSegment = drawTrendSegment(ctx, xScale, yScale, xMin, Math.min(stationStart, xMax), anchorYear, anchorValue, station.up_mm_yr, color, 0.28, true, 1.5);
        const solidSegment = drawTrendSegment(ctx, xScale, yScale, stationStart, stationEnd, anchorYear, anchorValue, station.up_mm_yr, color, 0.95, false, 2.4);
        const rightSegment = drawTrendSegment(ctx, xScale, yScale, Math.max(stationEnd, xMin), xMax, anchorYear, anchorValue, station.up_mm_yr, color, 0.28, true, 1.5);

        [leftSegment, solidSegment, rightSegment].filter(Boolean).forEach(segment => {{
          tideGaugePlotElements.push({{
            ...segment,
            type: "gnss",
            station
          }});
        }});
      }});

      ctx.strokeStyle = "rgba(36, 71, 111, 0.85)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      series.forEach((point, index) => {{
        const x = xScale(point[0]);
        const y = yScale(point[1]);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      ctx.strokeStyle = rgba(tideColor, 0.95);
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(xScale(xMin), yScale(trendStart));
      ctx.lineTo(xScale(xMax), yScale(trendEnd));
      ctx.stroke();
      ctx.setLineDash([]);
      tideGaugePlotElements.push({{
        type: "tideGauge",
        x1: xScale(xMin),
        y1: yScale(trendStart),
        x2: xScale(xMax),
        y2: yScale(trendEnd),
        gauge
      }});

      ctx.fillStyle = "#24476f";
      for (const point of series) {{
        ctx.beginPath();
        ctx.arc(xScale(point[0]), yScale(point[1]), 2, 0, Math.PI * 2);
        ctx.fill();
      }}

      ctx.fillStyle = "#334155";
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      ctx.textAlign = "center";
      for (let i = 0; i <= 4; i += 1) {{
        const year = xMin + (xMax - xMin) * i / 4;
        ctx.fillText(Math.round(year).toString(), xScale(year), height - 18);
      }}

      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i += 1) {{
        const value = yMin + (yMax - yMin) * (4 - i) / 4;
        ctx.fillText(value.toFixed(0), margin.left - 8, margin.top + (plotHeight * i / 4) + 4);
      }}

      ctx.save();
      ctx.translate(16, margin.top + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("CSL-TG residual (mm)", 0, 0);
      ctx.restore();

      ctx.textAlign = "left";
      ctx.fillStyle = "#1f2937";
      ctx.font = "12px Segoe UI, Arial, sans-serif";
      ctx.fillText(`CSL-TG trend: ${{formatNumber(gauge.up_mm_yr, 2)}} mm/yr`, margin.left, 18);
      ctx.strokeStyle = rgba(tideColor, 0.95);
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(margin.left + 188, 14);
      ctx.lineTo(margin.left + 232, 14);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#475569";
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      ctx.fillText("red dashed = CSL-TG fitted trend", margin.left + 240, 18);

      if (nearbyGnss.length) {{
        const legendX = width - margin.right - 132;
        const legendY = 36;
        const gradient = ctx.createLinearGradient(legendX, legendY + 10, legendX + 110, legendY + 10);
        gradient.addColorStop(0, "rgb(30,125,96)");
        gradient.addColorStop(0.5, "rgb(237,177,65)");
        gradient.addColorStop(1, "rgb(169,74,141)");
        ctx.fillStyle = gradient;
        ctx.fillRect(legendX, legendY + 4, 110, 8);
        ctx.fillStyle = "#334155";
        ctx.font = "10px Segoe UI, Arial, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("0 km", legendX, legendY + 25);
        ctx.textAlign = "right";
        ctx.fillText("100 km", legendX + 110, legendY + 25);
      }}
    }}

    function getTideGaugeHistogramTrends(gauge) {{
      if (!gauge) return [];
      const trends = [];
      if (Number.isFinite(gauge.up_mm_yr)) {{
        trends.push({{
          type: "tideGauge",
          label: gauge.name || "CSL-TG",
          value: gauge.up_mm_yr,
          sigma: gauge.up_sigma_mm_yr,
          color: [207, 50, 45],
          subtitle: "CSL-TG trend"
        }});
      }}

      const showGnss = document.getElementById("tideGaugeGnssToggle").checked;
      if (showGnss) {{
        (gauge.nearby_gnss || []).forEach(station => {{
          if (!Number.isFinite(station.up_mm_yr)) return;
          trends.push({{
            type: "gnss",
            label: station.station,
            value: station.up_mm_yr,
            sigma: station.up_sigma_mm_yr,
            color: colorForDistance(station.distance_km),
            subtitle: `${{formatNumber(station.distance_km, 1)}} km from tide gauge`,
            station
          }});
        }});
      }}
      return trends;
    }}

    function initializeTideGaugeHistogramRange(gauge, force = false) {{
      const trends = getTideGaugeHistogramTrends(gauge);
      const values = trends
        .flatMap(trend => {{
          const sigma = Number.isFinite(trend.sigma) ? Math.max(Math.abs(trend.sigma), 0) : 0;
          return [trend.value - sigma, trend.value, trend.value + sigma];
        }})
        .filter(Number.isFinite);

      if (!values.length) {{
        tideGaugeHistogramState.dataMin = -1;
        tideGaugeHistogramState.dataMax = 1;
        tideGaugeHistogramState.xMin = -1;
        tideGaugeHistogramState.xMax = 1;
        return;
      }}

      let dataMin = Math.min(...values);
      let dataMax = Math.max(...values);
      if (dataMin === dataMax) {{
        dataMin -= 1;
        dataMax += 1;
      }}
      const pad = Math.max((dataMax - dataMin) * 0.18, 0.5);
      dataMin -= pad;
      dataMax += pad;
      tideGaugeHistogramState.dataMin = dataMin;
      tideGaugeHistogramState.dataMax = dataMax;

      if (force || !Number.isFinite(tideGaugeHistogramState.xMin) || !Number.isFinite(tideGaugeHistogramState.xMax)) {{
        tideGaugeHistogramState.xMin = dataMin;
        tideGaugeHistogramState.xMax = dataMax;
      }} else {{
        tideGaugeHistogramState.xMin = Math.max(dataMin, tideGaugeHistogramState.xMin);
        tideGaugeHistogramState.xMax = Math.min(dataMax, tideGaugeHistogramState.xMax);
        if (tideGaugeHistogramState.xMax <= tideGaugeHistogramState.xMin) {{
          tideGaugeHistogramState.xMin = dataMin;
          tideGaugeHistogramState.xMax = dataMax;
        }}
      }}
    }}

    function panTideGaugeHistogram(delta) {{
      const fullMin = tideGaugeHistogramState.dataMin;
      const fullMax = tideGaugeHistogramState.dataMax;
      let xMin = tideGaugeHistogramState.xMin + delta;
      let xMax = tideGaugeHistogramState.xMax + delta;
      const span = xMax - xMin;
      const fullSpan = fullMax - fullMin;
      if (span >= fullSpan) {{
        tideGaugeHistogramState.xMin = fullMin;
        tideGaugeHistogramState.xMax = fullMax;
        return;
      }}
      if (xMin < fullMin) {{
        xMin = fullMin;
        xMax = fullMin + span;
      }}
      if (xMax > fullMax) {{
        xMax = fullMax;
        xMin = fullMax - span;
      }}
      tideGaugeHistogramState.xMin = xMin;
      tideGaugeHistogramState.xMax = xMax;
    }}

    function zoomTideGaugeHistogram(factor, anchorRatio) {{
      const fullMin = tideGaugeHistogramState.dataMin;
      const fullMax = tideGaugeHistogramState.dataMax;
      const fullSpan = fullMax - fullMin;
      const currentMin = tideGaugeHistogramState.xMin;
      const currentMax = tideGaugeHistogramState.xMax;
      const currentSpan = currentMax - currentMin;
      const minSpan = Math.max(fullSpan * 0.04, 0.05);
      let newSpan = Math.max(minSpan, Math.min(fullSpan, currentSpan * factor));
      if (newSpan >= fullSpan * 0.995) {{
        tideGaugeHistogramState.xMin = fullMin;
        tideGaugeHistogramState.xMax = fullMax;
        return;
      }}
      const anchorValue = currentMin + currentSpan * anchorRatio;
      let xMin = anchorValue - newSpan * anchorRatio;
      let xMax = xMin + newSpan;
      if (xMin < fullMin) {{
        xMin = fullMin;
        xMax = fullMin + newSpan;
      }}
      if (xMax > fullMax) {{
        xMax = fullMax;
        xMin = fullMax - newSpan;
      }}
      tideGaugeHistogramState.xMin = xMin;
      tideGaugeHistogramState.xMax = xMax;
    }}

    function drawTideGaugeHistogram(gauge) {{
      const canvas = document.getElementById("tideGaugeHistogramCanvas");
      if (!canvas) return;
      const wrapper = canvas.parentElement;
      const width = Math.max(520, wrapper.clientWidth || 620);
      const height = 170;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${{width}}px`;
      canvas.style.height = `${{height}}px`;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      tideGaugeHistogramElements = [];
      initializeTideGaugeHistogramRange(gauge);

      const trends = getTideGaugeHistogramTrends(gauge);
      const gnssTrends = trends.filter(trend => trend.type === "gnss");
      const tideTrend = trends.find(trend => trend.type === "tideGauge");
      const xMin = tideGaugeHistogramState.xMin;
      const xMax = tideGaugeHistogramState.xMax;
      const margin = {{left: 50, right: 18, top: 18, bottom: 48}};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1e-6)) * plotWidth;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      const binCount = Math.max(6, Math.min(28, Math.round(plotWidth / 34)));
      const binWidth = (xMax - xMin) / binCount;
      const bins = Array.from({{length: binCount}}, (_, index) => ({{
        count: 0,
        start: xMin + index * binWidth,
        end: xMin + (index + 1) * binWidth,
        members: []
      }}));

      gnssTrends.forEach(trend => {{
        if (trend.value < xMin || trend.value > xMax) return;
        const index = Math.min(binCount - 1, Math.max(0, Math.floor((trend.value - xMin) / Math.max(binWidth, 1e-6))));
        bins[index].count += 1;
        bins[index].members.push(trend);
      }});

      const yMax = Math.max(1, ...bins.map(bin => bin.count));
      const yTickValues = Array.from(
        new Set(
          Array.from({{length: Math.min(yMax, 4) + 1}}, (_, index) =>
            Math.round((yMax * index) / Math.min(yMax, 4))
          )
        )
      ).filter(value => value >= 0 && value <= yMax);

      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      yTickValues.forEach(value => {{
        const y = margin.top + plotHeight * (1 - value / yMax);
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
      }});
      ctx.stroke();

      bins.forEach(bin => {{
        const x0 = xScale(bin.start);
        const x1 = xScale(bin.end);
        const barWidth = Math.max(1, x1 - x0 - 2);
        const barHeight = (bin.count / yMax) * plotHeight;
        const y = margin.top + plotHeight - barHeight;
        ctx.fillStyle = bin.count ? "rgba(74, 117, 162, 0.72)" : "rgba(226, 232, 240, 0.35)";
        ctx.fillRect(x0 + 1, y, barWidth, barHeight);
        if (bin.count) {{
          tideGaugeHistogramElements.push({{
            type: "histBin",
            x1: x0,
            x2: x1,
            y1: y,
            y2: margin.top + plotHeight,
            bin
          }});
        }}
      }});

      gnssTrends.forEach(trend => {{
        if (trend.value < xMin || trend.value > xMax) return;
        const x = xScale(trend.value);
        const color = trend.color;
        ctx.strokeStyle = rgba(color, 0.9);
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(x, margin.top + plotHeight + 1);
        ctx.lineTo(x, margin.top + plotHeight + 10);
        ctx.stroke();
        tideGaugeHistogramElements.push({{
          type: "gnssTrend",
          x,
          y1: margin.top + plotHeight + 1,
          y2: margin.top + plotHeight + 13,
          trend
        }});
      }});

      if (tideTrend && tideTrend.value >= xMin && tideTrend.value <= xMax) {{
        const x = xScale(tideTrend.value);
        ctx.strokeStyle = rgba(tideTrend.color, 0.98);
        ctx.lineWidth = 2.2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(x, margin.top - 2);
        ctx.lineTo(x, margin.top + plotHeight + 12);
        ctx.stroke();
        ctx.setLineDash([]);
        tideGaugeHistogramElements.push({{
          type: "tideGaugeTrend",
          x,
          y1: margin.top - 2,
          y2: margin.top + plotHeight + 12,
          trend: tideTrend
        }});
      }}

      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, margin.top + plotHeight);
      ctx.lineTo(width - margin.right, margin.top + plotHeight);
      ctx.stroke();

      ctx.fillStyle = "#334155";
      ctx.font = "10px Segoe UI, Arial, sans-serif";
      ctx.textAlign = "center";
      for (let i = 0; i <= 4; i += 1) {{
        const value = xMin + (xMax - xMin) * i / 4;
        ctx.fillText(formatNumber(value, 1), xScale(value), height - 25);
      }}

      ctx.textAlign = "right";
      yTickValues.forEach(value => {{
        const y = margin.top + plotHeight * (1 - value / yMax);
        ctx.fillText(value.toString(), margin.left - 7, y + 4);
      }});

      ctx.textAlign = "left";
      ctx.fillStyle = "#475569";
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      ctx.fillText("GNSS trend count", margin.left, 12);
      ctx.textAlign = "right";
      ctx.fillText("mm/yr", width - margin.right, height - 7);

      const rangeLabel = document.getElementById("tideGaugeHistogramRange");
      if (rangeLabel) {{
        const full = Math.abs((xMax - xMin) - (tideGaugeHistogramState.dataMax - tideGaugeHistogramState.dataMin)) < 0.01;
        rangeLabel.textContent = full
          ? "Wheel to zoom, drag to pan"
          : `${{formatNumber(xMin, 2)}} to ${{formatNumber(xMax, 2)}} mm/yr`;
      }}
    }}

    function renderNearbyGnssList(gauge) {{
      const list = document.getElementById("tideGaugeGnssList");
      const stations = gauge.nearby_gnss || [];
      if (!stations.length) {{
        list.innerHTML = `
          <div class="nearby-gnss-list-header">
            <span>Nearby GPS stations</span>
            <span>None within 100 km</span>
          </div>
        `;
        return;
      }}

      const rows = stations.map(station => {{
        const color = colorForDistance(station.distance_km);
        const swatch = `rgb(${{color[0]}}, ${{color[1]}}, ${{color[2]}})`;
        return `
          <div class="nearby-gnss-row">
            <span class="nearby-gnss-swatch" style="background:${{swatch}}"></span>
            <a href="${{stationPageUrl(station.station)}}" target="_blank" rel="noreferrer">${{escapeHtml(station.station)}}</a>
            <span>${{formatNumber(station.up_mm_yr, 2)}} +/- ${{formatNumber(station.up_sigma_mm_yr, 2)}} mm/yr</span>
            <span>${{formatNumber(station.distance_km, 1)}} km</span>
          </div>
        `;
      }}).join("");

      list.innerHTML = `
        <div class="nearby-gnss-list-header">
          <span>Nearby GPS stations</span>
          <span>closest ${{stations.length}} within 100 km</span>
        </div>
        ${{rows}}
      `;
    }}

    function openTideGaugeTimeSeries(gauge) {{
      activeTideGauge = gauge;
      const modalElement = document.getElementById("tideGaugeModal");
      const modal = bootstrap.Modal.getOrCreateInstance(modalElement);
      const nearbyGnss = gauge.nearby_gnss || [];
      document.getElementById("tideGaugeTitle").textContent = `${{gauge.name}} tide-gauge VLM`;
      document.getElementById("tideGaugeSubtitle").textContent = `PSMSL ID ${{gauge.psmsl_id ?? "N/A"}} | MDadj3 adjusted CSL-TG residual`;
      document.getElementById("tideGaugeTrendStat").textContent = `${{formatNumber(gauge.up_mm_yr, 2)}} mm/yr`;
      document.getElementById("tideGaugeSigmaStat").textContent = `${{formatNumber(gauge.up_sigma_mm_yr, 2)}} mm/yr`;
      document.getElementById("tideGaugePeriodStat").textContent = `${{formatNumber(gauge.first_year, 0)}}-${{formatNumber(gauge.last_year, 0)}}`;
      document.getElementById("tideGaugeSampleStat").textContent = `${{gauge.sample_count ?? "N/A"}}`;
      document.getElementById("tideGaugeGnssSummary").textContent =
        `${{nearbyGnss.length}} nearby GNSS station${{nearbyGnss.length === 1 ? "" : "s"}} within 100 km`;
      document.getElementById("tideGaugeGnssToggle").checked = true;
      initializeTideGaugeHistogramRange(gauge, true);
      renderNearbyGnssList(gauge);
      document.getElementById("tideGaugeSourceLink").href = TIDE_GAUGE_METADATA.record_url || "{TIDE_GAUGE_RECORD_URL}";
      document.getElementById("tideGaugeStatus").innerHTML =
        `The red dashed line is the fitted CSL-TG trend. GNSS lines are anchored to the tide-gauge fitted line at each station's first MIDAS epoch; dashed segments are extrapolated outside the station period.`;
      modal.show();
      window.setTimeout(() => {{
        drawTideGaugePlot(gauge);
        drawTideGaugeHistogram(gauge);
      }}, 80);
    }}

    function setupTideGaugePlotInteractions() {{
      const canvas = document.getElementById("tideGaugePlotCanvas");
      const histogramCanvas = document.getElementById("tideGaugeHistogramCanvas");
      const tooltip = document.getElementById("tideGaugePlotTooltip");
      const toggle = document.getElementById("tideGaugeGnssToggle");
      const histogramReset = document.getElementById("tideGaugeHistogramReset");

      toggle.addEventListener("change", () => {{
        if (activeTideGauge) {{
          drawTideGaugePlot(activeTideGauge);
          initializeTideGaugeHistogramRange(activeTideGauge, true);
          drawTideGaugeHistogram(activeTideGauge);
        }}
      }});

      canvas.addEventListener("mousemove", event => {{
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        let nearest = null;
        let nearestDistance = Infinity;

        for (const element of tideGaugePlotElements) {{
          const distance = distanceToSegment(x, y, element);
          if (distance < nearestDistance) {{
            nearest = element;
            nearestDistance = distance;
          }}
        }}

        if (!nearest || nearestDistance > 10) {{
          tooltip.style.display = "none";
          return;
        }}

        if (nearest.type === "gnss") {{
          const station = nearest.station;
          tooltip.innerHTML = `
            <div class="fw-bold">${{escapeHtml(station.station)}} GNSS</div>
            <div>Trend: ${{formatNumber(station.up_mm_yr, 2)}} +/- ${{formatNumber(station.up_sigma_mm_yr, 2)}} mm/yr</div>
            <div>Period: ${{formatNumber(station.first_year, 1)}}-${{formatNumber(station.last_year, 1)}}</div>
            <div>Distance: ${{formatNumber(station.distance_km, 1)}} km</div>
          `;
        }} else {{
          tooltip.innerHTML = `
            <div class="fw-bold">${{escapeHtml(nearest.gauge.name)}} tide gauge</div>
            <div>Trend: ${{formatNumber(nearest.gauge.up_mm_yr, 2)}} +/- ${{formatNumber(nearest.gauge.up_sigma_mm_yr, 2)}} mm/yr</div>
            <div>Period: ${{formatNumber(nearest.gauge.first_year, 0)}}-${{formatNumber(nearest.gauge.last_year, 0)}}</div>
          `;
        }}

        tooltip.style.left = `${{event.clientX + 14}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
        tooltip.style.display = "block";
      }});

      canvas.addEventListener("mouseleave", () => {{
        tooltip.style.display = "none";
      }});

      histogramReset.addEventListener("click", () => {{
        if (!activeTideGauge) return;
        initializeTideGaugeHistogramRange(activeTideGauge, true);
        drawTideGaugeHistogram(activeTideGauge);
      }});

      histogramCanvas.addEventListener("wheel", event => {{
        if (!activeTideGauge) return;
        event.preventDefault();
        const rect = histogramCanvas.getBoundingClientRect();
        const marginLeft = 50;
        const marginRight = 18;
        const plotWidth = Math.max(1, rect.width - marginLeft - marginRight);
        const anchorRatio = Math.max(0, Math.min(1, (event.clientX - rect.left - marginLeft) / plotWidth));
        const span = tideGaugeHistogramState.xMax - tideGaugeHistogramState.xMin;

        if (event.shiftKey) {{
          const panPixels = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
          panTideGaugeHistogram((panPixels / plotWidth) * span);
        }} else {{
          zoomTideGaugeHistogram(Math.exp(event.deltaY * 0.0015), anchorRatio);
        }}
        drawTideGaugeHistogram(activeTideGauge);
      }}, {{passive: false}});

      histogramCanvas.addEventListener("mousedown", event => {{
        if (!activeTideGauge) return;
        tideGaugeHistogramState.dragging = true;
        tideGaugeHistogramState.dragStartX = event.clientX;
        tideGaugeHistogramState.dragStartRange = [
          tideGaugeHistogramState.xMin,
          tideGaugeHistogramState.xMax
        ];
      }});

      window.addEventListener("mousemove", event => {{
        if (!tideGaugeHistogramState.dragging || !activeTideGauge) return;
        const rect = histogramCanvas.getBoundingClientRect();
        const plotWidth = Math.max(1, rect.width - 50 - 18);
        const [startMin, startMax] = tideGaugeHistogramState.dragStartRange || [
          tideGaugeHistogramState.xMin,
          tideGaugeHistogramState.xMax
        ];
        const span = startMax - startMin;
        tideGaugeHistogramState.xMin = startMin;
        tideGaugeHistogramState.xMax = startMax;
        panTideGaugeHistogram(-((event.clientX - tideGaugeHistogramState.dragStartX) / plotWidth) * span);
        drawTideGaugeHistogram(activeTideGauge);
      }});

      window.addEventListener("mouseup", () => {{
        tideGaugeHistogramState.dragging = false;
      }});

      histogramCanvas.addEventListener("mousemove", event => {{
        if (tideGaugeHistogramState.dragging) return;
        const rect = histogramCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        let nearest = null;
        let nearestDistance = Infinity;

        for (const element of tideGaugeHistogramElements) {{
          let distance = Infinity;
          if (element.type === "histBin") {{
            const inside = x >= element.x1 && x <= element.x2 && y >= element.y1 && y <= element.y2;
            distance = inside ? 0 : Infinity;
          }} else {{
            distance = Math.abs(x - element.x);
            if (y < element.y1 - 5 || y > element.y2 + 5) distance = Infinity;
          }}
          if (distance < nearestDistance) {{
            nearest = element;
            nearestDistance = distance;
          }}
        }}

        if (!nearest || nearestDistance > 7) {{
          tooltip.style.display = "none";
          return;
        }}

        if (nearest.type === "histBin") {{
          const names = nearest.bin.members.slice(0, 5).map(member => escapeHtml(member.label)).join(", ");
          const overflow = nearest.bin.members.length > 5 ? ` +${{nearest.bin.members.length - 5}} more` : "";
          tooltip.innerHTML = `
            <div class="fw-bold">GNSS trend bin</div>
            <div>${{formatNumber(nearest.bin.start, 2)}} to ${{formatNumber(nearest.bin.end, 2)}} mm/yr</div>
            <div>${{nearest.bin.count}} station${{nearest.bin.count === 1 ? "" : "s"}}</div>
            <div class="text-secondary">${{names}}${{overflow}}</div>
          `;
        }} else {{
          const trend = nearest.trend;
          tooltip.innerHTML = `
            <div class="fw-bold">${{escapeHtml(trend.label)}} ${{trend.type === "gnss" ? "GNSS" : "tide gauge"}}</div>
            <div>Trend: ${{formatNumber(trend.value, 2)}} +/- ${{formatNumber(trend.sigma, 2)}} mm/yr</div>
            <div>${{escapeHtml(trend.subtitle || "")}}</div>
          `;
        }}

        tooltip.style.left = `${{event.clientX + 14}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
        tooltip.style.display = "block";
      }});

      histogramCanvas.addEventListener("mouseleave", () => {{
        if (!tideGaugeHistogramState.dragging) tooltip.style.display = "none";
      }});
    }}

    function metadataRow(label, value, copyValue = null) {{
      const copyButton = copyValue ? `
        <button class="btn btn-outline-secondary btn-sm copy-metadata-btn" type="button" data-copy-value="${{escapeHtml(copyValue)}}" aria-label="Copy ${{escapeHtml(label)}}">
          <i class="bi bi-copy"></i>
        </button>
      ` : "";

      return `
        <div class="metadata-row">
          <div class="metadata-label">${{escapeHtml(label)}}</div>
          <div class="metadata-value">${{escapeHtml(value)}}</div>
          <div>${{copyButton}}</div>
        </div>
      `;
    }}

    function datasetInfoHtml(datasetId) {{
      const attrs = DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return "";

      const description = (attrs.short_description || [])
        .map(sentence => `<p class="mb-2">${{escapeHtml(sentence)}}</p>`)
        .join("");

      return `
        <div>
          ${{metadataRow("Authors", attrs.authors)}}
          ${{metadataRow("Citation", attrs.citation)}}
          ${{metadataRow("DOI", attrs.doi || "N/A", attrs.doi || "")}}
          ${{metadataRow("Original file", attrs.original_file_url, attrs.original_file_url)}}
          ${{metadataRow("Year", attrs.publication_year)}}
          ${{metadataRow("Time period", formatTimePeriod(attrs.time_period_covered))}}
          <div class="mt-3">${{description}}</div>
        </div>
      `;
    }}

    function externalDatasetInfoHtml(datasetId) {{
      const attrs = EXTERNAL_DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return "";

      return `
        <div>
          ${{metadataRow("Main author", attrs.main_author)}}
          ${{metadataRow("Year", attrs.year)}}
          ${{metadataRow("DOI", attrs.doi || "N/A", attrs.doi || "")}}
          ${{metadataRow("Website", attrs.website_link, attrs.website_link)}}
          ${{metadataRow("Original file", attrs.original_file_url, attrs.original_file_url)}}
        </div>
      `;
    }}

    function copyToClipboard(value, button) {{
      const done = () => {{
        const icon = button.querySelector("i");
        if (!icon) return;
        icon.className = "bi bi-check2";
        window.setTimeout(() => {{
          icon.className = "bi bi-copy";
        }}, 950);
      }};

      if (navigator.clipboard && window.isSecureContext) {{
        navigator.clipboard.writeText(value).then(done).catch(() => {{}});
      }} else {{
        const textarea = document.createElement("textarea");
        textarea.value = value;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {{
          document.execCommand("copy");
          done();
        }} finally {{
          textarea.remove();
        }}
      }}
    }}

    function openDatasetInfo(datasetId) {{
      const attrs = DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return;

      document.getElementById("datasetInfoTitle").textContent = `${{attrs.label}} dataset information`;
      document.getElementById("datasetInfoBody").innerHTML = datasetInfoHtml(datasetId);
      document.querySelectorAll("#datasetInfoBody [data-copy-value]").forEach(button => {{
        button.addEventListener("click", () => {{
          copyToClipboard(button.getAttribute("data-copy-value"), button);
        }});
      }});

      bootstrap.Modal.getOrCreateInstance(document.getElementById("datasetInfoModal")).show();
    }}

    function openExternalDatasetInfo(datasetId) {{
      const attrs = EXTERNAL_DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return;

      document.getElementById("datasetInfoTitle").textContent = `${{attrs.label}} external dataset`;
      document.getElementById("datasetInfoBody").innerHTML = externalDatasetInfoHtml(datasetId);
      document.querySelectorAll("#datasetInfoBody [data-copy-value]").forEach(button => {{
        button.addEventListener("click", () => {{
          copyToClipboard(button.getAttribute("data-copy-value"), button);
        }});
      }});

      bootstrap.Modal.getOrCreateInstance(document.getElementById("datasetInfoModal")).show();
    }}

    function downloadDatasetHtml(attrs, isExternal = false) {{
      const citation = attrs.citation || [
        attrs.main_author,
        attrs.year ? `(${{attrs.year}})` : "",
        attrs.doi || ""
      ].filter(Boolean).join(" ");
      const url = attrs.original_file_url || attrs.website_link || "";
      const doi = attrs.doi || "";

      return `
        <div>
          <p class="mb-2">
            You are about to leave this viewer and download the original source file from the data provider.
          </p>
          <div class="alert alert-warning py-2 small mb-3">
            Please cite the original datasource and associated paper or DOI when using these data.
            This viewer is only a visualization and preprocessing layer.
          </div>
          ${{metadataRow("Dataset", attrs.display_name || attrs.label || attrs.main_author || "Dataset")}}
          ${{metadataRow(isExternal ? "Main author" : "Citation", citation || "N/A")}}
          ${{metadataRow("DOI", doi || "N/A", doi || "")}}
          ${{metadataRow("Original file", url || "N/A", url || "")}}
        </div>
      `;
    }}

    function openDatasetDownload(datasetId) {{
      const attrs = DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return;
      const url = attrs.original_file_url || attrs.metadata_url || "";

      document.getElementById("datasetDownloadTitle").textContent = `${{attrs.label}} original download`;
      document.getElementById("datasetDownloadBody").innerHTML = downloadDatasetHtml(attrs, false);
      document.getElementById("datasetDownloadLink").href = url || "#";
      document.getElementById("datasetDownloadLink").classList.toggle("disabled", !url);
      document.querySelectorAll("#datasetDownloadBody [data-copy-value]").forEach(button => {{
        button.addEventListener("click", () => {{
          copyToClipboard(button.getAttribute("data-copy-value"), button);
        }});
      }});

      bootstrap.Modal.getOrCreateInstance(document.getElementById("datasetDownloadModal")).show();
    }}

    function openExternalDatasetDownload(datasetId) {{
      const attrs = EXTERNAL_DATASET_ATTRIBUTES[datasetId];
      if (!attrs) return;
      const url = attrs.original_file_url || attrs.website_link || "";

      document.getElementById("datasetDownloadTitle").textContent = `${{attrs.label}} original download`;
      document.getElementById("datasetDownloadBody").innerHTML = downloadDatasetHtml(attrs, true);
      document.getElementById("datasetDownloadLink").href = url || "#";
      document.getElementById("datasetDownloadLink").classList.toggle("disabled", !url);
      document.querySelectorAll("#datasetDownloadBody [data-copy-value]").forEach(button => {{
        button.addEventListener("click", () => {{
          copyToClipboard(button.getAttribute("data-copy-value"), button);
        }});
      }});

      bootstrap.Modal.getOrCreateInstance(document.getElementById("datasetDownloadModal")).show();
    }}

    function initDatasetInfoButtons() {{
      document.querySelectorAll("[data-dataset-info]").forEach(button => {{
        button.addEventListener("click", () => {{
          openDatasetInfo(button.getAttribute("data-dataset-info"));
        }});
      }});
      document.querySelectorAll("[data-external-info]").forEach(button => {{
        button.addEventListener("click", () => {{
          openExternalDatasetInfo(button.getAttribute("data-external-info"));
        }});
      }});
      document.querySelectorAll("[data-dataset-download]").forEach(button => {{
        button.addEventListener("click", () => {{
          openDatasetDownload(button.getAttribute("data-dataset-download"));
        }});
      }});
      document.querySelectorAll("[data-external-download]").forEach(button => {{
        button.addEventListener("click", () => {{
          openExternalDatasetDownload(button.getAttribute("data-external-download"));
        }});
      }});
    }}

    function syncRenderVariableControls() {{
      const uncertaintyMode = state.renderVariable === "uncertainty";
      const colorSlider = document.getElementById("color-scale-slider");
      const activeRange = uncertaintyMode ? state.uncertaintyLimit : state.colorLimit;
      colorSlider.min = uncertaintyMode ? "0.5" : "1";
      colorSlider.max = uncertaintyMode ? "20" : "100";
      colorSlider.step = "0.5";
      colorSlider.value = activeRange;
      document.getElementById("color-scale-label-text").textContent = uncertaintyMode ? "Uncertainty color range" : "Trend color range";
      document.getElementById("color-scale-prefix").textContent = uncertaintyMode ? "0-" : "+/-";
      document.querySelectorAll("[data-layer-row]").forEach(row => {{
        const datasetId = row.getAttribute("data-layer-row");
        const available = datasetHasActiveVariable(datasetId);
        row.classList.toggle("unavailable", !available);
        const toggle = row.querySelector("[data-dataset-toggle]");
        if (toggle) {{
          toggle.disabled = !available;
          toggle.title = available ? "" : "No uncertainty payload available";
        }}
      }});
      document.getElementById("legend-title").textContent = uncertaintyMode
        ? "Shared uncertainty color scale (mm/yr)"
        : "Shared VLM color scale (mm/yr)";
      document.getElementById("vlm-legend-ramp").classList.toggle("uncertainty-ramp", uncertaintyMode);
      document.getElementById("legend-mid").textContent = uncertaintyMode ? "" : "0";
    }}

    function updateStats(positiveData, negativeData) {{
      const positiveShown = positiveData.length;
      const negativeShown = negativeData.length;
      const colorScaleLabel = state.colorLimit.toFixed(1);
      const uncertaintyScaleLabel = state.uncertaintyLimit.toFixed(1);
      const activeColorScaleLabel = state.renderVariable === "uncertainty" ? uncertaintyScaleLabel : colorScaleLabel;
      const durationLabel = state.minDuration.toFixed(1);
      const firstEpochLabel = state.minFirstEpoch.toFixed(1);
      const lastEpochLabel = state.maxLastEpoch.toFixed(1);
      const modeLabel = state.renderMode === "bars" ? "Bars" : "Points";

      document.getElementById("stat-total").textContent = ALL_DATA.length.toLocaleString();
      document.getElementById("stat-positive").textContent = positiveShown.toLocaleString();
      document.getElementById("stat-negative").textContent = negativeShown.toLocaleString();
      document.getElementById("stat-mode").textContent = modeLabel;
      document.getElementById("stat-color-scale").textContent = activeColorScaleLabel;
      document.getElementById("stat-duration").textContent = durationLabel;
      document.getElementById("stat-first-epoch").textContent = firstEpochLabel;
      document.getElementById("stat-last-epoch").textContent = lastEpochLabel;
      document.getElementById("color-scale-value").textContent = activeColorScaleLabel;
      document.getElementById("gia-opacity-value").textContent = Math.round(state.giaOpacity * 100).toString();
      document.getElementById("insar-opacity-value").textContent = Math.round(state.insarOpacity * 100).toString();
      document.getElementById("population-opacity-value").textContent = Math.round(state.populationOpacity * 100).toString();
      document.getElementById("duration-value").textContent = durationLabel;
      document.getElementById("first-epoch-value").textContent = firstEpochLabel;
      document.getElementById("last-epoch-value").textContent = lastEpochLabel;
      document.getElementById("search-state").textContent = state.search ? state.search.toUpperCase() : "All";
      const variableLabel = state.renderVariable === "uncertainty" ? "Uncertainty" : modeLabel;
      document.getElementById("navbar-summary").textContent =
        `${{(positiveShown + negativeShown).toLocaleString()}} shown - ${{variableLabel}} - color ${{state.renderVariable === "uncertainty" ? `0-${{uncertaintyScaleLabel}}` : `+/- ${{colorScaleLabel}}`}} mm/yr`;
      if (state.renderVariable === "uncertainty") {{
        document.getElementById("legend-min").textContent = "0";
        document.getElementById("legend-max").textContent = uncertaintyScaleLabel;
      }} else {{
        document.getElementById("legend-min").textContent = `-${{colorScaleLabel}}`;
        document.getElementById("legend-max").textContent = `+${{colorScaleLabel}}`;
      }}
      document.getElementById("populationLegend").hidden = !state.showPopulation;
      document.getElementById("population-legend-min").textContent = formatNumber(POPULATION_METADATA.min_population, 0);
      document.getElementById("population-legend-mid").textContent = `p95 ${{formatNumber(POPULATION_METADATA.p95_population, 0)}}`;
      document.getElementById("population-legend-max").textContent = `p99+ ${{formatNumber(POPULATION_METADATA.p99_population, 0)}}`;
    }}

    function updateLayers() {{
      const positiveData = filteredPositiveData();
      const negativeData = filteredNegativeData();
      syncRenderVariableControls();
      loadActiveRenderPayloads();
      if (deckgl) {{
        deckgl.setProps({{
          layers: [
          ...makeGIALayers(),
          makeNglImagedLayer(),
          ...makeInSARLayers(),
          makePopulationLayer(),
          makeGNSLayer(),
          makeOelsmannHybridLayer(),
          makeTideGaugeLayer(),
          ...makeStationLayers(positiveData, negativeData),
          makeSelectionLayer()
          ].filter(Boolean)
        }});
      }}
      updateStats(positiveData, negativeData);
    }}

    function setMode(mode) {{
      state.renderMode = mode;
      updateLayers();
    }}

    function setRenderVariable(variable) {{
      state.renderVariable = variable;
      if (variable === "uncertainty") {{
        loadUncertaintyPayloads().then(() => {{
          computeSelectionRecords();
          updateLayers();
          if (document.getElementById("selectionHistogramModal").classList.contains("show")) {{
            renderSelectionDatasetToggles();
            drawSelectionHistogram();
          }}
        }});
      }} else {{
        computeSelectionRecords();
        updateLayers();
        if (document.getElementById("selectionHistogramModal").classList.contains("show")) {{
          renderSelectionDatasetToggles();
          drawSelectionHistogram();
        }}
      }}
    }}

    function syncSelectionToolControl() {{
      const button = document.getElementById("selection-toggle");
      const popover = document.getElementById("selector-tool-popover");
      button.classList.toggle("active", selectionState.enabled);
      button.setAttribute("aria-pressed", selectionState.enabled ? "true" : "false");
      popover.hidden = !selectionState.enabled;
    }}

    document.getElementById("selection-toggle").addEventListener("click", () => {{
      document.getElementById("selector-tool-hint").hidden = true;
      selectionState.enabled = !selectionState.enabled;
      if (map) {{
        map.getCanvas().style.cursor = selectionState.enabled ? "crosshair" : "";
      }}
      syncSelectionToolControl();
      updateSelectionStatus();
    }});

    document.getElementById("selection-radius-slider").addEventListener("input", event => {{
      selectionState.radiusKm = Number(event.target.value);
      if (selectionState.center) {{
        computeSelectionRecords();
        updateLayers();
        if (document.getElementById("selectionHistogramModal").classList.contains("show")) {{
          renderSelectionDatasetToggles();
          drawSelectionHistogram();
        }}
      }}
      updateSelectionStatus();
    }});

    document.getElementById("selection-open-histogram").addEventListener("click", openSelectionHistogram);
    document.getElementById("selectionHistogramModal").addEventListener("shown.bs.modal", drawSelectionHistogram);
    document.getElementById("selectionDownloadCsv").addEventListener("click", downloadSelectionCsv);
    document.getElementById("selectionHistogramCanvas").addEventListener("mousemove", event => {{
      const tooltip = document.getElementById("selectionHistogramTooltip");
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = selectionHistogramBars.slice().reverse().find(bar =>
        x >= bar.x && x <= bar.x + bar.width && y >= bar.y && y <= bar.y + bar.height
      );
      if (!hit) {{
        tooltip.style.display = "none";
        return;
      }}
      tooltip.innerHTML = `
        <strong>${{hit.dataset}}</strong><br>
        ${{hit.count.toLocaleString()}} records<br>
        ${{formatNumber(hit.xMin, 1)}} to ${{formatNumber(hit.xMax, 1)}} mm/yr<br>
        <span class="text-secondary">${{hit.axis}}</span>
      `;
      const parentRect = tooltip.parentElement.getBoundingClientRect();
      tooltip.style.left = `${{Math.min(parentRect.width - 238, Math.max(8, event.clientX - parentRect.left + 12))}}px`;
      tooltip.style.top = `${{Math.min(parentRect.height - 96, Math.max(8, event.clientY - parentRect.top + 12))}}px`;
      tooltip.style.display = "block";
    }});
    document.getElementById("selectionHistogramCanvas").addEventListener("mouseleave", () => {{
      document.getElementById("selectionHistogramTooltip").style.display = "none";
    }});
    syncSelectionToolControl();

    document.getElementById("toggle-gps").addEventListener("change", event => {{
      state.showGPS = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-ngl-imaged").addEventListener("change", event => {{
      state.showNglImaged = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-gia").addEventListener("change", event => {{
      state.showGIA = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-insar").addEventListener("change", event => {{
      state.showInSAR = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-gns").addEventListener("change", event => {{
      state.showGNS = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-tide-gauge").addEventListener("change", event => {{
      state.showTideGauge = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-oelsmann-hybrid").addEventListener("change", event => {{
      state.showOelsmannHybrid = event.target.checked;
      updateLayers();
    }});

    document.getElementById("toggle-population").addEventListener("change", event => {{
      state.showPopulation = event.target.checked;
      if (state.showPopulation) {{
        loadPopulationDataset().then(() => updateLayers());
      }}
      updateLayers();
    }});

    document.getElementById("showcaseDisclaimerClose").addEventListener("click", () => {{
      document.getElementById("showcaseDisclaimer").hidden = true;
    }});

    function collapseTechniqueGroupsOnSmallScreens() {{
      if (!window.matchMedia("(max-width: 640px)").matches || !window.bootstrap) return;
      document.querySelectorAll("#techniqueLayerGroups .accordion-collapse.show").forEach(element => {{
        bootstrap.Collapse.getOrCreateInstance(element, {{toggle: false}}).hide();
      }});
    }}
    collapseTechniqueGroupsOnSmallScreens();

    function syncLayerPanelResponsiveState() {{
      if (!window.bootstrap) return;
      const panel = document.getElementById("layerPanel");
      const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(panel, {{backdrop: false, scroll: true}});
      if (window.matchMedia("(max-width: 640px)").matches) {{
        offcanvas.hide();
      }} else {{
        offcanvas.show();
      }}
    }}
    syncLayerPanelResponsiveState();

    const vlmLegend = document.getElementById("vlmLegend");
    const vlmLegendToggle = document.getElementById("vlmLegendToggle");
    function setVlmLegendCollapsed(collapsed) {{
      vlmLegend.classList.toggle("collapsed", collapsed);
      vlmLegendToggle.setAttribute("aria-expanded", String(!collapsed));
      const icon = vlmLegendToggle.querySelector("i");
      icon.className = collapsed ? "bi bi-palette-fill" : "bi bi-chevron-down";
    }}
    if (vlmLegend && vlmLegendToggle) {{
      setVlmLegendCollapsed(window.matchMedia("(max-width: 640px)").matches);
      vlmLegendToggle.addEventListener("click", () => {{
        setVlmLegendCollapsed(!vlmLegend.classList.contains("collapsed"));
      }});
    }}

    document.querySelectorAll("input[name='render-mode']").forEach(input => {{
      input.addEventListener("change", event => {{
        if (event.target.checked) setMode(event.target.value);
      }});
    }});

    document.querySelectorAll("input[name='render-variable']").forEach(input => {{
      input.addEventListener("change", event => {{
        if (event.target.checked) setRenderVariable(event.target.value);
      }});
    }});

    const colorScaleSlider = document.getElementById("color-scale-slider");
    colorScaleSlider.value = state.colorLimit;
    colorScaleSlider.addEventListener("input", event => {{
      if (state.renderVariable === "uncertainty") {{
        state.uncertaintyLimit = Number(event.target.value);
      }} else {{
        state.colorLimit = Number(event.target.value);
      }}
      updateLayers();
    }});

    const giaOpacitySlider = document.getElementById("gia-opacity-slider");
    giaOpacitySlider.value = Math.round(state.giaOpacity * 100);
    giaOpacitySlider.addEventListener("input", event => {{
      state.giaOpacity = Number(event.target.value) / 100;
      updateLayers();
    }});

    const insarOpacitySlider = document.getElementById("insar-opacity-slider");
    insarOpacitySlider.value = Math.round(state.insarOpacity * 100);
    insarOpacitySlider.addEventListener("input", event => {{
      state.insarOpacity = Number(event.target.value) / 100;
      updateLayers();
    }});

    const populationOpacitySlider = document.getElementById("population-opacity-slider");
    populationOpacitySlider.value = Math.round(state.populationOpacity * 100);
    populationOpacitySlider.addEventListener("input", event => {{
      state.populationOpacity = Number(event.target.value) / 100;
      updateLayers();
    }});

    document.getElementById("zoom-in").addEventListener("click", () => {{
      changeZoom(0.5);
    }});

    document.getElementById("zoom-out").addEventListener("click", () => {{
      changeZoom(-0.5);
    }});

    document.getElementById("duration-slider").addEventListener("input", event => {{
      state.minDuration = Number(event.target.value);
      updateLayers();
    }});

    const firstEpochSlider = document.getElementById("first-epoch-slider");
    firstEpochSlider.min = METADATA.first_epoch_min;
    firstEpochSlider.max = METADATA.first_epoch_max;
    firstEpochSlider.value = state.minFirstEpoch;
    firstEpochSlider.addEventListener("input", event => {{
      state.minFirstEpoch = Number(event.target.value);
      updateLayers();
    }});

    const lastEpochSlider = document.getElementById("last-epoch-slider");
    lastEpochSlider.min = METADATA.last_epoch_min;
    lastEpochSlider.max = METADATA.last_epoch_max;
    lastEpochSlider.value = state.maxLastEpoch;
    lastEpochSlider.addEventListener("input", event => {{
      state.maxLastEpoch = Number(event.target.value);
      updateLayers();
    }});

    document.getElementById("station-search").addEventListener("input", event => {{
      state.search = event.target.value.trim().toLowerCase();
      updateLayers();
    }});

    function showStartupDisclaimerOnce() {{
      const key = "global-vlm-startup-disclaimer-seen-v1";
      let alreadySeen = false;
      try {{
        alreadySeen = window.localStorage.getItem(key) === "true";
      }} catch (error) {{
        alreadySeen = false;
      }}
      if (alreadySeen) return;
      const modalElement = document.getElementById("startupDisclaimerModal");
      if (!modalElement) return;
      bootstrap.Modal.getOrCreateInstance(modalElement).show();
      modalElement.addEventListener("hidden.bs.modal", () => {{
        try {{
          window.localStorage.setItem(key, "true");
        }} catch (error) {{}}
      }}, {{once: true}});
    }}

    initDatasetInfoButtons();
    setupTideGaugePlotInteractions();
    showStartupDisclaimerOnce();
    updateStats(filteredPositiveData(), filteredNegativeData());
  </script>
  <script data-goatcounter="https://global-vlm.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""


def build_catalog_html(dataset_attributes: dict[str, dict]) -> str:
    datasets = sorted(
        dataset_attributes.values(),
        key=lambda item: (item.get("technique", ""), item.get("display_name", "")),
    )
    years = []
    for dataset in datasets:
        period = dataset.get("time_period_covered")
        if isinstance(period, dict):
            for key in ("min", "max"):
                value = period.get(key)
                if isinstance(value, (int, float)):
                    years.append(float(value))

    year_min = math.floor(min(years)) if years else 1900
    year_max = math.ceil(max(years)) if years else datetime.now(timezone.utc).year

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Global Vertical Land Motion Catalogue</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg">

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <style>
    :root {
      color-scheme: light;
      --nav-height: 58px;
      --line: #d7dde5;
      --ink: #1c2430;
      --muted: #687386;
      --panel: #ffffff;
      --soft: #f5f7fa;
      --accent: #2f6fed;
    }

    html,
    body {
      min-height: 100%;
      background: #eef2f6;
      color: var(--ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .app-navbar {
      height: var(--nav-height);
      background: rgba(255, 255, 255, 0.98);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 12px rgba(20, 31, 49, 0.08);
    }

    .brand-title {
      color: #1d2733;
      font-size: 15px;
      font-weight: 800;
      line-height: 1;
      white-space: nowrap;
    }

    .brand-title a {
      color: inherit;
      text-decoration: none;
    }
    .brand-short {
      display: none;
    }
    .nav-link {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      height: 36px;
      padding: 0 4px;
      color: #344559;
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
      line-height: 1;
    }
    .nav-link:hover,
    .nav-link:focus {
      color: #0f5ca8;
      text-decoration: none;
    }

    .showcase-disclaimer {
      position: fixed;
      top: calc(var(--nav-height) + 10px);
      left: 50%;
      transform: translateX(-50%);
      z-index: 1080;
      width: min(940px, calc(100vw - 32px));
      padding: 8px 12px;
      border: 1px solid #d8c88e;
      border-radius: 8px;
      background: rgba(255, 248, 218, 0.98);
      color: #5a4a16;
      box-shadow: 0 8px 22px rgba(20, 32, 46, 0.12);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.35;
      text-align: center;
    }

    .brand-title a:hover {
      color: var(--accent);
      text-decoration: underline;
    }

    main {
      padding-top: calc(var(--nav-height) + 74px);
      padding-bottom: 28px;
    }

    .catalog-shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 18px;
    }

    .filter-panel,
    .table-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(20, 31, 49, 0.07);
    }

    .filter-panel {
      padding: 16px;
    }

    .filter-pair-box {
      height: 100%;
      padding: 12px;
      border: 1px solid #d7dde8;
      border-radius: 8px;
      background: #f8fafc;
    }

    .filter-pair-title {
      margin-bottom: 10px;
      color: #344559;
      font-size: 12px;
      font-weight: 850;
      letter-spacing: .02em;
      text-transform: uppercase;
    }

    label {
      color: #536074;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      margin-bottom: 5px;
    }

    .form-control,
    .form-select {
      border-color: #cfd6e1;
      font-size: 13px;
    }

    .time-readout {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
    }

    .table-panel {
      overflow: hidden;
    }

    .table {
      margin-bottom: 0;
      font-size: 13px;
      vertical-align: middle;
    }

    .table thead th {
      background: #f6f8fb;
      border-bottom: 1px solid var(--line);
      color: #4c586b;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
      white-space: nowrap;
    }

    .dataset-title {
      font-weight: 750;
      line-height: 1.2;
    }

    .dataset-id {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      border: 1px solid #d4dae5;
      border-radius: 999px;
      background: #f8fafc;
      color: #354156;
      font-size: 12px;
      font-weight: 650;
      line-height: 1;
      padding: 5px 8px;
      white-space: nowrap;
    }

    .catalog-modal .modal-dialog {
      max-width: min(1040px, calc(100vw - 32px));
    }

    .catalog-modal .modal-body {
      max-height: calc(100vh - 210px);
      overflow: auto;
    }

    .attr-grid {
      display: grid;
      grid-template-columns: minmax(190px, 0.35fr) minmax(0, 1fr);
      gap: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .attr-key,
    .attr-value {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      font-size: 13px;
    }

    .attr-key {
      background: #f7f9fc;
      color: #4f5c70;
      font-weight: 750;
      overflow-wrap: anywhere;
    }

    .attr-value {
      background: #fff;
      color: #1e2938;
      overflow-wrap: anywhere;
    }

    .attr-key:last-of-type,
    .attr-value:last-of-type {
      border-bottom: 0;
    }

    .nested-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    .nested-table td {
      border-bottom: 1px solid #e7ebf1;
      padding: 5px 6px;
      vertical-align: top;
    }

    .nested-table tr:last-child td {
      border-bottom: 0;
    }

    .copy-btn {
      margin-left: 8px;
      padding: 2px 7px;
      font-size: 11px;
    }

    @media (max-width: 760px) {
      html {
        font-size: 13px;
      }

      .brand-long {
        display: inline;
      }

      .brand-short {
        display: none;
      }

      .app-navbar {
        padding-left: 9px !important;
        padding-right: 9px !important;
      }

      .nav-link {
        padding-left: 8px;
        padding-right: 8px;
      }

      .nav-link span {
        display: none !important;
      }

      main {
        padding-top: 118px;
      }

      .catalog-shell {
        padding: 0 10px;
      }

      .attr-grid {
        grid-template-columns: 1fr;
      }

      .attr-key {
        border-bottom: 0;
        padding-bottom: 4px;
      }

      .attr-value {
        padding-top: 4px;
      }
    }
  </style>
</head>
<body>
  <nav class="navbar app-navbar fixed-top px-3">
    <div class="container-fluid px-0">
      <div class="brand-title me-auto"><a href="index.html"><span class="brand-long">Global Vertical Land Motion</span></a></div>
      <a class="nav-link ms-3" href="index.html">
        <i class="bi bi-globe-americas"></i>
        <span class="d-none d-sm-inline">Map</span>
      </a>
      <a class="nav-link ms-3 active" href="catalogue.html">
        <i class="bi bi-table"></i>
        <span class="d-none d-sm-inline">Catalogue</span>
      </a>
      <a class="nav-link ms-3" href="compare.html">
        <i class="bi bi-columns-gap"></i>
        <span class="d-none d-sm-inline">Compare</span>
      </a>
      <a class="nav-link ms-3" href="about.html">
        <i class="bi bi-info-circle"></i>
        <span class="d-none d-sm-inline">About</span>
      </a>
    </div>
  </nav>

  <main>
    <div class="catalog-shell">
      <div class="showcase-disclaimer">
        Preliminary showcase: this site is largely AI-generated and intended to motivate real community development, review, and shared stewardship of VLM data. Catalogue metadata and classifications were assigned with AI support and have not been reviewed. Please cite original data sources, DOIs, and associated papers when using any dataset.
      </div>

      <section class="filter-panel mb-3">
        <div class="row g-3 align-items-end">
          <div class="col-12 col-lg-3">
            <label for="search">Search</label>
            <input id="search" class="form-control" type="search" placeholder="Dataset, author, DOI, description">
          </div>
          <div class="col-6 col-lg-2">
            <label for="technique-filter">Technique</label>
            <select id="technique-filter" class="form-select"></select>
          </div>
          <div class="col-6 col-lg-2">
            <label for="type-filter">Measurement type</label>
            <select id="type-filter" class="form-select"></select>
          </div>
          <div class="col-6 col-lg-2">
            <label for="frame-filter">Reference frame</label>
            <select id="frame-filter" class="form-select"></select>
          </div>
          <div class="col-6 col-lg-3">
            <label for="benchmark-filter">Benchmark dependence</label>
            <select id="benchmark-filter" class="form-select"></select>
          </div>

          <div class="col-12 col-lg-5">
            <div class="filter-pair-box">
              <div class="filter-pair-title">Observation sensitivity</div>
              <div class="row g-2">
                <div class="col-7">
                  <label for="sensitivity-field-filter">Field</label>
                  <select id="sensitivity-field-filter" class="form-select">
                    <option value="">Any field</option>
                    <option value="observes_land_surface_directly">Observes land surface directly</option>
                    <option value="observes_infrastructure_motion">Observes infrastructure motion</option>
                    <option value="observes_geocentric_land_motion">Observes geocentric land motion</option>
                    <option value="sensitive_to_shallow_vlm">Sensitive to shallow VLM</option>
                    <option value="sensitive_to_deep_vlm">Sensitive to deep VLM</option>
                    <option value="sensitive_to_vertical_accretion">Sensitive to vertical accretion</option>
                  </select>
                </div>
                <div class="col-5">
                  <label for="sensitivity-value-filter">Value</label>
                  <select id="sensitivity-value-filter" class="form-select">
                    <option value="all">Any value</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                    <option value="uncertain">uncertain</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
          <div class="col-12 col-lg-7">
            <div class="filter-pair-box">
              <div class="filter-pair-title">Process influence</div>
              <div class="row g-2">
                <div class="col-8">
                  <label for="process-field-filter">Field</label>
                  <select id="process-field-filter" class="form-select"></select>
                </div>
                <div class="col-4">
                  <label for="process-value-filter">Value</label>
                  <select id="process-value-filter" class="form-select">
                    <option value="all">Any value</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                    <option value="uncertain">uncertain</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <div class="col-12 col-lg-6">
            <div class="d-flex justify-content-between align-items-center">
              <label class="mb-1">Time period overlap</label>
              <span class="time-readout" id="time-readout"></span>
            </div>
            <div class="row g-2">
              <div class="col-6">
                <input id="year-min-filter" class="form-range" type="range" min="__YEAR_MIN__" max="__YEAR_MAX__" value="__YEAR_MIN__" step="1">
              </div>
              <div class="col-6">
                <input id="year-max-filter" class="form-range" type="range" min="__YEAR_MIN__" max="__YEAR_MAX__" value="__YEAR_MAX__" step="1">
              </div>
            </div>
          </div>
          <div class="col-12 col-lg-6 d-flex justify-content-lg-end align-items-center gap-2">
            <span class="text-secondary small fw-semibold" id="result-count"></span>
            <button id="reset-filters" class="btn btn-outline-secondary btn-sm" type="button">
              <i class="bi bi-arrow-counterclockwise"></i>
              Reset
            </button>
          </div>
        </div>
      </section>

      <section class="table-panel">
        <div class="table-responsive">
          <table class="table table-hover align-middle">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Technique</th>
                <th>Type</th>
                <th>Reference frame</th>
                <th>Time period</th>
                <th>Confidence</th>
                <th class="text-end">Details</th>
              </tr>
            </thead>
            <tbody id="catalog-table-body"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>

  <div class="modal fade catalog-modal" id="datasetModal" tabindex="-1" aria-labelledby="datasetModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
          <div>
            <h5 class="modal-title" id="datasetModalLabel">Dataset</h5>
            <div class="text-secondary small" id="datasetModalSubtitle"></div>
          </div>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" id="datasetModalBody"></div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">
            Close
          </button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal fade catalog-modal" id="downloadModal" tabindex="-1" aria-labelledby="downloadModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <div>
            <h5 class="modal-title" id="downloadModalLabel">Download original dataset</h5>
            <div class="text-secondary small" id="downloadModalSubtitle"></div>
          </div>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" id="downloadModalBody"></div>
        <div class="modal-footer">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal">Cancel</button>
          <a class="btn btn-primary btn-sm" id="downloadModalLink" href="#" target="_blank" rel="noreferrer">
            <i class="bi bi-download"></i>
            Download from source
          </a>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATASETS = __DATASETS_JSON__;
    const PROCESS_FIELDS = __PROCESS_FIELDS_JSON__;
    const YEAR_MIN = __YEAR_MIN__;
    const YEAR_MAX = __YEAR_MAX__;

    const modal = new bootstrap.Modal(document.getElementById("datasetModal"));
    const downloadModal = new bootstrap.Modal(document.getElementById("downloadModal"));

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function titleCaseKey(key) {
      return key
        .replace(/^influenced_by_/, "process: ")
        .replace(/_/g, " ")
        .replace(/\b\w/g, char => char.toUpperCase());
    }

    function formatPeriod(period) {
      if (!period || typeof period !== "object") return "not applicable";
      const min = period.min ?? "unknown";
      const max = period.max ?? "unknown";
      return `${min} - ${max}`;
    }

    function formatCoverage(coverage) {
      if (!coverage || typeof coverage !== "object") return "unknown";
      const parts = ["min_lon", "max_lon", "min_lat", "max_lat"].map(key => {
        const value = Number(coverage[key]);
        return Number.isFinite(value) ? value.toFixed(2) : "unknown";
      });
      return `lon ${parts[0]} to ${parts[1]}, lat ${parts[2]} to ${parts[3]}`;
    }

    function copyButton(value) {
      return `<button class="btn btn-outline-secondary btn-sm copy-btn" type="button" data-copy="${escapeHtml(value)}"><i class="bi bi-copy"></i> Copy</button>`;
    }

    function renderPrimitive(key, value) {
      if (value === true || value === false) return String(value);
      if (value === null || value === undefined || value === "") return "not available";
      const text = String(value);
      if (key === "doi" || key === "associated_article_doi") {
        const href = `https://doi.org/${encodeURIComponent(text)}`;
        return `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>${copyButton(text)}`;
      }
      if (key === "original_file_url" || /^https?:\/\//.test(text)) {
        return `<a href="${escapeHtml(text)}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>${copyButton(text)}`;
      }
      return escapeHtml(text);
    }

    function renderValue(key, value) {
      if (Array.isArray(value)) {
        return `<ul class="mb-0 ps-3">${value.map(item => `<li>${renderValue(key, item)}</li>`).join("")}</ul>`;
      }
      if (value && typeof value === "object") {
        return `<table class="nested-table"><tbody>${Object.entries(value).map(([nestedKey, nestedValue]) => `
          <tr>
            <td class="text-secondary fw-semibold">${escapeHtml(titleCaseKey(nestedKey))}</td>
            <td>${renderValue(nestedKey, nestedValue)}</td>
          </tr>
        `).join("")}</tbody></table>`;
      }
      return renderPrimitive(key, value);
    }

    function populateSelect(id, values, firstLabel) {
      const select = document.getElementById(id);
      select.innerHTML = `<option value="">${firstLabel}</option>` + values
        .filter(value => value !== undefined && value !== null && value !== "")
        .map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
        .join("");
    }

    function populateFilters() {
      populateSelect("technique-filter", [...new Set(DATASETS.map(d => d.technique))].sort(), "All techniques");
      populateSelect("type-filter", [...new Set(DATASETS.map(d => d.vlm_measurement_type))].sort(), "All types");
      populateSelect("frame-filter", [...new Set(DATASETS.map(d => d.reference_frame))].sort(), "All frames");
      populateSelect("benchmark-filter", [...new Set(DATASETS.map(d => d.benchmark_dependence))].sort(), "All levels");

      const processSelect = document.getElementById("process-field-filter");
      processSelect.innerHTML = `<option value="">Any process</option>` + PROCESS_FIELDS
        .map(field => `<option value="${escapeHtml(field)}">${escapeHtml(titleCaseKey(field))}</option>`)
        .join("");
    }

    function normalizedText(dataset) {
      return JSON.stringify(dataset).toLowerCase();
    }

    function datasetIntersectsTime(dataset, minYear, maxYear) {
      const period = dataset.time_period_covered;
      if (!period || typeof period !== "object") return true;
      const start = Number(period.min);
      const end = Number(period.max);
      if (!Number.isFinite(start) || !Number.isFinite(end)) return true;
      return end >= minYear && start <= maxYear;
    }

    function currentFilters() {
      let minYear = Number(document.getElementById("year-min-filter").value);
      let maxYear = Number(document.getElementById("year-max-filter").value);
      if (minYear > maxYear) [minYear, maxYear] = [maxYear, minYear];

      return {
        search: document.getElementById("search").value.trim().toLowerCase(),
        technique: document.getElementById("technique-filter").value,
        type: document.getElementById("type-filter").value,
        frame: document.getElementById("frame-filter").value,
        benchmark: document.getElementById("benchmark-filter").value,
        sensitivityField: document.getElementById("sensitivity-field-filter").value,
        sensitivityValue: document.getElementById("sensitivity-value-filter").value,
        processField: document.getElementById("process-field-filter").value,
        processValue: document.getElementById("process-value-filter").value,
        minYear,
        maxYear
      };
    }

    function filteredDatasets() {
      const filters = currentFilters();
      document.getElementById("time-readout").textContent = `${filters.minYear} - ${filters.maxYear}`;

      return DATASETS.filter(dataset => {
        if (filters.search && !normalizedText(dataset).includes(filters.search)) return false;
        if (filters.technique && dataset.technique !== filters.technique) return false;
        if (filters.type && dataset.vlm_measurement_type !== filters.type) return false;
        if (filters.frame && dataset.reference_frame !== filters.frame) return false;
        if (filters.benchmark && dataset.benchmark_dependence !== filters.benchmark) return false;
        if (filters.sensitivityField && filters.sensitivityValue !== "all" && dataset[filters.sensitivityField] !== filters.sensitivityValue) return false;
        if (filters.processField && filters.processValue !== "all" && dataset[filters.processField] !== filters.processValue) return false;
        return datasetIntersectsTime(dataset, filters.minYear, filters.maxYear);
      });
    }

    function renderTable() {
      const rows = filteredDatasets();
      const body = document.getElementById("catalog-table-body");
      document.getElementById("result-count").textContent = `${rows.length} of ${DATASETS.length} datasets`;

      if (!rows.length) {
        body.innerHTML = `<tr><td colspan="7" class="text-center text-secondary py-4">No datasets match the current filters.</td></tr>`;
        return;
      }

      body.innerHTML = rows.map(dataset => `
        <tr>
          <td>
            <div class="dataset-title">${escapeHtml(dataset.display_name || dataset.label || dataset.id)}</div>
            <div class="dataset-id">${escapeHtml(dataset.id)}</div>
          </td>
          <td><span class="pill">${escapeHtml(dataset.technique || "unknown")}</span></td>
          <td>${escapeHtml(dataset.vlm_measurement_type || "unknown")}</td>
          <td>${escapeHtml(dataset.reference_frame || "unknown")}</td>
          <td>${escapeHtml(formatPeriod(dataset.time_period_covered))}</td>
          <td>${escapeHtml(dataset.classification_confidence || "unknown")}</td>
          <td class="text-end">
            <div class="d-inline-flex align-items-center justify-content-end gap-2">
              <button class="btn btn-outline-secondary btn-sm" type="button" data-download-dataset="${escapeHtml(dataset.id)}" aria-label="Download original source for ${escapeHtml(dataset.display_name || dataset.label || dataset.id)}">
                <i class="bi bi-download"></i>
              </button>
              <button class="btn btn-outline-primary btn-sm" type="button" data-open-dataset="${escapeHtml(dataset.id)}">
                <i class="bi bi-window"></i>
                Details
              </button>
            </div>
          </td>
        </tr>
      `).join("");

      document.querySelectorAll("[data-open-dataset]").forEach(button => {
        button.addEventListener("click", () => openDataset(button.getAttribute("data-open-dataset")));
      });
      document.querySelectorAll("[data-download-dataset]").forEach(button => {
        button.addEventListener("click", () => openDownload(button.getAttribute("data-download-dataset")));
      });
    }

    function openDataset(datasetId) {
      const dataset = DATASETS.find(item => item.id === datasetId);
      if (!dataset) return;

      document.getElementById("datasetModalLabel").textContent = dataset.display_name || dataset.label || dataset.id;
      document.getElementById("datasetModalSubtitle").textContent = `${dataset.technique || "unknown technique"} | ${formatPeriod(dataset.time_period_covered)} | ${formatCoverage(dataset.coverage)}`;
      document.getElementById("datasetModalBody").innerHTML = `
        <div class="attr-grid">
          ${Object.entries(dataset).map(([key, value]) => `
            <div class="attr-key">${escapeHtml(titleCaseKey(key))}</div>
            <div class="attr-value">${renderValue(key, value)}</div>
          `).join("")}
        </div>
      `;

      document.querySelectorAll("[data-copy]").forEach(button => {
        button.addEventListener("click", async () => {
          const text = button.getAttribute("data-copy") || "";
          try {
            await navigator.clipboard.writeText(text);
            button.innerHTML = `<i class="bi bi-check2"></i> Copied`;
            setTimeout(() => { button.innerHTML = `<i class="bi bi-copy"></i> Copy`; }, 1200);
          } catch (error) {
            button.textContent = "Copy failed";
          }
        });
      });

      modal.show();
    }

    function openDownload(datasetId) {
      const dataset = DATASETS.find(item => item.id === datasetId);
      if (!dataset) return;

      const url = dataset.original_file_url || dataset.metadata_url || "";
      document.getElementById("downloadModalLabel").textContent = dataset.display_name || dataset.label || dataset.id;
      document.getElementById("downloadModalSubtitle").textContent = "Original source download";
      document.getElementById("downloadModalBody").innerHTML = `
        <p class="mb-2">
          You are about to leave this catalogue and download the original source file from the data provider.
        </p>
        <div class="alert alert-warning py-2 small mb-3">
          Please cite the original datasource and associated paper or DOI when using these data.
          This catalogue is only a metadata and visualization companion.
        </div>
        <div class="attr-grid">
          <div class="attr-key">Citation</div>
          <div class="attr-value">${renderValue("citation", dataset.citation || "not available")}</div>
          <div class="attr-key">DOI</div>
          <div class="attr-value">${renderValue("doi", dataset.doi || "not available")}</div>
          <div class="attr-key">Original File</div>
          <div class="attr-value">${renderValue("original_file_url", url || "not available")}</div>
        </div>
      `;
      const link = document.getElementById("downloadModalLink");
      link.href = url || "#";
      link.classList.toggle("disabled", !url);

      document.querySelectorAll("#downloadModalBody [data-copy]").forEach(button => {
        button.addEventListener("click", async () => {
          const text = button.getAttribute("data-copy") || "";
          try {
            await navigator.clipboard.writeText(text);
            button.innerHTML = `<i class="bi bi-check2"></i> Copied`;
            setTimeout(() => { button.innerHTML = `<i class="bi bi-copy"></i> Copy`; }, 1200);
          } catch (error) {
            button.textContent = "Copy failed";
          }
        });
      });

      downloadModal.show();
    }

    function resetFilters() {
      document.getElementById("search").value = "";
      document.getElementById("technique-filter").value = "";
      document.getElementById("type-filter").value = "";
      document.getElementById("frame-filter").value = "";
      document.getElementById("benchmark-filter").value = "";
      document.getElementById("sensitivity-field-filter").value = "";
      document.getElementById("sensitivity-value-filter").value = "all";
      document.getElementById("process-field-filter").value = "";
      document.getElementById("process-value-filter").value = "all";
      document.getElementById("year-min-filter").value = YEAR_MIN;
      document.getElementById("year-max-filter").value = YEAR_MAX;
      renderTable();
    }

    populateFilters();
    document.querySelectorAll("input, select").forEach(element => {
      element.addEventListener("input", renderTable);
      element.addEventListener("change", renderTable);
    });
    document.getElementById("reset-filters").addEventListener("click", resetFilters);
    renderTable();
  </script>
  <script data-goatcounter="https://global-vlm.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""
    return (
        html.replace("__DATASETS_JSON__", html_escape_json(datasets))
        .replace("__PROCESS_FIELDS_JSON__", html_escape_json(PROCESS_FIELDS))
        .replace("__YEAR_MIN__", str(year_min))
        .replace("__YEAR_MAX__", str(year_max))
    )


def build_about_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>About | Global Vertical Land Motion</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root {
      --nav-height: 58px;
      --bg: #0f1720;
      --panel: #121c28;
      --soft: #dbe7f3;
      --muted: #97a9bc;
      --line: #2b3a4d;
      --accent: #63b3ff;
      --accent-2: #7ae4c3;
      --warn: #ffd06b;
      --danger: #ff7369;
      --white: #f7fbff;
      --shadow: 0 20px 50px rgba(0,0,0,.28);
      --radius: 18px;
      --max: 1200px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    html, body {
      margin: 0;
      width: 100%;
      min-height: 100%;
      background:
        radial-gradient(1200px 700px at 80% -10%, rgba(99,179,255,.14), transparent 55%),
        radial-gradient(900px 500px at 10% 0%, rgba(122,228,195,.1), transparent 45%),
        linear-gradient(180deg, #0e1620 0%, #101823 100%);
      color: var(--soft);
      font-family: "Segoe UI", Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.5;
    }
    a { color: #9fd0ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .app-navbar {
      height: var(--nav-height);
      background: rgba(255, 255, 255, 0.97);
      border-bottom: 1px solid #d8dee8;
      box-shadow: 0 1px 12px rgba(20, 32, 46, 0.08);
      z-index: 1100;
    }
    .brand-title {
      font-size: 15px;
      font-weight: 800;
      color: #1c2b39;
    }
    .brand-title a {
      color: inherit;
      text-decoration: none;
    }
    .app-navbar .nav-link {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #2b3a4d;
      font-weight: 700;
      font-size: 14px;
      line-height: 1;
      white-space: nowrap;
    }
    .app-navbar .nav-link:hover,
    .app-navbar .nav-link.active {
      color: #0d6efd;
      text-decoration: none;
    }
    .showcase-disclaimer {
      position: fixed;
      top: calc(var(--nav-height) + 10px);
      left: 50%;
      transform: translateX(-50%);
      z-index: 1080;
      width: min(940px, calc(100vw - 32px));
      padding: 8px 12px;
      border: 1px solid #d8c88e;
      border-radius: 8px;
      background: rgba(255, 248, 218, 0.98);
      color: #5a4a16;
      box-shadow: 0 8px 22px rgba(20, 32, 46, 0.14);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.35;
      text-align: center;
    }
    main {
      padding-top: calc(var(--nav-height) + 54px);
    }
    .container-page {
      width: min(var(--max), calc(100vw - 32px));
      margin: 0 auto;
    }
    .section { padding: 34px 0; }
    .hero {
      padding: 36px 0 18px;
      border-bottom: 1px solid rgba(255,255,255,.06);
    }
    .brand-panel {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 20px;
    }
    .favicon-card {
      width: 42px;
      height: 42px;
      border-radius: 12px;
      background: #eaf3fb;
      display: grid;
      place-items: center;
      box-shadow: inset 0 0 0 1px rgba(15,23,32,.08);
      flex: 0 0 auto;
    }
    h1 {
      margin: 0 0 12px;
      max-width: 980px;
      color: var(--white);
      font-size: clamp(2rem, 3.8vw, 3.8rem);
      line-height: 1.04;
      letter-spacing: -.03em;
    }
    h2, h3, h4 { color: var(--white); }
    h3 { margin: 0 0 12px; font-size: 1.28rem; }
    p { margin: 0 0 12px; }
    .sub {
      max-width: 900px;
      color: var(--muted);
      font-size: clamp(1.02rem, 1.6vw, 1.15rem);
      margin-bottom: 16px;
    }
    .muted { color: var(--muted); }
    .hero-actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 18px 0 22px;
    }
    .pill-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      border-radius: 999px;
      padding: 11px 16px;
      font-weight: 800;
      border: 1px solid rgba(255,255,255,.09);
      background: rgba(255,255,255,.03);
      color: var(--white);
      transition: .2s ease;
    }
    .pill-button:hover {
      text-decoration: none;
      transform: translateY(-1px);
      background: rgba(255,255,255,.06);
    }
    .pill-button.primary {
      background: linear-gradient(135deg, rgba(99,179,255,.22), rgba(122,228,195,.16));
      border-color: rgba(99,179,255,.25);
    }
    .card-panel {
      background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.025));
      border: 1px solid rgba(255,255,255,.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .panel { padding: 24px; }
    .architecture {
      display: grid;
      grid-template-columns: 1fr 86px 1.06fr 86px 1fr;
      gap: 14px;
      align-items: start;
      margin-top: 14px;
    }
    .node {
      padding: 18px;
      min-height: 194px;
      transition: .2s ease;
      cursor: default;
    }
    .node:hover {
      transform: translateY(-3px);
      border-color: rgba(99,179,255,.24);
      background: linear-gradient(180deg, rgba(99,179,255,.08), rgba(255,255,255,.03));
    }
    .tag {
      display: inline-block;
      margin-bottom: 8px;
      color: #99d0ff;
      font-size: .72rem;
      font-weight: 900;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .node-head {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .node-icon {
      width: 34px;
      height: 34px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.08);
      flex: 0 0 auto;
    }
    .node-icon svg,
    .node-icon img {
      width: 22px;
      height: 22px;
      display: block;
      object-fit: contain;
    }
    .node h4 { margin: 0; font-size: 1.1rem; }
    .node p { margin: 0; color: #d4e0ec; font-size: .97rem; }
    .arrow {
      display: grid;
      place-items: center;
      min-height: 194px;
      color: #8fb6d8;
      font-weight: 800;
      position: relative;
    }
    .arrow .shaft {
      width: 100%;
      height: 2px;
      background: linear-gradient(90deg, rgba(99,179,255,.12), rgba(99,179,255,.7), rgba(99,179,255,.12));
      border-radius: 999px;
      position: relative;
    }
    .arrow .shaft::after {
      content: "";
      position: absolute;
      right: -1px;
      top: 50%;
      width: 12px;
      height: 12px;
      border-top: 2px solid #9fd0ff;
      border-right: 2px solid #9fd0ff;
      transform: translateY(-50%) rotate(45deg);
    }
    .arrow-label {
      position: absolute;
      top: calc(50% - 28px);
      background: rgba(15,23,32,.9);
      border: 1px solid rgba(255,255,255,.08);
      padding: 6px 9px;
      border-radius: 999px;
      font-size: .82rem;
      color: #d7e7f7;
    }
    .explainer {
      margin-top: 18px;
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,.035);
      border: 1px dashed rgba(255,255,255,.14);
      min-height: 98px;
    }
    .explainer h4 { margin: 0 0 8px; }
    .explainer p { margin: 0; color: #d0deea; }
    .flow-legend,
    .inline-links,
    .logos {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .flow-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.08);
      padding: 8px 10px;
      border-radius: 999px;
      color: #dce7f2;
      font-size: .92rem;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot.blue { background: #63b3ff; }
    .dot.green { background: #7ae4c3; }
    .dot.gold { background: #ffd06b; }
    .dot.red { background: #ff7369; }
    .important-banner {
      margin-top: 22px;
      padding: 22px;
      border-radius: 22px;
    }
    .ipls-banner {
      background: linear-gradient(135deg, rgba(255,115,105,.12), rgba(255,208,107,.12));
      border: 1px solid rgba(255,208,107,.22);
    }
    .disclaimer-banner {
      background: linear-gradient(135deg, rgba(99,179,255,.12), rgba(122,228,195,.12));
      border: 1px solid rgba(99,179,255,.22);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: .12em;
      font-size: .76rem;
      font-weight: 900;
    }
    .ipls-banner .eyebrow { color: #ffd88f; }
    .disclaimer-banner .eyebrow { color: #9fd0ff; }
    .important-banner h2 {
      margin: 0 0 8px;
      font-size: clamp(1.3rem,2.1vw,1.9rem);
      line-height: 1.1;
    }
    .important-banner p {
      margin: 0 0 14px;
      color: #d4dfeb;
      max-width: 980px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    .logo-chip {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 250px;
      padding: 14px 16px;
      background: rgba(255,255,255,.03);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 18px;
      color: inherit;
    }
    .logo-chip:hover {
      text-decoration: none;
      background: rgba(255,255,255,.055);
    }
    .logo-chip img,
    .logo-chip svg {
      width: 58px;
      height: 42px;
      object-fit: contain;
      flex: 0 0 auto;
      background: #fff;
      border-radius: 8px;
      padding: 6px;
    }
    .logo-chip strong { display: block; color: var(--white); }
    .logo-chip small { display: block; color: var(--muted); }
    .features {
      display: grid;
      grid-template-columns: repeat(4, minmax(0,1fr));
      gap: 14px;
    }
    .feature {
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,.03);
      border: 1px solid rgba(255,255,255,.08);
    }
    .feature h4 { margin: 0 0 6px; font-size: 1.04rem; }
    .feature p { margin: 0; color: #cedbe7; font-size: .95rem; }
    .footer {
      padding: 24px 0 48px;
      color: var(--muted);
      font-size: .95rem;
    }
    @media (max-width: 980px) {
      .grid-2,
      .architecture,
      .features { grid-template-columns: 1fr; }
      .arrow { min-height: 64px; }
      .arrow .shaft { transform: rotate(90deg); width: 64px; }
      .arrow-label { top: 10px; }
    }
    @media (max-width: 640px) {
      html { font-size: 13px; }
      .brand-long { display: inline; }
      .brand-short { display: none; }
      .app-navbar { padding-left: 9px !important; padding-right: 9px !important; }
      .app-navbar .nav-link { padding-left: 8px; padding-right: 8px; }
      .app-navbar .nav-link span { display: none !important; }
      .logo-chip { min-width: unset; width: 100%; }
    }
  </style>
</head>
<body>
  <nav class="navbar app-navbar fixed-top px-3">
    <div class="container-fluid px-0">
      <div class="brand-title me-auto"><a href="index.html"><span class="brand-long">Global Vertical Land Motion</span></a></div>
      <a class="nav-link ms-3" href="index.html"><i class="bi bi-globe-americas"></i><span class="d-none d-sm-inline">Map</span></a>
      <a class="nav-link ms-3" href="catalogue.html"><i class="bi bi-table"></i><span class="d-none d-sm-inline">Catalogue</span></a>
      <a class="nav-link ms-3" href="compare.html"><i class="bi bi-columns-gap"></i><span class="d-none d-sm-inline">Compare</span></a>
      <a class="nav-link ms-3 active" href="about.html"><i class="bi bi-info-circle"></i><span class="d-none d-sm-inline">About</span></a>
    </div>
  </nav>
  <div class="showcase-disclaimer">
    Preliminary showcase: this site is largely AI-generated and intended to motivate real community development, review, and shared stewardship of VLM data. Please cite original data sources, DOIs, and associated papers when using any dataset.
  </div>
  <main>
    <header class="hero">
      <div class="container-page">
        <div class="brand-panel">
          <div class="favicon-card" aria-hidden="true">
            <img src="favicon.svg" width="28" height="28" alt="">
          </div>
          <div>
            <div style="font-size:1.02rem;color:var(--white);font-weight:800">Global VLM Data Platform</div>
            <div class="muted" style="font-size:.92rem">Open vertical land motion data for sea-level science</div>
          </div>
        </div>

        <h1>A global platform for vertical land motion data in support of relative sea-level science.</h1>
        <p class="sub">
          This platform is designed to make heterogeneous VLM observations, models, and hybrid coastal estimates easier to discover, compare, document, and reuse in one open, community-oriented place.
        </p>

        <div class="hero-actions">
          <a class="pill-button primary" href="index.html">Open Global Map</a>
          <a class="pill-button" href="#how-it-works">Explore the platform concept</a>
          <a class="pill-button" href="https://sites.google.com/view/iplsubsidence/home" target="_blank" rel="noopener">Visit IPLS</a>
          <a class="pill-button" href="https://forms.gle/" target="_blank" rel="noopener">Join the IPLS mailing list</a>
        </div>

        <section class="card-panel panel" id="how-it-works">
          <h3>Interactive concept: how the platform works</h3>
          <p class="muted">
            Hover any box or arrow to update the explanation panel. The concept below makes the architecture explicit: community members work directly on two GitHub repositories, while open archives such as Zenodo remain the authoritative external source environment for published datasets and supplements.
          </p>

          <div class="architecture">
            <div class="node card-panel hoverable" data-title="Open data archives" data-text="External archives such as Zenodo, project supplements, and institutional repositories host raw or published datasets. The platform should link back to those original records wherever possible, rather than replacing them.">
              <div class="tag">External source</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-database-fill"></i></div>
                <h4>Zenodo & archives</h4>
              </div>
              <p>Original files, supplements, release records, DOI pages, metadata, and citation targets.</p>
            </div>

            <div class="arrow hoverable" data-title="Document and ingest" data-text="This arrow means the community documents where datasets come from, which variables are used, and how those data are transformed into web-ready formats while preserving provenance.">
              <div class="arrow-label">Ingest</div>
              <div class="shaft"></div>
            </div>

            <div class="node card-panel hoverable" data-title="GitHub repository 1" data-text="This is the shared data-and-code repository. Community contributors work here directly through pull requests, scripts, JSON metadata, manifests, and reproducible preprocessing logic.">
              <div class="tag">GitHub repo 1</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-github"></i></div>
                <h4>Data & code</h4>
              </div>
              <p>Dataset folders, metadata JSON, scripts, notebooks, processing logic, schema, manifests, attribution.</p>
            </div>

            <div class="arrow hoverable" data-title="Build and publish" data-text="This arrow represents automated checks, validation, data packaging, and publishing static payloads that the website can render without a backend server.">
              <div class="arrow-label">Build</div>
              <div class="shaft"></div>
            </div>

            <div class="node card-panel hoverable" data-title="GitHub repository 2" data-text="This is the public-facing website repository, for example GitHub Pages. It renders the static interface in the browser: maps, filters, about pages, downloads, comparison tools, comments, and documentation.">
              <div class="tag">GitHub repo 2</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-window-sidebar"></i></div>
                <h4>Pages website</h4>
              </div>
              <p>Static web app with maps, filters, comments, concept pages, downloads, and side-by-side comparison.</p>
            </div>
          </div>

          <div class="architecture" style="margin-top:14px">
            <div class="node card-panel hoverable" data-title="Community contribution" data-text="Community members work directly on the repositories. They can add datasets, improve metadata, fix code, document provenance, and propose interface improvements. The repos are not just storage; they are the collaboration space.">
              <div class="tag">Community</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-people-fill" style="color:#7ae4c3"></i></div>
                <h4>Community contributors</h4>
              </div>
              <p>Directly edit repo 1 and repo 2 through issues, pull requests, dataset submissions, and metadata review.</p>
            </div>

            <div class="arrow hoverable" data-title="Review and merge" data-text="This arrow represents technical and scientific review before updates are merged. That can include metadata checks, source attribution checks, and lightweight quality control.">
              <div class="arrow-label">Review</div>
              <div class="shaft"></div>
            </div>

            <div class="node card-panel hoverable" data-title="Shared standards" data-text="A common schema lets GNSS, InSAR, tide gauges, GIA models, hybrid estimates, and contextual layers be described in one coherent system. That includes original-source links, provenance, units, variables, and uncertainty notes.">
              <div class="tag">Standards</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-list-check" style="color:#ffd06b"></i></div>
                <h4>Common metadata model</h4>
              </div>
              <p>Technique, source, reference frame, variables, units, uncertainty, provenance, download targets.</p>
            </div>

            <div class="arrow hoverable" data-title="Scientific use and feedback" data-text="The platform should support interpretation, not just display. Users compare products, identify gaps, check provenance, and feed corrections or new data back into the shared repositories.">
              <div class="arrow-label">Use</div>
              <div class="shaft"></div>
            </div>

            <div class="node card-panel hoverable" data-title="Scientific and public value" data-text="The result is a public, interpretable, and reusable VLM discovery layer that supports assessment, communication, and coordination, especially for sea-level and coastal subsidence work.">
              <div class="tag">Outcome</div>
              <div class="node-head">
                <div class="node-icon"><i class="bi bi-layers-fill" style="color:#ff7369"></i></div>
                <h4>Decision-ready discovery</h4>
              </div>
              <p>Find data, compare products, document origin, identify gaps, and download subsets for reuse.</p>
            </div>
          </div>

          <div class="explainer" id="explainer">
            <h4>How to read the diagram</h4>
            <p>
              Hover a concept box or arrow to see what it means. The boxes represent data sources, repositories, roles, or outcomes; the arrows represent ingest, build, review, and scientific reuse.
            </p>
          </div>

          <div class="flow-legend">
            <span class="flow-pill"><span class="dot blue"></span>External source</span>
            <span class="flow-pill"><span class="dot green"></span>GitHub collaboration</span>
            <span class="flow-pill"><span class="dot gold"></span>Shared standards</span>
            <span class="flow-pill"><span class="dot red"></span>Public-facing impact</span>
          </div>
        </section>

        <section class="important-banner ipls-banner" id="ipls">
          <div class="eyebrow">Important alignment</div>
          <h2>This platform should contribute directly and visibly to the IPLS effort.</h2>
          <p>
            The <strong>International Panel on Land Subsidence (IPLS)</strong> aims to unite the global subsidence research community, close knowledge gaps on coastal subsidence and relative sea-level rise, foster collaboration across disciplines, consolidate existing knowledge, and support the first assessment report on these issues. This platform is intended to help that effort by making VLM datasets easier to find, inspect, compare, and document in a shared, web-native environment.
          </p>
          <div class="inline-links">
            <a class="pill-button primary" href="https://sites.google.com/view/iplsubsidence/home" target="_blank" rel="noopener">IPLS home</a>
            <a class="pill-button" href="https://forms.gle/" target="_blank" rel="noopener">Join the mailing list</a>
          </div>
        </section>

        <section class="important-banner disclaimer-banner">
          <div class="eyebrow">Important disclaimer</div>
          <h2>This is a preliminary showcase website.</h2>
          <p>
            The platform concept shown here is an early public-facing showcase, not yet a complete operational data system. All datasets displayed should remain linked to their <strong>original source</strong>, archive, DOI page, repository, or provider record. The website should help users discover, compare, and understand datasets, but it must not obscure provenance, replace required citation, or detach products from their original custodianship.
          </p>
        </section>
      </div>
    </header>

    <section class="section" id="about-project">
      <div class="container-page grid-2">
        <article class="card-panel panel">
          <h3>About the project</h3>
          <p>
            The platform centers on one practical idea: vertical land motion data matter for coastal risk, but they are scattered across disciplines, formats, repositories, and scales. The goal is to create an accessible web layer on top of this fragmented ecosystem so researchers can quickly see what exists, where it comes from, how it was processed, and how different products compare.
          </p>
          <p>
            It is intentionally designed around static deployment, portable metadata, transparent provenance, and community contribution. That lowers operational overhead, improves long-term persistence, and makes the platform easier to maintain even as contributors change over time.
          </p>
          <p class="muted">
            Project context: <a href="https://oelsmann.github.io//portfolio/portfolio-1/" target="_blank" rel="noopener">VLM-SLC project page</a>
          </p>
        </article>

        <article class="card-panel panel">
          <h3>Funding and institutional context</h3>
          <p>
            The scientific context of this work is linked to the <strong>Marie Sklodowska-Curie Actions (MSCA)</strong>, part of Horizon Europe. The platform reflects that spirit by connecting research infrastructure, open data, transparent workflows, and international collaboration around sea-level change and coastal subsidence.
          </p>
          <p>
            This showcase was developed by Julius Oelsmann at the Technical University of Munich in the context of a Marie Sklodowska-Curie project. The views and prototype implementation are those of the author and do not necessarily represent an official TUM or European Commission product.
          </p>

          <div class="logos">
            <a class="logo-chip" href="https://www.tum.de/en/" target="_blank" rel="noopener">
              <img src="tum_logo.svg" alt="TUM logo">
              <span><strong>TUM</strong><small>Technical University of Munich</small></span>
            </a>

            <a class="logo-chip" href="https://tulane.edu/" target="_blank" rel="noopener">
              <img src="tulane_logo.svg" alt="Tulane University logo">
              <span><strong>Tulane</strong><small>Tulane University</small></span>
            </a>

            <a class="logo-chip" href="https://commission.europa.eu/index_en" target="_blank" rel="noopener">
              <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <rect width="48" height="48" rx="10" fill="#004494"/>
                <g fill="#FFCC00">
                  <circle cx="24" cy="9.5" r="1.7"/><circle cx="31.5" cy="12" r="1.7"/><circle cx="36.5" cy="17.2" r="1.7"/>
                  <circle cx="38" cy="24" r="1.7"/><circle cx="36.5" cy="30.8" r="1.7"/><circle cx="31.5" cy="36" r="1.7"/>
                  <circle cx="24" cy="38.5" r="1.7"/><circle cx="16.5" cy="36" r="1.7"/><circle cx="11.5" cy="30.8" r="1.7"/>
                  <circle cx="10" cy="24" r="1.7"/><circle cx="11.5" cy="17.2" r="1.7"/><circle cx="16.5" cy="12" r="1.7"/>
                </g>
              </svg>
              <span><strong>European Commission</strong><small>EU / Horizon Europe / MSCA context</small></span>
            </a>
          </div>

        </article>
      </div>
    </section>

    <section class="section" id="community-contact">
      <div class="container-page grid-2">
        <article class="card-panel panel">
          <h3>Community co-development</h3>
          <p>
            This prototype is intended to evolve into a community effort where researchers and data providers from the VLM community are welcome to co-develop the code, metadata, and documentation through GitHub.
          </p>
          <p>
            Authors and custodians of original datasets will be contacted and invited to contribute, correct metadata, improve provenance, and help shape the platform into a joint project.
          </p>
        </article>

        <article class="card-panel panel">
          <h3>Contact and reuse</h3>
          <p>
            Contact: <a href="mailto:julius.oelsmann@tum.de">julius.oelsmann@tum.de</a><br>
            Julius Oelsmann, Technical University of Munich
          </p>
          <p>
            Copyright (c) 2026 Julius Oelsmann. Code and website source are released under the MIT License. Third-party datasets remain governed by their original providers, licenses, DOIs, terms of use, and citation requirements.
          </p>
          <p>
            This website uses GoatCounter for lightweight, aggregate website usage statistics. No advertising tracking is used.
          </p>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="container-page">
        <div class="features">
          <div class="feature">
            <h4>Multi-technique comparison</h4>
            <p>GNSS, InSAR, tide gauges, GIA models, hybrid estimates, and contextual layers can be viewed in one interface.</p>
          </div>
          <div class="feature">
            <h4>Static by design</h4>
            <p>GitHub Pages-style delivery reduces infrastructure needs while preserving public access and transparency.</p>
          </div>
          <div class="feature">
            <h4>Reproducible data handling</h4>
            <p>Each layer should link back to documented sources, scripts, metadata, and citation guidance.</p>
          </div>
          <div class="feature">
            <h4>Built for assessment efforts</h4>
            <p>The platform is meant to support cross-dataset synthesis and broader coordination, especially within IPLS.</p>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer class="footer">
    <div class="container-page">
      <div class="card-panel panel">
        <p style="margin:0">
          This About page is an initial, editable concept page for the Global VLM Data Platform. It can later be refined with final institutional language, official logo lockups, and direct links to live contribution workflows. Contact: <a href="mailto:julius.oelsmann@tum.de">julius.oelsmann@tum.de</a>.
        </p>
      </div>
    </div>
  </footer>

  <script>
    const explainer = document.getElementById('explainer');
    const hoverables = document.querySelectorAll('.hoverable');
    const defaultTitle = 'How to read the diagram';
    const defaultText = 'Hover a concept box or arrow to see what it means. The boxes represent data sources, repositories, roles, or outcomes; the arrows represent ingest, build, review, and scientific reuse.';

    hoverables.forEach(el => {
      el.addEventListener('mouseenter', () => {
        const title = el.dataset.title || defaultTitle;
        const text = el.dataset.text || defaultText;
        explainer.innerHTML = `<h4>${title}</h4><p>${text}</p>`;
      });
      el.addEventListener('mouseleave', () => {
        explainer.innerHTML = `<h4>${defaultTitle}</h4><p>${defaultText}</p>`;
      });
    });
  </script>
  <script data-goatcounter="https://global-vlm.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""


def build_compare_html(
    records: list[dict],
    metadata: dict,
    ngl_imaged_values: list[float | None],
    ngl_imaged_metadata: dict,
    gia_values: list[list[float]],
    gia_metadata: dict,
    insar_grids: list[dict],
    insar_metadata: dict,
    gns_records: list[dict],
    gns_metadata: dict,
    tide_gauge_records: list[dict],
    tide_gauge_metadata: dict,
    oelsmann_hybrid_records: list[dict],
    oelsmann_hybrid_metadata: dict,
    population_metadata: dict,
    render_urls: dict[str, str],
    uncertainty_urls: dict[str, str],
) -> str:
    replacements = {
        "__METADATA_JSON__": html_escape_json(metadata),
        "__NGL_IMAGED_METADATA_JSON__": html_escape_json(
            {key: value for key, value in ngl_imaged_metadata.items() if key != "zeta_values"}
        ),
        "__GIA_METADATA_JSON__": html_escape_json(gia_metadata),
        "__INSAR_METADATA_JSON__": html_escape_json(insar_metadata),
        "__GNS_METADATA_JSON__": html_escape_json(gns_metadata),
        "__TIDE_GAUGE_METADATA_JSON__": html_escape_json(tide_gauge_metadata),
        "__OELSMANN_HYBRID_METADATA_JSON__": html_escape_json(oelsmann_hybrid_metadata),
        "__POPULATION_METADATA_JSON__": html_escape_json(population_metadata),
        "__POPULATION_DATA_URL__": POPULATION_PAYLOAD_JS.as_posix(),
        "__RENDER_URLS_JSON__": html_escape_json(render_urls),
        "__UNCERTAINTY_URLS_JSON__": html_escape_json(uncertainty_urls),
    }
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Compare | Global Vertical Land Motion</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg">
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css" />
  <script src="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root {
      --nav-height: 58px;
      --bottom-height: 164px;
      --panel-border: #d8dee8;
    }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #34383d;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .app-navbar {
      height: var(--nav-height);
      background: rgba(255, 255, 255, 0.97);
      border-bottom: 1px solid var(--panel-border);
      box-shadow: 0 1px 12px rgba(20, 32, 46, 0.08);
      z-index: 1200;
    }
    .brand-title {
      font-size: 15px;
      font-weight: 800;
      color: #1c2b39;
      white-space: nowrap;
    }
    .brand-title a {
      color: inherit;
      text-decoration: none;
    }
    .compare-shell {
      position: absolute;
      top: var(--nav-height);
      bottom: var(--bottom-height);
      left: 0;
      right: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: #26313b;
    }
    body.compare-bottom-collapsed {
      --bottom-height: 0px;
    }
    .compare-pane {
      position: relative;
      min-width: 0;
      overflow: hidden;
      background: #2f2f2f;
    }
    .compare-map {
      position: absolute;
      inset: 0;
    }
    .pane-title {
      position: absolute;
      top: 10px;
      left: 12px;
      z-index: 20;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 6px 9px;
      font-size: 12px;
      font-weight: 800;
      color: #223044;
      box-shadow: 0 4px 16px rgba(20, 32, 46, 0.12);
    }
    .compare-pane[data-side="right"] .pane-title {
      left: auto;
      right: 12px;
    }
    .layer-panel {
      position: absolute;
      top: 48px;
      left: 12px;
      z-index: 20;
      width: min(285px, calc(100% - 24px));
      max-height: calc(100% - 70px);
      overflow: auto;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(20, 32, 46, 0.18);
      padding: 10px 12px;
    }
    .compare-pane[data-side="right"] .layer-panel {
      left: auto;
      right: 12px;
    }
    .layer-panel.collapsed .layer-panel-body {
      display: none;
    }
    .panel-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: #263241;
      font-size: 12px;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: .02em;
      margin-bottom: 8px;
    }
    .panel-toggle {
      border: 0;
      background: transparent;
      color: #526478;
      display: grid;
      place-items: center;
      padding: 2px;
    }
    .layer-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 0;
      border-top: 1px solid #e7ebf0;
    }
    .layer-row:first-of-type {
      border-top: 0;
    }
    .layer-name {
      font-size: 13px;
      font-weight: 700;
      color: #263241;
    }
    .layer-row.unavailable .layer-name,
    .layer-row.unavailable .layer-detail {
      color: #9aa6b5;
    }
    .layer-detail {
      font-size: 11px;
      color: #657487;
    }
    .layer-loading {
      display: none;
      width: 14px;
      height: 14px;
      border: 2px solid #cbd5e1;
      border-top-color: #24476f;
      border-radius: 999px;
      animation: compare-layer-spin 0.75s linear infinite;
      flex: 0 0 auto;
    }
    .layer-row.loading .layer-loading {
      display: inline-block;
    }
    @keyframes compare-layer-spin {
      to { transform: rotate(360deg); }
    }
    .bottom-controller {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: var(--bottom-height);
      max-height: calc(var(--nav-height) * 3);
      z-index: 1100;
      background: rgba(255, 255, 255, 0.98);
      border-top: 1px solid var(--panel-border);
      box-shadow: 0 -6px 24px rgba(20, 32, 46, 0.14);
      overflow: auto;
      padding: 10px 14px 12px;
    }
    .bottom-controller-toggle {
      position: absolute;
      top: 8px;
      right: 10px;
      z-index: 1;
      width: 34px;
      height: 32px;
      border: 0;
      border-radius: 7px;
      background: #eef3f8;
      color: #263241;
      display: grid;
      place-items: center;
      padding: 0;
    }
    .bottom-controller-toggle:hover,
    .bottom-controller-toggle:focus-visible {
      background: #e2eaf3;
    }
    .bottom-controller-body {
      padding-right: 42px;
    }
    .bottom-controller.collapsed {
      left: auto;
      right: 12px;
      bottom: 12px;
      width: auto;
      height: auto;
      max-height: none;
      overflow: visible;
      padding: 6px;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
    }
    .bottom-controller.collapsed .bottom-controller-body {
      display: none;
    }
    .bottom-controller.collapsed .bottom-controller-toggle {
      position: static;
      width: 38px;
      height: 36px;
    }
    .control-grid {
      display: grid;
      grid-template-columns: 170px repeat(6, minmax(120px, 1fr));
      gap: 10px 12px;
      align-items: end;
    }
    .control-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: #344559;
      font-size: 11px;
      font-weight: 800;
      margin-bottom: 2px;
    }
    .control-value {
      color: #69798b;
      font-variant-numeric: tabular-nums;
    }
    .compare-variable-control {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
    }
    .compare-variable-control input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .compare-variable-control label {
      border: 1px solid #ccd6e2;
      border-radius: 6px;
      background: #fff;
      color: #344559;
      cursor: pointer;
      font-size: 12px;
      font-weight: 800;
      line-height: 1;
      padding: 7px 8px;
      text-align: center;
    }
    .compare-variable-control input:checked + label {
      background: #24476f;
      border-color: #24476f;
      color: #fff;
    }
    .nav-link {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #344559;
      font-size: 14px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
    }
    .nav-link:hover,
    .nav-link:focus {
      color: #0f5ca8;
      text-decoration: none;
    }
    .showcase-disclaimer {
      position: fixed;
      top: calc(var(--nav-height) + 10px);
      left: 50%;
      transform: translateX(-50%);
      z-index: 1250;
      width: min(940px, calc(100vw - 32px));
      padding: 8px 12px;
      border: 1px solid #d8c88e;
      border-radius: 8px;
      background: rgba(255, 248, 218, 0.96);
      color: #5a4a16;
      box-shadow: 0 8px 22px rgba(20, 32, 46, 0.14);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.35;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      pointer-events: auto;
    }
    .showcase-disclaimer-close {
      width: 22px;
      height: 22px;
      border: 0;
      border-radius: 999px;
      color: #5a4a16;
      background: rgba(90, 74, 22, 0.1);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      padding: 0;
    }
    .showcase-disclaimer-close:hover,
    .showcase-disclaimer-close:focus {
      background: rgba(90, 74, 22, 0.18);
    }
    .legend-pill {
      position: absolute;
      right: 12px;
      bottom: 44px;
      z-index: 20;
      width: min(220px, calc(100% - 24px));
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 6px 8px;
      box-shadow: 0 4px 16px rgba(20, 32, 46, 0.14);
    }
    .legend-ramp {
      height: 8px;
      border-radius: 999px;
      border: 1px solid #c6ced9;
      background: linear-gradient(90deg, rgb(34,104,209), rgb(246,246,242), rgb(207,50,45));
    }
    .legend-ramp.uncertainty-ramp {
      background: linear-gradient(90deg, rgb(255,247,188), rgb(254,196,79), rgb(217,95,14), rgb(127,0,0));
    }
    .legend-labels {
      display: flex;
      justify-content: space-between;
      color: #5d6d80;
      font-size: 9px;
      font-weight: 800;
      margin-top: 3px;
    }
    .legend-scale-control {
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px solid #e1e7ef;
    }
    .legend-scale-header {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 3px;
    }
    .legend-scale-toggle {
      width: 28px;
      height: 26px;
      border: 0;
      border-radius: 7px;
      background: #eef3f8;
      color: #263241;
      display: grid;
      place-items: center;
      padding: 0;
    }
    .legend-scale-toggle:hover,
    .legend-scale-toggle:focus-visible {
      background: #e2eaf3;
    }
    .legend-scale-control.collapsed {
      padding-top: 6px;
    }
    .legend-scale-control.collapsed .legend-scale-body {
      display: none;
    }
    .legend-scale-control.collapsed .legend-scale-toggle {
      width: 32px;
      height: 30px;
    }
    .legend-scale-control .control-label {
      margin-bottom: 3px;
    }
    .legend-scale-control .form-range {
      margin: 0;
    }
    @media (max-width: 900px) {
      :root { --bottom-height: 174px; }
      html { font-size: 13px; }
      .brand-long { display: inline; }
      .brand-short { display: none; }
      .app-navbar { padding-left: 9px !important; padding-right: 9px !important; }
      .nav-link { padding-left: 8px; padding-right: 8px; }
      .nav-link span { display: none !important; }
      .compare-shell { grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
      .control-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .layer-panel { width: min(245px, calc(100% - 24px)); }
      .legend-pill { bottom: 44px; width: min(190px, calc(100% - 24px)); }
    }
  </style>
</head>
<body>
  <nav class="navbar app-navbar fixed-top px-3">
    <div class="container-fluid px-0">
      <div class="brand-title me-auto"><a href="index.html"><span class="brand-long">Global Vertical Land Motion</span></a></div>
      <a class="nav-link ms-3" href="index.html"><i class="bi bi-globe-americas"></i><span class="d-none d-sm-inline">Map</span></a>
      <a class="nav-link ms-3" href="catalogue.html"><i class="bi bi-table"></i><span class="d-none d-sm-inline">Catalogue</span></a>
      <a class="nav-link ms-3 active" href="compare.html"><i class="bi bi-columns-gap"></i><span class="d-none d-sm-inline">Compare</span></a>
      <a class="nav-link ms-3" href="about.html"><i class="bi bi-info-circle"></i><span class="d-none d-sm-inline">About</span></a>
    </div>
  </nav>

  <div class="showcase-disclaimer" id="compareShowcaseDisclaimer">
    <span>Preliminary showcase: this site is largely AI-generated and intended to motivate real community development, review, and shared stewardship of VLM data. Please cite original data sources, DOIs, and associated papers when using any dataset.</span>
    <button class="showcase-disclaimer-close" type="button" id="compareShowcaseDisclaimerClose" aria-label="Dismiss preliminary showcase notice">
      <i class="bi bi-x-lg"></i>
    </button>
  </div>

  <main class="compare-shell">
    <section class="compare-pane" data-side="left">
      <div class="compare-map" id="compare-map-left"></div>
      <div class="pane-title">Left globe</div>
      <div class="layer-panel" id="layer-panel-left"></div>
    </section>
    <section class="compare-pane" data-side="right">
      <div class="compare-map" id="compare-map-right"></div>
      <div class="pane-title">Right globe</div>
      <div class="layer-panel" id="layer-panel-right"></div>
      <div class="legend-pill">
        <div class="legend-ramp"></div>
        <div class="legend-labels"><span id="legend-right-min">-5</span><span id="legend-right-mid">0</span><span id="legend-right-max">5</span></div>
        <div class="legend-scale-control" id="compareScaleControl">
          <div class="legend-scale-header">
            <button class="legend-scale-toggle" type="button" id="compareScaleToggle" aria-label="Toggle compare color range" aria-expanded="true">
              <i class="bi bi-chevron-down"></i>
            </button>
          </div>
          <div class="legend-scale-body">
            <label class="control-label" for="compare-color-limit"><span id="compare-range-label">Trend range</span><span class="control-value"><span id="compare-range-prefix">±</span><span id="compare-color-value">5</span></span></label>
            <input type="range" class="form-range" min="1" max="100" step="0.5" value="5" id="compare-color-limit">
          </div>
        </div>
      </div>
    </section>
  </main>

  <aside class="bottom-controller" id="compareBottomController" aria-label="Shared comparison controls">
    <button class="bottom-controller-toggle" type="button" id="compareBottomToggle" aria-label="Toggle comparison controls" aria-expanded="true">
      <i class="bi bi-chevron-down"></i>
    </button>
    <div class="bottom-controller-body">
      <div class="control-grid">
      <div>
        <div class="control-label"><span>Variable</span></div>
        <div class="compare-variable-control" role="radiogroup" aria-label="Compare render variable">
          <input type="radio" name="compare-render-variable" id="compare-variable-trend" value="trend" checked>
          <label for="compare-variable-trend">Trend</label>
          <input type="radio" name="compare-render-variable" id="compare-variable-uncertainty" value="uncertainty">
          <label for="compare-variable-uncertainty">Uncertainty</label>
        </div>
      </div>
      <div>
        <label class="control-label" for="compare-gia-opacity"><span>GIA opacity</span><span class="control-value"><span id="compare-gia-value">35</span>%</span></label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="35" id="compare-gia-opacity">
      </div>
      <div>
        <label class="control-label" for="compare-insar-opacity"><span>InSAR opacity</span><span class="control-value"><span id="compare-insar-value">85</span>%</span></label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="85" id="compare-insar-opacity">
      </div>
      <div>
        <label class="control-label" for="compare-pop-opacity"><span>Population opacity</span><span class="control-value"><span id="compare-pop-value">70</span>%</span></label>
        <input type="range" class="form-range" min="0" max="100" step="5" value="70" id="compare-pop-opacity">
      </div>
      <div>
        <label class="control-label" for="compare-duration"><span>Min duration</span><span class="control-value"><span id="compare-duration-value">3.0</span> yr</span></label>
        <input type="range" class="form-range" min="0" max="25" step="0.5" value="3" id="compare-duration">
      </div>
      <div>
        <label class="control-label" for="compare-first"><span>Min first</span><span class="control-value"><span id="compare-first-value">0</span></span></label>
        <input type="range" class="form-range" min="0" max="1" step="0.1" value="0" id="compare-first">
      </div>
      <div>
        <label class="control-label" for="compare-last"><span>Max last</span><span class="control-value"><span id="compare-last-value">0</span></span></label>
        <input type="range" class="form-range" min="0" max="1" step="0.1" value="1" id="compare-last">
      </div>
      <div>
        <label class="control-label" for="compare-search"><span>Station search</span></label>
        <input type="search" class="form-control form-control-sm" id="compare-search" placeholder="Station ID">
      </div>
      <div>
        <div class="control-label"><span>External context</span></div>
        <div class="d-flex gap-3 align-items-center small fw-semibold">
          <div class="form-check form-switch m-0">
            <input class="form-check-input" type="checkbox" role="switch" id="compare-pop-left">
            <label class="form-check-label" for="compare-pop-left">Left pop.</label>
          </div>
          <div class="form-check form-switch m-0">
            <input class="form-check-input" type="checkbox" role="switch" id="compare-pop-right">
            <label class="form-check-label" for="compare-pop-right">Right pop.</label>
          </div>
        </div>
      </div>
      </div>
    </div>
  </aside>

  <script>
    const {MapboxOverlay, BitmapLayer, ColumnLayer, PolygonLayer, ScatterplotLayer, COORDINATE_SYSTEM} = deck;
    let POSITIVE_DATA = [];
    let NEGATIVE_DATA = [];
    let ALL_DATA = [];
    const METADATA = __METADATA_JSON__;
    let NGL_IMAGED_GRID_VALUES = null;
    const NGL_IMAGED_METADATA = __NGL_IMAGED_METADATA_JSON__;
    let GIA_GRID_VALUES = null;
    const GIA_METADATA = __GIA_METADATA_JSON__;
    let INSAR_GRIDS = [];
    let GNS_DATA = [];
    let TIDE_GAUGE_DATA = [];
    let OELSMANN_HYBRID_DATA = [];
    const POPULATION_METADATA = __POPULATION_METADATA_JSON__;
    const POPULATION_DATA_URL = "__POPULATION_DATA_URL__";
    const RENDER_PAYLOAD_URLS = __RENDER_URLS_JSON__;
    const UNCERTAINTY_PAYLOAD_URLS = __UNCERTAINTY_URLS_JSON__;

    const shared = {
      renderMode: "points",
      renderVariable: "trend",
      colorLimit: 5,
      uncertaintyLimit: 4,
      giaOpacity: 0.35,
      insarOpacity: 0.85,
      populationOpacity: 0.7,
      minDuration: 3,
      minFirstEpoch: METADATA.first_epoch_min,
      maxLastEpoch: METADATA.last_epoch_max,
      search: "",
      zoomBucket: 2.25,
      markerScale: markerScaleForZoom(2.25),
      tideGaugeScale: tideGaugeScaleForZoom(2.25),
      gpsAltitudeOffset: gpsAltitudeOffsetForZoom(2.25)
    };
    const RENDER_DATASET_IDS = {
      showGPS: "gnss_blewitt_2018",
      showNglImaged: "gnss_imaged_hammond_2021",
      showGIA: "gia_caron_2020",
      showInSAR: "insar_ohenhen_2025",
      showGNS: "insar_gnss_hamling_2022",
      showOelsmannHybrid: "hybrid_oelsmann_2026",
      showTideGauge: "tide_gauge_dangendorf_2026"
    };
    const UNCERTAINTY_DATASET_IDS = new Set(Object.keys(UNCERTAINTY_PAYLOAD_URLS));
    const uncertaintyState = {
      loaded: false,
      loading: null,
      payloads: {},
      nglImagedValues: null,
      giaValues: null,
      max: 5
    };
    const INITIAL_VIEW_STATE = {
      longitude: -101,
      latitude: 34,
      zoom: 2.25,
      minZoom: 0.75,
      maxZoom: 20,
      bearing: 0,
      pitch: 0
    };
    const sideState = {
      left: {showGPS: true, showNglImaged: false, showGIA: false, showInSAR: true, showGNS: true, showOelsmannHybrid: true, showTideGauge: true, showPopulation: false},
      right: {showGPS: true, showNglImaged: false, showGIA: false, showInSAR: true, showGNS: true, showOelsmannHybrid: true, showTideGauge: true, showPopulation: false}
    };
    const maps = {};
    const overlays = {};
    let syncing = false;
    let populationPointCache = null;
    let populationLoadPromise = null;

    function markerScaleForZoom(zoom) {
      const scale = Math.pow(0.72, zoom - 1.32);
      return Math.max(0.001, Math.min(1.35, scale));
    }
    function tideGaugeScaleForZoom(zoom) {
      const scale = Math.pow(0.64, zoom - 1.32);
      return Math.max(0.001, Math.min(1.35, scale));
    }
    function gpsAltitudeOffsetForZoom(zoom) {
      const scale = Math.pow(0.62, zoom - 1.32);
      return Math.max(50, Math.min(3500, 2500 * scale));
    }
    function zoomBucketFor(zoom) {
      return Math.round(Number(zoom || 0) * 4) / 4;
    }
    function lerp(a, b, t) {
      return Math.round(a + (b - a) * t);
    }
    function colorForValue(value, limit, alpha = 230) {
      const safeLimit = Math.max(1, Number(limit) || 8);
      const clamped = Math.max(-safeLimit, Math.min(safeLimit, Number(value) || 0));
      const t = (clamped + safeLimit) / (2 * safeLimit);
      if (t < 0.5) {
        const k = t / 0.5;
        return [lerp(34, 246, k), lerp(104, 246, k), lerp(209, 242, k), alpha];
      }
      const k = (t - 0.5) / 0.5;
      return [lerp(246, 207, k), lerp(246, 50, k), lerp(242, 45, k), alpha];
    }
    function colorForUncertainty(value, alpha = 230) {
      const safeMax = Math.max(0.5, Number(shared.uncertaintyLimit) || 4);
      const t = Math.max(0, Math.min(1, Number(value) / safeMax));
      const stops = [[255,247,188], [254,196,79], [217,95,14], [127,0,0]];
      const scaled = t * (stops.length - 1);
      const index = Math.min(Math.floor(scaled), stops.length - 2);
      const k = scaled - index;
      const a = stops[index];
      const b = stops[index + 1];
      return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k), alpha];
    }
    function renderValue(record) {
      return shared.renderVariable === "uncertainty" ? Number(record.up_sigma_mm_yr) : Number(record.up_mm_yr);
    }
    function colorForRecord(record, alpha = 230) {
      const value = renderValue(record);
      if (!Number.isFinite(value)) return [180, 188, 198, Math.round(alpha * 0.45)];
      return shared.renderVariable === "uncertainty" ? colorForUncertainty(value, alpha) : colorForValue(value, shared.colorLimit, alpha);
    }
    function datasetHasActiveVariable(datasetId) {
      return shared.renderVariable !== "uncertainty" || UNCERTAINTY_DATASET_IDS.has(datasetId);
    }
    function loadJsonPayload(url) {
      return fetch(url, {cache: "force-cache"}).then(response => {
        if (!response.ok) throw new Error(`Could not load ${url}`);
        return response.json();
      });
    }
    function hydratePointUncertainty(records, idField, payload) {
      if (!payload || !Array.isArray(payload.values)) return;
      const lookup = new Map(payload.values.map(item => [String(item[0]), Number(item[1])]));
      for (const record of records) {
        const value = lookup.get(String(record[idField]));
        if (Number.isFinite(value)) record.up_sigma_mm_yr = value;
      }
    }
    const renderPayloadState = {
      payloads: {},
      loading: {},
      errors: {}
    };
    function setCompareDatasetLoading(datasetId, loading) {
      document.querySelectorAll(`.layer-row[data-dataset-id="${datasetId}"]`).forEach(row => {
        row.classList.toggle("loading", loading);
        row.setAttribute("aria-busy", loading ? "true" : "false");
      });
    }
    function hydrateLoadedUncertaintyForDataset(datasetId) {
      if (!uncertaintyState.loaded) return;
      if (datasetId === RENDER_DATASET_IDS.showGPS) {
        hydratePointUncertainty(ALL_DATA, "station", uncertaintyState.payloads.gnss_blewitt_2018);
      } else if (datasetId === RENDER_DATASET_IDS.showGNS) {
        hydratePointUncertainty(GNS_DATA, "id", uncertaintyState.payloads.insar_gnss_hamling_2022);
      } else if (datasetId === RENDER_DATASET_IDS.showOelsmannHybrid) {
        hydratePointUncertainty(OELSMANN_HYBRID_DATA, "id", uncertaintyState.payloads.hybrid_oelsmann_2026);
      } else if (datasetId === RENDER_DATASET_IDS.showTideGauge) {
        hydratePointUncertainty(TIDE_GAUGE_DATA, "id", uncertaintyState.payloads.tide_gauge_dangendorf_2026);
      }
    }
    function assignRenderPayload(datasetId, payload) {
      if (datasetId === RENDER_DATASET_IDS.showGPS) {
        POSITIVE_DATA = Array.isArray(payload.positive) ? payload.positive : [];
        NEGATIVE_DATA = Array.isArray(payload.negative) ? payload.negative : [];
        ALL_DATA = POSITIVE_DATA.concat(NEGATIVE_DATA);
      } else if (datasetId === RENDER_DATASET_IDS.showNglImaged) {
        NGL_IMAGED_GRID_VALUES = Array.isArray(payload.values) ? payload.values : null;
        nglImagedCellCache = null;
        nglImagedCellCacheVariable = null;
      } else if (datasetId === RENDER_DATASET_IDS.showGIA) {
        GIA_GRID_VALUES = Array.isArray(payload.values) ? payload.values : null;
        giaCellCache = null;
        giaCellCacheVariable = null;
      } else if (datasetId === RENDER_DATASET_IDS.showInSAR) {
        INSAR_GRIDS = Array.isArray(payload.grids) ? payload.grids : [];
      } else if (datasetId === RENDER_DATASET_IDS.showGNS) {
        GNS_DATA = Array.isArray(payload.records) ? payload.records : [];
      } else if (datasetId === RENDER_DATASET_IDS.showOelsmannHybrid) {
        OELSMANN_HYBRID_DATA = Array.isArray(payload.records) ? payload.records : [];
      } else if (datasetId === RENDER_DATASET_IDS.showTideGauge) {
        TIDE_GAUGE_DATA = Array.isArray(payload.records) ? payload.records : [];
      }
      renderPayloadState.payloads[datasetId] = payload;
      hydrateLoadedUncertaintyForDataset(datasetId);
    }
    function loadRenderPayload(datasetId) {
      if (renderPayloadState.payloads[datasetId]) return Promise.resolve(renderPayloadState.payloads[datasetId]);
      if (renderPayloadState.loading[datasetId]) return renderPayloadState.loading[datasetId];
      const url = RENDER_PAYLOAD_URLS[datasetId];
      if (!url) return Promise.resolve(null);
      setCompareDatasetLoading(datasetId, true);
      renderPayloadState.loading[datasetId] = loadJsonPayload(url).then(payload => {
        assignRenderPayload(datasetId, payload || {});
        delete renderPayloadState.errors[datasetId];
        return payload;
      }).catch(error => {
        renderPayloadState.errors[datasetId] = error;
        console.warn(error);
        return null;
      }).finally(() => {
        delete renderPayloadState.loading[datasetId];
        setCompareDatasetLoading(datasetId, false);
        updateSideLayers("left");
        updateSideLayers("right");
      });
      return renderPayloadState.loading[datasetId];
    }
    function loadActiveRenderPayloads(side = null) {
      const sides = side ? [side] : ["left", "right"];
      const activeDatasetIds = new Set();
      for (const currentSide of sides) {
        const s = sideState[currentSide];
        Object.entries(RENDER_DATASET_IDS).forEach(([key, datasetId]) => {
          if (s[key] && datasetHasActiveVariable(datasetId)) activeDatasetIds.add(datasetId);
        });
        if (s.showPopulation && !makePopulationPointData()) {
          loadPopulationDataset().then(() => updateSideLayers(currentSide));
        }
      }
      activeDatasetIds.forEach(datasetId => loadRenderPayload(datasetId));
    }
    function updateUncertaintyMax() {
      const maxValues = Object.values(uncertaintyState.payloads).map(payload => Number(payload && payload.max)).filter(Number.isFinite);
      uncertaintyState.max = maxValues.length ? Math.max(0.5, ...maxValues) : 5;
    }
    function loadUncertaintyPayloads() {
      if (uncertaintyState.loaded) return Promise.resolve(uncertaintyState.payloads);
      if (uncertaintyState.loading) return uncertaintyState.loading;
      uncertaintyState.loading = Promise.all(Object.entries(UNCERTAINTY_PAYLOAD_URLS).map(([datasetId, url]) =>
        loadJsonPayload(url).then(payload => [datasetId, payload])
      )).then(entries => {
        uncertaintyState.payloads = Object.fromEntries(entries);
        hydratePointUncertainty(ALL_DATA, "station", uncertaintyState.payloads.gnss_blewitt_2018);
        hydratePointUncertainty(GNS_DATA, "id", uncertaintyState.payloads.insar_gnss_hamling_2022);
        hydratePointUncertainty(OELSMANN_HYBRID_DATA, "id", uncertaintyState.payloads.hybrid_oelsmann_2026);
        hydratePointUncertainty(TIDE_GAUGE_DATA, "id", uncertaintyState.payloads.tide_gauge_dangendorf_2026);
        uncertaintyState.nglImagedValues = uncertaintyState.payloads.gnss_imaged_hammond_2021?.values || null;
        uncertaintyState.giaValues = uncertaintyState.payloads.gia_caron_2020?.values || null;
        updateUncertaintyMax();
        uncertaintyState.loaded = true;
        uncertaintyState.loading = null;
        return uncertaintyState.payloads;
      }).catch(error => {
        uncertaintyState.loading = null;
        console.warn(error);
        return uncertaintyState.payloads;
      });
      return uncertaintyState.loading;
    }
    function colorForUp(value) {
      return colorForValue(value, shared.colorLimit, 230);
    }
    function formatNumber(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return Number(value).toFixed(digits);
    }
    let giaCellCache = null;
    let giaCellCacheVariable = null;
    let nglImagedCellCache = null;
    let nglImagedCellCacheVariable = null;
    function makeGridCellData(metadata, values, variable) {
      const cells = [];
      if (!values) return cells;
      const width = metadata.width;
      const height = metadata.height;
      const bounds = metadata.bounds;
      const lonStep = (bounds[2] - bounds[0]) / width;
      const latStep = (bounds[3] - bounds[1]) / height;
      for (let y = 0; y < height; y += 1) {
        const south = Math.max(-89.999, bounds[1] + y * latStep);
        const north = Math.min(89.999, south + latStep);
        for (let x = 0; x < width; x += 1) {
          const value = values[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          const west = bounds[0] + x * lonStep;
          const east = west + lonStep;
          cells.push({
            up_mm_yr: variable === "uncertainty" ? null : value,
            up_sigma_mm_yr: variable === "uncertainty" ? value : null,
            polygon: [[west, south], [east, south], [east, north], [west, north]]
          });
        }
      }
      return cells;
    }
    function makeGIACellData() {
      if (giaCellCache && giaCellCacheVariable === shared.renderVariable) return giaCellCache;
      const cells = [];
      const width = GIA_METADATA.width;
      const height = GIA_METADATA.height;
      const values = shared.renderVariable === "uncertainty" ? uncertaintyState.giaValues : GIA_GRID_VALUES;
      if (!values) return cells;
      for (let y = 0; y < height; y += 1) {
        const north = Math.min(89.999, 90 - y);
        const south = Math.max(-89.999, 90 - y - 1);
        for (let x = 0; x < width; x += 1) {
          const value = values[y * width + x];
          if (value === null || value === undefined || !Number.isFinite(Number(value))) continue;
          const west = -180 + x;
          const east = west + 1;
          cells.push({
            up_mm_yr: shared.renderVariable === "uncertainty" ? null : value,
            up_sigma_mm_yr: shared.renderVariable === "uncertainty" ? value : null,
            polygon: [[west, south], [east, south], [east, north], [west, north]]
          });
        }
      }
      giaCellCache = cells;
      giaCellCacheVariable = shared.renderVariable;
      return giaCellCache;
    }
    function makeNglImagedCellData() {
      if (nglImagedCellCache && nglImagedCellCacheVariable === shared.renderVariable) return nglImagedCellCache;
      const values = shared.renderVariable === "uncertainty" ? uncertaintyState.nglImagedValues : NGL_IMAGED_GRID_VALUES;
      nglImagedCellCache = makeGridCellData(NGL_IMAGED_METADATA, values, shared.renderVariable);
      nglImagedCellCacheVariable = shared.renderVariable;
      return nglImagedCellCache;
    }
    function makeGridCanvas(grid, opacity) {
      const canvas = document.createElement("canvas");
      canvas.width = grid.width;
      canvas.height = grid.height;
      const context = canvas.getContext("2d");
      const image = context.createImageData(canvas.width, canvas.height);
      const alpha = Math.round(255 * opacity);
      for (let i = 0; i < grid.values.length; i += 1) {
        const value = grid.values[i];
        const offset = i * 4;
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          image.data[offset + 3] = 0;
          continue;
        }
        const color = colorForValue(value, shared.colorLimit, alpha);
        image.data[offset] = color[0];
        image.data[offset + 1] = color[1];
        image.data[offset + 2] = color[2];
        image.data[offset + 3] = color[3];
      }
      context.putImageData(image, 0, 0);
      return canvas;
    }
    function makePopulationPointData() {
      if (populationPointCache) return populationPointCache;
      const payload = window.GHSL_POPULATION_PAYLOAD;
      if (!payload || !Array.isArray(payload.values)) return null;
      populationPointCache = payload.values.map(record => ({
        longitude: record[0],
        latitude: record[1],
        population: record[2],
        radius_m: record[3]
      }));
      return populationPointCache;
    }
    function loadPopulationDataset() {
      if (window.GHSL_POPULATION_PAYLOAD) return Promise.resolve(window.GHSL_POPULATION_PAYLOAD);
      if (populationLoadPromise) return populationLoadPromise;
      setCompareDatasetLoading("ghsl_schiavina_2025", true);
      populationLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = POPULATION_DATA_URL;
        script.async = true;
        script.onload = () => resolve(window.GHSL_POPULATION_PAYLOAD);
        script.onerror = reject;
        document.head.appendChild(script);
      }).catch(error => {
        populationLoadPromise = null;
        console.warn(error);
        return null;
      }).finally(() => {
        setCompareDatasetLoading("ghsl_schiavina_2025", false);
      });
      return populationLoadPromise;
    }
    function populationColor(value) {
      const logValue = Math.log10(Math.max(1, Number(value) || 1));
      const maxLog = Math.max(1, Number(POPULATION_METADATA.max_log10_population) || 3.5);
      const t = Math.max(0, Math.min(1, logValue / maxLog));
      const stops = [[255,252,214], [253,179,75], [214,74,64], [84,39,143]];
      const scaled = t * (stops.length - 1);
      const index = Math.min(Math.floor(scaled), stops.length - 2);
      const k = scaled - index;
      const a = stops[index];
      const b = stops[index + 1];
      return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k), Math.round(210 * shared.populationOpacity)];
    }
    function stationMatches(record) {
      if (record.duration < shared.minDuration) return false;
      if (record.first_epoch_year < shared.minFirstEpoch) return false;
      if (record.last_epoch_year > shared.maxLastEpoch) return false;
      if (shared.search && !record.station.toLowerCase().includes(shared.search)) return false;
      return true;
    }
    function makeLayers(side) {
      const s = sideState[side];
      const positive = s.showGPS && datasetHasActiveVariable(RENDER_DATASET_IDS.showGPS) ? POSITIVE_DATA.filter(stationMatches) : [];
      const negative = s.showGPS && datasetHasActiveVariable(RENDER_DATASET_IDS.showGPS) ? NEGATIVE_DATA.filter(stationMatches) : [];
      const layers = [];
      if (s.showGIA && datasetHasActiveVariable(RENDER_DATASET_IDS.showGIA)) {
        layers.push(new PolygonLayer({
          id: `${side}-gia`,
          data: makeGIACellData(),
          getPolygon: d => d.polygon,
          getFillColor: d => colorForRecord(d, Math.round(255 * shared.giaOpacity)),
          stroked: false,
          filled: true,
          pickable: false,
          parameters: {depthTest: false, depthMask: false},
          updateTriggers: {
            getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.giaOpacity, shared.renderVariable]
          }
        }));
      }
      if (s.showNglImaged && datasetHasActiveVariable(RENDER_DATASET_IDS.showNglImaged)) {
        layers.push(new PolygonLayer({
          id: `${side}-ngl-gps-imaging`,
          data: makeNglImagedCellData(),
          getPolygon: d => d.polygon,
          getFillColor: d => colorForRecord(d, 215),
          stroked: false,
          filled: true,
          pickable: false,
          parameters: {depthTest: false, depthMask: false},
          updateTriggers: {
            getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
          }
        }));
      }
      if (s.showInSAR && datasetHasActiveVariable(RENDER_DATASET_IDS.showInSAR)) {
        for (const grid of INSAR_GRIDS) {
          layers.push(new BitmapLayer({
            id: `${side}-insar-${grid.id}`,
            image: makeGridCanvas(grid, shared.insarOpacity),
            bounds: grid.bounds,
            _imageCoordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
            pickable: false,
            parameters: {depthTest: false, depthMask: false}
          }));
        }
        layers.push(new PolygonLayer({
          id: `${side}-insar-footprints`,
          data: INSAR_GRIDS.map(grid => {
            const [west, south, east, north] = grid.bounds;
            return {type: "insar_delta_extent", delta_name: grid.name, author_year: "Ohenhen et al. 2025", observation_period: "2014-2023", polygon: [[west, south], [east, south], [east, north], [west, north]]};
          }),
          getPolygon: d => d.polygon,
          stroked: true,
          filled: true,
          getFillColor: [255, 255, 255, 1],
          getLineColor: [28, 38, 52, 225],
          lineWidthMinPixels: 1.4,
          pickable: true,
          parameters: {depthTest: false, depthMask: false}
        }));
      }
      if (s.showPopulation) {
        const popData = makePopulationPointData();
        if (popData) {
          layers.push(new ScatterplotLayer({
            id: `${side}-population`,
            data: popData,
            getPosition: d => [d.longitude, d.latitude, 900],
            getRadius: d => d.radius_m,
            radiusUnits: "meters",
            radiusMinPixels: 1,
            radiusMaxPixels: 5,
            getFillColor: d => populationColor(d.population),
            pickable: true,
            parameters: {depthTest: false, depthMask: false}
          }));
        }
      }
      if (s.showGNS && datasetHasActiveVariable(RENDER_DATASET_IDS.showGNS)) {
        layers.push(new ScatterplotLayer({
          id: `${side}-gns`,
          data: GNS_DATA,
          getPosition: d => [d.longitude, d.latitude, 1200],
          getRadius: d => d.point_radius_m * shared.markerScale,
          radiusUnits: "meters",
          radiusMinPixels: 3,
          radiusMaxPixels: Math.max(5, 16 * shared.markerScale),
          stroked: true,
          filled: true,
          lineWidthMinPixels: 0.7,
          getLineColor: [255, 255, 255, 190],
          getFillColor: d => colorForRecord(d, 255),
          pickable: true,
          parameters: {depthTest: false, depthMask: false},
          updateTriggers: {
            getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
          }
        }));
      }
      if (s.showOelsmannHybrid && datasetHasActiveVariable(RENDER_DATASET_IDS.showOelsmannHybrid)) {
        layers.push(new ScatterplotLayer({
          id: `${side}-oelsmann-hybrid`,
          data: OELSMANN_HYBRID_DATA,
          getPosition: d => [d.longitude, d.latitude, 1450],
          getRadius: d => d.point_radius_m * shared.markerScale,
          radiusUnits: "meters",
          radiusMinPixels: 4,
          radiusMaxPixels: Math.max(7, 24 * shared.markerScale),
          stroked: true,
          filled: true,
          lineWidthMinPixels: 0.8,
          getLineColor: [80, 80, 80, 210],
          getFillColor: d => colorForRecord(d, 255),
          pickable: true,
          parameters: {depthTest: false, depthMask: false},
          updateTriggers: {
            getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
          }
        }));
      }
      if (s.showTideGauge && datasetHasActiveVariable(RENDER_DATASET_IDS.showTideGauge)) {
        layers.push(new ColumnLayer({
          id: `${side}-tide-gauges`,
          data: TIDE_GAUGE_DATA,
          getPosition: d => [d.longitude, d.latitude],
          getElevation: 1,
          elevationScale: 0,
          radius: Math.round(120000 * shared.tideGaugeScale),
          diskResolution: 4,
          extruded: false,
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1,
          getLineColor: [28, 38, 52, 210],
          getFillColor: d => colorForRecord(d, 255),
          pickable: true,
          parameters: {depthTest: false, depthMask: false},
          updateTriggers: {
            getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
          }
        }));
      }
      if (shared.renderMode === "points") {
        for (const [name, data] of [["positive", positive], ["negative", negative]]) {
          layers.push(new ScatterplotLayer({
            id: `${side}-gps-${name}-points`,
            data,
            getPosition: d => [d.longitude, d.latitude, shared.gpsAltitudeOffset],
            getRadius: d => d.point_radius_m * shared.markerScale,
            radiusUnits: "meters",
            radiusMinPixels: 2,
            radiusMaxPixels: 13,
            stroked: true,
            filled: true,
            lineWidthMinPixels: 1,
            getLineColor: [120, 130, 142, 220],
            getFillColor: d => colorForRecord(d, 230),
            pickable: true,
            parameters: {depthTest: true, depthMask: false},
            updateTriggers: {
              getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
            }
          }));
        }
      } else {
        for (const [name, data] of [["positive", positive], ["negative", negative]]) {
          layers.push(new ColumnLayer({
            id: `${side}-gps-${name}-bars`,
            data,
            getPosition: d => [d.longitude, d.latitude],
            getElevation: d => d.bar_elevation,
            radius: Math.round(18000 * shared.markerScale),
            elevationScale: 1,
            diskResolution: 8,
            getFillColor: d => colorForRecord(d, 230),
            pickable: true,
            updateTriggers: {
              getFillColor: [shared.colorLimit, shared.uncertaintyLimit, shared.renderVariable]
            }
          }));
        }
      }
      return layers.filter(Boolean);
    }
    function tooltipFor(object) {
      if (!object) return null;
      if (object.station) {
        return {html: `<b>${object.station}</b><br>UP: ${formatNumber(object.up_mm_yr, 2)} mm/yr<br>Duration: ${formatNumber(object.duration, 1)} yr`};
      }
      if (object.type === "insar_delta_extent") {
        return {html: `<b>${object.delta_name} Delta</b><br>${object.author_year}<br>${object.observation_period}`};
      }
      if (object.dataset_id === "tide_gauge_dangendorf_2026") {
        return {html: `<b>${object.name}</b><br>Trend: ${formatNumber(object.up_mm_yr, 2)} mm/yr`};
      }
      if (object.dataset) {
        return {html: `<b>${object.dataset}</b><br>UP: ${formatNumber(object.up_mm_yr, 2)} mm/yr`};
      }
      if (object.population) {
        return {html: `<b>GHSL population</b><br>${formatNumber(object.population, 1)} people/cell`};
      }
      return null;
    }
    function updateZoomScales(zoom) {
      const bucket = zoomBucketFor(zoom);
      if (bucket === shared.zoomBucket) return;
      shared.zoomBucket = bucket;
      shared.markerScale = markerScaleForZoom(bucket);
      shared.tideGaugeScale = tideGaugeScaleForZoom(bucket);
      shared.gpsAltitudeOffset = gpsAltitudeOffsetForZoom(bucket);
      updateAllLayers();
    }
    function updateSideLayers(side) {
      loadActiveRenderPayloads(side);
      if (overlays[side]) overlays[side].setProps({layers: makeLayers(side)});
    }
    function syncCompareVariableControls() {
      const uncertaintyMode = shared.renderVariable === "uncertainty";
      const color = document.getElementById("compare-color-limit");
      color.min = uncertaintyMode ? "0.5" : "1";
      color.max = uncertaintyMode ? "20" : "100";
      color.step = "0.5";
      color.value = uncertaintyMode ? shared.uncertaintyLimit : shared.colorLimit;
      document.getElementById("compare-range-label").textContent = uncertaintyMode ? "Uncertainty range" : "Trend range";
      document.getElementById("compare-range-prefix").textContent = uncertaintyMode ? "0-" : "±";
      document.getElementById("compare-color-value").textContent = uncertaintyMode
        ? formatNumber(shared.uncertaintyLimit, 1)
        : formatNumber(shared.colorLimit, 1);
      document.querySelectorAll(".layer-row[data-layer-key]").forEach(row => {
        const datasetId = RENDER_DATASET_IDS[row.getAttribute("data-layer-key")];
        const available = datasetHasActiveVariable(datasetId);
        row.classList.toggle("unavailable", !available);
        const input = row.querySelector("input");
        if (input) {
          input.disabled = !available;
          input.title = available ? "" : "No uncertainty payload available";
        }
      });
      document.querySelectorAll(".legend-ramp").forEach(ramp => {
        ramp.classList.toggle("uncertainty-ramp", shared.renderVariable === "uncertainty");
      });
    }
    function updateAllLayers() {
      syncCompareVariableControls();
      updateSideLayers("left");
      updateSideLayers("right");
      if (shared.renderVariable === "uncertainty") {
        document.getElementById("legend-right-min").textContent = "0";
        document.getElementById("legend-right-mid").textContent = "";
        document.getElementById("legend-right-max").textContent = formatNumber(shared.uncertaintyLimit, 1);
      } else {
        document.getElementById("legend-right-min").textContent = `-${shared.colorLimit}`;
        document.getElementById("legend-right-mid").textContent = "0";
        document.getElementById("legend-right-max").textContent = `${shared.colorLimit}`;
      }
    }
    function createPanel(side) {
      const panel = document.getElementById(`layer-panel-${side}`);
      const rows = [
        ["showGPS", "GNSS", "NGL MIDAS station velocities"],
        ["showNglImaged", "NGL GPS Imaging", "Hammond interpolated GNSS grid"],
        ["showGIA", "GIA", "Caron and Ivins model"],
        ["showInSAR", "InSAR", "Ohenhen delta grids"],
        ["showGNS", "InSAR + GNSS", "New Zealand coastal VLM"],
        ["showOelsmannHybrid", "Hybrid estimates", "Oelsmann coastal VLM"],
        ["showTideGauge", "Tide gauges", "CSL-TG residual trends"]
      ];
      panel.innerHTML = `
        <div class="panel-heading">
          <span>Technique groups</span>
          <button class="panel-toggle" type="button" id="${side}-panel-toggle" aria-label="Collapse ${side} layer panel">
            <i class="bi bi-chevron-up"></i>
          </button>
        </div>
        <div class="layer-panel-body">
          ${rows.map(([key, name, detail]) => `
        <label class="layer-row" data-layer-key="${key}" data-dataset-id="${RENDER_DATASET_IDS[key]}" for="${side}-${key}">
          <span><span class="layer-name">${name}</span><br><span class="layer-detail">${detail}</span></span>
          <span class="d-inline-flex align-items-center gap-2"><span class="layer-loading" aria-hidden="true"></span><span class="form-check form-switch m-0"><input class="form-check-input" type="checkbox" role="switch" id="${side}-${key}" ${sideState[side][key] ? "checked" : ""}></span></span>
        </label>
          `).join("")}
        </div>
      `;
      if (window.matchMedia("(max-width: 900px)").matches) {
        panel.classList.add("collapsed");
        const icon = panel.querySelector(`#${side}-panel-toggle i`);
        if (icon) icon.className = "bi bi-chevron-down";
      }
      document.getElementById(`${side}-panel-toggle`).addEventListener("click", () => {
        panel.classList.toggle("collapsed");
        const icon = document.querySelector(`#${side}-panel-toggle i`);
        icon.className = panel.classList.contains("collapsed") ? "bi bi-chevron-down" : "bi bi-chevron-up";
      });
      rows.forEach(([key]) => {
        document.getElementById(`${side}-${key}`).addEventListener("change", event => {
          sideState[side][key] = event.target.checked;
          updateSideLayers(side);
        });
      });
    }
    function createMap(side) {
      const map = new maplibregl.Map({
        container: `compare-map-${side}`,
        style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        center: [INITIAL_VIEW_STATE.longitude, INITIAL_VIEW_STATE.latitude],
        zoom: INITIAL_VIEW_STATE.zoom,
        minZoom: INITIAL_VIEW_STATE.minZoom,
        maxZoom: INITIAL_VIEW_STATE.maxZoom,
        bearing: INITIAL_VIEW_STATE.bearing,
        pitch: INITIAL_VIEW_STATE.pitch
      });
      map.addControl(new maplibregl.NavigationControl(), side === "right" ? "top-left" : "top-right");
      map.on("style.load", () => {
        if (typeof map.setProjection === "function") {
          map.setProjection({type: "globe"});
        }
        if (typeof map.setFog === "function") {
          map.setFog({
            color: "rgba(255,255,255,0.03)",
            "high-color": "#3a3a3a",
            "horizon-blend": 0.03,
            "space-color": "#2f2f2f",
            "star-intensity": 0
          });
        }
      });
      map.on("move", () => {
        updateZoomScales(map.getZoom());
        if (syncing) return;
        syncing = true;
        const center = map.getCenter();
        const target = side === "left" ? maps.right : maps.left;
        if (target) {
          target.jumpTo({
            center: [center.lng, center.lat],
            zoom: map.getZoom(),
            bearing: map.getBearing(),
            pitch: map.getPitch()
          });
        }
        syncing = false;
      });
      map.once("load", () => {
        overlays[side] = new MapboxOverlay({
          interleaved: false,
          layers: makeLayers(side),
          parameters: {cull: true},
          getTooltip: ({object}) => tooltipFor(object)
        });
        map.addControl(overlays[side]);
        updateSideLayers(side);
      });
      maps[side] = map;
    }
    function wireBottomControls() {
      const compactView = window.matchMedia("(max-width: 900px)").matches;
      const bottomController = document.getElementById("compareBottomController");
      const bottomToggle = document.getElementById("compareBottomToggle");
      function setCompareBottomCollapsed(collapsed) {
        bottomController.classList.toggle("collapsed", collapsed);
        document.body.classList.toggle("compare-bottom-collapsed", collapsed);
        bottomToggle.setAttribute("aria-expanded", String(!collapsed));
        const icon = bottomToggle.querySelector("i");
        icon.className = collapsed ? "bi bi-sliders2" : "bi bi-chevron-down";
      }
      if (bottomController && bottomToggle) {
        setCompareBottomCollapsed(compactView);
        bottomToggle.addEventListener("click", () => {
          setCompareBottomCollapsed(!bottomController.classList.contains("collapsed"));
        });
      }

      const scaleControl = document.getElementById("compareScaleControl");
      const scaleToggle = document.getElementById("compareScaleToggle");
      function setCompareScaleCollapsed(collapsed) {
        scaleControl.classList.toggle("collapsed", collapsed);
        scaleToggle.setAttribute("aria-expanded", String(!collapsed));
        const icon = scaleToggle.querySelector("i");
        icon.className = collapsed ? "bi bi-palette2" : "bi bi-chevron-down";
      }
      if (scaleControl && scaleToggle) {
        setCompareScaleCollapsed(compactView);
        scaleToggle.addEventListener("click", () => {
          setCompareScaleCollapsed(!scaleControl.classList.contains("collapsed"));
        });
      }

      function setCompareVariable(variable) {
        shared.renderVariable = variable;
        if (variable === "uncertainty") {
          loadUncertaintyPayloads().then(updateAllLayers);
        } else {
          updateAllLayers();
        }
      }
      document.querySelectorAll("input[name='compare-render-variable']").forEach(input => {
        input.addEventListener("change", event => {
          if (event.target.checked) setCompareVariable(event.target.value);
        });
      });
      const color = document.getElementById("compare-color-limit");
      color.addEventListener("input", event => {
        if (shared.renderVariable === "uncertainty") {
          shared.uncertaintyLimit = Number(event.target.value);
        } else {
          shared.colorLimit = Number(event.target.value);
        }
        updateAllLayers();
      });
      const gia = document.getElementById("compare-gia-opacity");
      gia.addEventListener("input", event => {
        shared.giaOpacity = Number(event.target.value) / 100;
        document.getElementById("compare-gia-value").textContent = Math.round(shared.giaOpacity * 100);
        updateAllLayers();
      });
      const insar = document.getElementById("compare-insar-opacity");
      insar.addEventListener("input", event => {
        shared.insarOpacity = Number(event.target.value) / 100;
        document.getElementById("compare-insar-value").textContent = Math.round(shared.insarOpacity * 100);
        updateAllLayers();
      });
      const pop = document.getElementById("compare-pop-opacity");
      pop.addEventListener("input", event => {
        shared.populationOpacity = Number(event.target.value) / 100;
        document.getElementById("compare-pop-value").textContent = Math.round(shared.populationOpacity * 100);
        updateAllLayers();
      });
      const duration = document.getElementById("compare-duration");
      duration.addEventListener("input", event => {
        shared.minDuration = Number(event.target.value);
        document.getElementById("compare-duration-value").textContent = shared.minDuration.toFixed(1);
        updateAllLayers();
      });
      const first = document.getElementById("compare-first");
      first.min = METADATA.first_epoch_min;
      first.max = METADATA.first_epoch_max;
      first.value = shared.minFirstEpoch;
      document.getElementById("compare-first-value").textContent = formatNumber(shared.minFirstEpoch, 1);
      first.addEventListener("input", event => {
        shared.minFirstEpoch = Number(event.target.value);
        document.getElementById("compare-first-value").textContent = formatNumber(shared.minFirstEpoch, 1);
        updateAllLayers();
      });
      const last = document.getElementById("compare-last");
      last.min = METADATA.last_epoch_min;
      last.max = METADATA.last_epoch_max;
      last.value = shared.maxLastEpoch;
      document.getElementById("compare-last-value").textContent = formatNumber(shared.maxLastEpoch, 1);
      last.addEventListener("input", event => {
        shared.maxLastEpoch = Number(event.target.value);
        document.getElementById("compare-last-value").textContent = formatNumber(shared.maxLastEpoch, 1);
        updateAllLayers();
      });
      document.getElementById("compare-search").addEventListener("input", event => {
        shared.search = event.target.value.trim().toLowerCase();
        updateAllLayers();
      });
      document.getElementById("compare-pop-left").addEventListener("change", event => {
        sideState.left.showPopulation = event.target.checked;
        updateSideLayers("left");
      });
      document.getElementById("compare-pop-right").addEventListener("change", event => {
        sideState.right.showPopulation = event.target.checked;
        updateSideLayers("right");
      });
      document.getElementById("compareShowcaseDisclaimerClose").addEventListener("click", () => {
        document.getElementById("compareShowcaseDisclaimer").hidden = true;
      });
    }
    createPanel("left");
    createPanel("right");
    wireBottomControls();
    createMap("left");
    createMap("right");
  </script>
  <script data-goatcounter="https://global-vlm.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


def write_html(output: Path, html: str) -> None:
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a MapLibre globe MIDAS GPS UP velocity site."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_HTML,
        help=f"HTML output path, default: {OUTPUT_HTML}",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=CATALOG_HTML,
        help=f"Dataset catalogue HTML output path, default: {CATALOG_HTML}",
    )
    parser.add_argument(
        "--about-output",
        type=Path,
        default=ABOUT_HTML,
        help=f"About HTML output path, default: {ABOUT_HTML}",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=COMPARE_HTML,
        help=f"Comparison HTML output path, default: {COMPARE_HTML}",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Fetch the live MIDAS file even if a cache already exists.",
    )
    parser.add_argument(
        "--refresh-gia",
        action="store_true",
        help="Fetch the JPL VESL GIA grid even if a cache already exists.",
    )
    parser.add_argument(
        "--refresh-ngl-imaged",
        action="store_true",
        help="Fetch the NGL GPS Imaging interpolated VLM text grid.",
    )
    parser.add_argument(
        "--refresh-insar",
        action="store_true",
        help="Fetch and extract the Zenodo InSAR VLM GeoTIFF archive.",
    )
    parser.add_argument(
        "--refresh-gns",
        action="store_true",
        help="Fetch and extract the GNS New Zealand InSAR/GNSS VLM archive.",
    )
    parser.add_argument(
        "--refresh-tide-gauge",
        action="store_true",
        help="Fetch the Zenodo CSL-TG tide-gauge MAT file.",
    )
    parser.add_argument(
        "--refresh-hybrid",
        action="store_true",
        help="Fetch the Zenodo Oelsmann hybrid coastal VLM NetCDF file.",
    )
    parser.add_argument(
        "--refresh-external",
        action="store_true",
        help="Fetch and extract external context datasets such as GHSL population.",
    )
    args = parser.parse_args()

    raw_text, source = load_raw_midas(force_refresh=args.force_refresh)
    raw_ngl_imaged_text, ngl_imaged_source = load_raw_ngl_imaged_vlm(
        force_refresh=args.refresh_ngl_imaged
    )
    raw_gia_text, gia_source = load_raw_gia(force_refresh=args.refresh_gia)
    insar_source = ensure_insar_dataset(force_refresh=args.refresh_insar)
    gns_source = ensure_gns_dataset(force_refresh=args.refresh_gns)
    tide_gauge_source = ensure_tide_gauge_dataset(force_refresh=args.refresh_tide_gauge)
    oelsmann_hybrid_source = ensure_oelsmann_hybrid_dataset(force_refresh=args.refresh_hybrid)
    population_source = ensure_ghsl_population_dataset(force_refresh=args.refresh_external)
    records, metadata = parse_midas(raw_text)
    ngl_imaged_values, ngl_imaged_uncertainties, ngl_imaged_metadata = parse_ngl_imaged_vlm_grid(
        raw_ngl_imaged_text
    )
    gia_values, gia_uncertainties, gia_metadata = parse_gia_grid(raw_gia_text)
    insar_grids, insar_metadata = parse_insar_grids()
    gns_records, gns_metadata = parse_gns_coastal_vlm()
    tide_gauge_records, tide_gauge_metadata = parse_tide_gauge_vlm()
    oelsmann_hybrid_records, oelsmann_hybrid_metadata = parse_oelsmann_hybrid_vlm()
    nearby_gnss_metadata = attach_nearby_gnss_to_tide_gauges(tide_gauge_records, records)
    tide_gauge_metadata.update(nearby_gnss_metadata)
    population_metadata = build_ghsl_population_sidecar(force_refresh=args.refresh_external)
    dataset_attributes = build_dataset_attributes(
        records,
        ngl_imaged_metadata,
        gia_metadata,
        insar_grids,
        gns_records,
        gns_metadata,
        tide_gauge_records,
        tide_gauge_metadata,
        oelsmann_hybrid_records,
        oelsmann_hybrid_metadata,
    )
    external_dataset_attributes = build_external_dataset_attributes(population_metadata)
    write_dataset_attribute_files(dataset_attributes)
    write_external_dataset_attribute_files(external_dataset_attributes)
    uncertainty_urls = write_uncertainty_payloads(
        records,
        ngl_imaged_uncertainties,
        ngl_imaged_metadata,
        gia_uncertainties,
        gia_metadata,
        gns_records,
        tide_gauge_records,
        oelsmann_hybrid_records,
    )
    render_urls = write_render_payloads(
        records,
        ngl_imaged_values,
        gia_values,
        insar_grids,
        gns_records,
        tide_gauge_records,
        oelsmann_hybrid_records,
    )
    metadata["raw_data_source_used"] = source
    ngl_imaged_metadata["raw_data_source_used"] = ngl_imaged_source
    gia_metadata["raw_data_source_used"] = gia_source
    insar_metadata["raw_data_source_used"] = insar_source
    gns_metadata["raw_data_source_used"] = gns_source
    tide_gauge_metadata["raw_data_source_used"] = tide_gauge_source
    oelsmann_hybrid_metadata["raw_data_source_used"] = oelsmann_hybrid_source
    population_metadata["raw_data_source_used"] = population_source
    html = build_html(
        records,
        metadata,
        ngl_imaged_values,
        ngl_imaged_metadata,
        gia_values,
        gia_metadata,
        insar_grids,
        insar_metadata,
        gns_records,
        gns_metadata,
        tide_gauge_records,
        tide_gauge_metadata,
        oelsmann_hybrid_records,
        oelsmann_hybrid_metadata,
        dataset_attributes,
        population_metadata,
        external_dataset_attributes,
        render_urls,
        uncertainty_urls,
    )
    write_html(args.output, html)
    catalog_html = build_catalog_html(dataset_attributes)
    write_html(args.catalog_output, catalog_html)
    about_html = build_about_html()
    write_html(args.about_output, about_html)
    compare_html = build_compare_html(
        records,
        metadata,
        ngl_imaged_values,
        ngl_imaged_metadata,
        gia_values,
        gia_metadata,
        insar_grids,
        insar_metadata,
        gns_records,
        gns_metadata,
        tide_gauge_records,
        tide_gauge_metadata,
        oelsmann_hybrid_records,
        oelsmann_hybrid_metadata,
        population_metadata,
        render_urls,
        uncertainty_urls,
    )
    write_html(args.compare_output, compare_html)

    print(f"Created: {args.output}")
    print(f"Created: {args.catalog_output}")
    print(f"Created: {args.about_output}")
    print(f"Created: {args.compare_output}")
    print(f"Stations in render payload: {metadata['station_count']:,}")
    print(f"Skipped malformed rows: {metadata['malformed_rows_skipped']:,}")
    print(f"Raw data source used: {source}")
    print(f"NGL GPS Imaging grid cells in render payload: {ngl_imaged_metadata['value_count']:,}")
    print(f"NGL GPS Imaging raw data source used: {ngl_imaged_source}")
    print(f"GIA grid cells in render payload: {gia_metadata['value_count']:,}")
    print(f"GIA raw data source used: {gia_source}")
    print(f"InSAR grids in render payload: {insar_metadata['grid_count']:,}")
    print(f"InSAR valid pixels in render payload: {insar_metadata['valid_pixel_count']:,}")
    print(f"InSAR raw data source used: {insar_source}")
    print(f"GNS coastal VLM points in render payload: {gns_metadata['record_count']:,}")
    print(f"GNS raw data source used: {gns_source}")
    print(f"CSL-TG tide gauges in render payload: {tide_gauge_metadata['record_count']:,}")
    print(f"CSL-TG raw data source used: {tide_gauge_source}")
    print(f"Oelsmann hybrid coastal VLM points in render payload: {oelsmann_hybrid_metadata['record_count']:,}")
    print(f"Oelsmann hybrid raw data source used: {oelsmann_hybrid_source}")
    print(f"GHSL population pixels in sidecar: {population_metadata['valid_pixel_count']:,}")
    print(f"GHSL population raw data source used: {population_source}")
    print(f"Dataset attribute files written: {ATTRIBUTE_DIR}")
    print(f"External dataset attribute files written: {EXTERNAL_ATTRIBUTE_DIR}")
    print(f"Generated at UTC: {metadata['generated_at_utc']}")


if __name__ == "__main__":
    main()
