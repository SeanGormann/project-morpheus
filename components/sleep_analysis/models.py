"""
Data models for Project Morpheus sleep analysis.

Two data sources:
  - MorpheusSession / Epoch  — realtime 30-second sensor recordings from the watch
  - SleepNight / SleepRecord — Apple Health sleep-stage events
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Apple Health category label mapping
# ---------------------------------------------------------------------------
_AH_CATEGORY_MAP: dict[str, str] = {
    'HKCategoryValueSleepAnalysisAsleep':      'Asleep (Unspecified)',
    'HKCategoryValueSleepAnalysisInBed':       'In Bed',
    'HKCategoryValueSleepAnalysisAwake':       'Awake',
    'HKCategoryValueSleepAnalysisAsleepCore':  'Core Sleep',
    'HKCategoryValueSleepAnalysisAsleepDeep':  'Deep Sleep',
    'HKCategoryValueSleepAnalysisAsleepREM':   'REM Sleep',
}

SLEEP_STAGE_CATEGORIES = frozenset({
    'Deep Sleep', 'Core Sleep', 'REM Sleep', 'Asleep (Unspecified)'
})


# ===========================================================================
# Morpheus realtime sensor models
# ===========================================================================

@dataclass
class MotionSample:
    """
    One accelerometer snapshot inside a 30-second epoch.
    The raw JSON array is [time_offset_sec, x, y, z].
    Gravity pulls at ~1g so magnitude ≈ 1.0 during stationary sleep.
    """
    time_offset: float  # seconds from session start
    x: float
    y: float
    z: float

    @classmethod
    def from_list(cls, data: list[float]) -> MotionSample:
        return cls(time_offset=data[0], x=data[1], y=data[2], z=data[3])

    @property
    def magnitude(self) -> float:
        """Euclidean magnitude of the acceleration vector (≈1g at rest)."""
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def delta_to(self, other: MotionSample) -> float:
        """
        Euclidean distance between two accelerometer vectors.
        Large values indicate body movement between the samples.
        """
        return float(np.sqrt(
            (other.x - self.x)**2 +
            (other.y - self.y)**2 +
            (other.z - self.z)**2
        ))


@dataclass
class Epoch:
    """
    A single 30-second recording interval from the Morpheus watch session.

    heart_rates     — individual BPM readings sampled within the epoch
    hr_feature      — running-normalised HR signal used by the sleep-stage model;
                      starts near HR/1000 at session open, drifts toward a
                      z-score-like value as a baseline is established
    motion_sample_first / _last — accelerometer snapshots at the start and
                      end of the 30-second window
    """
    index:               int
    timestamp_utc:       datetime        # timezone-aware UTC
    heart_rates:         list[int]
    heart_rates_avg:     float
    heart_rates_count:   int
    hr_feature:          float
    motion_rows_count:   int
    motion_sample_first: MotionSample
    motion_sample_last:  MotionSample

    @classmethod
    def from_dict(cls, data: dict) -> Epoch:
        ts = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        return cls(
            index               = data['epochIndex'],
            timestamp_utc       = ts,
            heart_rates         = data.get('heartRates', []),
            heart_rates_avg     = data.get('heartRatesAvg', float('nan')),
            heart_rates_count   = data.get('heartRatesCount', 0),
            hr_feature          = data.get('hrFeature', float('nan')),
            motion_rows_count   = data.get('motionRowsCount', 0),
            motion_sample_first = MotionSample.from_list(
                data.get('motionSampleFirst', [0.0, 0.0, 0.0, 0.0])
            ),
            motion_sample_last  = MotionSample.from_list(
                data.get('motionSampleLast', [0.0, 0.0, 0.0, 0.0])
            ),
        )

    # ---- Computed signals ------------------------------------------------

    @property
    def timestamp_local(self) -> datetime:
        """Epoch timestamp in local system time (timezone-naive)."""
        return self.timestamp_utc.astimezone().replace(tzinfo=None)

    @property
    def motion_delta(self) -> float:
        """
        Change in accelerometer vector between start and end of epoch.
        High values → body repositioning occurred during this 30s window.
        """
        return self.motion_sample_first.delta_to(self.motion_sample_last)

    @property
    def motion_magnitude(self) -> float:
        """Magnitude of the final accelerometer sample (≈1g when stationary)."""
        return self.motion_sample_last.magnitude

    @property
    def motion_magnitude_deviation(self) -> float:
        """Absolute deviation from 1g — non-zero when accelerating/moving."""
        return abs(self.motion_magnitude - 1.0)

    @property
    def hr_variability(self) -> float:
        """Intra-epoch heart rate standard deviation."""
        if len(self.heart_rates) < 2:
            return 0.0
        return float(np.std(self.heart_rates))


@dataclass
class MorpheusSession:
    """
    A full recording session from the Morpheus watch.
    Typically covers one full night's sleep at 30-second epoch resolution.
    """
    session_id:    str           # derived from filename
    session_start: datetime      # UTC, timezone-aware
    session_end:   datetime      # UTC, timezone-aware
    epoch_count:   int
    epochs:        list[Epoch] = field(default_factory=list)

    # ---- Construction ----------------------------------------------------

    @classmethod
    def from_file(cls, path: Path | str) -> MorpheusSession:
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        epochs = [Epoch.from_dict(e) for e in data.get('epochs', [])]

        return cls(
            session_id    = path.stem,
            session_start = datetime.fromisoformat(
                data['sessionStart'].replace('Z', '+00:00')
            ),
            session_end   = datetime.fromisoformat(
                data['sessionEnd'].replace('Z', '+00:00')
            ),
            epoch_count   = data.get('epochCount', len(epochs)),
            epochs        = epochs,
        )

    # ---- Conversion ------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame of per-epoch signals, timestamps in local system time.

        Columns
        -------
        timestamp           datetime (local, naive)
        hr_avg              float — mean BPM for the epoch
        hr_feature          float — running-normalised HR model feature
        hr_variability      float — intra-epoch HR std dev
        motion_rows         int   — raw motion sample count
        motion_delta        float — Δ accel between epoch start and end
        motion_magnitude    float — |accel| of last sample (≈1 when still)
        motion_mag_dev      float — |magnitude − 1g|
        """
        rows = []
        for ep in self.epochs:
            rows.append({
                'timestamp':        ep.timestamp_local,
                'hr_avg':           ep.heart_rates_avg,
                'hr_feature':       ep.hr_feature,
                'hr_variability':   ep.hr_variability,
                'motion_rows':      ep.motion_rows_count,
                'motion_delta':     ep.motion_delta,
                'motion_magnitude': ep.motion_magnitude,
                'motion_mag_dev':   ep.motion_magnitude_deviation,
            })
        return pd.DataFrame(rows)

    # ---- Convenience -----------------------------------------------------

    @property
    def duration_hours(self) -> float:
        return (self.session_end - self.session_start).total_seconds() / 3600

    @property
    def start_local(self) -> datetime:
        return self.session_start.astimezone().replace(tzinfo=None)

    @property
    def end_local(self) -> datetime:
        return self.session_end.astimezone().replace(tzinfo=None)

    def __repr__(self) -> str:
        return (f"MorpheusSession(id={self.session_id!r}, "
                f"epochs={self.epoch_count}, "
                f"duration={self.duration_hours:.2f}h, "
                f"start={self.start_local.strftime('%Y-%m-%d %H:%M')} local)")


# ===========================================================================
# Apple Health sleep models
# ===========================================================================

@dataclass
class SleepRecord:
    """
    A single Apple Health sleep-analysis event — one contiguous block of a
    specific sleep stage reported by the watch or iPhone.

    Timestamps are timezone-naive local time (Apple Health export convention).
    """
    start:    datetime
    end:      datetime
    category: str   # 'Deep Sleep' | 'Core Sleep' | 'REM Sleep' |
                    # 'Awake' | 'In Bed' | 'Asleep (Unspecified)'
    source:   str   # e.g. 'Apple Watch', 'iPhone'

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    @property
    def is_sleep(self) -> bool:
        return self.category in SLEEP_STAGE_CATEGORIES

    def __repr__(self) -> str:
        return (f"SleepRecord({self.category!r}, "
                f"{self.start.strftime('%H:%M')}–{self.end.strftime('%H:%M')}, "
                f"{self.duration_minutes:.0f}min)")


@dataclass
class SleepNight:
    """
    All Apple Health sleep records belonging to a single night, grouped by
    the convention: sleep after noon → that calendar date; before noon →
    previous calendar date.
    """
    sleep_date: str          # 'YYYY-MM-DD'
    records:    list[SleepRecord] = field(default_factory=list)

    # ---- Construction ----------------------------------------------------

    @classmethod
    def from_apple_health_xml(
        cls,
        xml_path: Path | str,
        sleep_date: str,
    ) -> SleepNight:
        """Parse one night from an Apple Health export XML file."""
        nights = load_sleep_nights_from_xml(xml_path, target_dates=[sleep_date])
        return nights.get(sleep_date, cls(sleep_date=sleep_date, records=[]))

    # ---- Conversion ------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame matching the shape used by rem_detection_viz.

        Columns: start, end, category, source, duration_hours, sleep_date
        """
        if not self.records:
            return pd.DataFrame(
                columns=['start', 'end', 'category', 'source', 'duration_hours', 'sleep_date']
            )
        rows = [
            {
                'start':          r.start,
                'end':            r.end,
                'category':       r.category,
                'source':         r.source,
                'duration_hours': r.duration_hours,
                'sleep_date':     self.sleep_date,
            }
            for r in sorted(self.records, key=lambda r: r.start)
        ]
        return pd.DataFrame(rows)

    # ---- Computed properties ---------------------------------------------

    @property
    def sleep_records(self) -> list[SleepRecord]:
        return [r for r in self.records if r.is_sleep]

    @property
    def sleep_onset(self) -> Optional[datetime]:
        """Earliest transition into an actual sleep stage."""
        sr = self.sleep_records
        return min(r.start for r in sr) if sr else None

    @property
    def wake_time(self) -> Optional[datetime]:
        """Latest end time across all sleep records."""
        sr = self.sleep_records
        return max(r.end for r in sr) if sr else None

    @property
    def total_sleep_hours(self) -> float:
        return sum(r.duration_hours for r in self.sleep_records)

    @property
    def time_in_bed_hours(self) -> float:
        if not self.records:
            return 0.0
        return (
            max(r.end   for r in self.records) -
            min(r.start for r in self.records)
        ).total_seconds() / 3600

    @property
    def sleep_efficiency(self) -> float:
        tib = self.time_in_bed_hours
        return (self.total_sleep_hours / tib * 100) if tib > 0 else 0.0

    @property
    def rem_hours(self) -> float:
        return sum(r.duration_hours for r in self.records if r.category == 'REM Sleep')

    @property
    def deep_hours(self) -> float:
        return sum(r.duration_hours for r in self.records if r.category == 'Deep Sleep')

    @property
    def core_hours(self) -> float:
        return sum(r.duration_hours for r in self.records if r.category == 'Core Sleep')

    @property
    def awake_hours(self) -> float:
        return sum(r.duration_hours for r in self.records if r.category == 'Awake')

    @property
    def rem_cycles(self) -> list[list[SleepRecord]]:
        """
        Group REM records into ultradian cycles.
        REM periods separated by >30 min are considered distinct cycles.
        """
        rem_recs = sorted(
            [r for r in self.records if r.category == 'REM Sleep'],
            key=lambda r: r.start,
        )
        if not rem_recs:
            return []

        cycles: list[list[SleepRecord]] = []
        current: list[SleepRecord] = [rem_recs[0]]
        for rec in rem_recs[1:]:
            gap = (rec.start - current[-1].end).total_seconds() / 60
            if gap > 30:
                cycles.append(current)
                current = [rec]
            else:
                current.append(rec)
        cycles.append(current)
        return cycles

    def __repr__(self) -> str:
        return (f"SleepNight({self.sleep_date}, "
                f"records={len(self.records)}, "
                f"sleep={self.total_sleep_hours:.1f}h, "
                f"efficiency={self.sleep_efficiency:.0f}%)")


# ===========================================================================
# Loader helpers
# ===========================================================================

def load_sleep_nights_from_xml(
    xml_path: Path | str,
    target_dates: Optional[list[str]] = None,
) -> dict[str, SleepNight]:
    """
    Parse an Apple Health export XML and return a dict of {sleep_date: SleepNight}.

    Parameters
    ----------
    xml_path      Path to export.xml (or a zip containing it).
    target_dates  Optional list of 'YYYY-MM-DD' strings to filter to.
                  If None, all dates are returned.
    """
    xml_path = Path(xml_path)

    # Transparently handle export.zip
    if xml_path.suffix == '.zip':
        with zipfile.ZipFile(xml_path, 'r') as zf:
            inner = 'apple_health_export/export.xml'
            with zf.open(inner) as f:
                root = ET.parse(f).getroot()
    else:
        root = ET.parse(xml_path).getroot()

    nights: dict[str, SleepNight] = {}

    for rec in root.findall('.//Record[@type="HKCategoryTypeIdentifierSleepAnalysis"]'):
        start = datetime.strptime(rec.get('startDate'), '%Y-%m-%d %H:%M:%S %z').replace(tzinfo=None)
        end   = datetime.strptime(rec.get('endDate'),   '%Y-%m-%d %H:%M:%S %z').replace(tzinfo=None)

        sleep_date = (
            str(start.date()) if start.hour >= 12
            else str((start - timedelta(days=1)).date())
        )

        if target_dates is not None and sleep_date not in target_dates:
            continue

        category = _AH_CATEGORY_MAP.get(rec.get('value'), rec.get('value', ''))
        sr = SleepRecord(
            start    = start,
            end      = end,
            category = category,
            source   = rec.get('sourceName', 'Unknown'),
        )

        if sleep_date not in nights:
            nights[sleep_date] = SleepNight(sleep_date=sleep_date)
        nights[sleep_date].records.append(sr)

    return nights


def resolve_export_xml(base_dir: Path) -> Optional[Path]:
    """
    Find and, if necessary, unzip the Apple Health export XML.
    Checks base_dir for export.zip and apple_health_export/export.xml.
    """
    xml_path = base_dir / 'apple_health_export' / 'export.xml'
    zip_path = base_dir / 'export.zip'

    if xml_path.exists():
        return xml_path

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract('apple_health_export/export.xml', base_dir)
        if xml_path.exists():
            return xml_path

    return None
