#!/usr/bin/env python3
"""
REM Detection Visualization
Combines Apple Health sleep hypnogram with Morpheus realtime sensor data
for specific nights, saving combined plots to rem_detection/
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime, timedelta, date
from pathlib import Path

from models import (
    MorpheusSession,
    SleepNight,
    load_sleep_nights_from_xml,
    resolve_export_xml,
)

# ---------------------------------------------------------------------------
# Configuration — map Apple Health sleep_date labels to Morpheus session files
# ---------------------------------------------------------------------------
SESSIONS: dict[str, str] = {
    '2026-02-17': 'morpheus_session_1771401783.json',
    '2026-02-18': 'morpheus_session_1771491471.json',
}

DARK_BG   = '#0D1117'
PANEL_BG  = '#111820'
GRID_COL  = '#2A3040'
TEXT_COL  = '#C8D0DC'

STAGE_Y: dict[str, float] = {
    'Awake':                3,
    'REM Sleep':            2,
    'Core Sleep':           1,
    'Deep Sleep':           0,
    'In Bed':              -0.5,
    'Asleep (Unspecified)': 0.5,
}
STAGE_COLOR: dict[str, str] = {
    'Deep Sleep':           '#2E4057',
    'Core Sleep':           '#5C7A99',
    'REM Sleep':            '#A8DADC',
    'Awake':                '#E63946',
    'In Bed':               '#3A3A4A',
    'Asleep (Unspecified)': '#8B9D83',
}


def _style_ax(ax, plot_t0: datetime, plot_t1: datetime, midnight: datetime) -> None:
    """Apply shared dark-theme styling to an axes panel."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(True, axis='x', color=GRID_COL, linestyle='--', linewidth=0.6, alpha=0.8)
    ax.grid(True, axis='y', color=GRID_COL, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlim(plot_t0, plot_t1)
    if plot_t0 <= midnight <= plot_t1:
        ax.axvline(midnight, color='#555566', linestyle='--', linewidth=1.2, alpha=0.7)


def plot_combined(night: SleepNight, session: MorpheusSession, output_dir: Path) -> None:
    """
    Three-panel figure for one night:
      [0] Sleep hypnogram (Apple Health)
      [1] Heart rate per epoch (Morpheus)
      [2] Body movement per epoch (Morpheus)
    """
    night_df = night.to_dataframe().sort_values('start')
    morph_df = session.to_dataframe()

    t_start = night_df['start'].min()
    t_end   = night_df['end'].max()

    # Clip Morpheus data to the sleep window
    buf = timedelta(minutes=20)
    morph = morph_df[
        (morph_df['timestamp'] >= t_start - buf) &
        (morph_df['timestamp'] <= t_end   + buf)
    ].copy()

    pad      = timedelta(minutes=25)
    plot_t0  = t_start - pad
    plot_t1  = t_end   + pad
    midnight = datetime.combine(
        date.fromisoformat(night.sleep_date) + timedelta(days=1),
        datetime.min.time(),
    )

    # ---- Layout ----------------------------------------------------------
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3, 1.6, 1.6],
        hspace=0.06,
        top=0.93, bottom=0.09, left=0.07, right=0.97,
    )
    ax_hyp = fig.add_subplot(gs[0])
    ax_hr  = fig.add_subplot(gs[1], sharex=ax_hyp)
    ax_mov = fig.add_subplot(gs[2], sharex=ax_hyp)

    for ax in (ax_hyp, ax_hr, ax_mov):
        _style_ax(ax, plot_t0, plot_t1, midnight)

    # =====================================================================
    # Panel 0: Hypnogram
    # =====================================================================
    pts_t, pts_y, pts_c = [], [], []
    for _, row in night_df.iterrows():
        y = STAGE_Y.get(row['category'], 0)
        c = STAGE_COLOR.get(row['category'], '#7f7f7f')
        pts_t += [row['start'], row['end']]
        pts_y += [y, y]
        pts_c += [c, c]

    for i in range(len(pts_t) - 1):
        ax_hyp.plot([pts_t[i], pts_t[i + 1]], [pts_y[i], pts_y[i + 1]],
                    color=pts_c[i], linewidth=2.5, solid_capstyle='round')

    for _, row in night_df.iterrows():
        y = STAGE_Y.get(row['category'], 0)
        c = STAGE_COLOR.get(row['category'], '#7f7f7f')
        ax_hyp.fill_between([row['start'], row['end']], -0.8, y,
                             color=c, alpha=0.18, linewidth=0)

    ax_hyp.set_ylim(-0.8, 3.7)
    ax_hyp.set_yticks([0, 1, 2, 3])
    ax_hyp.set_yticklabels(['Deep', 'Core', 'REM', 'Awake'], color=TEXT_COL, fontsize=10)
    ax_hyp.set_ylabel('Sleep Stage', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_hyp.tick_params(labelbottom=False)

    if plot_t0 <= midnight <= plot_t1:
        ax_hyp.text(midnight, 3.45, 'Midnight',
                    ha='center', fontsize=8, color='#777788', style='italic')

    legend_els = [Patch(facecolor=STAGE_COLOR[s], label=s)
                  for s in ('Deep Sleep', 'Core Sleep', 'REM Sleep', 'Awake')]
    ax_hyp.legend(handles=legend_els, loc='upper left', framealpha=0.35,
                  fontsize=9, facecolor='#1A1E28', labelcolor=TEXT_COL, edgecolor=GRID_COL)

    # =====================================================================
    # Panel 1: Heart Rate
    # =====================================================================
    if len(morph) > 0:
        ax_hr.plot(morph['timestamp'], morph['hr_avg'],
                   color='#FF7070', linewidth=1.2, alpha=0.7, label='HR per epoch')
        ax_hr.fill_between(morph['timestamp'],
                           morph['hr_avg'].min() - 2, morph['hr_avg'],
                           color='#FF7070', alpha=0.08)

        if len(morph) >= 7:
            hr_smooth = morph['hr_avg'].rolling(7, center=True, min_periods=3).mean()
            ax_hr.plot(morph['timestamp'], hr_smooth,
                       color='#FFD700', linewidth=2.0, alpha=0.9,
                       label='HR smoothed (7-epoch)')

        # hr_feature on secondary axis — normalised model input signal
        ax_hr2 = ax_hr.twinx()
        ax_hr2.set_facecolor(PANEL_BG)
        ax_hr2.plot(morph['timestamp'], morph['hr_feature'],
                    color='#88DDFF', linewidth=1.0, alpha=0.55,
                    linestyle='--', label='hrFeature')
        ax_hr2.axhline(0, color='#88DDFF', linewidth=0.6, alpha=0.3, linestyle=':')
        ax_hr2.set_ylabel('hrFeature', color='#88DDFF', fontsize=9)
        ax_hr2.tick_params(colors='#88DDFF', labelsize=8)
        for spine in ax_hr2.spines.values():
            spine.set_color(GRID_COL)

        h1, l1 = ax_hr.get_legend_handles_labels()
        h2, l2 = ax_hr2.get_legend_handles_labels()
        ax_hr.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.35,
                     fontsize=8, facecolor='#1A1E28', labelcolor=TEXT_COL, edgecolor=GRID_COL)

    ax_hr.set_ylabel('Heart Rate (bpm)', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_hr.tick_params(labelbottom=False)

    # =====================================================================
    # Panel 2: Movement
    # =====================================================================
    if len(morph) > 0:
        ax_mov.fill_between(morph['timestamp'], 0, morph['motion_delta'],
                            color='#7EB8F7', alpha=0.45, label='Movement (Δ accel)')
        ax_mov.plot(morph['timestamp'], morph['motion_delta'],
                    color='#7EB8F7', linewidth=1.0, alpha=0.8)
        ax_mov.plot(morph['timestamp'], morph['motion_mag_dev'],
                    color='#B8A7F7', linewidth=1.2, alpha=0.7,
                    label='|accel magnitude − 1g|')
        ax_mov.legend(loc='upper right', framealpha=0.35, fontsize=8,
                      facecolor='#1A1E28', labelcolor=TEXT_COL, edgecolor=GRID_COL)

    ax_mov.set_ylabel('Motion', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_mov.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
    ax_mov.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    plt.setp(ax_mov.xaxis.get_majorticklabels(),
             rotation=45, ha='right', color=TEXT_COL, fontsize=9)
    ax_mov.set_xlabel('Time', color=TEXT_COL, fontsize=10, fontweight='bold')

    # =====================================================================
    # Title + summary footer
    # =====================================================================
    day_label = datetime.strptime(night.sleep_date, '%Y-%m-%d').strftime('%B %d, %Y')
    fig.suptitle(f'Sleep + Sensor Data  —  Night of {day_label}',
                 fontsize=14, fontweight='bold', color='white', y=0.97)

    summary = (
        f'Time in Bed: {night.time_in_bed_hours:.1f}h   |   '
        f'Total Sleep: {night.total_sleep_hours:.1f}h   |   '
        f'REM: {night.rem_hours:.1f}h   |   '
        f'Deep: {night.deep_hours:.1f}h   |   '
        f'Efficiency: {night.sleep_efficiency:.0f}%   |   '
        f'Sensor Epochs: {len(morph)}'
    )
    fig.text(0.5, 0.018, summary, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1A2535',
                       alpha=0.85, edgecolor=GRID_COL),
             color=TEXT_COL, fontweight='bold')

    out_file = output_dir / f'sleep_sensor_{night.sleep_date}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved → {out_file}")


def main() -> None:
    print("REM Detection Visualization")
    print("=" * 50)

    base_dir   = Path(__file__).parent
    output_dir = base_dir / 'rem_detection'
    output_dir.mkdir(exist_ok=True)

    export_xml = resolve_export_xml(base_dir)
    if export_xml is None:
        print("ERROR: Apple Health export not found. "
              "Place export.zip or apple_health_export/export.xml next to this script.")
        return

    target_dates = list(SESSIONS.keys())
    nights = load_sleep_nights_from_xml(export_xml, target_dates=target_dates)
    print(f"Loaded Apple Health data for {len(nights)} night(s)")

    for sleep_date, session_file in SESSIONS.items():
        print(f"\n── Night: {sleep_date} ──────────────────────────")

        night = nights.get(sleep_date)
        if night is None or not night.records:
            print(f"  No Apple Health data found for {sleep_date}, skipping.")
            continue

        session_path = base_dir / session_file
        if not session_path.exists():
            print(f"  Morpheus file not found: {session_file}, skipping.")
            continue

        session = MorpheusSession.from_file(session_path)

        print(f"  {night}")
        print(f"  {session}")
        print(f"  Sleep window : {night.sleep_onset.strftime('%H:%M')} → "
              f"{night.wake_time.strftime('%H:%M')} (local)")
        print(f"  Session      : {session.start_local.strftime('%H:%M')} → "
              f"{session.end_local.strftime('%H:%M')} (local)")

        plot_combined(night, session, output_dir)

    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
