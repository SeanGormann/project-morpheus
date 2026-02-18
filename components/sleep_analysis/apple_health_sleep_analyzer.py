#!/usr/bin/env python3
"""
Apple Health Sleep Data Analyzer
Extracts and visualizes sleep data from Apple Health export
"""

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import zipfile

def extract_sleep_data(xml_file):
    """Extract sleep data from Apple Health export XML"""
    print("Parsing XML file (this may take a minute)...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    sleep_records = []
    
    # Sleep categories mapping
    sleep_categories = {
        'HKCategoryValueSleepAnalysisAsleep': 'Asleep (Unspecified)',
        'HKCategoryValueSleepAnalysisInBed': 'In Bed',
        'HKCategoryValueSleepAnalysisAwake': 'Awake',
        'HKCategoryValueSleepAnalysisAsleepCore': 'Core Sleep',
        'HKCategoryValueSleepAnalysisAsleepDeep': 'Deep Sleep',
        'HKCategoryValueSleepAnalysisAsleepREM': 'REM Sleep',
    }
    
    for record in root.findall('.//Record[@type="HKCategoryTypeIdentifierSleepAnalysis"]'):
        start_date = datetime.strptime(record.get('startDate'), '%Y-%m-%d %H:%M:%S %z')
        end_date = datetime.strptime(record.get('endDate'), '%Y-%m-%d %H:%M:%S %z')
        value = record.get('value')
        
        sleep_records.append({
            'start': start_date,
            'end': end_date,
            'duration_hours': (end_date - start_date).total_seconds() / 3600,
            'category': sleep_categories.get(value, value),
            'source': record.get('sourceName', 'Unknown')
        })
    
    df = pd.DataFrame(sleep_records)
    print(f"Found {len(df)} sleep records")
    return df

def group_by_nights(df):
    """Group sleep records into individual nights"""
    # Convert to timezone-naive for easier processing
    df = df.copy()
    df['start'] = pd.to_datetime(df['start']).dt.tz_localize(None)
    df['end'] = pd.to_datetime(df['end']).dt.tz_localize(None)
    
    # Filter for 2025 onwards
    df = df[df['start'] >= '2025-01-01']
    print(f"Filtered to {len(df)} sleep records from 2025 onwards")
    
    # Create a "sleep date" - if sleep starts after 6 PM, it belongs to that day
    # If it starts before noon, it belongs to the previous day
    df['sleep_date'] = df['start'].apply(
        lambda x: x.date() if x.hour >= 12 else (x - timedelta(days=1)).date()
    )
    
    return df

def plot_single_night(night_data, sleep_date, output_dir):
    """Create a visualization for a single night's sleep"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Color mapping for sleep stages
    colors = {
        'Deep Sleep': '#2E4057',      # Dark blue
        'Core Sleep': '#5C7A99',       # Medium blue
        'REM Sleep': '#A8DADC',        # Light blue
        'Awake': '#E63946',            # Red
        'In Bed': '#F1FAEE',           # Off-white
        'Asleep (Unspecified)': '#8B9D83'  # Grey-green
    }
    
    # Stage to y-position mapping (only show main sleep stages)
    stage_positions = {
        'Awake': 3,
        'REM Sleep': 2,
        'Core Sleep': 1,
        'Deep Sleep': 0,
        'In Bed': -1,  # Keep for plotting but won't show on y-axis
        'Asleep (Unspecified)': 0.5  # Keep for plotting but won't show on y-axis
    }
    
    # Sort by start time
    night_data = night_data.sort_values('start')
    
    # Get the time bounds dynamically
    first_sleep = night_data['start'].min()
    last_wake = night_data['end'].max()
    
    # Add 1.5 hour padding
    plot_start = first_sleep - timedelta(hours=1.5)
    plot_end = last_wake + timedelta(hours=1.5)
    
    # Create continuous line data points for the hypnogram
    time_points = []
    stage_values = []
    stage_colors_line = []
    
    for _, row in night_data.iterrows():
        start_time = row['start']
        end_time = row['end']
        category = row['category']
        y_pos = stage_positions.get(category, 0)
        color = colors.get(category, '#7f7f7f')
        
        # Add start point
        time_points.append(start_time)
        stage_values.append(y_pos)
        stage_colors_line.append(color)
        
        # Add end point
        time_points.append(end_time)
        stage_values.append(y_pos)
        stage_colors_line.append(color)
    
    # Plot the continuous line through sleep stages
    for i in range(len(time_points) - 1):
        ax1.plot([time_points[i], time_points[i+1]], 
                [stage_values[i], stage_values[i+1]], 
                color=stage_colors_line[i], linewidth=3, 
                solid_capstyle='round')
    
    # Also fill the area under the line for better visualization
    for _, row in night_data.iterrows():
        start_time = row['start']
        end_time = row['end']
        category = row['category']
        y_pos = stage_positions.get(category, 0)
        color = colors.get(category, '#7f7f7f')
        
        # Fill area
        ax1.fill_between([start_time, end_time], 
                         -1.5, y_pos,
                         color=color, alpha=0.2, linewidth=0)
    
    # Configure the hypnogram
    ax1.set_xlim(plot_start, plot_end)
    ax1.set_ylim(-0.5, 3.5)
    
    # Only show main sleep stages on y-axis
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Deep Sleep', 'Core Sleep', 'REM Sleep', 'Awake'])
    ax1.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Format x-axis with 15-minute intervals
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Deep Sleep'], label='Deep Sleep'),
        Patch(facecolor=colors['Core Sleep'], label='Core Sleep'),
        Patch(facecolor=colors['REM Sleep'], label='REM Sleep'),
        Patch(facecolor=colors['Awake'], label='Awake')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax1.grid(True, alpha=0.2, axis='y', linestyle=':')
    
    # Add vertical line for midnight if it falls within the sleep period
    midnight = datetime.combine(sleep_date + timedelta(days=1), datetime.min.time())
    if plot_start <= midnight <= plot_end:
        ax1.axvline(midnight, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.text(midnight, 3.3, 'Midnight', ha='center', fontsize=9, alpha=0.7)
    
    # Title with date
    ax1.set_title(f'Sleep Hypnogram - Night of {sleep_date.strftime("%B %d, %Y")}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Plot 2: Sleep stage duration bar chart (exclude In Bed and Asleep Unspecified)
    relevant_stages = ['Deep Sleep', 'Core Sleep', 'REM Sleep', 'Awake']
    stage_durations = night_data[night_data['category'].isin(relevant_stages)].groupby('category')['duration_hours'].sum()
    
    # Sort by stage position for consistent ordering
    stage_order = ['Deep Sleep', 'Core Sleep', 'REM Sleep', 'Awake']
    stage_durations = stage_durations.reindex([s for s in stage_order if s in stage_durations.index])
    
    if len(stage_durations) > 0:
        stage_colors_list = [colors.get(stage, '#7f7f7f') for stage in stage_durations.index]
        bars = ax2.barh(range(len(stage_durations)), stage_durations.values, 
                        color=stage_colors_list, edgecolor='black', linewidth=1)
        ax2.set_yticks(range(len(stage_durations)))
        ax2.set_yticklabels(stage_durations.index)
        ax2.set_xlabel('Duration (hours)', fontsize=11, fontweight='bold')
        ax2.set_title('Time in Each Stage', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add duration labels on bars
        for i, (bar, duration) in enumerate(zip(bars, stage_durations.values)):
            ax2.text(duration + 0.05, i, f'{duration:.1f}h', 
                    va='center', fontsize=10, fontweight='bold')
    
    # Calculate sleep metrics
    sleep_stages = ['Deep Sleep', 'Core Sleep', 'REM Sleep', 'Asleep (Unspecified)']
    total_sleep = night_data[night_data['category'].isin(sleep_stages)]['duration_hours'].sum()
    time_in_bed = (last_wake - first_sleep).total_seconds() / 3600
    efficiency = (total_sleep / time_in_bed * 100) if time_in_bed > 0 else 0
    
    # Add summary text
    summary_text = (f'Time in Bed: {time_in_bed:.1f}h  |  '
                   f'Total Sleep: {total_sleep:.1f}h  |  '
                   f'Sleep Efficiency: {efficiency:.0f}%')
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.8),
             fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Save figure
    output_file = output_dir / f'sleep_{sleep_date}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return total_sleep, time_in_bed

def create_rem_analysis(detailed_df, sleep_summary_df, output_dir):
    """Analyze REM cycles and create prediction visualizations"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: REM onset time distribution (minutes after sleep onset)
    ax1 = fig.add_subplot(gs[0, 0])
    for cycle_num in sorted(detailed_df['rem_cycle_number'].unique()):
        cycle_data = detailed_df[detailed_df['rem_cycle_number'] == cycle_num]['rem_start_minutes_after_sleep']
        if len(cycle_data) > 0:
            ax1.hist(cycle_data, bins=20, alpha=0.6, label=f'Cycle {cycle_num}')
    
    ax1.set_xlabel('Minutes After Sleep Onset', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('When Does REM Sleep Occur After You Fall Asleep?', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: REM duration by cycle number
    ax2 = fig.add_subplot(gs[0, 1])
    cycle_stats = detailed_df.groupby('rem_cycle_number')['rem_duration_minutes'].agg(['mean', 'std', 'count'])
    ax2.bar(cycle_stats.index, cycle_stats['mean'], yerr=cycle_stats['std'], 
            capsize=5, alpha=0.7, color='#5C7A99', edgecolor='black')
    ax2.set_xlabel('REM Cycle Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Duration (minutes)', fontsize=11, fontweight='bold')
    ax2.set_title('Average REM Duration by Cycle', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, (idx, row) in enumerate(cycle_stats.iterrows()):
        ax2.text(idx, row['mean'] + (row['std'] if pd.notna(row['std']) else 0) + 2, 
                f"n={int(row['count'])}", ha='center', fontsize=9)
    
    # Plot 3: REM timing prediction - scatter plot
    ax3 = fig.add_subplot(gs[1, :])
    colors_cycles = plt.cm.viridis(np.linspace(0, 1, detailed_df['rem_cycle_number'].nunique()))
    
    for idx, cycle_num in enumerate(sorted(detailed_df['rem_cycle_number'].unique())):
        cycle_data = detailed_df[detailed_df['rem_cycle_number'] == cycle_num]
        ax3.scatter(cycle_data['sleep_onset_hour'], 
                   cycle_data['rem_start_minutes_after_sleep'],
                   alpha=0.6, s=60, label=f'Cycle {cycle_num}',
                   color=colors_cycles[idx])
    
    ax3.set_xlabel('Sleep Onset Time (24-hour)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Minutes Until REM Starts', fontsize=11, fontweight='bold')
    ax3.set_title('REM Cycle Timing vs Sleep Onset Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis as time
    hour_labels = [f"{int(h)}:00" if h == int(h) else f"{int(h)}:30" 
                   for h in np.arange(int(detailed_df['sleep_onset_hour'].min()), 
                                     int(detailed_df['sleep_onset_hour'].max()) + 1, 0.5)]
    
    # Plot 4: REM Timing Summary Statistics
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    # Calculate average timing for each cycle
    summary_text = "REM CYCLE PREDICTIONS\n" + "="*50 + "\n\n"
    
    for cycle_num in sorted(detailed_df['rem_cycle_number'].unique()):
        cycle_data = detailed_df[detailed_df['rem_cycle_number'] == cycle_num]
        mean_onset = cycle_data['rem_start_minutes_after_sleep'].mean()
        std_onset = cycle_data['rem_start_minutes_after_sleep'].std()
        mean_duration = cycle_data['rem_duration_minutes'].mean()
        
        summary_text += f"Cycle {cycle_num}:\n"
        summary_text += f"  • Starts: {mean_onset:.0f} ± {std_onset:.0f} min after sleep\n"
        summary_text += f"  • Duration: {mean_duration:.0f} min\n"
        summary_text += f"  • Occurrences: {len(cycle_data)} nights\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Total REM sleep by bedtime
    ax5 = fig.add_subplot(gs[2, 1])
    sleep_summary_df['sleep_onset_hour'] = pd.to_datetime(sleep_summary_df['sleep_onset']).dt.hour + \
                                           pd.to_datetime(sleep_summary_df['sleep_onset']).dt.minute / 60
    
    ax5.scatter(sleep_summary_df['sleep_onset_hour'], 
               sleep_summary_df['rem_sleep_hours'] * 60,
               alpha=0.6, s=80, color='#A8DADC', edgecolor='black')
    ax5.set_xlabel('Sleep Onset Time (24-hour)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Total REM Sleep (minutes)', fontsize=11, fontweight='bold')
    ax5.set_title('Total REM Sleep vs Bedtime', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('REM Sleep Cycle Analysis & Predictions', fontsize=16, fontweight='bold', y=0.995)
    
    output_file = output_dir / 'rem_cycle_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved REM analysis to {output_file}")

def create_summary_plot(sleep_summary_df, output_dir):
    """Create an overview plot of all sleep data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Sleep duration over time
    ax1.plot(sleep_summary_df['date'], sleep_summary_df['total_sleep_hours'], 
            marker='o', linestyle='-', linewidth=2, markersize=4, label='Total Sleep')
    ax1.plot(sleep_summary_df['date'], sleep_summary_df['time_in_bed_hours'], 
            marker='s', linestyle='--', linewidth=1.5, markersize=3, 
            alpha=0.6, label='Time in Bed')
    ax1.axhline(y=8, color='g', linestyle=':', alpha=0.5, label='8h target')
    ax1.axhline(y=7, color='orange', linestyle=':', alpha=0.5, label='7h target')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Hours', fontsize=12)
    ax1.set_title('Sleep Duration Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Sleep efficiency (sleep time / time in bed)
    efficiency = (sleep_summary_df['total_sleep_hours'] / sleep_summary_df['time_in_bed_hours'] * 100)
    ax2.plot(sleep_summary_df['date'], efficiency, 
            marker='o', linestyle='-', linewidth=2, markersize=4, color='purple')
    ax2.axhline(y=85, color='g', linestyle=':', alpha=0.5, label='Good efficiency (85%+)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Sleep Efficiency (%)', fontsize=12)
    ax2.set_title('Sleep Efficiency Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sleep_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Apple Health Sleep Data Analyzer")
    print("=" * 50)
    
    # Check for export file
    export_zip = Path('export.zip')
    export_xml = Path('export.xml')
    
    if export_zip.exists():
        print("Found export.zip, extracting...")
        with zipfile.ZipFile(export_zip, 'r') as zip_ref:
            zip_ref.extract('apple_health_export/export.xml', '.')
        export_xml = Path('apple_health_export/export.xml')
    elif not export_xml.exists():
        print("\nERROR: No export.zip or export.xml found!")
        print("\nTo export your health data:")
        print("1. Open Health app on iPhone")
        print("2. Tap your profile (top right)")
        print("3. Scroll down and tap 'Export All Health Data'")
        print("4. Save the export.zip file")
        print("5. Place it in the same directory as this script")
        return
    
    # Extract sleep data
    df = extract_sleep_data(export_xml)
    
    if df.empty:
        print("No sleep data found in export!")
        return
    
    # Group by nights
    df = group_by_nights(df)
    
    # Create output directory
    output_dir = Path('sleep_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Process each night
    sleep_dates = sorted(df['sleep_date'].unique())
    print(f"\nProcessing {len(sleep_dates)} nights of sleep data...")
    
    sleep_summary = []
    detailed_records = []
    
    for i, sleep_date in enumerate(sleep_dates, 1):
        night_data = df[df['sleep_date'] == sleep_date]
        
        if len(night_data) > 0:
            total_sleep, time_in_bed = plot_single_night(night_data, sleep_date, output_dir)
            
            # Calculate detailed metrics for this night
            first_sleep = night_data['start'].min()
            last_wake = night_data['end'].max()
            
            # Get stage-specific durations
            stage_durations = night_data.groupby('category')['duration_hours'].sum()
            deep_sleep = stage_durations.get('Deep Sleep', 0)
            core_sleep = stage_durations.get('Core Sleep', 0)
            rem_sleep = stage_durations.get('REM Sleep', 0)
            awake = stage_durations.get('Awake', 0)
            
            # Calculate sleep onset (first actual sleep stage, not just "In Bed")
            actual_sleep_stages = ['Deep Sleep', 'Core Sleep', 'REM Sleep', 'Asleep (Unspecified)']
            sleep_records = night_data[night_data['category'].isin(actual_sleep_stages)]
            
            if len(sleep_records) > 0:
                sleep_onset = sleep_records['start'].min()
                final_wake = sleep_records['end'].max()
                
                # Time to fall asleep (if there's "In Bed" before actual sleep)
                in_bed_records = night_data[night_data['category'] == 'In Bed']
                if len(in_bed_records) > 0:
                    first_in_bed = in_bed_records['start'].min()
                    sleep_latency = (sleep_onset - first_in_bed).total_seconds() / 60  # minutes
                else:
                    sleep_latency = 0
            else:
                sleep_onset = first_sleep
                final_wake = last_wake
                sleep_latency = 0
            
            # Analyze true ultradian sleep cycles (90-120 min cycles)
            # Each cycle should contain progression through sleep stages ending in REM
            rem_records = night_data[night_data['category'] == 'REM Sleep'].sort_values('start')
            rem_cycle_data = []
            
            if len(rem_records) > 0:
                # Group REM periods into true sleep cycles
                # Strategy: REM periods within ~90-120 minutes of each other belong to different cycles
                # If REM periods are very close (<30 min apart), they're part of the same cycle
                
                cycles = []
                current_cycle_rems = []
                last_rem_end = None
                
                for _, rem_record in rem_records.iterrows():
                    rem_start = rem_record['start']
                    rem_end = rem_record['end']
                    
                    # If this is the first REM or it's been >30 min since last REM ended,
                    # this is likely a new cycle
                    if last_rem_end is None or (rem_start - last_rem_end).total_seconds() / 60 > 30:
                        if current_cycle_rems:
                            cycles.append(current_cycle_rems)
                        current_cycle_rems = [rem_record]
                    else:
                        # Close REM periods belong to same cycle
                        current_cycle_rems.append(rem_record)
                    
                    last_rem_end = rem_end
                
                # Add the last cycle
                if current_cycle_rems:
                    cycles.append(current_cycle_rems)
                
                # Now create cycle data - one entry per cycle, using the FIRST REM as the marker
                for cycle_num, cycle_rems in enumerate(cycles, 1):
                    # Use the first REM period in the cycle as the cycle marker
                    first_rem = cycle_rems[0]
                    
                    # Calculate total REM duration in this cycle
                    total_rem_duration = sum(r['duration_hours'] * 60 for r in cycle_rems)
                    
                    time_since_sleep_onset = (first_rem['start'] - sleep_onset).total_seconds() / 60
                    
                    rem_cycle_data.append({
                        'cycle_number': cycle_num,
                        'rem_start_minutes_after_sleep': time_since_sleep_onset,
                        'rem_duration_minutes': total_rem_duration,
                        'rem_start_time': first_rem['start'],
                        'num_rem_periods': len(cycle_rems)
                    })
            
            sleep_summary.append({
                'date': sleep_date,
                'sleep_onset': sleep_onset,
                'wake_time': final_wake,
                'total_sleep_hours': total_sleep,
                'time_in_bed_hours': time_in_bed,
                'deep_sleep_hours': deep_sleep,
                'core_sleep_hours': core_sleep,
                'rem_sleep_hours': rem_sleep,
                'awake_hours': awake,
                'sleep_latency_minutes': sleep_latency,
                'num_rem_cycles': len(rem_records),
                'rem_cycles': rem_cycle_data
            })
            
            # Add detailed records for each REM cycle
            for cycle_info in rem_cycle_data:
                detailed_records.append({
                    'date': sleep_date,
                    'sleep_onset': sleep_onset,
                    'sleep_onset_hour': sleep_onset.hour + sleep_onset.minute / 60,
                    'rem_cycle_number': cycle_info['cycle_number'],
                    'rem_start_minutes_after_sleep': cycle_info['rem_start_minutes_after_sleep'],
                    'rem_duration_minutes': cycle_info['rem_duration_minutes'],
                    'rem_start_time': cycle_info['rem_start_time'],
                    'rem_start_hour': cycle_info['rem_start_time'].hour + cycle_info['rem_start_time'].minute / 60,
                    'total_sleep_hours': total_sleep,
                    'deep_sleep_hours': deep_sleep,
                    'core_sleep_hours': core_sleep,
                    'num_rem_periods_in_cycle': cycle_info['num_rem_periods']
                })
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(sleep_dates)} nights...")
    
    print(f"\nGenerated {len(sleep_summary)} individual night visualizations")
    
    # Print cycle detection info
    if detailed_records:
        total_cycles = len(detailed_records)
        avg_cycles_per_night = total_cycles / len(sleep_summary) if len(sleep_summary) > 0 else 0
        print(f"\n📊 Sleep Cycle Analysis:")
        print(f"   - Detected {total_cycles} true ultradian sleep cycles")
        print(f"   - Average {avg_cycles_per_night:.1f} cycles per night")
        print(f"   - Cycle detection: REM periods >30min apart = new cycle")
        print(f"   - Each cycle represents ~90-120min NREM→REM progression")
    
    # Create summary plot
    if sleep_summary:
        print("Creating summary visualization...")
        sleep_summary_df = pd.DataFrame(sleep_summary)
        
        # Save summary CSV
        summary_csv = output_dir / 'sleep_summary.csv'
        sleep_summary_df.drop('rem_cycles', axis=1).to_csv(summary_csv, index=False)
        print(f"Saved sleep summary to {summary_csv}")
        
        # Save detailed REM cycle CSV
        if detailed_records:
            detailed_df = pd.DataFrame(detailed_records)
            detailed_csv = output_dir / 'rem_cycle_details.csv'
            detailed_df.to_csv(detailed_csv, index=False)
            print(f"Saved REM cycle details to {detailed_csv}")
            
            # Create REM prediction visualizations
            print("Creating REM cycle analysis...")
            create_rem_analysis(detailed_df, sleep_summary_df, output_dir)
        
        create_summary_plot(sleep_summary_df, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"   - {len(sleep_summary)} individual night plots")
    print(f"   - 1 summary plot (sleep_summary.png)")
    print(f"   - 1 REM cycle analysis (rem_cycle_analysis.png)")
    print(f"   - 2 CSV files:")
    print(f"     • sleep_summary.csv - nightly sleep metrics")
    print(f"     • rem_cycle_details.csv - detailed REM cycle timing data")
    print(f"\n💡 Use rem_cycle_details.csv to predict your REM cycles!")
    print(f"   Example: If you fall asleep at 11 PM, look at 'rem_start_minutes_after_sleep'")
    print(f"   to see when each REM cycle typically starts for you.")

if __name__ == '__main__':
    main()