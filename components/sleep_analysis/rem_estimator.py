#!/usr/bin/env python3
"""
REM Cycle Estimator
Predicts when your REM cycles will occur based on historical sleep data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import sys

def load_rem_data(csv_path='sleep_visualizations/rem_cycle_details.csv'):
    """Load REM cycle historical data"""
    if not Path(csv_path).exists():
        print(f"❌ Error: {csv_path} not found!")
        print("Please run the sleep analyzer first to generate REM cycle data.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} REM cycle records from {len(df['date'].unique())} nights")
    return df

def calculate_cycle_stats(df):
    """Calculate statistics for each REM cycle number"""
    stats = {}
    
    for cycle_num in sorted(df['rem_cycle_number'].unique()):
        cycle_data = df[df['rem_cycle_number'] == cycle_num]
        
        # Handle single data point case (std will be NaN)
        mean_start = cycle_data['rem_start_minutes_after_sleep'].mean()
        std_start = cycle_data['rem_start_minutes_after_sleep'].std()
        mean_duration = cycle_data['rem_duration_minutes'].mean()
        std_duration = cycle_data['rem_duration_minutes'].std()
        
        # If only 1 data point, use 20% of mean as uncertainty estimate
        if pd.isna(std_start) or len(cycle_data) == 1:
            std_start = mean_start * 0.2
        if pd.isna(std_duration) or len(cycle_data) == 1:
            std_duration = mean_duration * 0.2
        
        stats[cycle_num] = {
            'mean_start_minutes': mean_start,
            'std_start_minutes': std_start,
            'mean_duration_minutes': mean_duration,
            'std_duration_minutes': std_duration,
            'count': len(cycle_data),
            'min_start': cycle_data['rem_start_minutes_after_sleep'].min(),
            'max_start': cycle_data['rem_start_minutes_after_sleep'].max(),
        }
    
    return stats

def rem_estimator(sleep_onset_time, df, output_dir='rem_guesstimations'):
    """
    Estimate REM cycles based on sleep onset time
    
    Args:
        sleep_onset_time: int in 24-hour format (e.g., 2300 for 11pm, 0100 for 1am)
        df: DataFrame with historical REM cycle data
        output_dir: Directory to save visualization
    
    Returns:
        predictions: List of dicts with REM cycle predictions
    """
    # Parse the input time from 24hr format (e.g., 2330 = 23:30)
    try:
        time_str = str(sleep_onset_time).zfill(4)  # Ensure 4 digits (e.g., 100 -> 0100)
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        
        if hour > 23 or minute > 59:
            raise ValueError("Invalid time")
        
        sleep_time = datetime.strptime(f"{hour:02d}:{minute:02d}", '%H:%M')
        time_display = f"{hour:02d}{minute:02d}"
        
    except (ValueError, IndexError):
        print(f"❌ Invalid time format: {sleep_onset_time}")
        print("Please use 24-hour format like 2300 (11pm), 0100 (1am), 0030 (12:30am)")
        sys.exit(1)
    
    # Use today's date as reference
    today = datetime.now().date()
    sleep_datetime = datetime.combine(today, sleep_time.time())
    
    # If time is before noon, assume it's actually tomorrow (sleeping past midnight)
    if sleep_time.hour < 12:
        sleep_datetime += timedelta(days=1)
    
    print(f"\n{'='*70}")
    print(f"REM CYCLE PREDICTIONS FOR SLEEP ONSET AT {hour:02d}:{minute:02d} ({time_display})")
    print(f"{'='*70}\n")
    
    # Calculate cycle statistics
    cycle_stats = calculate_cycle_stats(df)
    
    predictions = []
    
    print(f"{'Cycle':<8} {'Start Time':<15} {'Mid-REM Time':<15} {'Duration':<20} {'Confidence':<12}")
    print(f"{'-'*80}")
    
    for cycle_num in sorted(cycle_stats.keys()):
        stats = cycle_stats[cycle_num]
        
        # Predict start time
        mean_start_minutes = stats['mean_start_minutes']
        std_start_minutes = stats['std_start_minutes']
        
        predicted_start = sleep_datetime + timedelta(minutes=mean_start_minutes)
        predicted_end = predicted_start + timedelta(minutes=stats['mean_duration_minutes'])
        
        # Calculate mid-REM time (halfway through the REM period)
        mid_rem_time = predicted_start + timedelta(minutes=stats['mean_duration_minutes'] / 2)
        
        # Calculate confidence based on standard deviation and sample size
        # Lower std = higher confidence, more samples = higher confidence
        cv = std_start_minutes / mean_start_minutes if mean_start_minutes > 0 else 1  # coefficient of variation
        sample_weight = min(stats['count'] / 10, 1.0)  # max out at 10 samples
        
        # Confidence score (0-100%)
        # Penalize single data points heavily
        if stats['count'] == 1:
            confidence = 20.0  # Low confidence for single data point
        else:
            confidence = (1 - min(cv, 1.0)) * sample_weight * 100
        
        # Uncertainty range
        uncertainty_minutes = std_start_minutes
        
        predictions.append({
            'cycle': cycle_num,
            'start_time': predicted_start,
            'mid_rem_time': mid_rem_time,
            'end_time': predicted_end,
            'duration_minutes': stats['mean_duration_minutes'],
            'confidence': confidence,
            'uncertainty_minutes': uncertainty_minutes,
            'sample_count': stats['count']
        })
        
        # Format output
        start_str = predicted_start.strftime('%I:%M %p')
        mid_rem_str = mid_rem_time.strftime('%I:%M %p')
        duration_str = f"{stats['mean_duration_minutes']:.0f}±{stats['std_duration_minutes']:.0f} min"
        confidence_str = f"{confidence:.0f}%"
        
        print(f"#{cycle_num:<7} {start_str:<15} {mid_rem_str:<15} {duration_str:<20} {confidence_str:<12}")
    
    print(f"{'-'*80}")
    print(f"\n📊 Predictions based on {len(df['date'].unique())} nights of data")
    print(f"⚡ Higher confidence = more consistent timing in your historical data\n")
    
    # Create visualization
    create_prediction_plot(sleep_datetime, predictions, cycle_stats, output_dir, time_display)
    
    return predictions

def create_prediction_plot(sleep_datetime, predictions, cycle_stats, output_dir, sleep_onset_str):
    """Create visualization of REM cycle predictions"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Colors - gradient from dark to light for progression through night
    cycle_colors = ['#2E4057', '#3D5A80', '#5C7A99', '#7B9DB2', '#A8DADC', 
                    '#B8E5E8', '#C8F0F2', '#D8FAFA']
    
    # Plot 1: Timeline showing REM periods with confidence
    # Use SHORT rectangular bars to show duration
    
    for i, pred in enumerate(predictions):
        cycle = pred['cycle']
        start = pred['start_time']
        end = pred['end_time']
        confidence = pred['confidence']
        uncertainty = pred['uncertainty_minutes']
        duration = pred['duration_minutes']
        
        color = cycle_colors[min(cycle - 1, len(cycle_colors) - 1)]
        
        # Calculate the middle point of the REM period
        mid_point = start + timedelta(minutes=duration/2)
        
        # IMPORTANT: Bar width is in MINUTES not extending to end time
        # Create a small bar centered on mid_point, width = duration in minutes
        bar_start = mid_point - timedelta(minutes=duration/2)
        bar_end = mid_point + timedelta(minutes=duration/2)
        
        # Convert duration to timedelta for the bar width
        ax1.barh(confidence, timedelta(minutes=duration), left=bar_start, height=6, 
                color=color, alpha=0.85, edgecolor='black', 
                linewidth=2, zorder=5, label=f"Cycle {cycle}")
        
        # Add horizontal error bars for uncertainty
        left_err = bar_start - timedelta(minutes=uncertainty)
        right_err = bar_end + timedelta(minutes=uncertainty)
        
        # Horizontal line for uncertainty
        ax1.plot([left_err, right_err], [confidence, confidence], 
                color=color, alpha=0.4, linewidth=2, zorder=3)
        
        # Caps on the error bars
        ax1.plot([left_err, left_err], [confidence-2, confidence+2], 
                color=color, alpha=0.4, linewidth=2, zorder=3)
        ax1.plot([right_err, right_err], [confidence-2, confidence+2], 
                color=color, alpha=0.4, linewidth=2, zorder=3)
        
        # Add cycle number label in the center
        ax1.text(mid_point, confidence, f'{cycle}', 
                ha='center', va='center', fontweight='bold', fontsize=11,
                color='white', zorder=10)
    
    # Mark sleep onset
    ax1.axvline(sleep_datetime, color='green', linestyle='--', 
               linewidth=2.5, label='Sleep Onset', alpha=0.8, zorder=10)
    
    # Format timeline
    ax1.set_ylabel('Prediction Confidence (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_title(f'Predicted REM Cycles - Sleep Onset at {sleep_onset_str[:2]}:{sleep_onset_str[2:]}', 
                 fontsize=14, fontweight='bold')
    
    # Set reasonable time bounds - from sleep onset to latest REM end + 30 min
    latest_end = max([p['end_time'] for p in predictions])
    ax1.set_xlim(sleep_datetime - timedelta(minutes=30), 
                 latest_end + timedelta(minutes=30))
    
    # Format x-axis to show hourly intervals
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax1.grid(True, alpha=0.3, axis='both')
    ax1.legend(loc='upper left', ncol=2, fontsize=9)
    
    # Add horizontal confidence zones
    ax1.axhspan(0, 30, alpha=0.05, color='red', label='Low Confidence')
    ax1.axhspan(30, 60, alpha=0.05, color='yellow')
    ax1.axhspan(60, 100, alpha=0.05, color='green')
    
    # Plot 2: Summary statistics - confidence and duration bars
    cycles = [p['cycle'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    durations = [p['duration_minutes'] for p in predictions]
    
    x = np.arange(len(cycles))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, confidences, width, label='Confidence (%)',
                    color='#5C7A99', alpha=0.7, edgecolor='black')
    bars2 = ax2_twin.bar(x + width/2, durations, width, label='Duration (min)',
                        color='#A8DADC', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.0f}m', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('REM Cycle', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confidence (%)', fontsize=11, fontweight='bold', color='#5C7A99')
    ax2_twin.set_ylabel('Duration (minutes)', fontsize=11, fontweight='bold', color='#A8DADC')
    ax2.set_title('Prediction Confidence & Expected Duration', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Cycle {c}' for c in cycles])
    ax2.tick_params(axis='y', labelcolor='#5C7A99')
    ax2_twin.tick_params(axis='y', labelcolor='#A8DADC')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rem_prediction_{sleep_onset_str}_{timestamp}.png"
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualization saved to: {output_path}\n")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python rem_estimator.py <sleep_onset_time>")
        print("Examples:")
        print("  python rem_estimator.py 2300    # 11:00 PM")
        print("  python rem_estimator.py 0100    # 1:00 AM")
        print("  python rem_estimator.py 130     # 1:30 AM (will pad to 0130)")
        print("  python rem_estimator.py 2345    # 11:45 PM")
        sys.exit(1)
    
    sleep_onset_time = int(sys.argv[1])
    
    # Load historical data
    df = load_rem_data()
    
    # Make predictions
    predictions = rem_estimator(sleep_onset_time, df)
    
    print("✅ Done! Check the rem_guesstimations/ folder for your visualization.")

if __name__ == '__main__':
    main()