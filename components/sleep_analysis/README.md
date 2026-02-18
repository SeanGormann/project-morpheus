# Apple Health Sleep Data Analyzer

This script extracts and visualizes your sleep data from Apple Health.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Export your Apple Health data:**
   - Open the Health app on your iPhone
   - Tap your profile picture (top right)
   - Scroll down and tap "Export All Health Data"
   - Save the `export.zip` file
   - Transfer it to your Mac (AirDrop, iCloud, etc.)

3. **Place the export file:**
   - Put `export.zip` in the same directory as this script

## Usage

Run the script:
```bash
uv run python apple_health_sleep_analyzer.py
```

## Output

The script creates a `sleep_visualizations/` folder containing:

- **Individual night plots**: One PNG for each night showing:
  - Timeline of sleep stages (Deep, Core, REM, Awake, In Bed)
  - Bar chart of time spent in each stage
  - Summary statistics (total sleep, time in bed)

- **Summary plot** (`sleep_summary.png`):
  - Sleep duration over time
  - Sleep efficiency trends

- **CSV data** (`sleep_data.csv`):
  - Exportable data for further analysis


Run the script:
```bash
uv run python rem_estimator.py 2330
```

arg = time of sleep onset 




## Sleep Stages Explained

- **Deep Sleep**: Physically restorative sleep
- **Core Sleep**: Light sleep (formerly "Light Sleep" in older Apple Watch models)
- **REM Sleep**: Rapid Eye Movement sleep, important for memory and learning
- **Awake**: Periods you were awake during the night
- **In Bed**: Time in bed but not necessarily asleep

## Tips

- The script automatically groups sleep sessions into nights
- Sleep starting after 12 PM is counted as that day's sleep
- Sleep starting before 12 PM is counted as the previous day's sleep
- This handles staying up past midnight correctly

## Troubleshooting

- **No data found**: Make sure you're using an Apple Watch or iPhone that tracks sleep
- **Missing stages**: Older devices may not track all sleep stages
- **Large file**: The export.xml can be several GB for years of data - be patient during parsing!
