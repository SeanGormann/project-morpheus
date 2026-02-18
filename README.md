# project-morpheus

A system for Targeted Lucidity Reactivation (TLR) — detecting REM sleep via wearable sensors and delivering timed audio cues to induce lucid dreaming.

---

### Repo Structure

```
.
├── README.md
├── main.py
├── pyproject.toml
├── tasks/
│   └── plan.md                         # TLR local inference plan
├── public/
│   └── the_sandman.png
└── components/
    ├── data/
    │   └── tlr_data/
    │       └── walch-apple-watch/      # Walch et al. Apple Watch dataset
    │           ├── heart_rate/         # Per-subject heart rate CSVs
    │           ├── labels/             # Polysomnography sleep stage labels
    │           └── motion/             # Accelerometer CSVs
    ├── reader/
    │   ├── apple_watch/
    │   │   ├── feature_extraction.py   # Feature engineering from Watch data
    │   │   ├── rem_detector_training/  # Model training scripts (RF, XGBoost, MLP, LSTM, NB)
    │   │   ├── models/                 # Trained model artefacts (.pkl)
    │   │   │   ├── baseline/
    │   │   │   ├── full/
    │   │   │   ├── full_tuned/
    │   │   │   └── full_circadian/
    │   │   ├── data_splits/            # Train/test split artefacts (.pkl / .npz)
    │   │   ├── results/                # Experiment result JSONs / CSVs
    │   │   ├── visualizations/
    │   │   └── TLR/                    # Reference TLR app (upstream study code)
    │   │       ├── DreamApp/           # iOS + Watch app (Swift)
    │   │       └── TLR_server-gke/     # Original GKE inference server
    │   └── fitbit/
    │       ├── api_call.py
    │       └── sleep_stage_analysis.ipynb
    ├── sleep_analysis/
    │   ├── apple_health_sleep_analyzer.py
    │   ├── rem_estimator.py
    │   └── apple_health_export/        # Exported Apple Health XML
    ├── ui_mobile/                      # React Native mobile app (iOS / Android)
    │   ├── App.tsx
    │   ├── screens/
    │   │   ├── HomeScreen.tsx
    │   │   ├── HistoryScreen.tsx
    │   │   └── SettingsScreen.tsx
    │   ├── components/
    │   │   ├── SleepChart.tsx
    │   │   ├── BottomNav.tsx
    │   │   └── WheelPicker.tsx
    │   ├── hooks/
    │   │   └── useSleepData.ts
    │   ├── services/
    │   │   └── healthKitService.ts
    │   ├── types/
    │   │   └── health.ts
    │   └── docs/
    │       ├── healthkit-integration.md
    │       ├── ios-build.md
    │       └── configurable-settings.md
    └── writer/
        ├── src/
        │   ├── app.py                  # Main entry point
        │   ├── audio_player.py         # Plays scheduled audio cues
        │   ├── scheduler.py            # Timing / interval logic
        │   └── edit_data.py
        ├── acoustic-data/
        │   └── gamma-40hz-5min-mid.wav # 40 Hz binaural / gamma stimulus
        └── soundcore_a20/              # Soundcore A20 headband integration
```

---

### Components

`components/reader` — reads and processes wearable sensor data (Apple Watch, Fitbit). Includes the full REM detection model training pipeline and the reference TLR iOS/Watch app.

`components/writer` — delivers audio cues at the right moment. Schedules and plays 40 Hz gamma / binaural beat stimuli via connected audio hardware.

`components/sleep_analysis` — analyses personal Apple Health exports; estimates REM periods from heart rate and motion signals.

`components/ui_mobile` — React Native companion app. Displays sleep history charts (via HealthKit), configurable TLR settings, and session controls.

`components/data` — training data. Currently holds the [Walch et al. Apple Watch dataset](https://physionet.org/content/sleep-accel/1.0.0/) (heart rate, motion, PSG labels for 31 subjects).

---

### Research Background

#### Targeted Lucidity Reactivation (TLR)

TLR is the technique at the core of this project: deliver a brief sensory cue (sound, light) precisely during REM sleep to trigger lucid dreaming without waking the sleeper.

**Key papers:**

- **Northwestern / Konkoly et al. (2024)** — REM-contingent auditory cues increase lucid dream frequency in a home setting.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1053810024001260)

- **Apple Watch TLR app + REM detection algorithm (IJoDR)** — Full description of the wrist-worn REM detector, activity-count feature extraction, and the GKE inference server that this project is replacing with on-device CoreML.
  [Full paper (open access PDF)](https://journals.ub.uni-heidelberg.de/index.php/IJoDR/article/download/100797/102654/274840)

---

### Shortcuts
- Link to [Notion](https://www.notion.so/Teamspace-Home-1e1967b7b17680f28fc1d29cdcc88781)
- Mainly putting that there for the craic, I can't use anything on it without upgrading, so here we are lol.

---

### TODO

**Writer**
- Get suitable (or multiple) files for 40 Hz white noise binaural beats
- Create simple file with datetime for time tracking
- Create simple application to play sound at multiple times over a certain period
- Add config for multiple settings options
  - Write Duration
  - Write noise level
  - Write intervals

**Reader / Model**
- Strip HTTP from `ServerCommunication.swift` → local-only handler (Phase 1)
- Create `DataLogger.swift` — log epoch data, export JSON
- Wire `SleepView` to log data locally (no server call)
- Add debug UI: epoch count, HR stats, motion stats
- Export Gaussian NB model to CoreML (Phase 2)
- Build `FeatureExtractor.swift` + `REMDetector.swift`

---

![The Sandman](public/the_sandman.png)

---

> **Private / Proprietary** — All rights reserved. Not licensed for external use, reproduction, or distribution without explicit written permission.
