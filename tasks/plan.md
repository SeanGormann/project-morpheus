# TLR Local Inference Plan — Project Morpheus

**Objective:** Replace cloud-based REM inference with on-device CoreML, keeping Watch data collection and iPhone audio cue logic.

**Reference:** [TLR Paper (IJoDR)](https://journals.ub.uni-heidelberg.de/index.php/IJoDR/article/view/100797)

---

## Priority Order (Reprioritised)

| Phase | Focus | Status |
|-------|-------|--------|
| **1. Data Capture** | Remove server, receive & store Watch data locally | ← **DO FIRST** |
| **2. Model Integration** | CoreML, FeatureExtractor, inference | ← **LAST** (mock for now) |

**Rationale:** See what data we actually get from the Watch before building the model pipeline. All communication stays local (Watch ↔ iPhone). No cloud.

---

## Phase 1: Data Capture & Storage (Do First)

**Goal:** App receives data from Watch, stores it locally, displays it. No server. No model yet.

### 1.1 Remove Server Dependency

| Action | File | Change |
|--------|------|--------|
| Remove HTTP | `ServerCommunication.swift` | Delete all `URLSession`, `uploadTask`, `http://34.102.41.199` code |
| Replace with local handler | `ServerCommunication.swift` or new `SessionManager.swift` | Process incoming data locally; no network |
| Keep | `StimulusActivation`, `SoundPlayer` | Use later; for now can mock or skip |

### 1.2 Receive & Store Data

| Action | Details |
|--------|---------|
| Create `DataLogger.swift` | Log each epoch: timestamp, hrFeature, heartRates (count + values), motionData (count + summary stats) |
| Persist to disk | JSON or CSV per session; store in app Documents or temp |
| Session lifecycle | Start on SleepView appear; end on SummaryView; export on demand |

### 1.3 Debug UI — See What We Get

| Display | Purpose |
|---------|---------|
| Epoch count | How many 30s chunks received |
| Last HR | hrFeature value + heartRates.count, avg of heartRates |
| Last motion | motionData.count rows, sample x/y/z from first & last |
| Status | "Waiting" / "Received epoch N" / "Session ended" |
| Export button | Share session file (JSON) for inspection |

### 1.4 Task Checklist — Phase 1

| ID | Task | Est. | Notes |
|----|------|------|-------|
| 1.1 | Strip HTTP from ServerCommunication → local-only handler | 30m | Keep same interface for SleepView |
| 1.2 | Create DataLogger.swift (log epoch data, export JSON) | 1h | Include full payload for inspection |
| 1.3 | Wire SleepView: on gotData → log + update UI (no server call) | 30m | Replace sendDataToServer with log + local process |
| 1.4 | Add debug UI: epoch count, HR stats, motion stats | 45m | So we can see live data |
| 1.5 | Add export: save session to file, share from SummaryView | 30m | JSON preferred for Python analysis |
| 1.6 | Build & run on Simulator (WatchConnectivity may not work) | — | May need physical devices |
| 1.7 | Run on physical iPhone + Watch, capture 5–10 min of data | 15m | Validate data shape and volume |

### 1.5 Data Schema (What We Log)

Per epoch:

```json
{
  "epoch_index": 0,
  "timestamp": "2025-02-12T22:30:00Z",
  "hr_feature": 0.123,
  "heart_rates_count": 60,
  "heart_rates_avg": 62.5,
  "heart_rates": [61, 62, 63, ...],
  "motion_rows_count": 900,
  "motion_sample_first": [0.0, 0.01, -0.02, 0.98],
  "motion_sample_last": [30.0, 0.02, 0.01, 0.99]
}
```

Store full `heart_rates` and `motion_data` if size is OK; otherwise sampled. Goal: inspect real payload later.

---

## Phase 2: Model Integration (Later — Mock First)

**Goal:** Once we have real data, add CoreML + FeatureExtractor. For now, mock inference (e.g. always false, or random).

### 2.1 Mock Inference (Placeholder)

- `isInREM` = false always, or random for testing
- No CoreML, no FeatureExtractor yet
- Lets us verify: receive → process → (mock) trigger → UI flow

### 2.2 Real Model (After Data Validation)

- Export Gaussian NB to CoreML
- FeatureExtractor.swift
- REMDetector.swift
- Wire to LocalInference

---

## Appendix A: Architecture (Verified)

### Data Flow (TLR → Morpheus)

```
Watch (Driver.swift)
  ├─ WorkoutManager: HR every ~0.5s → getHRs(), getAvgHR()
  ├─ MotionManager: 30Hz accelerometer → getMotionData()
  └─ Every 30s: WatchConnectivityManager.send(motionData, hrFeature, heartRates)

iPhone (SleepView.swift) — NEW
  ├─ Timer every 1s checks connectivityManager.gotData
  ├─ On new data: DataLogger.log() + update debug UI (NO server)
  └─ Optional: mock REM check → trigger audio (placeholder)
```

### WatchConnectivity Payload

- `motion`: `[[Double]]` — each row [timestamp, x, y, z]
- `heart rate`: Double (TLR's hrFeature)
- `hrs`: `[Double]` — heart rate samples in epoch

---

## Appendix B: Feature & Activity (Deferred)

- **Activity counts:** TLR uses ActiGraph; we use magnitude² sum. Implement on iPhone when we add model.
- **Z-score:** Use rolling stats over session; see plan when we implement FeatureExtractor.

---

## Appendix C: Model Integration (Deferred)

When ready: Export CoreML → FeatureExtractor → REMDetector → LocalInference → wire audio trigger.

---

## Appendix D: Files Reference

```
TLR/DreamApp/DreamApp/
├── Models/
│   ├── ServerCommunication.swift   # REPLACE with local handler (Phase 1)
│   ├── DataLogger.swift           # CREATE (Phase 1)
│   ├── SoundPlayer.swift          # KEEP
│   └── StimulusActivation.swift   # KEEP
└── Views/
    ├── SleepView.swift            # EDIT: log + debug UI, no HTTP
    └── SummaryView.swift          # EDIT: export session
```
