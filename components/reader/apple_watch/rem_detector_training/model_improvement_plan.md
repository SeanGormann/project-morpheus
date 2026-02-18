# REM Detection Generalization Improvement Plan

## 🔴 Current Problem

**Catastrophic Overfitting to Individual Subjects**
- CV ROC AUC: 0.946 (amazing within training subjects)
- Test ROC AUC: 0.481 (worse than random on new subjects)
- Gap of ~0.46 indicates models learned person-specific patterns, not generalizable REM features

**Root Cause:** 
- Only 31 subjects (24 train, 7 test)
- Features are absolute values (raw HR, activity) that vary wildly between people
- Models memorized "Subject A sleeps like this" instead of "REM looks like this"

---

## 🎯 Goals

1. **Primary:** Achieve test performance closer to CV performance (gap < 0.10)
2. **Target:** Test ROC AUC > 0.85 (usable for real-world REM detection)
3. **Maintain:** Keep recall high (>80%) for lucid dreaming triggers

---

## 📋 Solution Strategy

### Phase 1: Feature Engineering (HIGH IMPACT)

#### 1.1 Subject-Normalized Features
**Current:** Absolute values
```python
hr_feature = (hr_ema ** 3) / 1000  # Absolute heart rate
activity_feature = (activity_ema ** 2) / max_activity
```

**Proposed:** Relative/percentile features
```python
# Per-subject normalization
hr_baseline = subject_hr.median()
hr_std = subject_hr.std()
hr_feature_normalized = (hr - hr_baseline) / hr_std  # Z-score

# Or percentile within night
hr_percentile = percentileofscore(subject_night_hr, current_hr)

# Activity as % of personal max
activity_pct_max = activity / subject_max_activity
```

**Implementation:**
- Extract features per-subject BEFORE splitting
- Store subject-level stats (mean, std, min, max)
- Normalize all features to z-scores or percentiles
- Keep time_hours as relative (already good)

#### 1.2 Differential Features (Changes over time)
**Rationale:** REM transitions show characteristic patterns

```python
# Heart rate variability
hr_delta_30s = hr_current - hr_30s_ago
hr_trend = linear_slope(hr_last_5min)

# Activity change
activity_delta = activity_current - activity_1min_ago

# Stage transition features
time_since_last_movement = ...
time_in_current_state = ...
```

#### 1.3 Physiological Ratios (Scale-invariant)
**Rationale:** Ratios are more stable across individuals

```python
# Heart rate variability measures
rmssd = sqrt(mean(diff(hr_intervals)**2))
sdnn = std(hr_intervals)
pnn50 = percent(diff(hr_intervals) > 50ms)

# Activity/HR coupling
activity_hr_ratio = activity / hr
```

---

### Phase 2: Better Feature Extraction (MEDIUM IMPACT)

#### 2.1 Frequency-Domain Features (Already started with FFT model)
- VLF/LF/HF power bands
- Spectral entropy
- Dominant frequency
- LF/HF ratio (sympathetic/parasympathetic balance)

**Status:** FFT model exists but needs testing

#### 2.2 Temporal Context Windows
**Current:** 30-second epochs (isolated)

**Proposed:** Multi-scale windows
```python
# Short-term (30s): Immediate state
features_30s = extract_features(window_30s)

# Medium-term (2-5min): Local context
features_2min = extract_features(window_2min)
rolling_mean_hr_2min = ...
rolling_std_activity_2min = ...

# Long-term (10-15min): Cycle context
features_10min = extract_features(window_10min)
cycle_phase = estimate_cycle_position()
```

#### 2.3 Sequence Models (Future consideration)
- LSTM/GRU for temporal dependencies
- Transformer for attention over sleep history
- Currently: Random Forest treats each epoch independently

---

### Phase 3: Model Improvements (LOW-MEDIUM IMPACT)

#### 3.1 Try All Models with Feature Scaling
**Current:** Only MLP and LogReg use StandardScaler

**Proposed:** Scale for ALL models
- Tree-based models (RF, XGBoost) don't need scaling theoretically
- BUT with person-specific features, scaling might help generalization
- Test both scaled and unscaled versions

#### 3.2 Regularization & Hyperparameter Tuning
```python
# Random Forest
- Try max_depth limits (currently unlimited)
- Increase min_samples_leaf (currently 48)
- Try max_features='sqrt' instead of all features

# XGBoost
- Increase regularization (alpha, lambda)
- Try lower learning_rate + more n_estimators
- Add early_stopping_rounds

# MLP
- Add dropout layers
- Try different architectures (128->64->32)
- Experiment with batch normalization
```

#### 3.3 Ensemble Methods
```python
# Stack predictions
meta_features = [rf_proba, xgb_proba, mlp_proba, fft_proba]
meta_model = LogisticRegression()
final_prediction = meta_model.predict(meta_features)

# Weighted voting based on CV performance
final = 0.3*rf + 0.3*xgb + 0.2*mlp + 0.2*fft
```

---

### Phase 4: Data Augmentation (MEDIUM IMPACT)

#### 4.1 Synthetic Minority Oversampling (SMOTE)
**Issue:** 80/20 class imbalance (non-REM/REM)

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

#### 4.2 Time-Based Augmentation
```python
# Slight time shifts (±30s to ±2min)
# Simulate different sleep onset times
# Add gaussian noise to features (small σ)
```

---

### Phase 5: Evaluation & Validation (CRITICAL)

#### 5.1 Better Cross-Validation Strategy
**Current:** StratifiedKFold (splits samples randomly)

**Proposed:** Subject-level CV
```python
from sklearn.model_selection import GroupKFold

# Ensure same subject never in train AND val
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=subject_ids):
    # Train on some subjects, validate on completely different subjects
    ...
```

**This will give honest CV scores that match test performance**

#### 5.2 Comprehensive Metrics
Track more than just ROC AUC:
- **Precision/Recall curves:** Optimize threshold for lucid dreaming
- **Per-subject performance:** Identify which subjects generalize poorly
- **Confusion matrix:** Understand error types
- **Calibration:** Are probabilities reliable?

#### 5.3 Error Analysis
```python
# Find worst-performing test subjects
for subject_id in test_subjects:
    subject_performance = calculate_metrics(subject_id)
    # What's different about subjects with low performance?
    # Age? Sleep quality? Watch positioning?
```

---

## 🚀 Implementation Roadmap

### Week 1: Foundation (Feature Engineering)
- [ ] **Day 1-2:** Implement subject-normalized features
  - Calculate per-subject baselines (mean, std, percentiles)
  - Normalize all features to z-scores
  - Test train/test performance
  
- [ ] **Day 3:** Add differential features
  - HR/activity deltas
  - Trends over last N minutes
  - Time-in-state features

- [ ] **Day 4:** Test physiological ratios
  - HRV metrics (RMSSD, SDNN)
  - Activity/HR coupling
  - Compare with normalized features

- [ ] **Day 5:** Benchmark all feature sets
  - Baseline (current)
  - Normalized only
  - Normalized + differential
  - Normalized + differential + ratios
  - **Goal:** Test ROC AUC > 0.70

### Week 2: Models & Validation
- [ ] **Day 6:** Implement GroupKFold CV
  - Subject-level cross-validation
  - Get honest CV estimates
  - **Goal:** CV and Test within 0.10

- [ ] **Day 7-8:** Hyperparameter tuning
  - Grid search with GroupKFold
  - Test regularization strategies
  - **Goal:** Test ROC AUC > 0.80

- [ ] **Day 9:** SMOTE + data augmentation
  - Balance classes
  - Test impact on recall
  - **Goal:** Recall > 80%

- [ ] **Day 10:** Ensemble methods
  - Stack best models
  - Weighted voting
  - **Goal:** Test ROC AUC > 0.85

### Week 3: FFT Model & Polish
- [ ] **Day 11-12:** Debug and optimize FFT model
  - Currently extracts features but not tested
  - Compare frequency vs time domain
  - Combine both feature sets

- [ ] **Day 13:** Comprehensive evaluation
  - Error analysis per subject
  - Calibration plots
  - Feature importance analysis

- [ ] **Day 14:** Documentation & final testing
  - Create comparison report
  - Visualizations for all models
  - Select best model for deployment

---

## 📊 Success Criteria

### Minimum Viable Performance
- ✅ Test ROC AUC > 0.80
- ✅ Test Recall > 0.75
- ✅ CV/Test gap < 0.15

### Target Performance
- 🎯 Test ROC AUC > 0.85
- 🎯 Test Recall > 0.80
- 🎯 CV/Test gap < 0.10
- 🎯 Works on 6/7 test subjects (>85% subjects)

### Stretch Goals
- 🚀 Test ROC AUC > 0.90
- 🚀 Test Recall > 0.85
- 🚀 Personalized fine-tuning API ready

---

## 🛠️ Technical Debt to Address

1. **Current feature extraction is crude**
   - Simple EMA smoothing
   - Cubic/square transformations are arbitrary
   - No domain knowledge applied

2. **30-second epochs too coarse?**
   - REM detection might need finer granularity
   - Or maybe need longer context (cycle-level)

3. **No validation of Apple Watch data quality**
   - Are HR/motion sensors reliable?
   - Any data quality flags we should use?

4. **Subject metadata not utilized**
   - Age, sex, BMI (if available) could improve generalization
   - Sleep disorders, medications, etc.

---

## 📝 Next Immediate Steps

**Priority 1 (Start NOW):**
```bash
# 1. Implement subject normalization
python create_normalized_features.py

# 2. Test with GroupKFold CV
python train_with_group_cv.py

# 3. Compare performance
python compare_feature_sets.py
```

**Expected Improvement:**
- Subject normalization alone should boost test ROC AUC from 0.48 → 0.65-0.75
- GroupKFold will give honest CV scores (expect CV to DROP to ~0.70-0.75)
- Adding differential features should push to 0.75-0.85

---

## 🔬 Experiments to Run

### Experiment 1: Feature Normalization
**Hypothesis:** Subject-normalized features will generalize better
- Baseline: Current absolute features
- Test 1: Z-score normalization per subject
- Test 2: Percentile features (0-100 scale)
- Test 3: Min-max normalization per subject
- **Metric:** Test ROC AUC improvement

### Experiment 2: Feature Engineering
**Hypothesis:** Differential features capture REM transitions better
- Test 1: Add HR/activity deltas
- Test 2: Add rolling statistics (mean, std over 2-5min)
- Test 3: Add HRV metrics
- **Metric:** Feature importance, Test ROC AUC

### Experiment 3: Model Architecture
**Hypothesis:** Deeper models capture non-linear patterns
- Test 1: Deeper MLP (128→64→32 vs current 64→32)
- Test 2: Gradient boosting depth (max_depth 10 vs 6)
- Test 3: Ensemble vs single best model
- **Metric:** Test ROC AUC, inference speed

### Experiment 4: Temporal Context
**Hypothesis:** Longer context windows improve accuracy
- Test 1: 1-minute epochs (vs 30s)
- Test 2: Multi-scale features (30s + 2min + 10min)
- Test 3: LSTM with 10-epoch history
- **Metric:** Test ROC AUC, latency

---

## 📚 References & Resources

### Sleep Science Papers
- Walch et al. 2019 - Original Apple Watch sleep staging paper
- TLR lucid dreaming papers - REM detection for cue delivery
- HRV and sleep stages - Physiological correlates

### Similar Approaches
- Consumer sleep trackers (Oura, Whoop) - likely use subject normalization
- Clinical polysomnography - gold standard features
- Wearable sleep research - best practices

### Tools
- YASA library - sleep staging algorithms
- MNE-Python - sleep EEG analysis (might have useful feature extraction)
- HeartPy - HRV analysis

---

## 💡 Key Insights

1. **Small dataset curse:** 31 subjects is tiny for ML. Each test subject is 14% of test set!
2. **Person variability >> state variability:** Baseline HR varies 50-90 bpm across people, REM only changes ±10 bpm within person
3. **Time-domain features likely insufficient:** Need frequency-domain (FFT) for robust patterns
4. **This is a known hard problem:** Even commercial devices struggle with generalization

---

## 🎯 Final Thoughts

**Realistic Expectations:**
- Getting to 0.80 test ROC AUC is achievable with feature engineering
- Getting to 0.90+ requires either:
  - More subjects (100+)
  - Better sensors (EEG, not just HR/motion)
  - Personalized calibration period

**For Your Use Case (Lucid Dreaming):**
- Even 0.80 ROC AUC with 0.80 recall is EXCELLENT
- You only need to be right ~4/5 times
- Can fine-tune on your own sleep data later
- False positives (playing sound in non-REM) are harmless

**Let's crush this! 💪**