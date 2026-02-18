/**
 * HealthKit Service
 * 
 * Uses @kingstinct/react-native-healthkit for Apple Health sleep data.
 * This library has better support for React Native's New Architecture.
 */

import Healthkit, {
  CategoryValueSleepAnalysis,
} from '@kingstinct/react-native-healthkit';
import type { CategorySampleTyped } from '@kingstinct/react-native-healthkit';

import {
  HealthKitSleepSample,
  SleepStage,
  SleepSegment,
  SleepSession,
  SleepSummary,
  SleepChartDataPoint,
  DateRange,
} from '../types/health';

// Sleep analysis category type identifier
const SLEEP_ANALYSIS = 'HKCategoryTypeIdentifierSleepAnalysis' as const;

// Map HealthKit sleep values to our enum
const mapSleepValue = (value: CategoryValueSleepAnalysis): SleepStage => {
  switch (value) {
    case CategoryValueSleepAnalysis.inBed:
      return SleepStage.InBed;
    case CategoryValueSleepAnalysis.asleepCore:
      return SleepStage.Core;
    case CategoryValueSleepAnalysis.asleepDeep:
      return SleepStage.Deep;
    case CategoryValueSleepAnalysis.asleepREM:
      return SleepStage.REM;
    case CategoryValueSleepAnalysis.awake:
      return SleepStage.Awake;
    case CategoryValueSleepAnalysis.asleepUnspecified:
    default:
      return SleepStage.Asleep;
  }
};

// Depth values for chart visualization (higher = higher on chart)
// Awake at top, Deep at bottom (inverted from traditional hypnogram)
const STAGE_DEPTH: Record<SleepStage, number> = {
  [SleepStage.Awake]: 100,
  [SleepStage.InBed]: 85,
  [SleepStage.REM]: 60,
  [SleepStage.Core]: 35,
  [SleepStage.Asleep]: 30,
  [SleepStage.Deep]: 0,
};

class HealthKitService {
  private initialized = false;

  /**
   * Check if HealthKit is available on this device
   */
  async isAvailable(): Promise<boolean> {
    console.log('[HealthKitService] isAvailable called');
    try {
      const available = await Healthkit.isHealthDataAvailable();
      console.log('[HealthKitService] isAvailable result:', available);
      return available;
    } catch (err) {
      console.log('[HealthKitService] isAvailable error:', err);
      return false;
    }
  }

  /**
   * Initialize HealthKit and request permissions
   */
  async initialize(): Promise<boolean> {
    console.log('[HealthKitService] initialize called');
    
    try {
      console.log('[HealthKitService] About to request authorization...');
      console.log('[HealthKitService] SLEEP_ANALYSIS:', SLEEP_ANALYSIS);
      console.log('[HealthKitService] Healthkit.requestAuthorization:', typeof Healthkit.requestAuthorization);
      
      // Request authorization for sleep data - toRead for reading sleep data
      const result = await Healthkit.requestAuthorization({
        toRead: [SLEEP_ANALYSIS],
      });
      
      console.log('[HealthKitService] Authorization result:', result);
      this.initialized = true;
      return true;
    } catch (err: any) {
      console.log('[HealthKitService] initialize error:', err);
      console.log('[HealthKitService] error message:', err?.message);
      console.log('[HealthKitService] error stack:', err?.stack);
      this.initialized = false;
      return false;
    }
  }

  /**
   * Get raw sleep samples for a date range
   */
  async getSleepSamples(range: DateRange): Promise<HealthKitSleepSample[]> {
    if (!this.initialized) {
      throw new Error('HealthKit not initialized. Call initialize() first.');
    }

    try {
      const samples = await Healthkit.queryCategorySamples(
        SLEEP_ANALYSIS,
        {
          limit: 0, // 0 or negative means no limit
          ascending: true,
          filter: {
            date: {
              startDate: range.startDate,
              endDate: range.endDate,
            },
          },
        }
      );

      console.log('[HealthKitService] Got sleep samples:', samples.length);

      return samples.map((sample: any) => ({
        id: sample.uuid,
        startDate: sample.startDate,
        endDate: sample.endDate,
        value: mapSleepValue(sample.value as CategoryValueSleepAnalysis),
        sourceId: sample.sourceRevision?.source?.bundleIdentifier || '',
        sourceName: sample.sourceRevision?.source?.name || 'Unknown',
      }));
    } catch (err) {
      console.log('[HealthKitService] getSleepSamples error:', err);
      throw err;
    }
  }

  /**
   * Get last night's sleep data
   */
  async getLastNightSleep(): Promise<SleepSession | null> {
    const now = new Date();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(18, 0, 0, 0);

    const samples = await this.getSleepSamples({
      startDate: yesterday,
      endDate: now,
    });

    if (samples.length === 0) {
      return null;
    }

    return this.buildSleepSession(samples, now);
  }

  /**
   * Get sleep history for the past N days
   */
  async getSleepHistory(days: number = 7): Promise<SleepSession[]> {
    const sessions: SleepSession[] = [];
    const today = new Date();

    for (let i = 0; i < days; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);

      const rangeStart = new Date(date);
      rangeStart.setDate(rangeStart.getDate() - 1);
      rangeStart.setHours(18, 0, 0, 0);

      const rangeEnd = new Date(date);
      rangeEnd.setHours(14, 0, 0, 0);

      try {
        const samples = await this.getSleepSamples({
          startDate: rangeStart,
          endDate: rangeEnd,
        });

        if (samples.length > 0) {
          const session = this.buildSleepSession(samples, date);
          if (session) {
            sessions.push(session);
          }
        }
      } catch (error) {
        console.warn(`Failed to get sleep for ${date.toDateString()}:`, error);
      }
    }

    return sessions;
  }

  /**
   * Convert raw samples into a SleepSession
   */
  private buildSleepSession(
    samples: HealthKitSleepSample[],
    attributedDate: Date
  ): SleepSession | null {
    if (samples.length === 0) return null;

    const sorted = [...samples].sort(
      (a, b) => new Date(a.startDate).getTime() - new Date(b.startDate).getTime()
    );

    const segments: SleepSegment[] = sorted.map((sample) => {
      const start = new Date(sample.startDate);
      const end = new Date(sample.endDate);
      return {
        startTime: start,
        endTime: end,
        stage: sample.value,
        durationMinutes: (end.getTime() - start.getTime()) / (1000 * 60),
      };
    });

    const summary = this.calculateSummary(segments);

    const bedTime = segments[0].startTime;
    const wakeTime = segments[segments.length - 1].endTime;

    return {
      id: `${attributedDate.toISOString().split('T')[0]}`,
      date: attributedDate,
      bedTime,
      wakeTime,
      totalDurationMinutes: summary.timeInBedMinutes,
      segments,
      summary,
    };
  }

  /**
   * Calculate sleep summary statistics
   */
  private calculateSummary(segments: SleepSegment[]): SleepSummary {
    const summary: SleepSummary = {
      totalSleepMinutes: 0,
      timeInBedMinutes: 0,
      awakeMinutes: 0,
      coreMinutes: 0,
      deepMinutes: 0,
      remMinutes: 0,
      sleepEfficiency: 0,
    };

    for (const segment of segments) {
      summary.timeInBedMinutes += segment.durationMinutes;

      switch (segment.stage) {
        case SleepStage.Awake:
          summary.awakeMinutes += segment.durationMinutes;
          break;
        case SleepStage.Core:
        case SleepStage.Asleep:
          summary.coreMinutes += segment.durationMinutes;
          summary.totalSleepMinutes += segment.durationMinutes;
          break;
        case SleepStage.Deep:
          summary.deepMinutes += segment.durationMinutes;
          summary.totalSleepMinutes += segment.durationMinutes;
          break;
        case SleepStage.REM:
          summary.remMinutes += segment.durationMinutes;
          summary.totalSleepMinutes += segment.durationMinutes;
          break;
        case SleepStage.InBed:
          break;
      }
    }

    summary.sleepEfficiency =
      summary.timeInBedMinutes > 0
        ? (summary.totalSleepMinutes / summary.timeInBedMinutes) * 100
        : 0;

    return summary;
  }

  /**
   * Convert a SleepSession to chart-ready data points
   */
  sessionToChartData(session: SleepSession): SleepChartDataPoint[] {
    const points: SleepChartDataPoint[] = [];

    for (const segment of session.segments) {
      points.push({
        time: segment.startTime,
        timeLabel: this.formatTimeLabel(segment.startTime),
        stage: segment.stage,
        depthValue: STAGE_DEPTH[segment.stage],
      });

      const justBefore = new Date(segment.endTime.getTime() - 1000);
      points.push({
        time: justBefore,
        timeLabel: this.formatTimeLabel(justBefore),
        stage: segment.stage,
        depthValue: STAGE_DEPTH[segment.stage],
      });
    }

    return points;
  }

  /**
   * Format time for chart labels
   */
  private formatTimeLabel(date: Date): string {
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;

    if (minutes === 0) {
      return `${displayHours} ${ampm}`;
    }
    return `${displayHours}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  }
}

export const healthKitService = new HealthKitService();
export { HealthKitService };
