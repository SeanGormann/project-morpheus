/**
 * HealthKit Service
 * 
 * Wraps react-native-health and provides clean async/Promise APIs
 * for accessing Apple Health sleep data.
 */

import AppleHealthKit, {
  HealthKitPermissions,
} from 'react-native-health';
import {
  HealthKitSleepSample,
  SleepStage,
  SleepSegment,
  SleepSession,
  SleepSummary,
  SleepChartDataPoint,
  DateRange,
} from '../types/health';

// Permissions we need
const HEALTH_PERMISSIONS: HealthKitPermissions = {
  permissions: {
    read: [AppleHealthKit.Constants.Permissions.SleepAnalysis],
    write: [], // We're only reading
  },
};

// Map HealthKit values to our enum
const SLEEP_VALUE_MAP: Record<string, SleepStage> = {
  INBED: SleepStage.InBed,
  ASLEEP: SleepStage.Asleep,
  AWAKE: SleepStage.Awake,
  CORE: SleepStage.Core,
  DEEP: SleepStage.Deep,
  REM: SleepStage.REM,
};

// Depth values for chart visualization
const STAGE_DEPTH: Record<SleepStage, number> = {
  [SleepStage.Awake]: 0,
  [SleepStage.InBed]: 15,
  [SleepStage.REM]: 40,
  [SleepStage.Core]: 65,
  [SleepStage.Asleep]: 70, // Generic "asleep" if no stages
  [SleepStage.Deep]: 100,
};

class HealthKitService {
  private initialized = false;

  /**
   * Initialize HealthKit and request permissions
   */
  async initialize(): Promise<boolean> {
    return new Promise((resolve) => {
      AppleHealthKit.initHealthKit(HEALTH_PERMISSIONS, (error) => {
        if (error) {
          console.error('HealthKit init error:', error);
          this.initialized = false;
          resolve(false);
        } else {
          this.initialized = true;
          resolve(true);
        }
      });
    });
  }

  /**
   * Check if HealthKit is available on this device
   */
  isAvailable(): Promise<boolean> {
    return new Promise((resolve) => {
      AppleHealthKit.isAvailable((error, available) => {
        resolve(!error && available);
      });
    });
  }

  /**
   * Get raw sleep samples for a date range
   */
  async getSleepSamples(range: DateRange): Promise<HealthKitSleepSample[]> {
    if (!this.initialized) {
      throw new Error('HealthKit not initialized. Call initialize() first.');
    }

    return new Promise((resolve, reject) => {
      const options = {
        startDate: range.startDate.toISOString(),
        endDate: range.endDate.toISOString(),
        ascending: true,
      };

      AppleHealthKit.getSleepSamples(options, (error, results) => {
        if (error) {
          reject(new Error(`Failed to get sleep samples: ${error}`));
          return;
        }

        const samples: HealthKitSleepSample[] = (results || []).map((sample: any) => ({
          id: sample.id || `${sample.startDate}-${sample.endDate}`,
          startDate: sample.startDate,
          endDate: sample.endDate,
          value: SLEEP_VALUE_MAP[sample.value] || SleepStage.Asleep,
          sourceId: sample.sourceId || '',
          sourceName: sample.sourceName || 'Unknown',
        }));

        resolve(samples);
      });
    });
  }

  /**
   * Get last night's sleep data
   */
  async getLastNightSleep(): Promise<SleepSession | null> {
    const now = new Date();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(18, 0, 0, 0); // Start from 6 PM yesterday

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

    // Sort by start time
    const sorted = [...samples].sort(
      (a, b) => new Date(a.startDate).getTime() - new Date(b.startDate).getTime()
    );

    // Build segments
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

    // Calculate summary
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
          // Don't count as sleep
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
      // Add point at segment start
      points.push({
        time: segment.startTime,
        timeLabel: this.formatTimeLabel(segment.startTime),
        stage: segment.stage,
        depthValue: STAGE_DEPTH[segment.stage],
      });

      // Add point just before segment ends (to create step effect)
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

// Export singleton instance
export const healthKitService = new HealthKitService();

// Export class for testing
export { HealthKitService };

