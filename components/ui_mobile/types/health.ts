// HealthKit Sleep Stage Values
export enum SleepStage {
  InBed = 'INBED',
  Asleep = 'ASLEEP',
  Awake = 'AWAKE',
  Core = 'CORE',
  Deep = 'DEEP',
  REM = 'REM',
}

// Raw sample from react-native-health
export interface HealthKitSleepSample {
  id: string;
  startDate: string; // ISO string
  endDate: string;   // ISO string
  value: SleepStage;
  sourceId: string;
  sourceName: string;
}

// Processed sleep segment for chart rendering
export interface SleepSegment {
  startTime: Date;
  endTime: Date;
  stage: SleepStage;
  durationMinutes: number;
}

// A single night's sleep session
export interface SleepSession {
  id: string;
  date: Date; // The calendar date this sleep is attributed to
  bedTime: Date;
  wakeTime: Date;
  totalDurationMinutes: number;
  segments: SleepSegment[];
  summary: SleepSummary;
}

// Aggregated stats for a sleep session
export interface SleepSummary {
  totalSleepMinutes: number;
  timeInBedMinutes: number;
  awakeMinutes: number;
  coreMinutes: number;
  deepMinutes: number;
  remMinutes: number;
  sleepEfficiency: number; // percentage: totalSleep / timeInBed
}

// Chart data point for visualization
export interface SleepChartDataPoint {
  time: Date;
  timeLabel: string;
  stage: SleepStage;
  // Numeric depth for chart Y-axis (0-100)
  // Deep = 100, Core = 66, REM = 33, Awake = 0
  depthValue: number;
}

// Props for the updated SleepChart component
export interface SleepChartProps {
  accentColor: string;
  isDark?: boolean;
  sleepData: SleepChartDataPoint[] | null;
  isLoading?: boolean;
  error?: string | null;
}

// Date range for querying
export interface DateRange {
  startDate: Date;
  endDate: Date;
}

// Hook return type
export interface UseSleepDataReturn {
  // Last night's sleep
  lastNight: SleepSession | null;
  // Historical sessions (for history screen)
  history: SleepSession[];
  // Chart-ready data for last night
  chartData: SleepChartDataPoint[] | null;
  // Loading states
  isLoading: boolean;
  isLoadingMore: boolean;
  isInitialized: boolean;
  // Pagination
  loadedDays: number;
  hasMoreData: boolean;
  // Error handling
  error: string | null;
  // Permission status
  hasPermission: boolean | null;
  // Actions
  requestPermission: () => Promise<boolean>;
  refreshData: () => Promise<void>;
  loadMore: () => Promise<void>;
  fetchDateRange: (range: DateRange) => Promise<SleepSession[]>;
}

