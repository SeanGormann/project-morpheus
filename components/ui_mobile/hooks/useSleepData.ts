/**
 * useSleepData Hook
 * 
 * React hook that provides sleep data from HealthKit with
 * automatic initialization, error handling, and refresh capabilities.
 */

import { useState, useEffect, useCallback } from 'react';
import { Platform } from 'react-native';
import { healthKitService } from '../services/healthKitService';
import {
  SleepSession,
  SleepChartDataPoint,
  DateRange,
  UseSleepDataReturn,
} from '../types/health';

export function useSleepData(): UseSleepDataReturn {
  // State
  const [lastNight, setLastNight] = useState<SleepSession | null>(null);
  const [history, setHistory] = useState<SleepSession[]>([]);
  const [chartData, setChartData] = useState<SleepChartDataPoint[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  /**
   * Request HealthKit permissions
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    if (Platform.OS !== 'ios') {
      setError('HealthKit is only available on iOS');
      setHasPermission(false);
      return false;
    }

    try {
      const available = await healthKitService.isAvailable();
      if (!available) {
        setError('HealthKit is not available on this device');
        setHasPermission(false);
        return false;
      }

      const success = await healthKitService.initialize();
      setHasPermission(success);
      setIsInitialized(success);

      if (!success) {
        setError('Failed to get HealthKit permissions');
      }

      return success;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      setHasPermission(false);
      return false;
    }
  }, []);

  /**
   * Fetch last night's sleep and update chart data
   */
  const fetchLastNight = useCallback(async () => {
    if (!isInitialized) return;

    try {
      const session = await healthKitService.getLastNightSleep();
      setLastNight(session);

      if (session) {
        const data = healthKitService.sessionToChartData(session);
        setChartData(data);
      } else {
        setChartData(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sleep data';
      setError(message);
    }
  }, [isInitialized]);

  /**
   * Fetch sleep history
   */
  const fetchHistory = useCallback(async (days: number = 7) => {
    if (!isInitialized) return;

    try {
      const sessions = await healthKitService.getSleepHistory(days);
      setHistory(sessions);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sleep history';
      setError(message);
    }
  }, [isInitialized]);

  /**
   * Fetch sleep for a specific date range
   */
  const fetchDateRange = useCallback(async (range: DateRange): Promise<SleepSession[]> => {
    if (!isInitialized) {
      throw new Error('HealthKit not initialized');
    }

    const samples = await healthKitService.getSleepSamples(range);
    // This is simplified - in production you'd want to properly
    // group samples into sessions by night
    return [];
  }, [isInitialized]);

  /**
   * Refresh all data
   */
  const refreshData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await Promise.all([fetchLastNight(), fetchHistory()]);
    } finally {
      setIsLoading(false);
    }
  }, [fetchLastNight, fetchHistory]);

  /**
   * Initialize on mount (iOS only)
   */
  useEffect(() => {
    let mounted = true;

    async function init() {
      if (Platform.OS !== 'ios') {
        setError('HealthKit is only available on iOS');
        setIsLoading(false);
        return;
      }

      const success = await requestPermission();

      if (mounted && success) {
        await refreshData();
      }

      if (mounted) {
        setIsLoading(false);
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, []);

  return {
    lastNight,
    history,
    chartData,
    isLoading,
    isInitialized,
    error,
    hasPermission,
    requestPermission,
    refreshData,
    fetchDateRange,
  };
}

