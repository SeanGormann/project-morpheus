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

const DAYS_PER_PAGE = 14;

export function useSleepData(): UseSleepDataReturn {
  // State
  const [lastNight, setLastNight] = useState<SleepSession | null>(null);
  const [history, setHistory] = useState<SleepSession[]>([]);
  const [chartData, setChartData] = useState<SleepChartDataPoint[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [loadedDays, setLoadedDays] = useState(DAYS_PER_PAGE);
  const [hasMoreData, setHasMoreData] = useState(true);

  /**
   * Request HealthKit permissions
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    console.log('[HealthKit] requestPermission called, Platform:', Platform.OS);
    setIsLoading(true);
    setError(null);
    
    if (Platform.OS !== 'ios') {
      console.log('[HealthKit] Not iOS, skipping');
      setError('HealthKit is only available on iOS');
      setHasPermission(false);
      setIsLoading(false);
      return false;
    }

    try {
      console.log('[HealthKit] Checking availability...');
      const available = await healthKitService.isAvailable();
      console.log('[HealthKit] Available:', available);
      
      if (!available) {
        setError('HealthKit is not available on this device');
        setHasPermission(false);
        setIsLoading(false);
        return false;
      }

      console.log('[HealthKit] Initializing...');
      const success = await healthKitService.initialize();
      console.log('[HealthKit] Initialize result:', success);
      
      setHasPermission(success);
      setIsInitialized(success);

      if (!success) {
        setError('Failed to get HealthKit permissions');
        setIsLoading(false);
        return false;
      }

      // Fetch data after successful permission
      console.log('[HealthKit] Permission granted, fetching data...');
      try {
        const [lastNightData, historyData] = await Promise.all([
          healthKitService.getLastNightSleep(),
          healthKitService.getSleepHistory(DAYS_PER_PAGE),
        ]);
        
        console.log('[HealthKit] Data fetched:', { lastNight: !!lastNightData, historyCount: historyData.length });
        
        setLastNight(lastNightData);
        setHistory(historyData);
        setLoadedDays(DAYS_PER_PAGE);
        setHasMoreData(historyData.length >= DAYS_PER_PAGE);
        
        if (lastNightData) {
          setChartData(healthKitService.sessionToChartData(lastNightData));
        }
      } catch (fetchErr) {
        console.log('[HealthKit] Fetch error:', fetchErr);
        // Don't fail the whole thing if fetch fails
      }

      setIsLoading(false);
      return true;
    } catch (err) {
      console.log('[HealthKit] Error:', err);
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      setHasPermission(false);
      setIsLoading(false);
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
   * Fetch sleep history (replaces current history)
   */
  const fetchHistory = useCallback(async (days: number = DAYS_PER_PAGE) => {
    if (!isInitialized) return;

    try {
      const sessions = await healthKitService.getSleepHistory(days);
      setHistory(sessions);
      setLoadedDays(days);
      setHasMoreData(sessions.length >= days);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sleep history';
      setError(message);
    }
  }, [isInitialized]);

  /**
   * Load more historical data (appends to existing history)
   */
  const loadMore = useCallback(async () => {
    if (!isInitialized || isLoadingMore || !hasMoreData) return;

    setIsLoadingMore(true);
    const newDays = loadedDays + DAYS_PER_PAGE;

    try {
      const sessions = await healthKitService.getSleepHistory(newDays);
      setHistory(sessions);
      setLoadedDays(newDays);
      // If we got fewer sessions than days requested, we've reached the end
      setHasMoreData(sessions.length >= history.length + DAYS_PER_PAGE * 0.5);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load more history';
      setError(message);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isInitialized, isLoadingMore, hasMoreData, loadedDays, history.length]);

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
   * Refresh all data (resets to first page)
   */
  const refreshData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setLoadedDays(DAYS_PER_PAGE);
    setHasMoreData(true);

    try {
      await Promise.all([fetchLastNight(), fetchHistory(DAYS_PER_PAGE)]);
    } finally {
      setIsLoading(false);
    }
  }, [fetchLastNight, fetchHistory]);

  /**
   * Initialize on mount (iOS only) - but don't auto-request permission
   * Just set loading to false and let user trigger permission request
   */
  useEffect(() => {
    let mounted = true;

    async function init() {
      console.log('[useSleepData] init called, Platform:', Platform.OS);
      
      if (Platform.OS !== 'ios') {
        setError('HealthKit is only available on iOS');
        setIsLoading(false);
        return;
      }

      // Don't auto-request permissions - let user tap the button
      // This prevents crashes from happening automatically
      if (mounted) {
        setIsLoading(false);
        setHasPermission(false); // Show permission prompt
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
    isLoadingMore,
    isInitialized,
    loadedDays,
    hasMoreData,
    error,
    hasPermission,
    requestPermission,
    refreshData,
    loadMore,
    fetchDateRange,
  };
}

