/**
 * SleepChart Component
 * 
 * Visualizes sleep data from Apple Health/HealthKit.
 * Displays sleep stages over time with a smooth area chart.
 * 
 * Supports two modes:
 * 1. Real HealthKit data via sleepData prop
 * 2. Mock data via sleepStartHour/sleepDuration props (legacy)
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet, Dimensions, ActivityIndicator } from 'react-native';
import Svg, {
  Path,
  Defs,
  LinearGradient,
  Stop,
  Line,
  Circle,
  Text as SvgText,
} from 'react-native-svg';
import { SleepChartDataPoint, SleepStage } from '../types/health';

interface SleepChartProps {
  accentColor: string;
  isDark?: boolean;
  // New props for real HealthKit data
  sleepData?: SleepChartDataPoint[] | null;
  isLoading?: boolean;
  error?: string | null;
  // Legacy props for mock data mode
  sleepStartHour?: number;
  sleepDuration?: number;
}

// Colors for each sleep stage
const STAGE_COLORS: Record<SleepStage, string> = {
  [SleepStage.Awake]: '#f97316',    // Orange
  [SleepStage.InBed]: '#94a3b8',    // Gray
  [SleepStage.REM]: '#a78bfa',      // Purple
  [SleepStage.Core]: '#60a5fa',     // Blue
  [SleepStage.Asleep]: '#60a5fa',   // Blue
  [SleepStage.Deep]: '#2563eb',     // Deep blue
};

const STAGE_LABELS: Record<SleepStage, string> = {
  [SleepStage.Awake]: 'Awake',
  [SleepStage.InBed]: 'In Bed',
  [SleepStage.REM]: 'REM',
  [SleepStage.Core]: 'Core',
  [SleepStage.Asleep]: 'Asleep',
  [SleepStage.Deep]: 'Deep',
};

/**
 * Format time for display
 */
function formatTime(date: Date): string {
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 || 12;

  if (minutes === 0) {
    return `${displayHours}${ampm}`;
  }
  return `${displayHours}:${minutes.toString().padStart(2, '0')}`;
}

export const SleepChart: React.FC<SleepChartProps> = ({
  accentColor,
  isDark = true,
  sleepData,
  isLoading = false,
  error = null,
  sleepStartHour,
  sleepDuration,
}) => {
  const width = Dimensions.get('window').width - 64;
  const height = 160;
  const padding = { top: 20, right: 10, bottom: 30, left: 10 };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Check if we're using real HealthKit data or mock mode
  const useRealData = sleepData !== undefined;

  // Process real HealthKit data for rendering
  const chartElements = useMemo(() => {
    if (useRealData) {
      // Real HealthKit data mode
      if (!sleepData || sleepData.length === 0) return null;

      const startTime = sleepData[0].time.getTime();
      const endTime = sleepData[sleepData.length - 1].time.getTime();
      const timeRange = endTime - startTime;

      // Convert data points to chart coordinates
      const points = sleepData.map((d) => ({
        x: padding.left + ((d.time.getTime() - startTime) / timeRange) * chartWidth,
        y: padding.top + chartHeight - (d.depthValue / 100) * chartHeight,
        stage: d.stage,
        time: d.time,
      }));

      // Create step-like paths for each sleep stage (colored segments)
      let linePath = '';
      const coloredSegments: { path: string; color: string }[] = [];
      
      for (let i = 0; i < points.length; i++) {
        const point = points[i];
        if (i === 0) {
          linePath = `M ${point.x} ${point.y}`;
        } else {
          const prev = points[i - 1];
          // Create colored segment for the horizontal line
          const segmentPath = `M ${prev.x} ${prev.y} L ${point.x} ${prev.y}`;
          coloredSegments.push({ path: segmentPath, color: STAGE_COLORS[prev.stage] });
          
          // Create colored segment for the vertical transition
          const transitionPath = `M ${point.x} ${prev.y} L ${point.x} ${point.y}`;
          coloredSegments.push({ path: transitionPath, color: STAGE_COLORS[point.stage] });
          
          linePath += ` L ${point.x} ${prev.y} L ${point.x} ${point.y}`;
        }
      }

      // Create area path
      const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

      // Generate time labels (show ~5 labels)
      const labelCount = 5;
      const labelInterval = Math.floor(sleepData.length / labelCount);
      const timeLabels = sleepData
        .filter((_, i) => i % labelInterval === 0 || i === sleepData.length - 1)
        .map((d) => ({
          x: padding.left + ((d.time.getTime() - startTime) / timeRange) * chartWidth,
          label: formatTime(d.time),
        }));

      return { linePath, areaPath, points, timeLabels, coloredSegments, isRealData: true };
    } else {
      // Legacy mock data mode
      const numDataPoints = 11;
      const startHour = sleepStartHour ?? 22;
      const duration = sleepDuration ?? 480;

      const data = Array.from({ length: numDataPoints }, (_, i) => {
        const progress = i / (numDataPoints - 1);
        const totalMinutes = startHour * 60 + (progress * duration);
        const hours = Math.floor(totalMinutes / 60) % 24;
        const minutes = totalMinutes % 60;

        const ampm = hours >= 12 ? 'PM' : 'AM';
        const displayHours = hours % 12 || 12;
        const timeLabel = minutes === 0
          ? `${displayHours} ${ampm}`
          : `${displayHours}:${Math.floor(minutes / 10) * 10} ${ampm}`;

        // Generate sample sleep depth values
        const value = Math.sin(progress * Math.PI * 2) * 30 + 50 + Math.sin(progress * Math.PI * 4) * 10;

        return {
          time: timeLabel,
          value: Math.max(0, Math.min(100, value)),
        };
      });

      // Create path for the area chart
      const points = data.map((d, i) => ({
        x: padding.left + (i / (data.length - 1)) * chartWidth,
        y: padding.top + chartHeight - (d.value / 100) * chartHeight,
      }));

      // Create smooth curve path
      const linePath = points.reduce((path, point, i) => {
        if (i === 0) return `M ${point.x} ${point.y}`;
        const prev = points[i - 1];
        const cpX = (prev.x + point.x) / 2;
        return `${path} C ${cpX} ${prev.y}, ${cpX} ${point.y}, ${point.x} ${point.y}`;
      }, '');

      // Create area path
      const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

      // Reference line positions
      const refLine1X = padding.left + (0.4 * chartWidth);
      const refLine2X = padding.left + (0.6 * chartWidth);

      const timeLabels = data.filter((_, i) => i % 2 === 0).map((d, i) => ({
        x: padding.left + ((i * 2) / (data.length - 1)) * chartWidth,
        label: d.time,
      }));

      return {
        linePath,
        areaPath,
        points,
        timeLabels,
        refLine1X,
        refLine2X,
        isRealData: false,
      };
    }
  }, [sleepData, useRealData, sleepStartHour, sleepDuration, chartWidth, chartHeight]);

  // Loading state (only for real data mode)
  if (useRealData && isLoading) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator color={accentColor} size="small" />
        <Text style={[styles.statusText, { color: isDark ? '#94a3b8' : '#64748b' }]}>
          Loading sleep data...
        </Text>
      </View>
    );
  }

  // Error state (only for real data mode)
  if (useRealData && error) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={[styles.errorText, { color: '#ef4444' }]}>
          {error}
        </Text>
      </View>
    );
  }

  // No data state (only for real data mode)
  if (useRealData && !chartElements) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={[styles.statusText, { color: isDark ? '#94a3b8' : '#64748b' }]}>
          No sleep data available
        </Text>
        <Text style={[styles.subText, { color: isDark ? '#64748b' : '#94a3b8' }]}>
          Wear your Apple Watch to bed to track sleep
        </Text>
      </View>
    );
  }

  if (!chartElements) return null;

  const { linePath, areaPath, timeLabels } = chartElements;

  return (
    <View style={styles.container}>
      {/* Legend - only show for real data mode */}
      {chartElements.isRealData && (
        <View style={styles.legend}>
          {[SleepStage.Deep, SleepStage.Core, SleepStage.REM, SleepStage.Awake].map((stage) => (
            <View key={stage} style={styles.legendItem}>
              <View style={[styles.legendBar, { backgroundColor: STAGE_COLORS[stage] }]} />
              <Text style={[styles.legendText, { color: isDark ? '#94a3b8' : '#64748b' }]}>
                {STAGE_LABELS[stage]}
              </Text>
            </View>
          ))}
        </View>
      )}

      <Svg width={width} height={height}>
        <Defs>
          <LinearGradient id="sleepGradient" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor={accentColor} stopOpacity={0.5} />
            <Stop offset="50%" stopColor="#60a5fa" stopOpacity={0.3} />
            <Stop offset="100%" stopColor="#2563eb" stopOpacity={0.1} />
          </LinearGradient>
          <LinearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor="#92b7c9" stopOpacity={0.4} />
            <Stop offset="100%" stopColor="#92b7c9" stopOpacity={0} />
          </LinearGradient>
        </Defs>

        {/* Y-axis labels for real data mode - Awake at top, Deep at bottom */}
        {chartElements.isRealData && (
          <>
            <SvgText
              x={padding.left}
              y={padding.top - 5}
              fontSize={9}
              fill={isDark ? '#64748b' : '#94a3b8'}
            >
              Awake
            </SvgText>
            <SvgText
              x={padding.left}
              y={height - padding.bottom + 15}
              fontSize={9}
              fill={isDark ? '#64748b' : '#94a3b8'}
            >
              Deep
            </SvgText>
          </>
        )}

        {/* Horizontal grid lines */}
        {[0, 0.33, 0.66, 1].map((ratio, i) => (
          <Line
            key={i}
            x1={padding.left}
            y1={padding.top + chartHeight * (1 - ratio)}
            x2={width - padding.right}
            y2={padding.top + chartHeight * (1 - ratio)}
            stroke={isDark ? '#334155' : '#e2e8f0'}
            strokeWidth={0.5}
            strokeDasharray="2,4"
          />
        ))}

        {/* Reference lines for mock mode */}
        {!chartElements.isRealData && chartElements.refLine1X && (
          <>
            <Line
              x1={chartElements.refLine1X}
              y1={padding.top}
              x2={chartElements.refLine1X}
              y2={height - padding.bottom}
              stroke={accentColor}
              strokeWidth={1}
              strokeDasharray="3,3"
            />
            <Line
              x1={chartElements.refLine2X}
              y1={padding.top}
              x2={chartElements.refLine2X}
              y2={height - padding.bottom}
              stroke={accentColor}
              strokeWidth={1}
              strokeDasharray="3,3"
            />
          </>
        )}

        {/* Area fill */}
        <Path d={areaPath} fill={chartElements.isRealData ? "url(#sleepGradient)" : "url(#areaGradient)"} />

        {/* Line stroke - colored segments for real data, single color for mock */}
        {chartElements.isRealData && chartElements.coloredSegments ? (
          // Render each segment with its own color
          chartElements.coloredSegments.map((segment, i) => (
            <Path
              key={i}
              d={segment.path}
              stroke={segment.color}
              strokeWidth={2.5}
              fill="none"
              strokeLinecap="round"
            />
          ))
        ) : (
          <Path
            d={linePath}
            stroke="#92b7c9"
            strokeWidth={2}
            fill="none"
            strokeLinejoin="round"
          />
        )}

        {/* Accent dots for mock mode only */}
        {!chartElements.isRealData && chartElements.refLine1X && chartElements.points && (
          <>
            <Circle
              cx={chartElements.refLine1X}
              cy={chartElements.points[Math.floor(0.4 * (chartElements.points.length - 1))].y}
              r={4}
              fill={accentColor}
            />
            <Circle
              cx={chartElements.refLine2X}
              cy={chartElements.points[Math.floor(0.6 * (chartElements.points.length - 1))].y}
              r={4}
              fill={accentColor}
            />
          </>
        )}

        {/* X-axis time labels */}
        {timeLabels.map((label, i) => (
          <SvgText
            key={i}
            x={label.x}
            y={height - 5}
            fontSize={10}
            fill={isDark ? '#64748b' : '#94a3b8'}
            textAnchor="middle"
          >
            {label.label}
          </SvgText>
        ))}
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    alignItems: 'center',
    marginTop: 8,
    minHeight: 160,
  },
  centered: {
    justifyContent: 'center',
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 8,
    flexWrap: 'wrap',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendBar: {
    width: 16,
    height: 3,
    borderRadius: 1.5,
  },
  legendText: {
    fontSize: 10,
  },
  statusText: {
    fontSize: 14,
    marginTop: 8,
  },
  subText: {
    fontSize: 12,
    marginTop: 4,
  },
  errorText: {
    fontSize: 14,
  },
});
