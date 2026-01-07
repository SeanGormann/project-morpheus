import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Svg, { Path, Defs, LinearGradient, Stop, Line, Circle, Text as SvgText } from 'react-native-svg';

interface SleepChartProps {
  accentColor: string;
  isDark?: boolean;
  sleepStartHour: number;
  sleepDuration: number;
}

export const SleepChart: React.FC<SleepChartProps> = ({ accentColor, isDark = true, sleepStartHour, sleepDuration }) => {
  // Generate data points dynamically based on sleep schedule
  const numDataPoints = 11;
  const data = Array.from({ length: numDataPoints }, (_, i) => {
    const progress = i / (numDataPoints - 1);
    const totalMinutes = sleepStartHour * 60 + (progress * sleepDuration);
    const hours = Math.floor(totalMinutes / 60) % 24;
    const minutes = totalMinutes % 60;
    
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const timeLabel = minutes === 0 
      ? `${displayHours} ${ampm}` 
      : `${displayHours}:${Math.floor(minutes / 10) * 10} ${ampm}`;
    
    // Generate sample sleep depth values (you can replace this with real data)
    // Using a deterministic function based on progress for consistent rendering
    const value = Math.sin(progress * Math.PI * 2) * 30 + 50 + Math.sin(progress * Math.PI * 4) * 10;
    
    return {
      time: timeLabel,
      value: Math.max(0, Math.min(100, value)),
    };
  });
  const width = Dimensions.get('window').width - 64;
  const height = 140;
  const padding = { top: 10, right: 10, bottom: 25, left: 10 };
  
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
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
  
  // Create area path (line path + close to bottom)
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;
  
  // Reference line positions (at 40% and 60% of sleep duration)
  const refLine1X = padding.left + (0.4 * chartWidth);
  const refLine2X = padding.left + (0.6 * chartWidth);

  return (
    <View style={styles.container}>
      <Svg width={width} height={height}>
        <Defs>
          <LinearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor="#92b7c9" stopOpacity={0.4} />
            <Stop offset="100%" stopColor="#92b7c9" stopOpacity={0} />
          </LinearGradient>
        </Defs>
        
        {/* Reference lines */}
        <Line
          x1={refLine1X}
          y1={padding.top}
          x2={refLine1X}
          y2={height - padding.bottom}
          stroke={accentColor}
          strokeWidth={1}
          strokeDasharray="3,3"
        />
        <Line
          x1={refLine2X}
          y1={padding.top}
          x2={refLine2X}
          y2={height - padding.bottom}
          stroke={accentColor}
          strokeWidth={1}
          strokeDasharray="3,3"
        />
        
        {/* Area fill */}
        <Path d={areaPath} fill="url(#areaGradient)" />
        
        {/* Line stroke */}
        <Path d={linePath} stroke="#92b7c9" strokeWidth={2} fill="none" />
        
        {/* Accent dots */}
        <Circle cx={refLine1X} cy={points[Math.floor(0.4 * (points.length - 1))].y} r={4} fill={accentColor} />
        <Circle cx={refLine2X} cy={points[Math.floor(0.6 * (points.length - 1))].y} r={4} fill={accentColor} />
        
        {/* X-axis labels */}
        {data.filter((_, i) => i % 2 === 0).map((d, i) => (
          <SvgText
            key={d.time}
            x={padding.left + ((i * 2) / (data.length - 1)) * chartWidth}
            y={height - 5}
            fontSize={10}
            fill="#64748b"
            textAnchor="middle"
          >
            {d.time}
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
  },
});
