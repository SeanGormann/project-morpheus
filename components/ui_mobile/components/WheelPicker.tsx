import React, { useRef, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  PanResponder,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface WheelPickerProps {
  values: string[];
  selectedIndex: number;
  onValueChange: (index: number) => void;
  itemHeight?: number;
  visibleItems?: number;
  textColor?: string;
  selectedColor?: string;
  backgroundColor?: string;
  accentColor?: string;
}

const ITEM_HEIGHT = 56;
const VISIBLE_ITEMS = 5;

export const WheelPicker: React.FC<WheelPickerProps> = ({
  values,
  selectedIndex,
  onValueChange,
  itemHeight = ITEM_HEIGHT,
  visibleItems = VISIBLE_ITEMS,
  textColor = '#94a3b8',
  selectedColor = '#ffffff',
  backgroundColor = '#1e293b',
  accentColor = '#13a4ec',
}) => {
  const scrollY = useRef(new Animated.Value(0)).current;
  const lastOffset = useRef(selectedIndex * itemHeight);
  const [currentIndex, setCurrentIndex] = useState(selectedIndex);
  
  const containerHeight = itemHeight * visibleItems;
  const paddingVertical = (containerHeight - itemHeight) / 2;

  useEffect(() => {
    // Animate to selected index when it changes externally
    Animated.spring(scrollY, {
      toValue: selectedIndex * itemHeight,
      useNativeDriver: true,
      tension: 100,
      friction: 12,
    }).start();
    lastOffset.current = selectedIndex * itemHeight;
    setCurrentIndex(selectedIndex);
  }, [selectedIndex]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        scrollY.stopAnimation();
      },
      onPanResponderMove: (_, gestureState) => {
        const newValue = lastOffset.current - gestureState.dy;
        const clampedValue = Math.max(0, Math.min(newValue, (values.length - 1) * itemHeight));
        scrollY.setValue(clampedValue);
        
        // Update current index for visual feedback
        const index = Math.round(clampedValue / itemHeight);
        if (index !== currentIndex && index >= 0 && index < values.length) {
          setCurrentIndex(index);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        const velocity = gestureState.vy;
        const currentValue = lastOffset.current - gestureState.dy;
        
        // Calculate target with momentum
        let targetValue = currentValue - velocity * 150;
        
        // Snap to nearest item
        targetValue = Math.round(targetValue / itemHeight) * itemHeight;
        
        // Clamp to valid range
        targetValue = Math.max(0, Math.min(targetValue, (values.length - 1) * itemHeight));
        
        const targetIndex = Math.round(targetValue / itemHeight);
        
        Animated.spring(scrollY, {
          toValue: targetValue,
          useNativeDriver: true,
          tension: 80,
          friction: 10,
          velocity: -velocity,
        }).start(() => {
          lastOffset.current = targetValue;
          onValueChange(targetIndex);
          setCurrentIndex(targetIndex);
        });
      },
    })
  ).current;

  const renderItem = (value: string, index: number) => {
    const inputRange = [
      (index - 2) * itemHeight,
      (index - 1) * itemHeight,
      index * itemHeight,
      (index + 1) * itemHeight,
      (index + 2) * itemHeight,
    ];

    const scale = scrollY.interpolate({
      inputRange,
      outputRange: [0.7, 0.85, 1, 0.85, 0.7],
      extrapolate: 'clamp',
    });

    const opacity = scrollY.interpolate({
      inputRange,
      outputRange: [0.3, 0.5, 1, 0.5, 0.3],
      extrapolate: 'clamp',
    });

    const rotateX = scrollY.interpolate({
      inputRange,
      outputRange: ['45deg', '25deg', '0deg', '-25deg', '-45deg'],
      extrapolate: 'clamp',
    });

    const translateY = scrollY.interpolate({
      inputRange,
      outputRange: [10, 5, 0, -5, -10],
      extrapolate: 'clamp',
    });

    const isSelected = index === currentIndex;

    return (
      <Animated.View
        key={index}
        style={[
          styles.item,
          {
            height: itemHeight,
            transform: [
              { perspective: 1000 },
              { rotateX },
              { scale },
              { translateY },
            ],
            opacity,
          },
        ]}
      >
        <Text
          style={[
            styles.itemText,
            {
              color: isSelected ? selectedColor : textColor,
              fontSize: isSelected ? 32 : 22,
              fontWeight: isSelected ? '700' : '500',
            },
          ]}
        >
          {value}
        </Text>
      </Animated.View>
    );
  };

  const translateY = Animated.multiply(scrollY, -1);

  return (
    <View style={[styles.container, { height: containerHeight }]}>
      {/* Selection indicator */}
      <View
        style={[
          styles.selectionIndicator,
          {
            top: paddingVertical,
            height: itemHeight,
            borderColor: accentColor,
          },
        ]}
      >
        <View style={[styles.indicatorGlow, { backgroundColor: accentColor }]} />
      </View>
      
      {/* Side indicators */}
      <View style={[styles.sideIndicator, styles.sideIndicatorLeft, { backgroundColor: accentColor }]} />
      <View style={[styles.sideIndicator, styles.sideIndicatorRight, { backgroundColor: accentColor }]} />

      {/* Picker content */}
      <View
        style={[styles.pickerContent, { paddingVertical }]}
        {...panResponder.panHandlers}
      >
        <Animated.View
          style={[
            styles.itemsContainer,
            {
              transform: [{ translateY }],
            },
          ]}
        >
          {values.map((value, index) => renderItem(value, index))}
        </Animated.View>
      </View>

      {/* Top gradient fade */}
      <LinearGradient
        colors={[backgroundColor, `${backgroundColor}00`]}
        style={[styles.gradient, styles.gradientTop]}
        pointerEvents="none"
      />

      {/* Bottom gradient fade */}
      <LinearGradient
        colors={[`${backgroundColor}00`, backgroundColor]}
        style={[styles.gradient, styles.gradientBottom]}
        pointerEvents="none"
      />

      {/* Decorative tick marks */}
      <View style={styles.tickContainer}>
        {[...Array(11)].map((_, i) => (
          <View
            key={i}
            style={[
              styles.tick,
              {
                backgroundColor: i === 5 ? accentColor : `${textColor}30`,
                width: i === 5 ? 3 : 1,
                height: i === 5 ? 20 : (i % 2 === 0 ? 12 : 8),
              },
            ]}
          />
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    maxWidth: 280,
    alignSelf: 'center',
    overflow: 'hidden',
    borderRadius: 20,
    position: 'relative',
  },
  pickerContent: {
    flex: 1,
    overflow: 'hidden',
  },
  itemsContainer: {
    alignItems: 'center',
  },
  item: {
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  itemText: {
    fontVariant: ['tabular-nums'],
    letterSpacing: 1,
  },
  selectionIndicator: {
    position: 'absolute',
    left: 20,
    right: 20,
    borderRadius: 12,
    borderWidth: 2,
    zIndex: 1,
    pointerEvents: 'none',
  },
  indicatorGlow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 10,
    opacity: 0.08,
  },
  sideIndicator: {
    position: 'absolute',
    width: 4,
    height: 40,
    borderRadius: 2,
    top: '50%',
    marginTop: -20,
    zIndex: 2,
    opacity: 0.6,
  },
  sideIndicatorLeft: {
    left: 4,
  },
  sideIndicatorRight: {
    right: 4,
  },
  gradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 60,
    zIndex: 2,
    pointerEvents: 'none',
  },
  gradientTop: {
    top: 0,
  },
  gradientBottom: {
    bottom: 0,
  },
  tickContainer: {
    position: 'absolute',
    bottom: 8,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
    zIndex: 3,
  },
  tick: {
    borderRadius: 1,
  },
});

