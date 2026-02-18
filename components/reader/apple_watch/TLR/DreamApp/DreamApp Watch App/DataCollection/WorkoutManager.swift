//
//  WorkoutManager.swift
//  DreamApp Watch App
//
//  Created by Rishabh Mallela on 6/14/23.
//

import Foundation
import HealthKit

class WorkoutManager: NSObject {
    
    var healthStore = HKHealthStore()
    var session: HKWorkoutSession?
    var builder: HKLiveWorkoutBuilder?
    var heartRates: [Double] = []
    var heartRateSum: Double = 0
    var numHeartRates = 0
    
//    override init() {
////        super.init()
//        self.healthStore = HKHealthStore()
//    }
    
    // Start the workout.
    func startWorkout() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit not available on this device")
            return
        }
        print("here")
        let configuration = HKWorkoutConfiguration()
        configuration.activityType = .other
        configuration.locationType = .indoor

        do {
            session = try HKWorkoutSession(healthStore: healthStore, configuration: configuration)
            builder = session?.associatedWorkoutBuilder()
            print("started workout")
        } catch {
            print("couldn't initiate workout builder: \(error)")
            return
        }

        builder?.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore,
                                                     workoutConfiguration: configuration)
        session?.delegate = self
        builder?.delegate = self

        let startDate = Date()
        session?.startActivity(with: startDate)
        builder?.beginCollection(withStart: startDate) { [weak self] success, error in
            if let error = error {
                print("HKLiveWorkoutBuilder beginCollection failed: \(error)")
            }
            if success {
                print("Workout collection started successfully")
            }
        }
    }
    
    func requestAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit not available")
            return
        }
        let typesToShare: Set = [HKQuantityType.workoutType()]
        let typesToRead: Set = [HKQuantityType.quantityType(forIdentifier: .heartRate)!]

        healthStore.requestAuthorization(toShare: typesToShare, read: typesToRead) { success, error in
            if let error = error {
                print("HealthKit auth error: \(error)")
            }
            print("HealthKit auth completed: success=\(success)")
        }
    }
    
    @Published var running = false
    func endWorkout() {
        session?.end()
        resetWorkout()
        print("woke up")
    }
    
    // MARK: - Workout Stuff and Stimuli Activation
    @Published var heartRate: Double = 0
    
    func updateForStatistics(_ statistics: HKStatistics?) {
        guard let statistics = statistics else { return }

        DispatchQueue.main.async {
            switch statistics.quantityType {
            case HKQuantityType.quantityType(forIdentifier: .heartRate):
                let heartRateUnit = HKUnit.count().unitDivided(by: HKUnit.minute())
                self.heartRate = statistics.mostRecentQuantity()?.doubleValue(for: heartRateUnit) ?? 0
//                print(self.heartRate)
                self.heartRateSum += self.heartRate
                self.numHeartRates += 1
                self.heartRates.append(self.heartRate)
            default:
                return
            }
        }
    }
    
    func getHRs() -> [Double] {
        let temp = self.heartRates
        self.heartRates = []
        return temp
    }
    
    func getAvgHR() -> Double {
        let a = self.heartRateSum
        let b = self.numHeartRates
        self.heartRateSum = 0
        self.numHeartRates = 0
        let avg = Double(a/Double(b))
        return (avg) //average heart rate over epoch
    }
    
    
    func activateStimuli() { //sound, interval, light, interval, sound, interval, light
//        print("activating")
//        bluetoothConnection!.writeData(val: 10)
//        soundPlayer.playSound()
//        print("sound")
//        DispatchQueue.main.asyncAfter(deadline: .now() + 18) { //sleep interval
//            self.soundPlayer.playSound()
//            print("sound")
//            DispatchQueue.main.asyncAfter(deadline: .now() + 13) { //sleep interval
//                print("done with activation")
//                DispatchQueue.main.asyncAfter(deadline: .now() + 480) { //sleep after activation to let lucid dream happen //MOKAMVALUE make 7 min (420)
//                    self.activatingStimuli = false
//                    print("done with sleep")
//                }
//            }
//        }
        
    }
    
    func resetWorkout() {
        builder = nil
        session = nil
        heartRate = 0
    }
}

// MARK: - HKWorkoutSessionDelegate
extension WorkoutManager: HKWorkoutSessionDelegate {
    func workoutSession(_ workoutSession: HKWorkoutSession, didChangeTo toState: HKWorkoutSessionState,
                        from fromState: HKWorkoutSessionState, date: Date) {
        DispatchQueue.main.async {
            self.running = toState == .running
        }

        // Wait for the session to transition states before ending the builder.
        if toState == .ended {
            builder?.endCollection(withEnd: date) { (success, error) in
                self.builder?.finishWorkout { (workout, error) in
                }
            }
        }
    }

    func workoutSession(_ workoutSession: HKWorkoutSession, didFailWithError error: Error) {
        print("HKWorkoutSession failed: \(error)")
    }
}

// MARK: - HKLiveWorkoutBuilderDelegate
extension WorkoutManager: HKLiveWorkoutBuilderDelegate {
    func workoutBuilderDidCollectEvent(_ workoutBuilder: HKLiveWorkoutBuilder) {

    }

    func workoutBuilder(_ workoutBuilder: HKLiveWorkoutBuilder, didCollectDataOf collectedTypes: Set<HKSampleType>) {
        for type in collectedTypes {
            guard let quantityType = type as? HKQuantityType else {
                return // Nothing to do.
            }

            let statistics = workoutBuilder.statistics(for: quantityType)

            // Update the published values.
            updateForStatistics(statistics)
        }
    }
}

