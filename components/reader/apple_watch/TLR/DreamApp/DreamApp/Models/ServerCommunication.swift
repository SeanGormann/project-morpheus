//
//  ServerCommunication.swift
//  DreamApp
//
//  Project Morpheus — local-only, no HTTP
//

import UIKit

/// Local session manager. Processes Watch data, logs to DataLogger.
/// No server, no cloud. REM trigger mocked for now.
class ServerCommunication: NSObject, ObservableObject {
    
    var startTime: Date
    var stimulusActivator = StimulusActivation()
    var numberOfTimesActivated = 0
    
    let dataLogger = DataLogger()
    
    // MARK: - Published state for debug UI
    @Published var epochCount: Int = 0
    @Published var status: String = "Waiting for connection"
    @Published var lastHrFeature: Double = 0
    @Published var lastHeartRatesCount: Int = 0
    @Published var lastHeartRatesAvg: Double = 0
    @Published var lastMotionRowsCount: Int = 0
    @Published var lastMotionSample: String = "—"
    
    private var done = false
    private var epochIndex = 0
    
    init(date: Date) {
        self.startTime = date
        super.init()
        dataLogger.startSession()
    }
    
    /// Process epoch data locally. No HTTP.
    func sendDataToServer(motionData: [[Double]], hrFeat: Double, time: Int, heartRates: [Double]) {
        if done { return }
        
        // Log for inspection
        dataLogger.log(
            epochIndex: epochIndex,
            hrFeature: hrFeat,
            heartRates: heartRates,
            motionData: motionData
        )
        
        // Update UI state
        epochCount = dataLogger.epochCount
        status = "Received epoch \(epochCount)"
        lastHrFeature = hrFeat
        lastHeartRatesCount = heartRates.count
        lastHeartRatesAvg = heartRates.isEmpty ? 0 : heartRates.reduce(0, +) / Double(heartRates.count)
        lastMotionRowsCount = motionData.count
        
        if let first = motionData.first, first.count >= 4 {
            lastMotionSample = String(format: "[%.2f, %.3f, %.3f, %.3f]", first[0], first[1], first[2], first[3])
        } else {
            lastMotionSample = "—"
        }
        
        // TODO Phase 2: Run local inference, trigger audio if REM
        // For now: no trigger
        epochIndex += 1
    }
    
    /// End session. No HTTP. Returns export URL for sharing.
    func endServer() {
        done = true
        status = "Session ended"
    }
    
    /// Export session JSON for inspection.
    func exportSession() -> URL? {
        dataLogger.exportSession()
    }
}
