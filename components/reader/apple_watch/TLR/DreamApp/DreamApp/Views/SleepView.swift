//
//  SleepView.swift
//  DreamApp
//
//  Project Morpheus — Phase 1: Local data capture + debug UI
//

import SwiftUI

struct SleepView: View {
    
    @StateObject private var sessionManager = ServerCommunication(date: Date())
    var soundPlayer = SoundPlayer(rand: 5)
    @ObservedObject private var connectivityManager = WatchConnectivityManager.shared
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button {
                    soundPlayer.playSound()
                } label: {
                    Text("Training Audio")
                }
                .padding(.bottom, 8)

                NavigationLink(destination: SummaryView(sessionManager: sessionManager)) {
                    Text("Press here when done sleeping to see sleep summary")
                        .padding()
                }
                .navigationBarBackButtonHidden(true)
                
                Divider()
                
                // MARK: - Debug UI
                Text(sessionManager.status)
                    .font(.headline)
                    .foregroundColor(sessionManager.epochCount > 0 ? .green : .secondary)
                
                if sessionManager.epochCount > 0 {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Epochs received: \(sessionManager.epochCount)")
                            .font(.subheadline)
                        Text("Last HR feature: \(sessionManager.lastHrFeature, specifier: "%.4f")")
                            .font(.subheadline)
                        Text("Last heart rates: \(sessionManager.lastHeartRatesCount) samples, avg \(sessionManager.lastHeartRatesAvg, specifier: "%.1f") bpm")
                            .font(.subheadline)
                        Text("Last motion: \(sessionManager.lastMotionRowsCount) rows")
                            .font(.subheadline)
                        Text("Motion sample: \(sessionManager.lastMotionSample)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }
            }
            .padding()
        }
        .onAppear(perform: start)
    }
    
    func start() {
        UIApplication.shared.isIdleTimerDisabled = true
        var time = 30
        
        let timer = Timer(fire: Date(), interval: 1, repeats: true) { _ in
            if connectivityManager.gotData {
                connectivityManager.gotData = false
                
                if let md = connectivityManager.notificationMessage?.motionData as? [[Double]],
                   let hf = connectivityManager.notificationMessage?.hrFeature as? Double,
                   let rt = connectivityManager.notificationMessage?.heartRates as? [Double] {
                    
                    sessionManager.sendDataToServer(motionData: md, hrFeat: hf, time: time, heartRates: rt)
                }
                time += 30
            }
        }
        RunLoop.current.add(timer, forMode: .default)
    }
}

struct SleepView_Previews: PreviewProvider {
    static var previews: some View {
        SleepView()
    }
}
