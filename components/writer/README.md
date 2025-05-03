# Writer

More specific documentation on specific devices and code interfaces.


What I've looked at so far:

- [Sleeppals Micro Sleep Buds](https://www.amazon.com/Sleep-Earbuds-Bluetooth-Headphones-Reduction/dp/B0DF7QFDG9/ref=sr_1_15?crid=25J9X8ATNVUVH&dib=eyJ2IjoiMSJ9.SRVpPMjwWwBoGMQEgd-c_98U5-i8Gggx1XiucBG2yxvQS3set1ZagP_7k83Y1I3iHxpylD9kX1COzxBHpku8wmql-eRsaN2YnsuVZgTFTT6p1qdq9hW8G-yH2N7X8WDNAvqpddWD5rb1lbUkyxPljbBNpr9dolJ181nX1MEiO_mNaCjOcJM2I13iYWyb2VQjSnxxwlUDzA0sO1FlOF4F-mXD1whcgwVcNVS4ck8w-dM.5GGGa-EjQRI1bvVQBYYQGOcTUENNQ1-pVaMckTrIMb0&dib_tag=se&keywords=soundcore%2BA20%2Bearbuds&qid=1745950822&sprefix=soundcore%2Ba20earbuds%2Caps%2C295&sr=8-15&th=1): Look just as comfortable as sleepcore ones and a third of the price.

- [Perytong Sleep Headband](https://www.amazon.com/Sleeping-Headphones-Bluetooth-Perytong-Headbands/dp/B0D31L1M5G/ref=sr_1_1_sspa?crid=122SDQDOJTNZQ&dib=eyJ2IjoiMSJ9.-76ghpb4BaRnHnudb3ljHGoK8fNwU-sQEaFC2Zbr1mO2OfO5qH_o_NFUp4VRj6dgDX4ptXN_9xT_3rZKTLX69maoxezKSHdJsoBwmixiu3uOeXATVLpVf0naTxiYm5pEYir2Hu3VRaozf3dN71kpGofH5Mei6vjqv6MaceOawzvqq3lxMGcyY8g79VkuZL-mqNN77nGQre-dW3T78PtaZ7Z20zhoECIVyFH_3FGdopM.HtHhdXKDsDKdbD0Xv6Q7Bsi-j4KsWgFKtQILV_PWk9Y&dib_tag=se&keywords=perytong%2Bsleep%2Bheadphones&qid=1745950759&sprefix=perytong%2Bsleep%2Bheadphones%2Caps%2C326&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1): Also dirt cheap and may just be genuinely more comftorable, also a far lower rick of it falling off during the night.


Both of the above are cheap and present two different methods of using sound during sleeping. They should both be connectable to a desktop/laptop via bluetooth. 

# But why use audio to influence brain activity?
It's not fucking electrocuting your brain for one. There's evidence to support it works too:

## 40 Hz Auditory Stimulation and Brain Entrainment

### 1. Auditory Steady-State Responses (ASSRs)
- Amplitude-modulated white-noise and click-train stimuli at 40 Hz reliably evoke an auditory steady-state response, reflecting phase-locked gamma-band activity across widespread cortical networks. Source localization and EEG topographies show the strongest power increases over frontal and prefrontal electrodes corresponding to the dorsolateral prefrontal cortex (DLPFC) when participants listen with eyes closed to sinusoidal 40 Hz stimulation [Han et al. (2023)](https://doi.org/10.1007/s11571-022-09834-x).  

### 2. 40 Hz Binaural Beats
- Listening to binaural beats with a 40 Hz difference tone induces robust frequency-following responses in EEG, with significant power increases in the gamma band over fronto-central regions. These neural changes correlate with improved working memory performance and modulations of emotional state after 15–30 minutes of exposure [Jirakittayakorn & Wongsawat (2017)](https://pubmed.ncbi.nlm.nih.gov/28739482/).  

### 3. Systematic Review of Binaural-Beat Entrainment
- A recent systematic review of 14 EEG/MEG studies finds mixed evidence for true brainwave entrainment by binaural beats. While some report time-locked gamma ASSRs, methodological heterogeneity (stimulus parameters, analysis methods, sample characteristics) limits definitive conclusions about reliable gamma entrainment via binaural beats [Ingendoh et al. (2023)](https://doi.org/10.1371/journal.pone.0286023). This will be important to dig into to refine our choices and check limitations.


---

### References

- Han, C., Zhao, X., Li, M., et al. **Enhancement of the neural response during 40 Hz auditory entrainment in closed-eye state in human prefrontal region.** *Cogn Neurodyn.* 2023 Apr;17(2):399–410. [https://doi.org/10.1007/s11571-022-09834-x](https://doi.org/10.1007/s11571-022-09834-x)  
- Jirakittayakorn, N. & Wongsawat, Y. **Brain responses to 40-Hz binaural beat and effects on emotion and memory.** *Int J Psychophysiol.* 2017; PMID: 28739482. [https://pubmed.ncbi.nlm.nih.gov/28739482/](https://pubmed.ncbi.nlm.nih.gov/28739482/)  
- Ingendoh, R. M., Posny, E. S., & Heine, A. **Binaural beats to entrain the brain? A systematic review of the effects of binaural beat stimulation on brain oscillatory activity.** *PLoS One.* 2023 May 19;18(5):e0286023. [https://doi.org/10.1371/journal.pone.0286023](https://doi.org/10.1371/journal.pone.0286023)  



---

## Transcranial Stimulation buisness

### 1. Direct DLPFC Modulation via 40 Hz tACS
- Concurrent 40 Hz transcranial alternating-current stimulation (tACS) applied over bilateral DLPFC increases BOLD activity locally in the stimulated DLPFC and remotely in premotor and anterior cingulate cortices, demonstrating causal engagement of frontal executive networks at gamma frequency [Helfrich et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35880231/).  


### References

- Helfrich, R. F., et al. **Local and distributed fMRI changes induced by 40 Hz gamma tACS of the bilateral dorsolateral prefrontal cortex: a pilot study.** *Neural Plast.* 2022; PMID: 35880231. [https://pubmed.ncbi.nlm.nih.gov/35880231/](https://pubmed.ncbi.nlm.nih.gov/35880231/)  


---


# Morpheus Console

Morpheus Console is a Python-based scheduler and audio player for binaural beats, designed to support lucid dreaming induction by entraining brainwaves during sleep.

## Features

* **Timeline UI**: Visualize and schedule playback times between 10:00 PM and 9:00 AM.
* **Configurable Duration**: Set playback length per event (e.g., 2, 5, or 15 minutes).
* **Audio Playback**: Uses PyQt5's QMediaPlayer for seamless system audio output.
* **Automatic Scheduling**: APScheduler triggers events daily at specified times.

## Installation

```bash
git clone <repo-url>
cd project-morpheus
pip install uv
uv venv
source .venv/bin/activate
uv sync
# Ensure ffmpeg or system audio support is available
```

May need to:

```bash
brew install python-tk@3.12 ffmpeg
```


## Usage

```bash
python src/app.py
```

1. **Time (HH\:MM):** Enter the hour and minute for playback.
2. **Duration (min):** Select how many minutes to play the binaural beat.
3. **Add / Remove:** Schedule or delete events; dots appear on the timeline.
4. **Play Now:** Test playback immediately at the chosen duration.

---

## Neuroscience Background & Deep Dive

### Lucid Dreaming Brainwaves

Lucid dreaming—becoming aware that you’re dreaming while still asleep—occurs during **REM sleep**, characterized by:

* **Theta (4–7 Hz):** Dominant REM rhythm in hippocampus and cortex supporting memory and dream narrative.
* **Gamma (30–60 Hz, peak \~40 Hz):** Frontolateral gamma enhancements correlate with self-awareness and cognitive control in dreams.
* **Reduced Delta (<4 Hz):** Lower slow-wave power compared to non-lucid REM.
* **Alpha & Beta (8–30 Hz):** Altered coherence and power reflecting modulated cortical communication.

These patterns suggest a reactivation of prefrontal networks—normally offline in REM—enabling metacognition and insight into the dream state.

### Binaural-Beat Entrainment Meta-Analysis

A systematic review (Ingendoh et al., 2023) of 14 EEG/MEG studies evaluated binaural beats (BB) for neural entrainment:

| Frequency Band   | Entrainment Duration | Effectiveness                                   |
| ---------------- | -------------------- | ----------------------------------------------- |
| Theta (4–8 Hz)   | 6–10 min             | Reliable theta power ↑                          |
| Alpha (8–12 Hz)  | \~5 min              | Alpha power ↑                                   |
| Gamma (30–70 Hz) | \~15 min             | Auditory steady-state responses (ASSR) at 40 Hz |
| Beta (13–30 Hz)  | —                    | No consistent entrainment                       |

**Optimal Parameters:**

* **Carrier Tones:** \~400 Hz pure tones, no pink noise embedding.
* **Beat Differences:** ≤30 Hz for clarity; 40 Hz difference still triggers ASSR.
* **Delivery:** Over-ear or in-ear isolation to each ear.

**Implications:**

* Strongest entrainment occurs in theta and gamma bands—key signatures of lucid REM.
* Methodological heterogeneity demands standardized protocols for reliable induction.

### Two-Phase Binaural-Beat Protocol for Lucidity

1. **Pre-Sleep Theta Priming:**

   * **Stimulation:** 7 Hz BB (400 vs. 407 Hz) for **6–10 minutes** immediately before lights out.
   * **Goal:** Bias the brain toward REM-like theta oscillations at sleep onset.

2. **REM Gamma Boost:**

   * **Timing:** After \~90 minutes (first REM cycle).
   * **Stimulation:** 40 Hz BB (400 vs. 440 Hz) for **15 minutes**.
   * **Goal:** Amplify frontal gamma rhythms associated with self-reflective awareness.

**Tips:**

* Keep volume low (below arousal threshold).
* Combine with mnemonic induction (MILD) and reality checks for synergistic effects.

---

## References

* Ingendoh, R. M., Posny, E. S., & Heine, A. (2023). *Binaural beats to entrain the brain?* PLoS One, 18(5)\:e0286023. doi:10.1371/journal.pone.0286023
* Jirakittayakorn, N., & Wongsawat, Y. (2017). *Brain responses to 40-Hz binaural beat.* Int J Psychophysiol. PMID:28739482
* Han, C., Zhao, X., Li, M., et al. (2023). *Enhancement of the neural response during 40 Hz auditory entrainment.* Cogn Neurodyn. doi:10.1007/s11571-022-09834-x
* Helfrich, R. F., et al. (2022). *40 Hz gamma tACS of the DLPFC.* Neural Plast. PMID:35880231

---

## Future Directions

* **Empirical Validation:** Test two-phase BB protocol with high-density EEG and subjective lucidity measures.
* **Personalization:** Adapt frequencies and durations based on individual EEG responsiveness.
* **Integration:** Combine BB delivery with sleep-trackers or smart earbuds for closed-loop stimulation.
