# kd-lucid-dream-lab

## How to run streaming from OpenBCI GUI

1. Open OpenBCI GUI
2. System Control Panel -> Ganglion Live
2. Select the following settings:
- Pick Transfer Protocol: BLED112 Dongle
- BLE Device: select your ganglion board
- Session Data: BDF+
- Brainflow Streamer: Choose "Network", Set IP address to 225.1.1.1, Set port to 6677
3. Start Session 
4. Start Data Stream
5. In a new terminal, run the script `stream-from-openbci-GUI.py`

<div>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <p>Stream to brainflow demo - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/7c4b133287134a08a924a850928adf90-866d7ca113c60d57-full-play.gif">
    </a>
  </div>

