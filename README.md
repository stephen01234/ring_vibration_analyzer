# Ring Vibration Analyzer 圓環振動模態分析器

這個專案提供一個基於 Python 的工具 `RingVibrationAnalyzer`，用於從影片中觀察和分析圓環的振動特性。它可以追蹤圓環的運動、測量其屬性，並對其振動進行頻率分析。

---
## 🌟 Features 功能特色

* **Video-based Analysis**: Analyzes ring vibrations from a video file.
    影片分析：從影片檔案分析圓環振動。
* **Interactive Calibration Tools**:
    互動式校準工具：
    * **Origin Calibration**: Interactively set the coordinate system origin.
        原點校準：互動式設定座標系原點。
    * **Scale Calibration**: Interactively calibrate pixel-to-real-world distance ratio.
        比例尺校準：互動式校準像素與實際距離的比例。
    * **Region of Interest (ROI) Selection**: Interactively select an ROI for analysis.
        感興趣區域 (ROI) 選擇：互動式選擇分析區域。
    * **Perspective Calibration**: Interactively correct perspective distortion (optional).
        透視校準：互動式校正透視變形 (可選)。
* **Ring Detection & Tracking**:
    圓環偵測與追蹤：
    * Standard and enhanced thin ring detection algorithms.
        標準及增強型細環偵測演算法。
    * Automatic ring type detection (normal/thin).
        自動偵測環形類型 (普通/細環)。
    * Kalman filter for smoother tracking.
        卡爾曼濾波器用於平滑追蹤。
* **Vibration Analysis**:
    振動分析：
    * Real-time data recording of position, diameter, thickness, etc.
        即時記錄位置、直徑、厚度等數據。
    * Fast Fourier Transform (FFT) for frequency domain analysis.
        快速傅立葉變換 (FFT) 用於頻域分析。
    * Generation of time-domain and frequency-domain plots.
        產生時域和頻域圖表。
    * Vibration trajectory plotting and mode analysis.
        振動軌跡繪製與模態分析。
* **Radial Vibration Analysis**:
    徑向振動分析：
    * Calculation and analysis of radial deformations at multiple points around the ring.
        計算並分析環上多個點的徑向變形。
    * Polar plots for radial vibration amplitude and frequency.
        徑向振動幅度和頻率的極座標圖。
* **Lowest Point Tracking**: Tracks the lowest point of the ring contour.
    最低點追蹤：追蹤圓環輪廓的最低點。
* **Data Export**: Saves vibration data and analysis summaries to CSV and text files.
    數據匯出：將振動數據和分析摘要儲存為 CSV 和文字檔案。
* **Contour Snapshots**:
    輪廓快照：
    * Save aligned contour snapshots (images centered on the ring).
        儲存對齊的輪廓快照 (以環心為中心的影像)。
    * Create a timelapse video from saved snapshots.
        從儲存的快照建立縮時影片。
* **Settings Management**: Save and load analysis settings (calibration data, ROI, etc.) to/from JSON files.
    設定管理：將分析設定 (校準數據、ROI 等) 儲存至 JSON 檔案或從中載入。
* **Command-Line Interface**: Control various options through command-line arguments.
    命令列介面：透過命令列參數控制各種選項。
* **Interactive Controls**: In-program hotkeys for pausing, adjusting speed, toggling views, and more.
    互動控制：程式內建熱鍵用於暫停、調整速度、切換視圖等。

---
## ⚙️ Requirements 需求

* Python 3.9.21
* OpenCV (`cv2`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* SciPy (`scipy`)
* Pandas (`pandas`)

---
## 🛠️ Installation 安裝

1.  Clone the repository:
    複製儲存庫：
    
bash
    
    git clone [https://github.com/stephen01234/ring_vibration_analyzer](https://github.com/stephen01234/ring_vibration_analyzer)
    
    cd your-repository-directory
    

2.  Install the required Python packages:
    安裝所需的 Python 套件：
    
bash
    
    pip install opencv-python numpy matplotlib scipy pandas
    


---
## 🚀 Usage 使用方法

Run the script from the command line, providing the path to the video file.
從命令列執行腳本，並提供影片檔案的路徑。

```bash
python thin_ring_vibration_analyzer.py path/to/your/video.mp4 [options]
````

**Command-Line Arguments 命令列參數:**

  * `video`: Path to the video file (required).
    `video`：影片檔案路徑 (必需)。
  * `--output-dir DIRECTORY`: Specify the output directory for saved data and plots.
    `--output-dir DIRECTORY`：指定儲存數據和圖表的輸出目錄。
  * `--debug`: Enable debug mode (prints more information).
    `--debug`：啟用除錯模式 (印出更多資訊)。
  * `--min-area PIXELS`: Minimum detection area for the ring in pixels (default: 100).
    `--min-area PIXELS`：圓環的最小偵測面積 (像素) (預設：100)。
  * `--speed FACTOR`: Initial playback speed (0.1-5.0, default: 1.0).
    `--speed FACTOR`：初始播放速度 (0.1-5.0，預設：1.0)。
  * `--calibrate`: Perform interactive scale calibration.
    `--calibrate`：執行互動式比例尺校準。
  * `--roi`: Perform interactive ROI selection.
    `--roi`：執行互動式 ROI 選擇。
  * `--origin`: Perform interactive origin calibration.
    `--origin`：執行互動式原點校準。
  * `--apply-perspective`: Perform interactive perspective calibration.
    `--apply-perspective`：執行互動式透視校準。
  * `--calibrate-all`: Perform all calibrations sequentially (perspective, ROI, scale, origin).
    `--calibrate-all`：依序執行所有校準 (透視、ROI、比例尺、原點)。
  * `--frame-range`: Interactively select the frame range to process.
    `--frame-range`：互動式選擇要處理的影格範圍。
  * `--start-frame FRAME_NUMBER`: Specify the starting frame number.
    `--start-frame FRAME_NUMBER`：指定起始影格編號。
  * `--end-frame FRAME_NUMBER`: Specify the ending frame number.
    `--end-frame FRAME_NUMBER`：指定結束影格編號。
  * `--no-auto-record`: Disable automatic data recording at the start.
    `--no-auto-record`：禁用開始時的自動數據記錄。
  * `--snapshot`: Enable saving of aligned contour snapshots.
    `--snapshot`：啟用儲存對齊的輪廓快照。
  * `--snapshot-fps FPS`: Set FPS for the timelapse video created from snapshots (default: 10).
    `--snapshot-fps FPS`：設定從快照建立的縮時影片的幀率 (預設：10)。
  * `--only-calibrate`: Perform calibrations and exit without analysis.
    `--only-calibrate`：執行校準後退出，不進行分析。
  * `--save-settings [FILENAME.json]`: Save current settings (calibrations, ROI) to a JSON file. If no filename is provided, it's auto-generated.
    `--save-settings [FILENAME.json]`：將目前設定 (校準、ROI) 儲存到 JSON 檔案。若未提供檔名，則自動產生。
  * `--load-settings FILENAME.json`: Load settings from a specified JSON file.
    `--load-settings FILENAME.json`：從指定的 JSON 檔案載入設定。
  * `--save-after-calibration`: Automatically save settings after completing any calibration step.
    `--save-after-calibration`：完成任何校準步驟後自動儲存設定。
  * `--settings-only`: Adjust and save settings then exit (use with --save-settings and calibration flags).
    `--settings-only`：調整並儲存設定後退出 (與 --save-settings 及校準旗標一同使用)。
  * `--thin-ring`: Enable analysis mode specifically for thin rings.
    `--thin-ring`：啟用專為細環設計的分析模式。
  * `--auto-detect-ring`: Automatically detect ring type (normal/thin) and choose the appropriate analysis method.
    `--auto-detect-ring`：自動偵測環的類型 (普通/細環) 並選擇合適的分析方法。
  * `--min-thin-area PIXELS`: Minimum detection area for thin ring mode (default: 10).
    `--min-thin-area PIXELS`：細環模式的最小偵測面積 (像素) (預設：10)。
  * `--thin-ring-threshold VALUE`: Area ratio threshold for thin ring determination (default: 1.5).
    `--thin-ring-threshold VALUE`：用於判斷細環的面積比例閾值 (預設：1.5)。
  * `--multi-layer`: Optimize for multi-layer gradient rings (used with thin ring mode).
    `--multi-layer`：針對多層漸層環進行優化 (與細環模式一同使用)。
  * `--enhanced-edges`: Use enhanced edge detection (used with thin ring mode).
    `--enhanced-edges`：使用增強的邊緣偵測 (與細環模式一同使用)。
  * `--radial-points NUMBER`: Number of points for radial tracking (default: 24).
    `--radial-points NUMBER`：徑向追蹤點的數量 (預設：24)。

**Example 範例:**

bash
python thin_ring_vibration_analyzer.py my_video.mp4 --roi --calibrate --origin --save-settings my_settings.json

This command will run the analyzer on `my_video.mp4`, allow you to select an ROI, calibrate the scale, set the origin, and then save these settings to my_settings.json before starting the analysis.
此命令將在 my_video.mp4 上執行分析器，允許您選擇 ROI、校準比例尺、設定原點，然後在開始分析之前將這些設定儲存到 `my_settings.json`。

-----

## ⌨️ Interactive Controls 互動控制 (During Video Processing 影片處理期間)

  * **Spacebar**: Pause/Resume video playback.
    **空格鍵**：暫停/繼續影片播放。
  * **s**: Slow down playback speed.
    **s**：減慢播放速度。
  * **f**: Speed up playback speed.
    **f**：加快播放速度。
  * **b**: Toggle binary image view.
    **b**：切換二進制影像檢視。
  * **r**: Start/Stop vibration data recording.
    **r**：開始/停止振動數據記錄。
  * **t**: Toggle thin ring detection mode (during processing, if not set by args).
    **t**：切換細環偵測模式 (處理期間，若未透過參數設定)。
  * **l**: Toggle tracking-only display mode (shows only video and tracked ring).
    **l**：切換僅追蹤顯示模式 (僅顯示影片和追蹤的圓環)。
  * **c**: Toggle contour snapshot saving.
    **c**：切換輪廓快照儲存。
  * **v**: Generate timelapse video from existing snapshots / Toggle radial analysis window visibility.
    **v**：從現有快照產生縮時影片 / 切換徑向分析視窗的可見性。
  * **a**: Analyze recorded vibration data and show plots.
    **a**：分析已記錄的振動數據並顯示圖表。
  * **d**: Analyze recorded radial vibration data and show plots.
    **d**：分析已記錄的徑向振動數據並顯示圖表。
      * (Note: t was previously for trajectory plot, now used for thin-ring toggle. Trajectory plot might be part of 'a' or 'd', or needs a new key if separate).
        (注意：`t` 先前用於軌跡圖，現用於切換細環模式。軌跡圖可能是 'a' 或 'd' 的一部分，如果分開則需要新的按鍵)。
  * **q** or **Esc**: Quit the program.
    **q** 或 **Esc**：退出程式。

**Calibration Window Controls 校準視窗控制:**

  * **Mouse Wheel**: Zoom in/out on the image.
    **滑鼠滾輪**：放大/縮小影像。
  * **Mouse Drag (Left Button)**: Pan the image.
    **滑鼠拖曳 (左鍵)**：平移影像。
  * **Shift + Left Click**: Select points for calibration (origin, scale points).
    **Shift + 左鍵點擊**：選擇校準點 (原點、比例尺點)。
  * **Enter**: Confirm selections in calibration.
    **Enter**：確認校準中的選擇。
  * **Esc**: Cancel calibration or current selection.
    **Esc**：取消校準或目前選擇。

**ROI (Region of Interest) Selection 感興趣區域 (ROI) 選擇:**
* **Mouse Drag (Left Button)**: Draw the rectangular ROI. **滑鼠拖曳 (左鍵)**：繪製矩形 ROI。
* **Enter**: Confirm the selected ROI. **Enter**：確認選擇的 ROI。
* **Esc**: Cancel ROI selection. **Esc**：取消 ROI 選擇。
  
**Origin Calibration 特定控制 (除了通用控制外)**
* Instructions are primarily displayed in the terminal. Shift+Click sets the origin. 操作說明主要顯示在終端機中。Shift+點擊設定原點。
* The window displays the current view with pixel grid at higher zoom levels. 視窗在高縮放級別下顯示帶有像素網格的目前視圖。
  
**Frame Range Selection 影格範圍選擇**
* **Left/Right Arrow Keys**: Move one frame backward/forward. **左/右方向鍵**：向後/前移動一個影格。
* **PgUp or 's'**: Move backward by approximately one second of frames. **PgUp 或 's'**：向後移動約一秒鐘的影格數。
* **PgDn or 'w'**: Move forward by approximately one second of frames. **PgDn 或 'w'**：向前移動約一秒鐘的影格數。
* **Keys '1' through '9'**: Jump to 10% through 90% of the video duration, respectively. **數字鍵 '1' 到 '9'**：分別跳轉到影片長度的 10% 到 90% 位置。
* **'0'**: Jump to the beginning of the video. **'0'**：跳轉到影片開頭。
* **'-'**: Jump to the end of the video. **'-'**：跳轉到影片結尾。
* **'R'**: Set the current frame as the start frame for processing. **'R'**：將目前影格設定為處理的起始影格。
* **'E'**: Set the current frame as the end frame for processing. **'E'**：將目前影格設定為處理的結束影格。
* **Enter**: Confirm the selected start and end frames. **Enter**：確認選擇的起始和結束影格。
* **Esc**: Cancel frame range selection. **Esc**：取消影格範圍選擇。
* The window displays the current frame, selected range on a progress bar, and time information. 視窗會顯示目前影格、進度條上選擇的範圍以及時間資訊。
-----

## 📊 Output 輸出

  * **CSV files**: Detailed frame-by-frame vibration data, radial deformation data, and lowest point tracking data.
    **CSV 檔案**：逐幀的詳細振動數據、徑向變形數據和最低點追蹤數據。
  * **PNG images**: Vibration analysis plots (time-domain, frequency-domain, trajectory, radial analysis).
    **PNG 圖片**：振動分析圖表 (時域、頻域、軌跡、徑向分析)。
  * **Text files**: Summary of the vibration analysis.
    **文字檔案**：振動分析摘要。
  * **JSON files**: Saved settings for calibrations, ROI, etc.
    **JSON 檔案**：儲存的設定，用於校準、ROI 等。
  * **MP4 vide**o: Timelapse video from contour snapshots (if snapshots are enabled and video is generated). **MP4 影片**：從輪廓快照產生的縮時影片 (如果啟用快照並產生影片)。
  * All outputs are saved in a timestamped directory within the specified output directory (or a vibration_output_YYYYMMDD_HHMMSS directory by default). 所有輸出都儲存在指定輸出目錄中的帶時間戳的目錄內 (預設情況下為 vibration_output_YYYYMMDD_HHMMSS 目錄)。
