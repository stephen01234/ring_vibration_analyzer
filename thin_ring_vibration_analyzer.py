import cv2
import numpy as np
import argparse
import os
from datetime import datetime
import json
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import traceback
import time

class RingVibrationAnalyzer:
    """
    圓環振動模態分析器
    用於觀測並分析圓環的振動模態特性，包括振幅、頻率、振動方向等
    """
    
    def calibrate_origin(self):
        """
        互動式座標原點校正方法 - 終端機指引版
        允許使用者選擇一個點作為座標系原點
        固定視窗大小，滾輪只控制影片的視野範圍而非像素縮放
        所有文字指引改為在終端機顯示
        - 使用Shift+點擊設定原點和座標軸
        - 使用一般點擊拖曳移動影片位置
        """
        origin_point = None
        temp_frame = None
        calibration_done = False
        
        # 固定窗口大小
        fixed_window_width = 1024
        fixed_window_height = 768
        
        # 初始化縮放因子和偏移量
        scale_factor = 1.0
        offset_x = 0
        offset_y = 0
        
        # 拖曳相關變數
        is_dragging = False
        drag_start_x = 0
        drag_start_y = 0
        
        # 追蹤鼠標位置
        mouse_x = 0
        mouse_y = 0
        
        # 合并鼠標回調函數，處理點擊、拖曳和滾輪事件
        def combined_mouse_callback(event, x, y, flags, param):
            nonlocal origin_point, temp_frame, scale_factor, offset_x, offset_y
            nonlocal is_dragging, drag_start_x, drag_start_y, mouse_x, mouse_y
            
            # 更新鼠標在原始影像上的坐標
            mouse_x = int((x - offset_x) / scale_factor)
            mouse_y = int((y - offset_y) / scale_factor)
            
            # 處理滾輪事件 - 調整視野範圍
            if event == cv2.EVENT_MOUSEWHEEL:
                # 獲取滾輪方向 (向上為正，向下為負)
                wheel_direction = 1 if flags > 0 else -1
                
                # 記錄鼠標在影像上的原始位置
                img_x = (x - offset_x) / scale_factor
                img_y = (y - offset_y) / scale_factor
                
                # 調整比例因子 - 控制視野範圍，允許更高的放大倍率
                old_factor = scale_factor
                
                # 根據當前縮放因子調整縮放步長 (高縮放時步長更大)
                if scale_factor < 5.0:
                    zoom_step = 0.5
                elif scale_factor < 10.0:
                    zoom_step = 1.0
                else:
                    zoom_step = 2.0
                
                scale_factor = max(1.0, min(30.0, scale_factor + wheel_direction * zoom_step))
                
                # 以鼠標位置為中心調整偏移量
                offset_x = x - img_x * scale_factor
                offset_y = y - img_y * scale_factor
                
                # 確保影像在窗口中居中（僅當影像小於窗口時）
                view_width = int(frame_width / scale_factor)
                view_height = int(frame_height / scale_factor)
                
                if view_width < fixed_window_width:
                    offset_x = (fixed_window_width - view_width) // 2
                if view_height < fixed_window_height:
                    offset_y = (fixed_window_height - view_height) // 2
                
                # 在終端機顯示目前的縮放比例
                print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                    
                # 重新繪製顯示
                redraw_display()
            
            # SHIFT + 左鍵點擊：設定原點和座標軸
            elif event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                # 將窗口坐標轉換回原始比例坐標
                original_x = int((x - offset_x) / scale_factor)
                original_y = int((y - offset_y) / scale_factor)
                
                # 確認點擊位置在影像範圍內
                if 0 <= original_x < frame_width and 0 <= original_y < frame_height:
                    # 設置新原點
                    origin_point = (original_x, original_y)
                    print(f"\n已選擇原點: ({original_x}, {original_y})")
                    print("按Enter確認選擇，或Shift+點擊其他位置重新選擇")
                    
                    # 重新繪製顯示
                    redraw_display()
                
            # 普通左鍵點擊：開始拖曳
            elif event == cv2.EVENT_LBUTTONDOWN:
                is_dragging = True
                drag_start_x = x
                drag_start_y = y
                
            # 鼠標移動：處理拖曳和坐標顯示
            elif event == cv2.EVENT_MOUSEMOVE:
                # 更新鼠標坐標顯示
                mouse_x = int((x - offset_x) / scale_factor)
                mouse_y = int((y - offset_y) / scale_factor)
                
                # 處理拖曳
                if is_dragging:
                    # 計算移動距離
                    dx = x - drag_start_x
                    dy = y - drag_start_y
                    
                    # 更新拖曳起點
                    drag_start_x = x
                    drag_start_y = y
                    
                    # 更新偏移量
                    offset_x += dx
                    offset_y += dy
                
                # 重新繪製顯示
                redraw_display()
                
            # 釋放左鍵：結束拖曳
            elif event == cv2.EVENT_LBUTTONUP:
                is_dragging = False
        
        def redraw_display():
            nonlocal temp_frame, origin_point
            
            # 創建一個黑色背景
            temp_frame = np.zeros((fixed_window_height, fixed_window_width, 3), dtype=np.uint8)
            
            # 計算實際顯示區域的大小
            view_width = int(fixed_window_width / scale_factor)
            view_height = int(fixed_window_height / scale_factor)
            
            # 計算中心點
            center_x = int((-offset_x / scale_factor) + (fixed_window_width / (2 * scale_factor)))
            center_y = int((-offset_y / scale_factor) + (fixed_window_height / (2 * scale_factor)))
            
            # 計算要從原始影像中裁剪的區域
            start_x = max(0, center_x - view_width // 2)
            start_y = max(0, center_y - view_height // 2)
            end_x = min(frame_width, start_x + view_width)
            end_y = min(frame_height, start_y + view_height)
            
            # 裁剪原始影像的指定區域
            crop_width = end_x - start_x
            crop_height = end_y - start_y
            
            if crop_width > 0 and crop_height > 0:
                view_region = frame[start_y:end_y, start_x:end_x]
                
                # 縮放裁剪區域以適應窗口
                dest_width = int(crop_width * scale_factor)
                dest_height = int(crop_height * scale_factor)
                
                # 計算在窗口中的顯示位置
                dest_x = max(0, int((fixed_window_width - dest_width) / 2))
                dest_y = max(0, int((fixed_window_height - dest_height) / 2))
                
                if dest_width > 0 and dest_height > 0:
                    # 使用最近鄰插值確保像素清晰可見
                    resized_region = cv2.resize(view_region, (dest_width, dest_height), interpolation=cv2.INTER_NEAREST)
                    # 確保不會超出範圍
                    if dest_y + dest_height <= fixed_window_height and dest_x + dest_width <= fixed_window_width:
                        temp_frame[dest_y:dest_y+dest_height, dest_x:dest_x+dest_width] = resized_region
                    
                    # 繪製像素網格 (在倍率大於3.0時)
                    if scale_factor >= 3.0:
                        # 繪製垂直網格線
                        for x in range(dest_x, dest_x + dest_width + 1, int(scale_factor)):
                            cv2.line(temp_frame, (x, dest_y), (x, dest_y + dest_height), (40, 40, 40), 1)
                        
                        # 繪製水平網格線
                        for y in range(dest_y, dest_y + dest_height + 1, int(scale_factor)):
                            cv2.line(temp_frame, (dest_x, y), (dest_x + dest_width, y), (40, 40, 40), 1)
                        
                        # 繪製10x10像素的粗網格 (每10個像素一條粗線)
                        for x in range(dest_x, dest_x + dest_width + 1, int(scale_factor * 10)):
                            cv2.line(temp_frame, (x, dest_y), (x, dest_y + dest_height), (80, 80, 80), 1)
                        
                        for y in range(dest_y, dest_y + dest_height + 1, int(scale_factor * 10)):
                            cv2.line(temp_frame, (dest_x, y), (dest_x + dest_width, y), (80, 80, 80), 1)
                    
                    # 高亮顯示鼠標所在的像素
                    if 0 <= mouse_x - start_x < crop_width and 0 <= mouse_y - start_y < crop_height:
                        pixel_x = dest_x + int((mouse_x - start_x) * scale_factor)
                        pixel_y = dest_y + int((mouse_y - start_y) * scale_factor)
                        
                        # 繪製當前像素的高亮框
                        pixel_size = max(1, int(scale_factor))
                        cv2.rectangle(temp_frame, 
                                    (pixel_x, pixel_y), 
                                    (pixel_x + pixel_size, pixel_y + pixel_size), 
                                    (0, 255, 255), 1)
                        
                        # 在高放大率下顯示RGB值
                        if scale_factor >= 6.0 and 0 <= mouse_x < frame_width and 0 <= mouse_y < frame_height:
                            b, g, r = frame[mouse_y, mouse_x]
                            rgb_text = f"RGB: ({r},{g},{b})"
                            cv2.putText(temp_frame, rgb_text, (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 如果已設置原點，繪製原點和坐標軸
            if origin_point:
                # 轉換原點坐標為視圖坐標
                view_orig_x = origin_point[0] - start_x
                view_orig_y = origin_point[1] - start_y
                
                # 轉換為顯示坐標
                display_x = int(view_orig_x * scale_factor) + dest_x
                display_y = int(view_orig_y * scale_factor) + dest_y
                
                # 確保坐標在顯示範圍內
                if (0 <= display_x < fixed_window_width and 
                    0 <= display_y < fixed_window_height):
                    # 繪製原點
                    cv2.circle(temp_frame, (display_x, display_y), 3, (0, 255, 255), -1)
                    
                    # 繪製座標軸 - X軸和Y軸
                    line_length = int(50)
                    cv2.arrowedLine(temp_frame, 
                            (display_x, display_y), 
                            (display_x + line_length, display_y), 
                            (0, 0, 255), 2, tipLength=0.1)  # X軸
                    cv2.arrowedLine(temp_frame, 
                            (display_x, display_y), 
                            (display_x, display_y - line_length), 
                            (0, 255, 0), 2, tipLength=0.1)  # Y軸
                    
                    # 添加坐標軸標籤
                    cv2.putText(temp_frame, "X", 
                            (display_x + line_length + 10, display_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(temp_frame, "Y", 
                            (display_x, display_y - line_length - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 顯示當前縮放比例和像素坐標
            scale_text = f"縮放: {scale_factor:.1f}x"
            cv2.putText(temp_frame, scale_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 顯示鼠標位置的像素坐標
            coord_text = f"像素坐標: ({mouse_x}, {mouse_y})"
            cv2.putText(temp_frame, coord_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加操作說明
            cv2.putText(temp_frame, "Shift+點擊設定原點, 拖曳移動視圖, 滾輪縮放", 
                    (10, fixed_window_height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 顯示結果
            cv2.imshow('Origin Calibration', temp_frame)
        
        try:
            # 讀取視頻第一幀
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取視頻幀")
                return None
            
            # 儲存原始影像的尺寸，不應用ROI裁剪，確保使用完整的影像進行校正
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            
            # 如果有透視變換矩陣並且需要應用透視校正，則應用透視變換
            if hasattr(self, 'perspective_matrix') and self.perspective_matrix is not None and getattr(self, 'apply_perspective', False):
                frame = cv2.warpPerspective(
                    frame, 
                    self.perspective_matrix, 
                    (self.warped_width, self.warped_height)
                )
                # 更新影像尺寸
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
            
            # 創建窗口
            cv2.namedWindow('Origin Calibration', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Origin Calibration', fixed_window_width, fixed_window_height)
            
            # 初始化偏移量使影像居中
            offset_x = (fixed_window_width - frame_width) // 2
            offset_y = (fixed_window_height - frame_height) // 2
            
            # 設置鼠標回調
            cv2.setMouseCallback('Origin Calibration', combined_mouse_callback)
            
            # 在終端機顯示操作指南
            print("\n==== 座標原點校準 ====")
            print("操作說明:")
            print("  - Shift+點擊影像上的點設定為座標原點")
            print("  - 拖曳滑鼠可平移影像位置")
            print("  - 使用滑鼠滾輪放大/縮小視野 (像素保持1:1)")
            print("  - 放大倍率超過3倍時會顯示像素網格")
            print("  - 放大倍率超過6倍時會顯示像素RGB值")
            print("  - 確定選擇後按Enter確認")
            print("  - 按ESC取消校準")
            print("請使用Shift+點擊選擇一個點作為座標系原點...\n")
            
            # 初始繪製顯示
            redraw_display()
            
            while not calibration_done:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    print("\n取消座標原點校準")
                    cv2.destroyAllWindows()
                    return None
                elif key == 13 and origin_point:  # Enter
                    print("\n確認選擇座標原點")
                    break
                elif key == ord('+') or key == ord('='):  # + 鍵放大
                    # 記錄視圖中心
                    center_x = fixed_window_width / 2
                    center_y = fixed_window_height / 2
                    
                    # 計算中心在原始影像中的坐標
                    img_center_x = (center_x - offset_x) / scale_factor
                    img_center_y = (center_y - offset_y) / scale_factor
                    
                    # 根據當前縮放因子調整縮放步長
                    if scale_factor < 5.0:
                        zoom_step = 0.5
                    elif scale_factor < 10.0:
                        zoom_step = 1.0
                    else:
                        zoom_step = 2.0
                    
                    # 增加縮放因子
                    old_factor = scale_factor
                    scale_factor = min(30.0, scale_factor + zoom_step)
                    
                    # 以中心點為基準調整偏移量
                    offset_x = center_x - img_center_x * scale_factor
                    offset_y = center_y - img_center_y * scale_factor
                    
                    # 在終端機顯示目前的縮放比例
                    print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                    
                    # 重繪顯示
                    redraw_display()
                    
                elif key == ord('-'):  # - 鍵縮小
                    # 記錄視圖中心
                    center_x = fixed_window_width / 2
                    center_y = fixed_window_height / 2
                    
                    # 計算中心在原始影像中的坐標
                    img_center_x = (center_x - offset_x) / scale_factor
                    img_center_y = (center_y - offset_y) / scale_factor
                    
                    # 根據當前縮放因子調整縮放步長
                    if scale_factor <= 5.0:
                        zoom_step = 0.5
                    elif scale_factor <= 10.0:
                        zoom_step = 1.0
                    else:
                        zoom_step = 2.0
                    
                    # 減小縮放因子
                    old_factor = scale_factor
                    scale_factor = max(1.0, scale_factor - zoom_step)
                    
                    # 以中心點為基準調整偏移量
                    offset_x = center_x - img_center_x * scale_factor
                    offset_y = center_y - img_center_y * scale_factor
                    
                    # 確保影像在窗口中居中（僅當影像小於窗口時）
                    view_width = int(frame_width / scale_factor)
                    view_height = int(frame_height / scale_factor)
                    
                    if view_width < fixed_window_width:
                        offset_x = (fixed_window_width - view_width) // 2
                    if view_height < fixed_window_height:
                        offset_y = (fixed_window_height - view_height) // 2
                    
                    # 在終端機顯示目前的縮放比例
                    print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                    
                    # 重繪顯示
                    redraw_display()
            
            # 詢問確認
            print(f"是否接受座標原點 ({origin_point[0]}, {origin_point[1]})? (y/n):")
            confirm = input().lower()
            if confirm == 'y':
                self.origin_x = origin_point[0]
                self.origin_y = origin_point[1]
                self.use_custom_origin = True
                print(f"座標原點設定為: ({self.origin_x}, {self.origin_y})")
                calibration_done = True
            else:
                print("座標原點校正取消")
                return None
            self.origin_x = origin_point[0]
            self.origin_y = origin_point[1]
            self.use_custom_origin = True
            
            # 明確記錄這是絕對座標系中的原點
            self.origin_is_absolute = True
            
            # 清理
            cv2.destroyAllWindows()
            for i in range(5):  # 確保窗口關閉
                cv2.waitKey(1)
            
            return (self.origin_x, self.origin_y)
            
        except Exception as e:
            print(f"座標原點校正過程中發生錯誤：{e}")
            import traceback
            traceback.print_exc()
            
            # 確保清理
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            return None
        
        if hasattr(self, 'origin_x') and hasattr(self, 'origin_y'):
            print(f"座標原點已設定為: ({self.origin_x}, {self.origin_y})")
            self.use_custom_origin = True  # 確保此標誌被設置為 True
            
            # 初始化原點對象（如果尚未初始化）
            if not hasattr(self, 'custom_origin_initialized'):
                self.custom_origin_initialized = True
                print("自定義座標原點已初始化並激活")
            
            return (self.origin_x, self.origin_y)
        else:
            # 如果未能設置原點，使用默認值
            self.origin_x = 0
            self.origin_y = 0
            self.use_custom_origin = True
            print("未能設置自定義原點，使用默認值 (0, 0)")
            return (0, 0)
    
    def __init__(self, video_path, output_dir=None, debug=False, detection_area_min=100, play_speed=1.0, auto_record=True,thin_ring_mode=True, thin_ring_params=None):
        """
        初始化振動分析器，添加自定義座標原點的初始化
        
        參數:
        video_path (str): 影片文件路徑
        output_dir (str): 輸出目錄
        debug (bool): 是否開啟調試模式
        detection_area_min (int): 最小檢測面積（像素）
        play_speed (float): 播放速度倍率
        auto_record (bool): 是否自動開始記錄數據
        """
        self.num_radial_points = 24

        self.reference_radial_distances = {}
        # 參數驗證
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"視頻文件不存在: {video_path}")
        self.show_radial_window = True  # 默认显示径向分析窗口
        self.video_path = video_path
        self.debug = debug
        
        # 追蹤參數設定
        self.detection_area_min = max(10, detection_area_min)
        self.play_speed = max(0.1, min(play_speed, 5.0))
        self.is_paused = False
        self.show_binary = True  # 控制是否顯示二值化圖像
        
        # 添加座標原點相關屬性並設置默認值
        self.use_custom_origin = True
        self.origin_x = 0
        self.origin_y = 0
        self.custom_origin_initialized = True  # 標記已初始化
        
        
        self.thin_ring_mode = thin_ring_mode
        # 追蹤模式設定
        self.tracking_mode_only = False  # 默认为显示所有元素
        
        # 振動分析相關參數
        self.tracking_history = []  # 存儲圓環位置歷史
        self.max_history_length = 200  # 顯示的歷史軌跡長度
        self.vibration_samples = []  # 儲存振動樣本資料
        self.max_vibration_samples = 3000  # 最大振動樣本數量
        self.is_recording = auto_record  # 默認自動開始記錄
        self.reference_position = None  # 振動參考位置

        self.thin_ring_params = {
            'min_area': 10,                # 最小檢測面積
            'circularity_range': (0.3, 1.0),  # 圓形度範圍
            'center_distance_factor': 0.7,    # 中心距離閾值因子
            'canny_thresholds': (50, 150),    # Canny邊緣檢測的閾值
            'area_ratio_target': 1.5          # 目標面積比例
        }
        
        # 設定輸出目錄
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir is None:
            self.output_dir = f"vibration_output_{current_datetime}"
        else:
            self.output_dir = os.path.join(output_dir, f"vibration_{current_datetime}")

        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"結果將保存到目錄: {self.output_dir}")
        if thin_ring_params:
            self.thin_ring_params.update(thin_ring_params)
        # 初始化影片捕獲
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise IOError(f"無法打開影片文件: {video_path}")
        except Exception as e:
            print(f"視頻打開失敗: {e}")
            sys.exit(1)
        
        # 獲取影片信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 安全性檢查
        if self.total_frames <= 0 or self.frame_width <= 0 or self.frame_height <= 0:
            print("無效的視頻文件：無法獲取視頻信息")
            sys.exit(1)
        
        print(f"影片信息: {self.frame_width}x{self.frame_height}, FPS: {self.fps}, 總幀數: {self.total_frames}")
        
        # 初始化結果數據結構
        self.data = {
            'frame': [],
            'time': [],
            'x': [],
            'y': [],
            'rel_x': [],  # 添加相對座標欄位
            'rel_y': [],  # 添加相對座標欄位
            'angle': [],
            'inner_diameter': [],
            'outer_diameter': [],
            'ring_thickness': []
        }
        
        # 像素到厘米的轉換比例 (預設為1)
        self.pixel_to_cm = 1.0
        
        # 影像處理相關
        self.roi = None
        self.use_roi = False
        
        # 初始化卡爾曼濾波器
        self.kalman = cv2.KalmanFilter(4, 2)  # 狀態: [x, y, dx, dy], 測量: [x, y]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman_initialized = False
        
        # 振動分析視覺化參數
        self.vibration_plot_data = {
            'times': [],
            'x_pos': [],
            'y_pos': [],
            'x_vibration': [],
            'y_vibration': []
        }
        
        # 即時頻率分析的窗口大小
        self.fft_window_size = int(self.fps * 3)  # 3秒資料窗口
        
        # 用於存儲圓環屬性的資料
        self.current_inner_diameter = 0
        self.current_outer_diameter = 0
        self.current_ring_thickness = 0
        
        # 自動記錄模式
        if auto_record:
            print("自動記錄模式已開啟 - 將自動記錄所有幀")
    
    def enhanced_thin_ring_detection(self, frame, previous_position=None):
        """
        增強版細環檢測方法，主要針對二值化和輪廓檢測進行優化
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        previous_position (tuple): 上一幀檢測到的位置
        
        返回:
        tuple: (position, angle, contour, binary_img, success, inner_diameter, outer_diameter)
        """
        # 步驟1: 轉為灰度圖像並增強對比度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用自適應直方圖均衡化以增強對比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 步驟2: 多種降噪方法以適應不同場景
        # 保留邊緣的雙邊濾波
        bilateral = cv2.bilateralFilter(enhanced, 7, 50, 50)
        
        # 一般高斯模糊 - 較輕微
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 非局部均值降噪 - 對保留結構很有效
        nlm = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # 步驟3: 多種二值化方法
        binary_results = []
        
        # 自適應閾值 - 不同參數組合
        thresholds = [
            {"block_size": 7, "c": 2, "image": bilateral},
            {"block_size": 9, "c": 2, "image": nlm},
            {"block_size": 11, "c": 3, "image": gaussian},
            {"block_size": 15, "c": 4, "image": enhanced}
        ]
        
        for params in thresholds:
            try:
                binary = cv2.adaptiveThreshold(
                    params["image"], 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 
                    params["block_size"], 
                    params["c"]
                )
                binary_results.append(binary)
            except Exception as e:
                if self.debug:
                    print(f"自適應閾值錯誤: {e}")
        
        # Canny 邊緣檢測 - 對細環非常有效
        canny_params = [
            {"low": 20, "high": 60, "image": bilateral},
            {"low": 30, "high": 90, "image": bilateral},
            {"low": 40, "high": 120, "image": gaussian},
            {"low": 50, "high": 150, "image": enhanced}
        ]
        
        for params in canny_params:
            try:
                edges = cv2.Canny(params["image"], params["low"], params["high"])
                binary_results.append(edges)
            except Exception as e:
                if self.debug:
                    print(f"Canny檢測錯誤: {e}")
        
        # 基於梯度的二值化
        try:
            # 計算 Sobel 梯度
            sobelx = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
            
            # 計算梯度幅度
            gradient = np.sqrt(sobelx**2 + sobely**2)
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 閾值處理梯度
            _, gradient_binary = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
            binary_results.append(gradient_binary)
        except Exception as e:
            if self.debug:
                print(f"梯度計算錯誤: {e}")
        
        # Otsu閾值方法
        try:
            for img in [enhanced, bilateral, nlm]:
                _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                binary_results.append(otsu)
        except Exception as e:
            if self.debug:
                print(f"Otsu閾值錯誤: {e}")
        
        # 步驟4: 合併所有二值化結果
        combined_binary = np.zeros_like(gray)
        valid_count = 0
        
        for binary in binary_results:
            if binary is not None and binary.shape == combined_binary.shape:
                combined_binary = cv2.bitwise_or(combined_binary, binary)
                valid_count += 1
        
        # 如果沒有有效的二值化結果，使用備用方法
        if valid_count == 0:
            _, backup_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
            combined_binary = backup_binary
        
        # 步驟5: 形態學處理改進二值圖像
        # 使用不同方向的結構元素連接斷點
        # 水平連接
        kernel_h = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
        morph_h = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_h)
        
        # 垂直連接
        kernel_v = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
        morph_v = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_v)
        
        # 對角線連接 (45度)
        kernel_d1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8)
        morph_d1 = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_d1)
        
        # 對角線連接 (135度)
        kernel_d2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8)
        morph_d2 = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_d2)
        
        # 合併所有方向的結果
        direction_combined = cv2.bitwise_or(cv2.bitwise_or(morph_h, morph_v), 
                                        cv2.bitwise_or(morph_d1, morph_d2))
        
        # 使用小核的閉操作進一步連接斷點
        kernel_close = np.ones((2, 2), np.uint8)
        closed_binary = cv2.morphologyEx(direction_combined, cv2.MORPH_CLOSE, kernel_close)
        
        # 針對環狀結構進行優化，使用形態學處理
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 從5x5改為3x3
        circular_binary = cv2.morphologyEx(closed_binary, cv2.MORPH_CLOSE, kernel_circle)
        
        # 使用開運算去除噪點，但保留較大結構
        kernel_open = np.ones((2, 2), np.uint8)
        opened_binary = cv2.morphologyEx(circular_binary, cv2.MORPH_OPEN, kernel_open)
        
        # 步驟6: 霍夫圓檢測增強
        circles_mask = np.zeros_like(gray)
        
        try:
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=40,     # 降低這個值可能有助於檢測邊緣不太明顯的圓
                param2=20,     # 降低此參數以檢測更多可能的圓
                minRadius=10,
                maxRadius=150
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                for circle in circles[0, :]:
                    x, y, r = circle
                    # 只繪製輪廓線
                    cv2.circle(circles_mask, (x, y), r, 255, 2)
        except Exception as e:
            if self.debug:
                print(f"霍夫圓檢測錯誤: {e}")
        
        # 合併霍夫圓結果
        final_binary = cv2.bitwise_or(opened_binary, circles_mask)
        
        # 步驟7: 連通域分析優化
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_binary, connectivity=8)
        
        # 過濾連通域，保留有效的物件
        filtered_binary = np.zeros_like(final_binary)
        min_area = max(10, self.detection_area_min * 0.3)  # 降低最小面積閾值
        
        for i in range(1, num_labels):  # 跳過背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 使用更寬鬆的篩選條件
            if area > min_area:
                # 計算寬高比
                aspect_ratio = max(width, height) / (min(width, height) + 0.1)
                
                # 允許更多形狀，但過濾極度細長的物件
                if aspect_ratio < 8.0:
                    filtered_binary[labels == i] = 255
        
        # 最終的二值化圖像用於輪廓檢測
        final_binary = filtered_binary
        
        # 步驟8: 創建彩色二值圖像用於顯示
        binary_img = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
        
        # 步驟9: 查找輪廓
        contours, _ = cv2.findContours(final_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在二值化圖像上繪製所有輪廓
        cv2.drawContours(binary_img, contours, -1, (0, 100, 100), 1)
        
        # 步驟10: 評估輪廓
        valid_contours = []
        min_area = max(10, self.detection_area_min * 0.3)  # 再次確認最小面積閾值
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 過濾過小的輪廓
            if area < min_area:
                continue
            
            # 計算圓形度
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # 使用非常寬鬆的圓形度條件
            if 0.2 < circularity < 1.5:
                # 計算中心點
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                    
                    # 保存這個輪廓的詳細信息
                    valid_contours.append({
                        'contour': contour,
                        'area': area,
                        'circularity': circularity,
                        'center': center
                    })
        
        # 步驟11: 如果沒有找到有效輪廓，考慮合成輪廓或返回失敗
        if not valid_contours:
            # 如果有前一幀位置，創建一個人工合成的輪廓
            if previous_position is not None:
                # 使用上一幀位置創建一個人工圓環
                cv2.circle(binary_img, (int(previous_position[0]), int(previous_position[1])), 30, (0, 0, 255), 2)
                
                # 建立一個假輪廓
                synthetic_contour = []
                for angle in np.linspace(0, 2*np.pi, 36):
                    x = int(previous_position[0] + 30 * np.cos(angle))
                    y = int(previous_position[1] + 30 * np.sin(angle))
                    synthetic_contour.append([[x, y]])
                
                synthetic_contour = np.array(synthetic_contour, dtype=np.int32)
                return previous_position, 0, synthetic_contour, binary_img, True, 50, 60
            
            # 如果沒有前一幀位置，返回失敗
            return None, 0, None, binary_img, False, 0, 0
        
        # 步驟12: 如果有前一幀位置，優先選擇接近的輪廓
        selected_contour = None
        
        if previous_position is not None:
            # 計算所有輪廓到前一位置的距離
            for contour_info in valid_contours:
                center = contour_info['center']
                dist = np.sqrt((center[0] - previous_position[0])**2 + 
                            (center[1] - previous_position[1])**2)
                contour_info['prev_distance'] = dist
            
            # 按距離排序
            valid_contours.sort(key=lambda x: x['prev_distance'])
            
            # 選擇最近的輪廓
            for contour_info in valid_contours:
                if contour_info['prev_distance'] < 100:  # 距離閾值
                    selected_contour = contour_info
                    break
        
        # 如果沒有找到接近的輪廓，選擇圓形度最高的
        if selected_contour is None:
            valid_contours.sort(key=lambda x: -x['circularity'])  # 按圓形度降序排序
            selected_contour = valid_contours[0]
        
        # 步驟13: 最終處理和繪製
        contour = selected_contour['contour']
        center = selected_contour['center']
        
        # 新增: 重採樣和平滑關鍵區域
        try:
            if len(contour) >= 5:
                # 首先進行重採樣 - 均勻分佈點
                # 計算輪廓總長度
                perimeter = cv2.arcLength(contour, True)
                
                # 重新採樣為100個點
                resampled_contour = []
                contour_points = contour.reshape(-1, 2)
                cumulative_length = 0
                segment_length = perimeter / 100
                
                for i in range(len(contour_points)):
                    p1 = contour_points[i]
                    p2 = contour_points[(i + 1) % len(contour_points)]
                    
                    segment_dist = np.sqrt(np.sum((p2 - p1) ** 2))
                    
                    while cumulative_length + segment_dist >= segment_length:
                        t = (segment_length - cumulative_length) / segment_dist
                        new_point = p1 + t * (p2 - p1)
                        resampled_contour.append([[new_point[0], new_point[1]]])
                        
                        segment_dist = segment_dist - (segment_length - cumulative_length)
                        cumulative_length = 0
                        p1 = new_point
                        
                    cumulative_length += segment_dist
                
                # 確保有足夠的點
                if len(resampled_contour) >= 5:
                    resampled_contour = np.array(resampled_contour, dtype=np.int32)
                    
                    # 然後平滑關鍵區域 - 特別處理頂部和底部區域
                    # 獲取輪廓點
                    points = resampled_contour.reshape(-1, 2)
                    
                    # 計算每個點相對中心的角度
                    angles = []
                    for point in points:
                        dx = point[0] - center[0]
                        dy = point[1] - center[1]
                        angle = np.rad2deg(np.arctan2(dy, dx))
                        angles.append(angle)
                    
                    # 定義需要平滑的角度區域
                    critical_regions = [(-10, 10), (170, 190), (80, 100), (-100, -80)]
                    
                    # 對每個關鍵區域進行平滑
                    for region in critical_regions:
                        start_angle, end_angle = region
                        # 找出該區域的點
                        region_indices = []
                        for i, angle in enumerate(angles):
                            if start_angle <= angle <= end_angle or (start_angle < -90 and angle >= 360 + start_angle):
                                region_indices.append(i)
                                
                        if len(region_indices) > 3:
                            # 計算該區域點的平均半徑
                            avg_radius = 0
                            for idx in region_indices:
                                dx = points[idx][0] - center[0]
                                dy = points[idx][1] - center[1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                avg_radius += dist
                            
                            if len(region_indices) > 0:
                                avg_radius /= len(region_indices)
                                
                                # 將該區域的所有點調整為沿著與中心的徑向方向
                                for idx in region_indices:
                                    # 計算徑向方向
                                    dx = points[idx][0] - center[0]
                                    dy = points[idx][1] - center[1]
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    
                                    # 正規化方向向量
                                    if dist > 0:
                                        dx /= dist
                                        dy /= dist
                                        
                                        # 更新點的位置
                                        points[idx][0] = center[0] + avg_radius * dx
                                        points[idx][1] = center[1] + avg_radius * dy
                    
                    # 重建輪廓
                    smoothed_contour = np.array([points], dtype=np.int32)
                    contour = smoothed_contour
        except Exception as e:
            if self.debug:
                print(f"輪廓處理錯誤: {e}")
        
        # 在二值化圖像上繪製選中的輪廓
        cv2.drawContours(binary_img, [contour], 0, (0, 255, 0), 2)
        cv2.circle(binary_img, center, 3, (0, 0, 255), -1)
        
        # 嘗試擬合橢圓以獲取角度和尺寸
        angle = 0
        inner_diameter = 0
        outer_diameter = 0
        
        # 增強的橢圓擬合方法
        try:
            if len(contour) >= 5:
                # 標準橢圓擬合
                try:
                    standard_ellipse = cv2.fitEllipse(contour)
                    
                    # 基於標準橢圓計算每個點的誤差
                    ellipse_center = standard_ellipse[0]
                    ellipse_axes = standard_ellipse[1]
                    ellipse_angle = standard_ellipse[2]
                    
                    # 獲取輪廓點
                    points = contour.reshape(-1, 2)
                    
                    # 計算每個點到擬合橢圓的距離
                    distances = []
                    for point in points:
                        # 計算點到橢圓中心的向量
                        dx = point[0] - ellipse_center[0]
                        dy = point[1] - ellipse_center[1]
                        
                        # 計算向量長度
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # 計算該方向上橢圓的半徑
                        angle_rad = np.arctan2(dy, dx)
                        # 調整角度以匹配橢圓旋轉
                        angle_ellipse = angle_rad - np.deg2rad(ellipse_angle)
                        
                        # 計算橢圓在該方向的半徑
                        a = ellipse_axes[0] / 2
                        b = ellipse_axes[1] / 2
                        r_ellipse = (a * b) / np.sqrt((b * np.cos(angle_ellipse))**2 + (a * np.sin(angle_ellipse))**2)
                        
                        # 計算點到橢圓的距離
                        distance = abs(dist - r_ellipse)
                        distances.append(distance)
                    
                    # 計算距離的閾值 (平均距離 + 2 * 標準差)
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    threshold = mean_dist + 2 * std_dist
                    
                    # 移除距離過大的點
                    good_indices = [i for i, d in enumerate(distances) if d <= threshold]
                    if len(good_indices) >= 5 and len(good_indices) < len(points):
                        optimized_contour = np.array([points[i] for i in good_indices]).reshape(-1, 1, 2)
                        
                        # 使用優化後的輪廓進行擬合
                        optimized_ellipse = cv2.fitEllipse(optimized_contour)
                        
                        # 在二值化圖像上繪製優化後的橢圓
                        cv2.ellipse(binary_img, optimized_ellipse, (255, 0, 255), 1)
                        
                        # 使用優化後的橢圓參數
                        center = optimized_ellipse[0]
                        angle = optimized_ellipse[2]
                        
                        # 計算內外徑
                        axes = optimized_ellipse[1]
                        max_axis = max(axes)
                        min_axis = min(axes)
                        avg_axis = (max_axis + min_axis) / 2
                        outer_diameter = avg_axis
                        inner_diameter = avg_axis * 0.8
                    else:
                        # 使用標準橢圓擬合結果
                        center = standard_ellipse[0]
                        angle = standard_ellipse[2]
                        
                        # 計算內外徑
                        axes = standard_ellipse[1]
                        max_axis = max(axes)
                        min_axis = min(axes)
                        avg_axis = (max_axis + min_axis) / 2
                        outer_diameter = avg_axis
                        inner_diameter = avg_axis * 0.8
                        
                        # 在二值化圖像上繪製標準橢圓
                        cv2.ellipse(binary_img, standard_ellipse, (255, 0, 0), 1)
                except Exception as e:
                    if self.debug:
                        print(f"橢圓擬合錯誤: {e}")
                    raise Exception("橢圓擬合失敗，嘗試圓形擬合")
            else:
                raise Exception("輪廓點數不足，嘗試圓形擬合")
        except Exception as e:
            if self.debug:
                print(f"橢圓擬合過程錯誤: {e}")
            
            # 嘗試圓形擬合作為後備選項
            try:
                # 計算到中心的平均距離作為半徑
                points = contour.reshape(-1, 2)
                distances = [np.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2) for p in points]
                radius = np.mean(distances)
                
                # 在二值化圖像上畫出擬合的圓
                cv2.circle(binary_img, (int(center[0]), int(center[1])), int(radius), (0, 200, 200), 1)
                
                # 設置直徑和角度
                outer_diameter = radius * 2
                inner_diameter = outer_diameter * 0.8
                angle = 0
            except Exception as sub_e:
                if self.debug:
                    print(f"圓形擬合也失敗: {sub_e}")
                
                # 使用面積估計直徑
                diameter = 2 * np.sqrt(selected_contour['area'] / np.pi)
                outer_diameter = diameter
                inner_diameter = diameter * 0.8
                angle = 0
        
        # 強化環形可視化
        cv2.circle(binary_img, (int(center[0]), int(center[1])), int(outer_diameter/2), (255, 100, 100), 1)
        
        return center, angle, contour, binary_img, True, inner_diameter, outer_diameter
    
    def calibrate_scale(self):
        """
        互動式比例尺校準方法 - 增強版
        允許使用者選擇兩個點，並輸入其實際距離
        加入像素級精確縮放功能，便於精準定位
        """
        points = []
        temp_frame = None
        
        # 初始化固定視窗大小
        window_width = 1024
        window_height = 768
        
        # 拖曳相關變數
        is_dragging = False
        drag_start_x = 0
        drag_start_y = 0
        
        # 初始化縮放因子和偏移量
        scale_factor = 1.0
        offset_x = 0
        offset_y = 0
        
        # 追蹤鼠標位置
        mouse_x = 0
        mouse_y = 0
        
        # 重置像素到厘米的轉換率
        self.pixel_to_cm = None
        calibration_done = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, temp_frame, scale_factor, offset_x, offset_y
            nonlocal is_dragging, drag_start_x, drag_start_y, mouse_x, mouse_y
            
            # 更新鼠標在原始影像上的坐標
            mouse_x = int((x - offset_x) / scale_factor)
            mouse_y = int((y - offset_y) / scale_factor)
            
            # 處理滾輪事件 - 調整視野範圍
            if event == cv2.EVENT_MOUSEWHEEL:
                # 獲取滾輪方向 (向上為正，向下為負)
                wheel_direction = 1 if flags > 0 else -1
                
                # 記錄鼠標在影像上的原始位置
                img_x = (x - offset_x) / scale_factor
                img_y = (y - offset_y) / scale_factor
                
                # 調整比例因子 - 控制視野範圍，允許更高的放大倍率
                old_factor = scale_factor
                
                # 根據當前縮放因子調整縮放步長 (高縮放時步長更大)
                if scale_factor < 5.0:
                    zoom_step = 0.5
                elif scale_factor < 10.0:
                    zoom_step = 1.0
                else:
                    zoom_step = 2.0
                
                scale_factor = max(1.0, min(30.0, scale_factor + wheel_direction * zoom_step))
                
                # 以鼠標位置為中心調整偏移量
                offset_x = x - img_x * scale_factor
                offset_y = y - img_y * scale_factor
                
                # 確保影像在窗口中居中（僅當影像小於窗口時）
                view_width = int(frame.shape[1] / scale_factor)
                view_height = int(frame.shape[0] / scale_factor)
                
                if view_width < window_width:
                    offset_x = (window_width - view_width) // 2
                if view_height < window_height:
                    offset_y = (window_height - view_height) // 2
                
                # 在終端機顯示目前的縮放比例
                print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                
                # 重新繪製顯示
                redraw_view()
            
            # SHIFT + 左鍵點擊選擇參考點
            elif event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                if len(points) < 2:
                    # 將窗口坐標轉換回原始比例坐標
                    original_x = int((x - offset_x) / scale_factor)
                    original_y = int((y - offset_y) / scale_factor)
                    
                    # 確認點擊位置在影像範圍內
                    if 0 <= original_x < frame.shape[1] and 0 <= original_y < frame.shape[0]:
                        points.append((original_x, original_y))
                        print(f"選擇點 #{len(points)}: ({original_x}, {original_y})")
                    
                    # 重繪顯示
                    redraw_view()
            
            # 普通左鍵點擊：開始拖曳
            elif event == cv2.EVENT_LBUTTONDOWN:
                is_dragging = True
                drag_start_x = x
                drag_start_y = y
                
            # 鼠標移動：處理拖曳和坐標顯示
            elif event == cv2.EVENT_MOUSEMOVE:
                # 更新鼠標坐標顯示
                mouse_x = int((x - offset_x) / scale_factor)
                mouse_y = int((y - offset_y) / scale_factor)
                
                # 處理拖曳
                if is_dragging:
                    # 計算移動距離
                    dx = x - drag_start_x
                    dy = y - drag_start_y
                    
                    # 更新拖曳起點
                    drag_start_x = x
                    drag_start_y = y
                    
                    # 更新偏移量
                    offset_x += dx
                    offset_y += dy
                
                # 重新繪製顯示
                redraw_view()
                
            # 釋放左鍵：結束拖曳
            elif event == cv2.EVENT_LBUTTONUP:
                is_dragging = False
        
        def redraw_view():
            nonlocal temp_frame, points
            
            # 創建一個黑色背景
            temp_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            # 計算實際顯示區域的大小
            view_width = int(fixed_window_width / scale_factor)
            view_height = int(fixed_window_height / scale_factor)
            
            # 計算中心點
            center_x = int((-offset_x / scale_factor) + (fixed_window_width / (2 * scale_factor)))
            center_y = int((-offset_y / scale_factor) + (fixed_window_height / (2 * scale_factor)))
            
            # 計算要從原始影像中裁剪的區域
            start_x = max(0, center_x - view_width // 2)
            start_y = max(0, center_y - view_height // 2)
            end_x = min(frame.shape[1], start_x + view_width)
            end_y = min(frame.shape[0], start_y + view_height)
            
            # 裁剪原始影像的指定區域
            crop_width = end_x - start_x
            crop_height = end_y - start_y
            
            if crop_width > 0 and crop_height > 0:
                view_region = frame[start_y:end_y, start_x:end_x]
                
                # 縮放裁剪區域以適應窗口
                dest_width = int(crop_width * scale_factor)
                dest_height = int(crop_height * scale_factor)
                
                # 計算在窗口中的顯示位置
                dest_x = max(0, int((fixed_window_width - dest_width) / 2))
                dest_y = max(0, int((fixed_window_height - dest_height) / 2))
                
                if dest_width > 0 and dest_height > 0:
                    # 使用最近鄰插值確保像素清晰可見
                    resized_region = cv2.resize(view_region, (dest_width, dest_height), interpolation=cv2.INTER_NEAREST)
                    # 確保不會超出範圍
                    if dest_y + dest_height <= fixed_window_height and dest_x + dest_width <= fixed_window_width:
                        temp_frame[dest_y:dest_y+dest_height, dest_x:dest_x+dest_width] = resized_region
                    
                    # 繪製像素網格 (在倍率大於3.0時)
                    if scale_factor >= 3.0:
                        # 繪製垂直網格線
                        for x in range(dest_x, dest_x + dest_width + 1, int(scale_factor)):
                            cv2.line(temp_frame, (x, dest_y), (x, dest_y + dest_height), (40, 40, 40), 1)
                        
                        # 繪製水平網格線
                        for y in range(dest_y, dest_y + dest_height + 1, int(scale_factor)):
                            cv2.line(temp_frame, (dest_x, y), (dest_x + dest_width, y), (40, 40, 40), 1)
                        
                        # 繪製10x10像素的粗網格 (每10個像素一條粗線)
                        for x in range(dest_x, dest_x + dest_width + 1, int(scale_factor * 10)):
                            cv2.line(temp_frame, (x, dest_y), (x, dest_y + dest_height), (80, 80, 80), 1)
                        
                        for y in range(dest_y, dest_y + dest_height + 1, int(scale_factor * 10)):
                            cv2.line(temp_frame, (dest_x, y), (dest_x + dest_width, y), (80, 80, 80), 1)
                    
                    # 高亮顯示鼠標所在的像素
                    if 0 <= mouse_x - start_x < crop_width and 0 <= mouse_y - start_y < crop_height:
                        pixel_x = dest_x + int((mouse_x - start_x) * scale_factor)
                        pixel_y = dest_y + int((mouse_y - start_y) * scale_factor)
                        
                        # 繪製當前像素的高亮框
                        pixel_size = max(1, int(scale_factor))
                        cv2.rectangle(temp_frame, 
                                    (pixel_x, pixel_y), 
                                    (pixel_x + pixel_size, pixel_y + pixel_size), 
                                    (0, 255, 255), 1)
                        
                        # 在高放大率下顯示RGB值
                        if scale_factor >= 6.0 and 0 <= mouse_x < frame.shape[1] and 0 <= mouse_y < frame.shape[0]:
                            b, g, r = frame[mouse_y, mouse_x]
                            rgb_text = f"RGB: ({r},{g},{b})"
                            cv2.putText(temp_frame, rgb_text, (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 繪製已選擇的點
            for i, point in enumerate(points):
                # 計算點在顯示視圖中的位置
                view_point_x = point[0] - start_x
                view_point_y = point[1] - start_y
                
                if 0 <= view_point_x < crop_width and 0 <= view_point_y < crop_height:
                    # 轉換為屏幕坐標
                    screen_x = dest_x + int(view_point_x * scale_factor)
                    screen_y = dest_y + int(view_point_y * scale_factor)
                    
                    # 繪製點
                    cv2.circle(temp_frame, (screen_x, screen_y), max(3, int(scale_factor / 2)), (0, 0, 255), -1)
                    cv2.putText(temp_frame, f"Point {i+1}", 
                            (screen_x + 10, screen_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 如果有兩個點，繪製連線
            if len(points) == 2:
                # 檢查兩個點是否在當前視圖中
                p1_x = points[0][0] - start_x
                p1_y = points[0][1] - start_y
                p2_x = points[1][0] - start_x
                p2_y = points[1][1] - start_y
                
                # 至少一個點在視圖中時才繪製線
                if ((0 <= p1_x < crop_width and 0 <= p1_y < crop_height) or 
                    (0 <= p2_x < crop_width and 0 <= p2_y < crop_height)):
                    
                    # 轉換為屏幕坐標
                    screen_p1_x = dest_x + int(p1_x * scale_factor)
                    screen_p1_y = dest_y + int(p1_y * scale_factor)
                    screen_p2_x = dest_x + int(p2_x * scale_factor)
                    screen_p2_y = dest_y + int(p2_y * scale_factor)
                    
                    # 繪製線
                    cv2.line(temp_frame, (screen_p1_x, screen_p1_y), (screen_p2_x, screen_p2_y), (0, 255, 0), 2)
                    
                    # 計算像素距離
                    pixel_distance = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
                    
                    # 顯示距離
                    cv2.putText(temp_frame, f"distance: {pixel_distance:.2f} pixels", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 顯示當前縮放比例和鼠標位置的像素坐標
            scale_text = f"縮放: {scale_factor:.1f}x"
            cv2.putText(temp_frame, scale_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            coord_text = f"像素坐標: ({mouse_x}, {mouse_y})"
            cv2.putText(temp_frame, coord_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 顯示操作說明
            cv2.putText(temp_frame, "SHIFT+點擊選擇參考點 (選擇兩個點), 按Enter確認", 
                    (10, window_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(temp_frame, "拖曳移動視圖, 滾輪縮放, ESC取消", 
                    (10, window_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 顯示影像
            cv2.imshow('Scale Calibration', temp_frame)
        
        try:
            # 讀取視頻第一幀
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取視頻幀")
                return None
            
            # 應用ROI裁剪(如果啟用)
            if self.use_roi and self.roi is not None:
                x1, y1, x2, y2 = self.roi
                frame = frame[y1:y2, x1:x2]
            
            # 固定窗口大小
            fixed_window_width = window_width
            fixed_window_height = window_height
            
            # 創建顯示窗口
            cv2.namedWindow('Scale Calibration', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Scale Calibration', window_width, window_height)
            offset_x = (window_width - frame.shape[1]) // 2
            offset_y = (window_height - frame.shape[0]) // 2

            # 設置鼠標回調
            cv2.setMouseCallback('Scale Calibration', mouse_callback)
            
            # 初始顯示
            redraw_view()
            
            print("\n==== 比例尺校準 ====")
            print("操作說明:")
            print("  - 按住 SHIFT + 左鍵點擊來選擇參考點 (需選擇兩個點)")
            print("  - 拖曳滑鼠可平移影像位置")
            print("  - 使用滑鼠滾輪放大/縮小視野")
            print("  - 放大倍率超過3倍時會顯示像素網格")
            print("  - 放大倍率超過6倍時會顯示像素RGB值")
            print("  - 按 ENTER 確認選擇")
            print("  - 按 ESC 退出校準")
            print("請選擇兩個已知實際距離的參考點...\n")
            
            # 等待使用者選擇兩個點
            while not calibration_done:
                key = cv2.waitKey(100) & 0xFF
                
                if key == 27:  # ESC
                    print("\n取消校準")
                    cv2.destroyAllWindows()
                    return None
                elif key == ord('+') or key == ord('='):  # + 鍵放大
                    # 記錄視圖中心
                    center_x = window_width / 2
                    center_y = window_height / 2
                    
                    # 計算中心在原始影像中的坐標
                    img_center_x = (center_x - offset_x) / scale_factor
                    img_center_y = (center_y - offset_y) / scale_factor
                    
                    # 根據當前縮放因子調整縮放步長
                    if scale_factor < 5.0:
                        zoom_step = 0.5
                    elif scale_factor < 10.0:
                        zoom_step = 1.0
                    else:
                        zoom_step = 2.0
                    
                    # 增加縮放因子
                    old_factor = scale_factor
                    scale_factor = min(30.0, scale_factor + zoom_step)
                    
                    # 以中心點為基準調整偏移量
                    offset_x = center_x - img_center_x * scale_factor
                    offset_y = center_y - img_center_y * scale_factor
                    
                    # 在終端機顯示目前的縮放比例
                    print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                    
                    # 重繪顯示
                    redraw_view()
                    
                elif key == ord('-'):  # - 鍵縮小
                    # 記錄視圖中心
                    center_x = window_width / 2
                    center_y = window_height / 2
                    
                    # 計算中心在原始影像中的坐標
                    img_center_x = (center_x - offset_x) / scale_factor
                    img_center_y = (center_y - offset_y) / scale_factor
                    
                    # 根據當前縮放因子調整縮放步長
                    if scale_factor <= 5.0:
                        zoom_step = 0.5
                    elif scale_factor <= 10.0:
                        zoom_step = 1.0
                    else:
                        zoom_step = 2.0
                    
                    # 減小縮放因子
                    old_factor = scale_factor
                    scale_factor = max(1.0, scale_factor - zoom_step)
                    
                    # 以中心點為基準調整偏移量
                    offset_x = center_x - img_center_x * scale_factor
                    offset_y = center_y - img_center_y * scale_factor
                    
                    # 確保影像在窗口中居中（僅當影像小於窗口時）
                    view_width = int(frame.shape[1] / scale_factor)
                    view_height = int(frame.shape[0] / scale_factor)
                    
                    if view_width < window_width:
                        offset_x = (window_width - view_width) // 2
                    if view_height < window_height:
                        offset_y = (window_height - view_height) // 2
                    
                    # 在終端機顯示目前的縮放比例
                    print(f"當前視野範圍: {100/scale_factor:.1f}%, 縮放倍率: {scale_factor:.1f}x", end="\r")
                    
                    # 重繪顯示
                    redraw_view()
                elif key == 13 and len(points) == 2:  # ENTER
                    # 計算像素距離
                    pixel_distance = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
                    
                    # 提示使用者輸入實際距離
                    print("\n" + "="*40)
                    print(f"兩點之間的像素距離為: {pixel_distance:.2f} 像素")
                    
                    # 循環直到獲得有效輸入
                    try:
                        real_distance = float(input("請輸入這兩點之間的實際距離 (cm)："))
                        if real_distance <= 0:
                            print("距離必須為正數，請重新選擇點並輸入。")
                            points = []
                            redraw_view()
                            continue
                        
                        # 計算比例尺
                        ratio = real_distance / pixel_distance
                        print(f"比例尺計算結果: 1像素 = {ratio:.6f} cm")
                        
                        # 詢問確認
                        confirm = input("是否接受這個比例尺? (y/n): ").lower()
                        if confirm == 'y':
                            self.pixel_to_cm = ratio
                            calibration_done = True
                            print(f"校準成功！比例尺設置為: 1像素 = {ratio:.6f} cm")
                        else:
                            print("重新開始校準過程...")
                            points = []  # 清除已選點
                            redraw_view()
                            
                    except ValueError:
                        print("請輸入有效的數字。")
                        points = []
                    except Exception as e:
                        print(f"輸入處理錯誤：{e}")
                        points = []
            
            # 清理
            cv2.destroyAllWindows()
            
            # 確認比例尺設置成功
            if self.pixel_to_cm is not None:
                return self.pixel_to_cm
            else:
                print("校準未完成，使用默認比例尺 1.0")
                self.pixel_to_cm = 1.0
                return 1.0
                
        except Exception as e:
            print(f"校準過程中發生錯誤：{e}")
            traceback.print_exc()
            
            # 確保清理
            cv2.destroyAllWindows()
            
            # 設置默認比例尺
            self.pixel_to_cm = 1.0
            return 1.0

    def select_roi(self):
        """
        互動式選擇感興趣區域 (ROI)
        允許使用者框選一個矩形作為分析區域
        """
        # 重置ROI相關變數
        self.roi = None
        self.use_roi = False
        
        # 創建窗口
        cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI Selection', 1024, 768)
        
        # 儲存滑鼠操作的變數
        roi_points = []
        dragging = False
        start_point = None
        
        # 滑鼠回調函數
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_points, dragging, start_point, frame_display
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # 開始拖曳
                dragging = True
                start_point = (x, y)
                # 重置已有的選擇
                roi_points = []
                
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                # 拖曳過程中，繪製臨時矩形
                temp_frame = frame_display.copy()
                cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('ROI Selection', temp_frame)
                
            elif event == cv2.EVENT_LBUTTONUP and dragging:
                # 拖曳結束，保存矩形區域
                dragging = False
                end_point = (x, y)
                
                # 確保矩形座標正確（左上和右下）
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                x2 = max(start_point[0], end_point[0])
                y2 = max(start_point[1], end_point[1])
                
                # 確保矩形大小合理
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    # 矩形太小，重置
                    temp_frame = frame_display.copy()
                    cv2.putText(temp_frame, "Selected area too small, try again", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('ROI Selection', temp_frame)
                    return
                
                # 保存ROI座標
                roi_points = [x1, y1, x2, y2]
                
                # 在影像上顯示最終矩形
                temp_frame = frame_display.copy()
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 顯示ROI資訊
                roi_info = f"ROI: ({x1}, {y1}), ({x2}, {y2}), Size: {x2-x1}x{y2-y1}"
                cv2.putText(temp_frame, roi_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(temp_frame, "Press ENTER to confirm, ESC to cancel, or select again", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 顯示裁剪後的預覽
                if len(roi_points) == 4:
                    cropped = frame[roi_points[1]:roi_points[3], roi_points[0]:roi_points[2]]
                    if cropped.size > 0:  # 確保裁剪區域有效
                        cv2.namedWindow('Cropped Preview', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Cropped Preview', (roi_points[2]-roi_points[0]), (roi_points[3]-roi_points[1]))
                        cv2.imshow('Cropped Preview', cropped)
                
                cv2.imshow('ROI Selection', temp_frame)
        
        try:
            # 讀取視頻第一幀
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取視頻幀")
                return None
            
            # 創建顯示用的幀
            frame_display = frame.copy()
            
            # 添加指導文字
            # cv2.putText(frame_display, "拖曳滑鼠選擇感興趣區域", 
            #     (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame_display, "按ESC退出，ENTER確認", 
            #     (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 顯示幀並設置鼠標回調
            cv2.imshow('ROI Selection', frame_display)
            cv2.setMouseCallback('ROI Selection', mouse_callback)
            
            # 等待用戶操作
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    # 取消選擇
                    roi_points = []
                    break
                elif key == 13 and len(roi_points) == 4:  # ENTER
                    # 確認選擇
                    print(f"已選擇ROI: ({roi_points[0]}, {roi_points[1]}), ({roi_points[2]}, {roi_points[3]})")
                    
                    self.roi = roi_points
                    self.use_roi = True
                    break
            
            # 清理窗口 
            cv2.destroyAllWindows()
            
            return self.roi
            
        except Exception as e:
            print(f"ROI選擇過程中發生錯誤：{e}")
            traceback.print_exc()
            
            # 確保清理
            cv2.destroyAllWindows()
            
            return None

    def detect_thin_ring_contours(self, frame, previous_position=None):
        """
        檢測細環的多層漸層輪廓 - 修正版
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        previous_position (tuple): 上一幀的位置
        
        返回:
        list: 包含輪廓信息的字典列表，按面積排序
        """
        # 轉為灰度圖像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用自適應直方圖均衡化以增強對比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 使用小核的高斯模糊以保留細節
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 組合多種二值化方法的結果 - 創建一個空的二值圖像
        binary_combined = np.zeros_like(gray)
        
        # 1. 自適應閾值 - 使用較小的窗口以捕獲細節
        try:
            binary_adaptive = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 7, 2
            )
            binary_combined = cv2.bitwise_or(binary_combined, binary_adaptive)
        except Exception as e:
            print(f"自適應閾值錯誤: {e}")
        
        # 2. Canny邊緣檢測 - 對細邊緣非常有效
        try:
            canny_edges = cv2.Canny(blurred, 30, 150)
            binary_combined = cv2.bitwise_or(binary_combined, canny_edges)
        except Exception as e:
            print(f"Canny檢測錯誤: {e}")
        
        # 3. Otsu二值化作為備用方法
        try:
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_combined = cv2.bitwise_or(binary_combined, binary_otsu)
        except Exception as e:
            print(f"Otsu二值化錯誤: {e}")
        
        # 使用很小的結構元素進行形態學操作，以連接斷裂的邊緣
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, kernel)
        
        # 顯示二值化結果以便調試
        # 注意: 這行不應影響函數的主要功能，只是用來檢查二值化結果
        cv2.imshow('Thin Ring Binary', binary)
        
        # 查找輪廓前先檢查二值圖像是否有效
        if np.sum(binary) == 0:
            print("警告: 二值圖像全黑，調整參數")
            # 使用更寬鬆的Canny參數重新嘗試
            binary = cv2.Canny(blurred, 10, 100)
        
        # 查找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 過濾和評估輪廓
        valid_contours = []
        
        # 降低面積閾值以捕獲更細的輪廓 (但設置最小值為10防止噪點)
        min_area = max(10, self.detection_area_min * 0.3)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 計算圓形度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # 放寬對細環的圓形度要求
                if 0.3 < circularity < 1.1:  # 調整為更寬鬆的範圍
                    # 計算中心點
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 儲存輪廓資訊
                        valid_contours.append({
                            'contour': contour,
                            'area': area,
                            'circularity': circularity,
                            'center': (cx, cy)
                        })
        
        # 按面積排序
        valid_contours.sort(key=lambda x: x['area'])
        
        return valid_contours
    
    def apply_roi_to_frame(self, frame):
        """
        將ROI應用到幀，裁剪出感興趣區域
        
        參數:
        frame (numpy.ndarray): 原始幀
        
        返回:
        numpy.ndarray: 裁剪後的幀
        """
        if self.use_roi and self.roi and len(self.roi) == 4:
            # 裁剪幀
            x1, y1, x2, y2 = self.roi
            cropped_frame = frame[y1:y2, x1:x2]
            return cropped_frame
        else:
            # 如果沒有設置ROI，返回原始幀
            return frame
    
    def initialize_kalman_filter(self):
        """改良的卡爾曼濾波器初始化，提高對快速移動物體的追蹤能力"""
        self.kalman = cv2.KalmanFilter(4, 2)  # 狀態: [x, y, dx, dy], 測量: [x, y]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
        # 增加過程噪聲以提高對快速變化的響應能力
        process_noise = 0.15  # 從0.03增加到0.1，使系統更快地響應變化
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * process_noise
        
        # 降低測量噪聲，使系統更相信測量值而非預測值
        measurement_noise = 0.05  # 從0.1降低到0.05
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
        
        self.kalman_initialized = False

    def update_kalman_filter(self, measurement):
        """
        改良的卡爾曼濾波器更新，包含自適應參數調整以處理快速運動
        
        參數:
        measurement (tuple): 測量的位置 (x, y)
        
        返回:
        tuple: 濾波後的位置 (x, y)
        """
        if not hasattr(self, 'kalman'):
            self.initialize_kalman_filter()
        
        # 保存前一個狀態，用於計算速度
        if self.kalman_initialized:
            previous_state = self.kalman.statePost.copy()
        
        # 如果是首次初始化
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
            self.kalman_initialized = True
            
            # 初始化速度估計和歷史測量值列表
            self.velocity_estimate = [0, 0]
            self.measurement_history = []
            
            return measurement
        
        # 添加當前測量值到歷史記錄
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > 5:  # 保留最近5個測量值
            self.measurement_history.pop(0)
        
        # 基於歷史測量值估計速度
        if len(self.measurement_history) >= 2:
            latest = self.measurement_history[-1]
            previous = self.measurement_history[-2]
            dx = latest[0] - previous[0]
            dy = latest[1] - previous[1]
            
            # 使用指數移動平均更新速度估計
            alpha = 0.3  # 權重因子
            self.velocity_estimate[0] = alpha * dx + (1 - alpha) * self.velocity_estimate[0]
            self.velocity_estimate[1] = alpha * dy + (1 - alpha) * self.velocity_estimate[1]
        
        # 根據估計速度調整過程噪聲
        velocity_magnitude = np.sqrt(self.velocity_estimate[0]**2 + self.velocity_estimate[1]**2)
        
        # 速度越快，使用越高的過程噪聲值
        adaptive_process_noise = min(0.5, 0.05 + velocity_magnitude * 0.01)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * adaptive_process_noise
        
        # 預測
        predicted = self.kalman.predict()
        
        # 測量更新
        measurement_array = np.array([[measurement[0]], [measurement[1]]], np.float32)
        
        # 檢查測量值與預測值的差距
        dx = abs(measurement[0] - predicted[0, 0])
        dy = abs(measurement[1] - predicted[1, 0])
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 根據速度大小和測量差距自適應調整測量噪聲
        if distance > 20 and velocity_magnitude < 5:
            # 靜止或低速下的大差距可能是檢測錯誤，增加測量噪聲
            adaptive_measurement_noise = distance / 10
        elif velocity_magnitude > 5:
            # 高速運動時，降低測量噪聲，更相信實時測量值
            adaptive_measurement_noise = 0.01
        else:
            # 正常情況
            adaptive_measurement_noise = 0.05
        
        # 暫時調整測量噪聲
        original_noise = self.kalman.measurementNoiseCov.copy()
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * adaptive_measurement_noise
        
        # 更新
        corrected = self.kalman.correct(measurement_array)
        
        # 恢復原始測量噪聲設置（可選）
        # self.kalman.measurementNoiseCov = original_noise
        
        # 返回修正後的位置
        return (corrected[0, 0], corrected[1, 0])
 
    def reset_kalman_filter(self):
        """重置卡尔曼滤波器"""
        if hasattr(self, 'kalman_initialized'):
            self.kalman_initialized = False
            
        # 如果需要，完全重新初始化濾波器
        if hasattr(self, 'kalman'):
            self.initialize_kalman_filter()

    def smooth_contour(self, contour, method="approx_poly", param=0.001):
        """
        對輪廓進行平滑處理以減少浮動
        
        參數:
        contour (numpy.ndarray): 原始輪廓點集
        method (str): 平滑方法，可選 "approx_poly", "spline", "moving_avg", "gaussian"
        param (float): 平滑參數，方法不同參數意義不同
        
        返回:
        numpy.ndarray: 平滑後的輪廓
        """
        if contour is None or len(contour) < 5:
            return contour
        
        try:
            if method == "approx_poly":
                # 使用多邊形近似，param是精度參數
                epsilon = 0.003 * cv2.arcLength(contour, True)  # 減小值可保留更多細節
                smooth = cv2.approxPolyDP(contour, epsilon, True)
                
                # 如果近似後點太少，調整精度重試
                if len(smooth) < 8:
                    epsilon = param * 0.5 * cv2.arcLength(contour, True)
                    smooth = cv2.approxPolyDP(contour, epsilon, True)
                
                return smooth
                
            elif method == "spline":
                # 使用B樣條進行平滑處理
                from scipy.interpolate import splprep, splev
                import numpy as np
                
                # 將輪廓轉換為點陣列
                contour_array = contour.squeeze().astype(np.float32)
                
                # 確保至少有4個點
                if len(contour_array) < 4:
                    return contour
                
                # 閉合曲線處理
                is_closed = np.allclose(contour_array[0], contour_array[-1])
                
                # 為閉合曲線準備數據
                if is_closed:
                    x = np.append(contour_array[:, 0], contour_array[0, 0])
                    y = np.append(contour_array[:, 1], contour_array[0, 1])
                else:
                    x = contour_array[:, 0]
                    y = contour_array[:, 1]
                
                # 擬合樣條曲線
                s_param = param * len(contour_array) * 0.05
                tck, u = splprep([x, y], s=s_param, per=is_closed)
                
                # 生成更多平滑的點
                u_new = np.linspace(0, 1, len(contour) * 2)
                x_new, y_new = splev(u_new, tck)
                
                # 轉換回OpenCV輪廓格式
                smooth_array = np.column_stack((x_new, y_new)).astype(np.int32)
                smooth = smooth_array.reshape((-1, 1, 2))
                
                return smooth
                
            elif method == "moving_avg":
                # 使用移動平均平滑，param是窗口大小
                window_size = max(3, int(param * len(contour)))
                
                # 確保窗口大小是奇數
                if window_size % 2 == 0:
                    window_size += 1
                
                # 將輪廓轉換為點陣列
                contour_array = contour.squeeze()
                
                # 創建閉合循環以處理邊界
                padded = np.vstack([contour_array[-window_size//2:], contour_array, contour_array[:window_size//2]])
                
                # 應用移動平均
                smooth_array = np.zeros_like(contour_array, dtype=np.float32)
                for i in range(len(contour_array)):
                    smooth_array[i] = np.mean(padded[i:i+window_size], axis=0)
                
                # 轉換回OpenCV輪廓格式
                smooth = smooth_array.astype(np.int32).reshape((-1, 1, 2))
                
                return smooth
                
            elif method == "gaussian":
                # 使用高斯濾波進行平滑，param是標準差
                import numpy as np
                from scipy.ndimage import gaussian_filter1d
                
                # 將輪廓轉換為點陣列
                contour_array = contour.squeeze().astype(np.float32)
                
                # 確保足夠多的點
                if len(contour_array) < 3:
                    return contour
                
                # 為閉合曲線準備處理
                is_closed = np.allclose(contour_array[0], contour_array[-1])
                
                # 創建閉合循環以處理邊界效應
                if is_closed:
                    # 添加前後的點以處理邊界
                    padding = int(len(contour_array) * 0.1)  # 使用10%的點作為padding
                    padded_x = np.concatenate([contour_array[-padding:, 0], contour_array[:, 0], contour_array[:padding, 0]])
                    padded_y = np.concatenate([contour_array[-padding:, 1], contour_array[:, 1], contour_array[:padding, 1]])
                    
                    # 應用高斯濾波
                    smoothed_x = gaussian_filter1d(padded_x, param)
                    smoothed_y = gaussian_filter1d(padded_y, param)
                    
                    # 提取中間部分
                    smooth_x = smoothed_x[padding:-padding]
                    smooth_y = smoothed_y[padding:-padding]
                else:
                    # 直接應用高斯濾波
                    smooth_x = gaussian_filter1d(contour_array[:, 0], param)
                    smooth_y = gaussian_filter1d(contour_array[:, 1], param)
                
                # 組合為點陣列
                smooth_array = np.column_stack((smooth_x, smooth_y)).astype(np.int32)
                
                # 轉換回OpenCV輪廓格式
                smooth = smooth_array.reshape((-1, 1, 2))
                
                return smooth
            
            else:
                # 默認返回原始輪廓
                return contour
                
        except Exception as e:
            if self.debug:
                print(f"輪廓平滑處理出錯: {e}")
            return contour  # 發生錯誤時返回原始輪廓

    def start_vibration_recording(self):
        """開始記錄振動數據"""
        self.is_recording = True
        self.vibration_samples = []
        print("開始記錄振動數據")
    
    def stop_vibration_recording(self):
        """停止記錄振動數據"""
        self.is_recording = False
        print(f"停止記錄振動數據，共記錄 {len(self.vibration_samples)} 個樣本")

    def update_vibration_plot_data(self, sample):
        """
        更新振動繪圖數據，使用自定義座標原點系統
        
        參數:
        sample (dict): 振動樣本
        """
        # 添加時間數據
        self.vibration_plot_data['times'].append(sample['time'])
        
        # 始終使用相對於自定義原點的座標
        self.vibration_plot_data['x_pos'].append(sample['rel_x'])
        self.vibration_plot_data['y_pos'].append(sample['rel_y'])
        
        # 添加振動數據（相對於參考位置的偏移）
        self.vibration_plot_data['x_vibration'].append(sample['dx'])
        self.vibration_plot_data['y_vibration'].append(sample['dy'])
        
        # 計算並添加徑向振動數據
        if self.reference_position is not None:
            # 計算當前點到參考點的距離
            dx = sample['x'] - self.reference_position[0]
            dy = sample['y'] - self.reference_position[1]
            current_radius = np.sqrt(dx*dx + dy*dy)
            
            # 計算參考半徑 (第一次記錄的半徑)
            if 'reference_radius' not in self.vibration_plot_data:
                self.vibration_plot_data['reference_radius'] = current_radius
                self.vibration_plot_data['radial_vibration'] = [0.0]  # 初始點的徑向振動為0
            else:
                # 計算徑向振動 (當前半徑 - 參考半徑)
                radial_vibration = current_radius - self.vibration_plot_data['reference_radius']
                self.vibration_plot_data['radial_vibration'].append(radial_vibration)
        
        # 如果有徑向變形數據，則計算平均徑向變形
        if 'radial_deformations' in sample and sample['radial_deformations']:
            # 計算所有角度的平均徑向變形
            deformations = [d['deformation'] for d in sample['radial_deformations']]
            avg_deformation = np.mean(deformations)
            
            # 添加到振動數據
            if 'avg_radial_deformation' not in self.vibration_plot_data:
                self.vibration_plot_data['avg_radial_deformation'] = []
            self.vibration_plot_data['avg_radial_deformation'].append(avg_deformation)
        
        # 限制數據長度，保持與 FFT 窗口大小一致
        max_length = max(self.fft_window_size, 200)  # 至少保留200個點
        if len(self.vibration_plot_data['times']) > max_length:
            self.vibration_plot_data['times'].pop(0)
            self.vibration_plot_data['x_pos'].pop(0)
            self.vibration_plot_data['y_pos'].pop(0)
            self.vibration_plot_data['x_vibration'].pop(0)
            self.vibration_plot_data['y_vibration'].pop(0)
            
            # 也移除徑向振動數據
            if 'radial_vibration' in self.vibration_plot_data and len(self.vibration_plot_data['radial_vibration']) > max_length:
                self.vibration_plot_data['radial_vibration'].pop(0)
            
            # 移除平均徑向變形數據
            if 'avg_radial_deformation' in self.vibration_plot_data and len(self.vibration_plot_data['avg_radial_deformation']) > max_length:
                self.vibration_plot_data['avg_radial_deformation'].pop(0)

    def perform_realtime_fft(self):
        """
        執行即時 FFT 分析，使用自定義座標系中的數據
        
        返回:
        dict: FFT 分析結果
        """
        # 確保有足夠的數據
        if len(self.vibration_plot_data['times']) < self.fft_window_size:
            return None
        
        # 獲取最近的數據窗口 - 注意這裡已確保使用相對座標
        times = np.array(self.vibration_plot_data['times'][-self.fft_window_size:])
        x_vibration = np.array(self.vibration_plot_data['x_vibration'][-self.fft_window_size:])
        y_vibration = np.array(self.vibration_plot_data['y_vibration'][-self.fft_window_size:])
        
        # 獲取徑向振動數據
        has_radial_data = False
        if 'radial_vibration' in self.vibration_plot_data and len(self.vibration_plot_data['radial_vibration']) >= self.fft_window_size:
            radial_vibration = np.array(self.vibration_plot_data['radial_vibration'][-self.fft_window_size:])
            has_radial_data = True
        elif 'avg_radial_deformation' in self.vibration_plot_data and len(self.vibration_plot_data['avg_radial_deformation']) >= self.fft_window_size:
            radial_vibration = np.array(self.vibration_plot_data['avg_radial_deformation'][-self.fft_window_size:])
            has_radial_data = True
        
        # 去除趨勢
        x_detrended = signal.detrend(x_vibration)
        y_detrended = signal.detrend(y_vibration)
        if has_radial_data:
            radial_detrended = signal.detrend(radial_vibration)
        
        # 應用窗口函數
        window = signal.windows.hann(len(x_detrended))
        x_windowed = x_detrended * window
        y_windowed = y_detrended * window
        if has_radial_data:
            radial_windowed = radial_detrended * window
        
        # 計算 FFT
        x_fft = fft(x_windowed)
        y_fft = fft(y_windowed)
        if has_radial_data:
            radial_fft = fft(radial_windowed)
        
        # 計算採樣率和頻率軸
        dt = np.mean(np.diff(times))
        fs = 1 / dt
        n = len(times)
        freqs = fftfreq(n, dt)
        
        # 只保留正頻率部分
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        x_fft_mag = np.abs(x_fft[pos_mask]) / n * 2  # 乘以2是因為只取了一半的頻譜
        y_fft_mag = np.abs(y_fft[pos_mask]) / n * 2
        if has_radial_data:
            radial_fft_mag = np.abs(radial_fft[pos_mask]) / n * 2
        
        # 找出主頻率
        x_peak_idx = np.argmax(x_fft_mag)
        y_peak_idx = np.argmax(y_fft_mag)
        if has_radial_data:
            radial_peak_idx = np.argmax(radial_fft_mag)
        
        x_peak_freq = freqs[x_peak_idx]
        y_peak_freq = freqs[y_peak_idx]
        if has_radial_data:
            radial_peak_freq = freqs[radial_peak_idx]
        
        x_peak_amp = x_fft_mag[x_peak_idx]
        y_peak_amp = y_fft_mag[y_peak_idx]
        if has_radial_data:
            radial_peak_amp = radial_fft_mag[radial_peak_idx]
        
        # 計算振幅（標準差）
        x_amplitude = np.std(x_vibration)
        y_amplitude = np.std(y_vibration)
        if has_radial_data:
            radial_amplitude = np.std(radial_vibration)
        
        result = {
            'freqs': freqs,
            'x_fft_mag': x_fft_mag,
            'y_fft_mag': y_fft_mag,
            'x_peak_freq': x_peak_freq,
            'y_peak_freq': y_peak_freq,
            'x_peak_amp': x_peak_amp,
            'y_peak_amp': y_peak_amp,
            'x_amplitude': x_amplitude,
            'y_amplitude': y_amplitude
        }
        
        # 添加徑向振動數據
        if has_radial_data:
            result.update({
                'radial_fft_mag': radial_fft_mag,
                'radial_peak_freq': radial_peak_freq,
                'radial_peak_amp': radial_peak_amp,
                'radial_amplitude': radial_amplitude
            })
        
        return result
    
    def analyze_and_plot_vibration(self):
        """
        分析並繪製完整的振動分析圖表
        
        返回:
        matplotlib.figure.Figure: 分析圖表
        """
        if len(self.vibration_samples) < 10:
            print("錯誤: 沒有足夠的振動數據進行分析")
            return None
        
        # 提取數據
        times = np.array([sample['time'] for sample in self.vibration_samples])
        x_pos = np.array([sample['x'] for sample in self.vibration_samples])
        y_pos = np.array([sample['y'] for sample in self.vibration_samples])
        
        # 計算振動（去趨勢）
        x_trend = signal.savgol_filter(x_pos, min(51, len(x_pos) // 4 * 2 + 1), 3)
        y_trend = signal.savgol_filter(y_pos, min(51, len(y_pos) // 4 * 2 + 1), 3)
        
        x_vibration = x_pos - x_trend
        y_vibration = y_pos - y_trend
        
        # 計算振幅
        x_amplitude = np.std(x_vibration)
        y_amplitude = np.std(y_vibration)
        
        # 計算峰對峰值
        x_peak_to_peak = np.max(x_vibration) - np.min(x_vibration)
        y_peak_to_peak = np.max(y_vibration) - np.min(y_vibration)
        
        # FFT 分析
        fft_results = self._perform_fft_analysis(times, x_vibration, y_vibration)
        
        # 創建圖表
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ring vibration analysis', fontsize=16)
        
        # 時域振動圖 - X 方向
        axs[0, 0].plot(times, x_pos, 'b-', alpha=0.5, label='raw data')
        axs[0, 0].plot(times, x_trend, 'r--', label='trend')
        axs[0, 0].plot(times, x_vibration + x_trend, 'g-', alpha=0.7, label='Vibration+Trend')
        axs[0, 0].set_title('X-direction position and vibration')
        axs[0, 0].set_xlabel('time (seconds)')
        axs[0, 0].set_ylabel('position (pixels)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # 時域振動圖 - Y 方向
        axs[0, 1].plot(times, y_pos, 'b-', alpha=0.5, label='raw data')
        axs[0, 1].plot(times, y_trend, 'r--', label='trend')
        axs[0, 1].plot(times, y_vibration + y_trend, 'g-', alpha=0.7, label='Vibration+Trend')
        axs[0, 1].set_title('Y -direction position and vibration')
        axs[0, 1].set_xlabel('time (seconds)')
        axs[0, 1].set_ylabel('position (pixels)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 頻域分析
        if fft_results:
            # X 方向頻譜
            axs[1, 0].plot(fft_results['freqs'], fft_results['x_fft_mag'], 'b-')
            axs[1, 0].axvline(x=fft_results['x_peak_freq'], color='r', linestyle='--',
                    label=f'Main frequency: {fft_results["x_peak_freq"]:.2f} Hz')
            axs[1, 0].set_title('X Directional frequency analysis')
            axs[1, 0].set_xlabel('frequency (Hz)')
            axs[1, 0].set_ylabel('amplitude')
            axs[1, 0].set_xlim(0, min(10, max(fft_results['freqs'])))
            axs[1, 0].legend()
            axs[1, 0].grid(True)
            
            # Y 方向頻譜
            axs[1, 1].plot(fft_results['freqs'], fft_results['y_fft_mag'], 'b-')
            axs[1, 1].axvline(x=fft_results['y_peak_freq'], color='r', linestyle='--',
                    label=f'Main frequency: {fft_results["y_peak_freq"]:.2f} Hz')
            axs[1, 1].set_title('Y Directional frequency analysis')
            axs[1, 1].set_xlabel('frequency (Hz)')
            axs[1, 1].set_ylabel('amplitude')
            axs[1, 1].set_xlim(0, min(10, max(fft_results['freqs'])))
            axs[1, 1].legend()
            axs[1, 1].grid(True)
        
        # 添加振幅和頻率資訊的文字標註
        amplitude_text = (f"X amplitude: {x_amplitude:.4f} px (std), {x_peak_to_peak:.4f} px (peak to peak)\n"
                        f"Y amplitude: {y_amplitude:.4f} px (std), {y_peak_to_peak:.4f} px (peak to peak)")
        
        frequency_text = ""
        if fft_results:
            frequency_text = (f"X Main frequency: {fft_results['x_peak_freq']:.4f} Hz (T: {1/fft_results['x_peak_freq']:.4f} sec)\n"
                            f"Y Main frequency: {fft_results['y_peak_freq']:.4f} Hz (T: {1/fft_results['y_peak_freq']:.4f} sec)")
        
        fig.text(0.1, 0.01, amplitude_text + "\n" + frequency_text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig
    
    def _perform_fft_analysis(self, times, x_vibration, y_vibration):
        """
        使用快速傅立葉變換 (FFT) 進行頻率分析
        
        參數:
        times (numpy.ndarray): 時間序列
        x_vibration (numpy.ndarray): X方向振動數據
        y_vibration (numpy.ndarray): Y方向振動數據
        
        返回:
        dict: FFT 分析結果
        """
        # 確保有足夠的數據點進行 FFT
        if len(times) < 8:
            print("警告: 數據點太少，無法進行有效的 FFT 分析")
            return None
        
        # 重新採樣到均勻時間間隔
        t_resampled = np.linspace(times[0], times[-1], len(times))
        x_resampled = np.interp(t_resampled, times, x_vibration)
        y_resampled = np.interp(t_resampled, times, y_vibration)
        
        # 計算採樣率 (Hz)
        sr = 1.0 / (t_resampled[1] - t_resampled[0])
        
        # 應用 Hanning 窗函數減少洩漏
        window = np.hanning(len(x_resampled))
        x_windowed = x_resampled * window
        y_windowed = y_resampled * window
        
        # 計算 FFT
        x_fft = fft(x_windowed)
        y_fft = fft(y_windowed)
        
        # 計算頻率軸
        n = len(x_resampled)
        freqs = fftfreq(n, 1/sr)
        
        # 只保留正頻率部分
        positive_freq_idx = np.arange(1, n // 2)
        freqs = freqs[positive_freq_idx]
        x_fft_mag = np.abs(x_fft[positive_freq_idx]) / n * 2  # 乘以2是因為只取了一半的頻譜
        y_fft_mag = np.abs(y_fft[positive_freq_idx]) / n * 2
        
        # 找出主頻率
        if len(freqs) > 0:
            x_peak_idx = np.argmax(x_fft_mag)
            y_peak_idx = np.argmax(y_fft_mag)
            
            x_peak_freq = freqs[x_peak_idx]
            y_peak_freq = freqs[y_peak_idx]
            
            x_peak_amp = x_fft_mag[x_peak_idx]
            y_peak_amp = y_fft_mag[y_peak_idx]
        else:
            x_peak_freq = y_peak_freq = x_peak_amp = y_peak_amp = 0
        
        return {
            'freqs': freqs,
            'x_fft_mag': x_fft_mag,
            'y_fft_mag': y_fft_mag,
            'x_peak_freq': x_peak_freq,
            'y_peak_freq': y_peak_freq,
            'x_peak_amp': x_peak_amp,
            'y_peak_amp': y_peak_amp
        }
        
    def save_vibration_analysis(self, figure=None, filename=None):
        """
        保存振動分析圖表
        
        參數:
        figure (matplotlib.figure.Figure): 圖表對象，如果為 None，則自動生成
        filename (str): 文件名，如果為 None，則自動生成
        
        返回:
        str: 保存的文件路徑
        """
        if figure is None:
            figure = self.analyze_and_plot_vibration()
            
        if figure is None:
            print("錯誤: 無法生成分析圖表")
            return None
        
        # 設定檔案名稱
        if filename is None:
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'vibration_analysis_{current_datetime}.png'
        
        # 確保有正確的副檔名
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            filename += '.png'
        
        # 完整的檔案路徑
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存圖表
        figure.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"振動分析圖表已保存至: {filepath}")
        
        # 同時保存一份摘要文本檔
        summary_path = filepath.replace('.png', '_summary.txt').replace('.jpg', '_summary.txt').replace('.jpeg', '_summary.txt').replace('.pdf', '_summary.txt')
        
        # 提取摘要數據
        samples = self.vibration_samples
        times = np.array([s['time'] for s in samples])
        x_pos = np.array([s['x'] for s in samples])
        y_pos = np.array([s['y'] for s in samples])
        
        # 計算振動（去趨勢）
        x_trend = signal.savgol_filter(x_pos, min(51, len(x_pos) // 4 * 2 + 1), 3)
        y_trend = signal.savgol_filter(y_pos, min(51, len(y_pos) // 4 * 2 + 1), 3)
        
        x_vibration = x_pos - x_trend
        y_vibration = y_pos - y_trend
        
        x_amplitude = np.std(x_vibration)
        y_amplitude = np.std(y_vibration)
        
        x_peak_to_peak = np.max(x_vibration) - np.min(x_vibration)
        y_peak_to_peak = np.max(y_vibration) - np.min(y_vibration)
        
        # 進行 FFT 分析
        fft_results = self._perform_fft_analysis(times, x_vibration, y_vibration)
        
        # 寫入摘要
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("圓環振動分析摘要\n")
            f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"數據點數: {len(samples)}\n")
            f.write(f"記錄時間: {times[-1] - times[0]:.2f} 秒\n")
            f.write(f"影片: {self.video_path}\n\n")
            
            f.write("========== 振幅統計 ==========\n")
            f.write(f"X方向振幅 (標準差): {x_amplitude:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({x_amplitude * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
                
            f.write(f"Y方向振幅 (標準差): {y_amplitude:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({y_amplitude * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
                
            f.write(f"X方向峰對峰值: {x_peak_to_peak:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({x_peak_to_peak * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
                
            f.write(f"Y方向峰對峰值: {y_peak_to_peak:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({y_peak_to_peak * self.pixel_to_cm:.6f} cm)\n\n")
            else:
                f.write("\n\n")
            
            if fft_results:
                f.write("========== 頻率分析 ==========\n")
                f.write(f"X方向主頻率: {fft_results['x_peak_freq']:.6f} Hz (週期: {1/fft_results['x_peak_freq']:.6f} 秒)\n")
                f.write(f"Y方向主頻率: {fft_results['y_peak_freq']:.6f} Hz (週期: {1/fft_results['y_peak_freq']:.6f} 秒)\n\n")
            
            f.write("========== 圓環尺寸統計 ==========\n")
            avg_inner = np.mean([s['inner_diameter'] for s in samples])
            avg_outer = np.mean([s['outer_diameter'] for s in samples])
            avg_thickness = np.mean([s['thickness'] for s in samples])
            
            f.write(f"平均內徑: {avg_inner:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({avg_inner * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
                
            f.write(f"平均外徑: {avg_outer:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({avg_outer * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
                
            f.write(f"平均厚度: {avg_thickness:.6f} 像素")
            if self.pixel_to_cm != 1.0:
                f.write(f" ({avg_thickness * self.pixel_to_cm:.6f} cm)\n")
            else:
                f.write("\n")
        
        print(f"振動分析摘要已保存至: {summary_path}")
        return filepath
    
    def analyze_vibration_modes(self):
        """
        分析振動模態，包括相位關係、軌跡形狀等
        
        返回:
        dict: 模態分析結果
        """
        if len(self.vibration_samples) < 20:
            print("錯誤: 沒有足夠的振動數據進行模態分析")
            return None
        
        # 提取數據
        times = np.array([sample['time'] for sample in self.vibration_samples])
        x_pos = np.array([sample['x'] for sample in self.vibration_samples])
        y_pos = np.array([sample['y'] for sample in self.vibration_samples])
        
        # 計算振動（去趨勢）
        x_trend = signal.savgol_filter(x_pos, min(51, len(x_pos) // 4 * 2 + 1), 3)
        y_trend = signal.savgol_filter(y_pos, min(51, len(y_pos) // 4 * 2 + 1), 3)
        
        x_vibration = x_pos - x_trend
        y_vibration = y_pos - y_trend
        
        # FFT 分析
        fft_results = self._perform_fft_analysis(times, x_vibration, y_vibration)
        if fft_results is None:
            return None
        
        # 主頻率及其振幅
        x_main_freq = fft_results['x_peak_freq']
        y_main_freq = fft_results['y_peak_freq']
        
        # 同頻率判定閾值
        freq_threshold = 0.05  # Hz
        
        # 振動模態判定
        if abs(x_main_freq - y_main_freq) < freq_threshold:
            # X和Y振動頻率相近，可能是單一模態
            # 計算相位差以判斷是線性、圓形還是橢圓軌跡
            
            # 在主頻率下重構信號
            t = np.linspace(0, 2*np.pi, len(times))
            
            # 擬合正弦波
            x_params, _ = curve_fit(lambda t, A, phi: A * np.sin(t + phi), 
                                t, x_vibration, p0=[np.std(x_vibration), 0])
            y_params, _ = curve_fit(lambda t, A, phi: A * np.sin(t + phi), 
                                t, y_vibration, p0=[np.std(y_vibration), 0])
            
            x_amplitude, x_phase = x_params
            y_amplitude, y_phase = y_params
            
            # 計算相位差
            phase_diff = abs(x_phase - y_phase) % (2*np.pi)
            if phase_diff > np.pi:
                phase_diff = 2*np.pi - phase_diff
            
            # 振幅比
            amplitude_ratio = max(x_amplitude, y_amplitude) / (min(x_amplitude, y_amplitude) + 1e-10)
            
            # 模態判定
            if phase_diff < 0.1 * np.pi or phase_diff > 0.9 * np.pi:
                # 近似線性振動
                mode = "線性振動"
                if phase_diff < 0.1 * np.pi:
                    orientation = np.arctan2(y_amplitude, x_amplitude)
                else:
                    orientation = np.arctan2(y_amplitude, -x_amplitude)
                
                orientation_deg = np.degrees(orientation) % 180
                detail = f"direction angle: {orientation_deg:.1f}°, amplitude ratio: {amplitude_ratio:.2f}"
            
            elif 0.4 * np.pi < phase_diff < 0.6 * np.pi:
                # 近似橢圓/圓形振動
                if 0.8 < amplitude_ratio < 1.2:
                    mode = "圓形振動"
                    detail = f"phase difference: {np.degrees(phase_diff):.1f}°,  amplitude ratio: {amplitude_ratio:.2f}"
                else:
                    mode = "橢圓振動"
                    semi_major = max(x_amplitude, y_amplitude)
                    semi_minor = min(x_amplitude, y_amplitude)
                    eccentricity = np.sqrt(1 - (semi_minor / semi_major) ** 2)
                    detail = f"Eccentricity: {eccentricity:.3f}, phase difference: {np.degrees(phase_diff):.1f}°"
            else:
                # 一般橢圓振動
                mode = "橢圓振動"
                detail = f"phase difference: {np.degrees(phase_diff):.1f}°, amplitude ratio: {amplitude_ratio:.2f}"
        else:
            # 頻率不同，可能是複合模態
            mode = "複合模態"
            detail = f"x_main_freq: {x_main_freq:.2f} Hz, y_main_freq: {y_main_freq:.2f} Hz"
        
        # 計算徑向和切向振動
        center_x = np.mean(x_trend)
        center_y = np.mean(y_trend)
        
        radial_vibration = []
        tangential_vibration = []
        
        for i in range(len(times)):
            # 計算位置向量
            dx = x_pos[i] - center_x
            dy = y_pos[i] - center_y
            r = np.sqrt(dx*dx + dy*dy)
            
            # 計算徑向單位向量
            if r > 1e-10:
                radial_x = dx / r
                radial_y = dy / r
            else:
                radial_x = 1.0
                radial_y = 0.0
            
            # 計算切向單位向量 (逆時針旋轉90度)
            tangential_x = -radial_y
            tangential_y = radial_x
            
            # 計算振動點的偏移向量
            vib_dx = x_vibration[i]
            vib_dy = y_vibration[i]
            
            # 投影到徑向和切向
            radial_component = vib_dx * radial_x + vib_dy * radial_y
            tangential_component = vib_dx * tangential_x + vib_dy * tangential_y
            
            radial_vibration.append(radial_component)
            tangential_vibration.append(tangential_component)
        
        radial_vibration = np.array(radial_vibration)
        tangential_vibration = np.array(tangential_vibration)
        
        # 徑向和切向振動的特性
        radial_amplitude = np.std(radial_vibration)
        tangential_amplitude = np.std(tangential_vibration)
        
        # 徑向/切向比
        rt_ratio = radial_amplitude / (tangential_amplitude + 1e-10)
        
        if rt_ratio > 2.0:
            direction = "主要徑向振動"
        elif rt_ratio < 0.5:
            direction = "主要切向振動"
        else:
            direction = "徑向和切向混合振動"
        
        # 返回分析結果
        return {
            'mode': mode,
            'detail': detail,
            'direction': direction,
            'rt_ratio': rt_ratio,
            'radial_amplitude': radial_amplitude,
            'tangential_amplitude': tangential_amplitude,
            'x_main_freq': x_main_freq,
            'y_main_freq': y_main_freq
        }
    
    def plot_vibration_trajectory(self):
        """
        繪製振動軌跡，使用自定義座標系
        
        返回:
        matplotlib.figure.Figure: 軌跡圖表
        """
        if len(self.vibration_samples) < 3:
            print("錯誤: 沒有足夠的振動數據繪製軌跡")
            return None
        
        # 提取數據 - 使用相對座標而非原始座標
        times = np.array([sample['time'] for sample in self.vibration_samples])
        
        # 使用相對於自定義原點的座標（向上為正）
        x_pos = np.array([sample['rel_x'] for sample in self.vibration_samples])
        y_pos = np.array([sample['rel_y'] for sample in self.vibration_samples])
        
        # 計算振動（去趨勢）
        x_trend = signal.savgol_filter(x_pos, min(51, len(x_pos) // 4 * 2 + 1), 3)
        y_trend = signal.savgol_filter(y_pos, min(51, len(y_pos) // 4 * 2 + 1), 3)
        
        x_vibration = x_pos - x_trend
        y_vibration = y_pos - y_trend
        
        # 創建圖表
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original trajectory
        axs[0].plot(x_pos, y_pos, 'b-', alpha=0.7, label='original trajectory')
        axs[0].plot(x_trend, y_trend, 'r--', label='average path')
        axs[0].set_title('Ring Movement Trajectory (Custom Origin)')
        axs[0].set_xlabel('X Position (pixels)')
        axs[0].set_ylabel('Y Position (pixels)')
        axs[0].legend()
        axs[0].axis('equal')
        axs[0].grid(True)

        # Plot vibration trajectory
        scatter = axs[1].scatter(x_vibration, y_vibration, c=times, cmap='viridis',
        alpha=0.7, s=10, label='vibration')
        axs[1].plot(x_vibration[0], y_vibration[0], 'ro', label='starting point')
        axs[1].set_title('Vibration Trajectory (Detrended)')
        axs[1].set_xlabel('X Vibration (pixels)')
        axs[1].set_ylabel('Y Vibration (pixels)')
        axs[1].legend()
        axs[1].axis('equal')
        axs[1].grid(True)

        # Add color bar to indicate time
        cbar = fig.colorbar(scatter, ax=axs[1])
        cbar.set_label('Time (seconds)')

        # Analyze vibration modes
        modes = self.analyze_vibration_modes()
        if modes:
            title = f"Ring Vibration Trajectory - {modes['mode']}"
            if 'detail' in modes:
                title += f"\n{modes['detail']}"
            fig.suptitle(title, fontsize=14)
            
            # Add additional information
            txt = f"Vibration direction: {modes['direction']}\n"
            txt += f"Radial/Tangential ratio: {modes['rt_ratio']:.2f}\n"
            txt += f"X main frequency: {modes['x_main_freq']:.2f} Hz, Y main frequency: {modes['y_main_freq']:.2f} Hz"
            fig.text(0.02, 0.01, txt, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5))
        else:
            fig.suptitle("Ring Vibration Trajectory (Custom Origin)", fontsize=14)
        plt.tight_layout()
        
        return fig

    def add_radial_vibration_sample(self, frame_number, time_sec, position, inner_diameter, outer_diameter, radial_points):
        """
        修改版的徑向振動樣本添加方法，確保角度一致性，使用類屬性中設定的點數，
        並添加自訂座標系套用比例尺的新座標
        """
        if self.is_recording:
            # 如果是第一次記錄，創建預定義的角度集合
            if not hasattr(self, 'predefined_angles'):
                # 使用類屬性的點數
                num_points = self.num_radial_points
                # 創建均勻分佈的角度
                angle_step = 360 / num_points
                self.predefined_angles = [i * angle_step for i in range(num_points)]
            
            # 將相對於 ROI 的座標轉換為絕對座標
            absolute_position = self.adjust_coordinates_for_roi(position, to_absolute=True) if self.use_roi else position
            
            # 如果這是第一個樣本，保存為參考樣本
            if len(self.vibration_samples) == 0:
                self.reference_position = absolute_position
                
                # 初始化參考徑向距離字典
                self.reference_radial_distances = {angle: 0 for angle in self.predefined_angles}
            
            # 計算相對於參考位置的偏移
            dx = absolute_position[0] - self.reference_position[0]
            dy = absolute_position[1] - self.reference_position[1]
            
            # 準備存儲徑向變形的列表
            adjusted_radial_deformations = []
            
            # 遍歷預定義的角度
            for angle in self.predefined_angles:
                # 在當前的徑向點中找最接近這個角度的點
                best_match = None
                min_angle_diff = float('inf')
                
                for point_info in radial_points:
                    point_angle = point_info.get('angle', 0)
                    angle_diff = abs((point_angle - angle + 360) % 360)
                    
                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_match = point_info
                
                # 如果找到匹配的點
                if best_match:
                    point_coords = best_match.get('point', (0, 0))
                    distance = best_match.get('distance', 0)
                    
                    # 獲取參考距離，如果沒有則使用當前距離
                    ref_distance = self.reference_radial_distances.get(angle, distance)
                    
                    # 計算變形
                    deformation = distance - ref_distance
                    
                    # 絕對坐標轉換為相對座標
                    absolute_point = self.adjust_coordinates_for_roi(point_coords, to_absolute=True) if self.use_roi else point_coords
                    adjusted_x = absolute_point[0] - self.origin_x
                    adjusted_y = self.origin_y - absolute_point[1]  # Y軸反轉
                    
                    adjusted_radial_deformations.append({
                        'angle': angle,
                        'distance': distance,
                        'deformation': deformation,
                        'point': absolute_point,  # 絕對屏幕坐標
                        'adjusted_point': (adjusted_x, adjusted_y)  # 相對座標
                    })
                else:
                    # 如果找不到匹配的點，使用默認值
                    adjusted_radial_deformations.append({
                        'angle': angle,
                        'distance': 0,
                        'deformation': 0,
                        'point': (0, 0),
                        'adjusted_point': (0, 0)
                    })
            
            # 確保按角度排序
            adjusted_radial_deformations.sort(key=lambda x: x['angle'])
            
            # 准备样本数据
            sample = {
                'frame': frame_number,
                'time': time_sec,
                'x': absolute_position[0],
                'y': absolute_position[1],
                'rel_x': absolute_position[0] - self.origin_x,
                'rel_y': self.origin_y - absolute_position[1],
                'dx': dx,
                'dy': dy,
                'inner_diameter': inner_diameter,
                'outer_diameter': outer_diameter,
                'thickness': (outer_diameter - inner_diameter) / 2,
                'radial_deformations': adjusted_radial_deformations
            }
            
            # 新增: 自訂座標系套用比例尺的新座標
            sample['custom_scaled_x'] = sample['rel_x'] * self.pixel_to_cm
            sample['custom_scaled_y'] = sample['rel_y'] * self.pixel_to_cm
            
            self.vibration_samples.append(sample)
            
            # 限制样本数量
            if len(self.vibration_samples) > self.max_vibration_samples:
                self.vibration_samples.pop(0)
            
            # 更新即时绘图数据
            self.update_vibration_plot_data(sample)

    def save_vibration_data(self, filename=None):
        """
        將徑向振動數據保存為 CSV 文件，將每個角度的資料轉為獨立欄位，並確保比例尺正確套用
        
        參數:
        filename (str): 文件名，如果為 None，則自動生成
        
        返回:
        str: 保存的文件路徑
        """
        if len(self.vibration_samples) < 1:
            print("錯誤: 沒有振動數據可保存")
            return None
        
        # 檢查是否有徑向變形數據
        has_radial_data = False
        for sample in self.vibration_samples:
            if 'radial_deformations' in sample and sample['radial_deformations']:
                has_radial_data = True
                break
        
        if not has_radial_data:
            print("錯誤: 沒有徑向變形數據可保存")
            return None
        
        # 獲取所有可能的角度
        all_angles = set()
        for sample in self.vibration_samples:
            if 'radial_deformations' in sample:
                for deform in sample['radial_deformations']:
                    all_angles.add(deform['angle'])
        angles = sorted(list(all_angles))
        
        # 準備數據列
        rows = []
        
        for sample in self.vibration_samples:
            # 創建基本行數據，包含標準座標系中的數據 (Y軸向上為正)
            row = {
                'frame': sample['frame'],
                'time': sample['time'],
                'x': sample['x'],
                'y': sample['y'],
                'rel_x': sample['rel_x'],  # 相對座標 (X)
                'rel_y': sample['rel_y'],  # 相對座標 (Y，向上為正)
                'dx': sample['dx'],
                'dy': sample['dy'],
                'inner_diameter': sample['inner_diameter'],
                'outer_diameter': sample['outer_diameter'],
                'thickness': sample['thickness']
            }
            
            # 新增: 自訂座標系套用比例尺的新座標
            row['custom_scaled_x'] = sample['rel_x'] * self.pixel_to_cm
            row['custom_scaled_y'] = sample['rel_y'] * self.pixel_to_cm
            
            # 如果有徑向變形數據，為每個角度創建單獨的欄位
            if 'radial_deformations' in sample and sample['radial_deformations']:
                # 創建角度到數據的映射
                angle_dict = {}
                for deform in sample['radial_deformations']:
                    angle_dict[deform['angle']] = deform
                
                # 為每個角度創建欄位
                for angle in angles:
                    if angle in angle_dict:
                        deform = angle_dict[angle]
                        # 添加變形值 
                        row[f'deform_{angle}'] = deform['deformation']
                for angle in angles:
                    if angle in angle_dict:
                        deform = angle_dict[angle]
                        #添加距離值
                        row[f'distance_{angle}'] = deform['distance']
                for angle in angles:
                    if angle in angle_dict:
                        deform = angle_dict[angle]
                        #添加距離值
                        row[f'point_x_{angle}'] = deform['point'][0]
                        row[f'point_y_{angle}'] = deform['point'][1]
                for angle in angles:
                    if angle in angle_dict:
                        deform = angle_dict[angle]
                        #添加距離值
                        if 'adjusted_point' in deform:
                            row[f'adj_point_x_{angle}'] = deform['adjusted_point'][0]
                            row[f'adj_point_y_{angle}'] = deform['adjusted_point'][1]
                for angle in angles:
                    if angle in angle_dict:
                        deform = angle_dict[angle]
                        #添加距離值
                        row[f'point_x_{angle}'] = deform['point'][0]
                        row[f'point_y_{angle}'] = deform['point'][1]
            
            rows.append(row)
        
        # 創建DataFrame
        df = pd.DataFrame(rows)
        
        # 添加厘米單位的資料(如果有校準)
        if self.pixel_to_cm != 1.0:
            # 確保所有基本位置和尺寸數據都有厘米單位版本
            df['x_cm'] = df['x'] * self.pixel_to_cm
            df['y_cm'] = df['y'] * self.pixel_to_cm
            df['rel_x_cm'] = df['rel_x'] * self.pixel_to_cm
            df['rel_y_cm'] = df['rel_y'] * self.pixel_to_cm
            df['dx_cm'] = df['dx'] * self.pixel_to_cm
            df['dy_cm'] = df['dy'] * self.pixel_to_cm
            df['inner_diameter_cm'] = df['inner_diameter'] * self.pixel_to_cm
            df['outer_diameter_cm'] = df['outer_diameter'] * self.pixel_to_cm
            df['thickness_cm'] = df['thickness'] * self.pixel_to_cm
            
            # 轉換每個角度的徑向資料到厘米
            for angle in angles:
                # 變形和距離資料
                deform_col = f'deform_{angle}'
                dist_col = f'distance_{angle}'
                
                if deform_col in df.columns:
                    df[f'{deform_col}_cm'] = df[deform_col] * self.pixel_to_cm
                
                if dist_col in df.columns:
                    df[f'{dist_col}_cm'] = df[dist_col] * self.pixel_to_cm
                
                # 點座標資料 (原始和調整後的)
                point_x_col = f'point_x_{angle}'
                point_y_col = f'point_y_{angle}'
                adj_x_col = f'adj_point_x_{angle}'
                adj_y_col = f'adj_point_y_{angle}'
                
                if point_x_col in df.columns:
                    df[f'{point_x_col}_cm'] = df[point_x_col] * self.pixel_to_cm
                
                if point_y_col in df.columns:
                    df[f'{point_y_col}_cm'] = df[point_y_col] * self.pixel_to_cm
                
                if adj_x_col in df.columns:
                    df[f'{adj_x_col}_cm'] = df[adj_x_col] * self.pixel_to_cm
                
                if adj_y_col in df.columns:
                    df[f'{adj_y_col}_cm'] = df[adj_y_col] * self.pixel_to_cm
        
        # 設定檔案名稱
        if filename is None:
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'radial_vibration_data_{current_datetime}.csv'
        
        # 確保有正確的副檔名
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # 完整的檔案路徑
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存 CSV
        df.to_csv(filepath, index=False)
        print(f"徑向振動數據已保存至: {filepath}")
        
        return filepath

    def analyze_radial_vibration(self):
        """
        分析徑向振動
        
        返回:
        dict: 徑向振動分析結果
        """
        if len(self.vibration_samples) < 10:
            print("錯誤: 沒有足夠的振動數據進行分析")
            return None
        
        # 初始化各角度的徑向振動數據
        angles = set()
        for sample in self.vibration_samples:
            if 'radial_deformations' in sample:
                for deformation in sample['radial_deformations']:
                    angles.add(deformation['angle'])
        
        angles = sorted(list(angles))
        angle_data = {angle: [] for angle in angles}
        times = []
        
        # 提取各角度的徑向變形數據
        for sample in self.vibration_samples:
            times.append(sample['time'])
            
            if 'radial_deformations' in sample:
                # 初始化此時間點的所有角度變形為None
                frame_deformations = {angle: None for angle in angles}
                
                # 填入有的數據
                for deformation in sample['radial_deformations']:
                    angle = deformation['angle']
                    frame_deformations[angle] = deformation['deformation']
                
                # 收集數據
                for angle in angles:
                    angle_data[angle].append(frame_deformations[angle])
        
        # 分析各角度的徑向振動
        results = {}
        for angle in angles:
            # 去除None值
            data = [d for d in angle_data[angle] if d is not None]
            if len(data) < 10:
                continue
            
            # 計算統計數據
            mean = np.mean(data)
            std = np.std(data)
            max_val = np.max(data)
            min_val = np.min(data)
            peak_to_peak = max_val - min_val
            
            # FFT分析
            if len(data) >= 16:  # 至少需要16個點做FFT
                # 重採樣到等間隔時間
                t_resampled = np.linspace(times[0], times[-1], len(data))
                try:
                    # 去趨勢
                    data_detrended = signal.detrend(data)
                    
                    # 應用窗函數
                    windowed = data_detrended * signal.windows.hann(len(data_detrended))
                    
                    # 計算FFT
                    fft_result = fft(windowed)
                    n = len(data)
                    sr = 1.0 / (t_resampled[1] - t_resampled[0])
                    freqs = fftfreq(n, 1/sr)
                    
                    # 只保留正頻率部分
                    positive_freq_idx = np.arange(1, n // 2)
                    freqs = freqs[positive_freq_idx]
                    fft_mag = np.abs(fft_result[positive_freq_idx]) / n * 2
                    
                    # 找出主頻率
                    if len(fft_mag) > 0:
                        peak_idx = np.argmax(fft_mag)
                        peak_freq = freqs[peak_idx]
                        peak_amp = fft_mag[peak_idx]
                    else:
                        peak_freq = 0
                        peak_amp = 0
                except:
                    peak_freq = 0
                    peak_amp = 0
            else:
                peak_freq = 0
                peak_amp = 0
            
            # 儲存分析結果
            results[angle] = {
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val,
                'peak_to_peak': peak_to_peak,
                'dominant_frequency': peak_freq,
                'frequency_amplitude': peak_amp
            }
        
        return {
            'times': times,
            'angle_data': angle_data,
            'analysis': results
        }

    def plot_radial_vibration(self, analysis_result=None):
        """
        繪製徑向振動的分析圖表
        
        參數:
        analysis_result (dict): 徑向振動分析結果，如果為None則自動生成
        
        返回:
        matplotlib.figure.Figure: 分析圖表
        """
        if analysis_result is None:
            analysis_result = self.analyze_radial_vibration()
        
        if analysis_result is None:
            print("無法生成徑向振動分析")
            return None
        
        # 創建極座標圖
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('環形結構徑向振動分析', fontsize=16)
        
        # 1. 極座標圖顯示各角度的振幅
        ax1 = fig.add_subplot(2, 2, 1, projection='polar')
        
        angles = sorted(analysis_result['analysis'].keys())
        amplitudes = [analysis_result['analysis'][angle]['std'] for angle in angles]
        peak_to_peaks = [analysis_result['analysis'][angle]['peak_to_peak'] for angle in angles]
        
        # 轉換為弧度
        angles_rad = [angle * np.pi / 180 for angle in angles]
        
        # Plot standard deviation (amplitude)
        ax1.plot(angles_rad, amplitudes, 'b-', label='Amplitude (Standard Deviation)')
        # Plot peak-to-peak values
        ax1.plot(angles_rad, peak_to_peaks, 'r-', label='Peak-to-Peak Value')
        # Fill polar plot
        ax1.fill(angles_rad, amplitudes, 'b', alpha=0.1)
        ax1.set_title('Vibration Amplitude Distribution by Angle')
        ax1.set_theta_zero_location('N')  # North is 0 degrees
        ax1.set_theta_direction(-1)  # Clockwise direction
        ax1.set_rlabel_position(0)  # Position of scale labels
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 2. Polar plot showing main frequency at each angle
        ax2 = fig.add_subplot(2, 2, 2, projection='polar')
        frequencies = [analysis_result['analysis'][angle]['dominant_frequency'] for angle in angles]
        freq_amplitudes = [analysis_result['analysis'][angle]['frequency_amplitude'] for angle in angles]
        # Main frequency
        ax2.plot(angles_rad, frequencies, 'g-', label='Main Frequency (Hz)')
        ax2.fill(angles_rad, frequencies, 'g', alpha=0.1)
        ax2.set_title('Main Frequency Distribution by Angle')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rlabel_position(0)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 3. Time series overlay of vibrations at all angles
        ax3 = fig.add_subplot(2, 1, 2)
        times = analysis_result['times']
        angle_data = analysis_result['angle_data']
        # Select some representative angles
        if len(angles) > 6:
            selected_angles = angles[::len(angles)//6][:6]  # Select about 6 angles
        else:
            selected_angles = angles
        for angle in selected_angles:
            data = angle_data[angle]
            # Filter out None values
            valid_indices = [i for i, val in enumerate(data) if val is not None]
            valid_times = [times[i] for i in valid_indices]
            valid_data = [data[i] for i in valid_indices]
            if len(valid_data) > 1:
                # Plot vibration at this angle
                ax3.plot(valid_times, valid_data, label=f'{angle}°')
        ax3.set_title('Radial Vibration Time Series for Representative Angles')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Radial Deformation (pixels)')
        ax3.grid(True)
        ax3.legend()

        # Add overall summary
        mean_amplitude = np.mean(amplitudes)
        max_amplitude = np.max(amplitudes)
        mean_frequency = np.mean([f for f in frequencies if f > 0])
        summary_text = (f"Average Amplitude: {mean_amplitude:.4f} pixels\n"
                        f"Maximum Amplitude: {max_amplitude:.4f} pixels\n")
        if not np.isnan(mean_frequency):
            summary_text += f"Average Main Frequency: {mean_frequency:.4f} Hz (Period: {1/mean_frequency:.4f} seconds)\n"
        # Find the angle with maximum amplitude
        max_amp_angle = angles[np.argmax(amplitudes)]
        summary_text += f"Angle with Maximum Amplitude: {max_amp_angle}°"
        fig.text(0.02, 0.01, summary_text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig

    def toggle_tracking_mode(self, event=None):
        """
        切换是否只显示影片和追踪的圆环
        可通过按下'l'键触发
        """
        self.tracking_mode_only = not self.tracking_mode_only
        
        # 在控制台输出当前模式以供参考
        mode_name = "仅追踪模式" if self.tracking_mode_only else "完整分析模式"
        print(f"切换至: {mode_name}")

    def show_tracking_window(self, original_frame, ring_position=None, ring_contour=None, highspeed_mode=False, lowest_point=None):
        """
        增强版的追踪状态窗口显示，添加最低点标记
        
        参数:
        original_frame (numpy.ndarray): 原始帧
        ring_position (tuple): 圆环中心位置 (x, y)
        ring_contour (numpy.ndarray): 圆环轮廓
        highspeed_mode (bool): 是否处于高速追踪模式
        lowest_point (tuple): 最低点坐标 (x, y)
        """
        # 设定固定窗口大小
        window_width = 1024  # 设定窗口固定宽度
        window_height = 768  # 设定窗口固定高度
        
        # 创建显示用的帧
        display_frame = original_frame.copy()
        
        # 检查是否启用了ROI裁剪
        roi_active = self.use_roi and self.roi is not None and len(self.roi) == 4
        roi_offset_x = 0
        roi_offset_y = 0
        
        if roi_active:
            # 获取ROI坐标
            x1, y1, x2, y2 = self.roi
            roi_offset_x = x1
            roi_offset_y = y1
            
            # 在原始帧上标记ROI区域
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, "ROI", (x1+5, y1+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 如果有检测到圆环，绘制轮廓和中心点
        if ring_position is not None and ring_contour is not None:
            # 如果使用ROI，需要调整坐标
            if roi_active:
                # 调整轮廓坐标
                adjusted_contour = ring_contour.copy()
                adjusted_contour[:, :, 0] += roi_offset_x
                adjusted_contour[:, :, 1] += roi_offset_y
                
                # 调整圆心坐标
                adjusted_position = (int(ring_position[0] + roi_offset_x), int(ring_position[1] + roi_offset_y))
                
                # 绘制轮廓 - 在高速模式下使用不同颜色
                contour_color = (0, 255, 255) if highspeed_mode else (0, 255, 0)  # 高速模式黄色，正常模式绿色
                cv2.drawContours(display_frame, [adjusted_contour], 0, contour_color, 1)
                
                # 绘制中心点
                cv2.circle(display_frame, adjusted_position, 1, (0, 0, 255), -1)
                
                # 绘制最低点（如果有）
                if lowest_point is not None and None not in lowest_point:
                    lowest_x, lowest_y = lowest_point
                    # 调整最低点坐标
                    adjusted_lowest = (int(lowest_x + roi_offset_x), int(lowest_y + roi_offset_y))
                    # 绘制最低点
                    cv2.circle(display_frame, adjusted_lowest, 1, (0, 255, 255), -1)  # 黄色点
                    cv2.putText(display_frame, "Lowest", (adjusted_lowest[0] + 5, adjusted_lowest[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 高速模式下添加速度向量预测（如果有速度信息）
                if hasattr(self, 'velocity_estimate') and highspeed_mode:
                    # 利用实例变量中的速度信息
                    vx, vy = self.velocity_estimate
                    # 绘制速度向量 (放大5倍以便于观察)
                    end_x = int(adjusted_position[0] + vx * 5)
                    end_y = int(adjusted_position[1] + vy * 5)
                    cv2.arrowedLine(display_frame, adjusted_position, (end_x, end_y), (255, 0, 0), 2)
            else:
                # 无ROI的情况，直接绘制
                contour_color = (0, 255, 255) if highspeed_mode else (0, 255, 0)
                cv2.drawContours(display_frame, [ring_contour], 0, contour_color, 1)
                cv2.circle(display_frame, (int(ring_position[0]), int(ring_position[1])), 3, (0, 0, 255), -1)
                
                # 绘制最低点（如果有）
                if lowest_point is not None and None not in lowest_point:
                    lowest_x, lowest_y = lowest_point
                    cv2.circle(display_frame, (int(lowest_x), int(lowest_y)), 5, (0, 255, 255), -1)  # 黄色点
                    cv2.putText(display_frame, "Lowest", (int(lowest_x) + 5, int(lowest_y) + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 高速模式下添加速度向量预测
                if hasattr(self, 'velocity_estimate') and highspeed_mode:
                    vx, vy = self.velocity_estimate
                    end_x = int(ring_position[0] + vx * 5)
                    end_y = int(ring_position[1] + vy * 5)
                    cv2.arrowedLine(display_frame, 
                                (int(ring_position[0]), int(ring_position[1])), 
                                (end_x, end_y), 
                                (255, 0, 0), 2)
            
            # 添加追踪状态信息
            status_text = "High speed tracking" if highspeed_mode else "Tracking"
            status_color = (0, 255, 255) if highspeed_mode else (0, 255, 0)
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # 如果有速度估计，显示速度
            if hasattr(self, 'velocity_estimate'):
                vx, vy = self.velocity_estimate
                speed = np.sqrt(vx*vx + vy*vy)
                cv2.putText(display_frame, f"Speed: {speed:.1f} px/frame", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示最低点信息（如果有）
            if lowest_point is not None and None not in lowest_point:
                lowest_x, lowest_y = lowest_point
                if hasattr(self, 'reference_lowest_point') and self.reference_lowest_point is not None:
                    ref_x, ref_y = self.reference_lowest_point
                    dx = lowest_x - ref_x
                    dy = lowest_y - ref_y
                    
                    # 显示最低点位置和偏移
                    lowest_info = f"lowest point coordinates: ({lowest_x}, {lowest_y})"
                    offset_info = f"lowest point offset: dx={dx:.1f}, dy={dy:.1f}"
                    
                    cv2.putText(display_frame, lowest_info, (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, offset_info, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # 追踪失败时显示提示
            cv2.putText(display_frame, "Ring not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示ROI状态
        if roi_active:
            cv2.putText(display_frame, "ROI start", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 计算影片与窗口的尺寸比例
        h, w = display_frame.shape[:2]
        
        # 缩放系数 - 可以调整这个值来改变影片在窗口中的大小
        scale_factor = 0.9  # 例如，0.9 表示影片会填充窗口的 90%
        
        # 计算保持原始比例下的显示尺寸
        scale = min(window_width / w, window_height / h) * scale_factor
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # 创建黑色背景画布
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # 计算置中位置
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2
        
        # 调整影片大小
        resized_frame = cv2.resize(display_frame, (new_width, new_height))
        
        # 将调整后的影片放在画布中心
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        # 创建固定大小的窗口
        cv2.namedWindow('Tracking Status', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracking Status', window_width, window_height)
        
        # 显示结果
        cv2.imshow('Tracking Status', canvas)

    def save_lowest_point_data(self, filename=None):
        """
        将最低点数据保存为CSV文件，确保比例尺正确应用
        
        参数:
        filename (str): 文件名，如果为None则自动生成
        
        返回:
        str: 保存的文件路径
        """
        if not hasattr(self, 'lowest_point_data') or len(self.lowest_point_data['frame']) == 0:
            print("错误: 没有最低点数据可保存")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame({
            'frame': self.lowest_point_data['frame'],
            'time': self.lowest_point_data['time'],
            'x': self.lowest_point_data['x'],
            'y': self.lowest_point_data['y'],
            'rel_x': self.lowest_point_data['rel_x'],  # 相对坐标
            'rel_y': self.lowest_point_data['rel_y'],  # 相对坐标（Y轴向上为正）
            'dx': self.lowest_point_data['dx'],
            'dy': self.lowest_point_data['dy']
        })
        
        # 如果需要，将像素转换为厘米
        if self.pixel_to_cm != 1.0:
            # 确保所有位置数据都有厘米单位版本
            df['x_cm'] = df['x'] * self.pixel_to_cm
            df['y_cm'] = df['y'] * self.pixel_to_cm
            df['rel_x_cm'] = df['rel_x'] * self.pixel_to_cm
            df['rel_y_cm'] = df['rel_y'] * self.pixel_to_cm
            df['dx_cm'] = df['dx'] * self.pixel_to_cm
            df['dy_cm'] = df['dy'] * self.pixel_to_cm
        
        # 设定文件名称
        if filename is None:
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lowest_point_data_{current_datetime}.csv'
        
        # 确保有正确的扩展名
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # 完整的文件路径
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存CSV
        df.to_csv(filepath, index=False)
        print(f"最低点数据已保存至: {filepath}")
        return filepath
    
    def refine_contour_for_thin_rings(self, frame, contour, center):
        """
        對細圓環輪廓進行精細化處理
        
        參數:
        frame (numpy.ndarray): 原始圖像幀
        contour (numpy.ndarray): 原始輪廓
        center (tuple): 中心點坐標
        
        返回:
        numpy.ndarray: 優化後的輪廓
        """
        try:
            # 轉為灰度圖像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 創建輪廓掩碼
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # 對掩碼區域應用Canny邊緣檢測
            edges = cv2.Canny(gray, 50, 150)
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            # 在邊緣上查找輪廓
            refined_contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(refined_contours) > 0:
                # 選擇最大的輪廓
                largest_contour = max(refined_contours, key=cv2.contourArea)
                
                # 確認輪廓與原始輪廓的中心大致相同
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    refined_center = (cx, cy)
                    
                    # 計算中心點距離
                    distance = np.sqrt((refined_center[0] - center[0])**2 + (refined_center[1] - center[1])**2)
                    
                    # 如果中心點接近，返回優化後的輪廓
                    if distance < 20:  # 可以調整此閾值
                        return largest_contour
            
            # 如果沒有找到合適的優化輪廓，返回原始輪廓
            return contour
        
        except Exception as e:
            if self.debug:
                print(f"輪廓精細化處理時出錯: {e}")
            return contour
    
    def determine_ring_type(self, frame):
        """
        自動判斷環的類型（粗環或細環）
        
        參數:
        frame (numpy.ndarray): 當前幀
        
        返回:
        str: 'thin' 表示細環，'normal' 表示普通環
        """
        # 轉為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用基本的檢測方法獲取潛在的輪廓
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 篩選有效的環形輪廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.detection_area_min:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if 0.4 < circularity < 1.0:
                    valid_contours.append({
                        'contour': contour,
                        'area': area
                    })
        
        # 如果找不到有效輪廓，默認為普通環
        if len(valid_contours) < 2:
            return 'normal'
        
        # 按面積排序
        sorted_contours = sorted(valid_contours, key=lambda x: x['area'])
        if len(sorted_contours) > 0:
            best_contour = sorted_contours[-1]  # 最大的輪廓
        
        # 如果最大和最小輪廓的面積比例較小，可能是細環
        if len(sorted_contours) >= 2:
            smallest = sorted_contours[0]['area']
            largest = sorted_contours[-1]['area']
            area_ratio = largest / smallest
            
            # 細環的特徵是內外環面積比例較小，且多個輪廓的面積接近
            if area_ratio < 3.0 and len(sorted_contours) > 3:
                # 還可以分析輪廓之間的面積差異：細環通常有多個漸層，面積差異較小
                areas = [c['area'] for c in sorted_contours]
                area_diffs = [areas[i+1] - areas[i] for i in range(len(areas)-1)]
                avg_diff_ratio = np.mean(area_diffs) / areas[-1] if areas[-1] > 0 else 1.0
                
                if avg_diff_ratio < 0.3:  # 如果平均差異較小，可能是細環
                    return 'thin'
        
        return 'normal'  # 默認為普通環
    
    def calculate_radial_points_for_thin_rings(self, frame, center, contour, num_points=36):
        """
        為細環計算徑向點 - 透過計算射線與輪廓線段的交點
        
        參數:
        frame (numpy.ndarray): 原始圖像幀
        center (tuple): 中心點坐標 (x, y)
        contour (numpy.ndarray): 輪廓點集
        num_points (int): 要採樣的徑向點數量，預設為36
        
        返回:
        list: 徑向點列表
        """
        if num_points is None:
            num_points = self.num_radial_points
        if contour is None or len(contour) < 5:
            return []
            
        center_x, center_y = center
        radial_points = []
        
        # 將輪廓轉換為線段
        contour_points = contour.reshape(-1, 2)
        segments = []
        for i in range(len(contour_points)):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % len(contour_points)]
            segments.append((p1, p2))
        
        # 對每個角度計算射線與線段的交點
        for i in range(num_points):
            angle_deg = i * (360 / num_points)
            angle_rad = np.deg2rad(angle_deg)
            
            # 計算射線方向
            ray_dx = np.cos(angle_rad)
            ray_dy = np.sin(angle_rad)
            
            # 射線的遠點 (確保足夠遠以覆蓋整個輪廓)
            far_distance = max(frame.shape) * 2
            far_x = center_x + ray_dx * far_distance
            far_y = center_y + ray_dy * far_distance
            
            # 找出射線與所有線段的交點
            intersections = []
            for segment in segments:
                s1, s2 = segment
                
                # 線段兩端點
                x1, y1 = s1
                x2, y2 = s2
                
                # 計算射線和線段的交點
                # 使用參數方程： 
                # 射線: p = (center_x, center_y) + t1 * (ray_dx, ray_dy), t1 >= 0
                # 線段: q = (x1, y1) + t2 * ((x2-x1), (y2-y1)), 0 <= t2 <= 1
                
                denom = (y2 - y1) * ray_dx - (x2 - x1) * ray_dy
                
                # 如果分母為0，射線與線段平行
                if abs(denom) < 1e-10:
                    continue
                    
                ua = ((x2 - x1) * (center_y - y1) - (y2 - y1) * (center_x - x1)) / denom
                ub = (ray_dx * (center_y - y1) - ray_dy * (center_x - x1)) / denom
                
                # 檢查交點是否在線段上且射線的正方向
                if ua >= 0 and 0 <= ub <= 1:
                    # 計算交點坐標
                    intersect_x = center_x + ua * ray_dx
                    intersect_y = center_y + ua * ray_dy
                    
                    # 計算交點到中心的距離
                    dist = np.sqrt((intersect_x - center_x)**2 + (intersect_y - center_y)**2)
                    
                    intersections.append({
                        'point': (int(intersect_x), int(intersect_y)),
                        'distance': dist
                    })
            
            # 如果找到交點，選擇最近的一個
            if intersections:
                # 按距離排序
                intersections.sort(key=lambda x: x['distance'])
                closest = intersections[0]
                
                radial_points.append({
                    'point': closest['point'],
                    'angle': angle_deg,
                    'distance': closest['distance']
                })
            else:
                # 如果沒有找到交點（例如，輪廓有間隔），使用估計值
                # 使用平均半徑，如果有的話
                avg_radius = 30  # 預設值
                if len(radial_points) > 0:
                    avg_radius = np.mean([p['distance'] for p in radial_points])
                    
                est_x = int(center_x + avg_radius * ray_dx)
                est_y = int(center_y + avg_radius * ray_dy)
                
                radial_points.append({
                    'point': (est_x, est_y),
                    'angle': angle_deg,
                    'distance': avg_radius,
                    'estimated': True  # 標記為估計點
                })
        
        return radial_points
    
    def update_lowest_point_data(self, frame_number, time_sec, lowest_point):
        """
        更新最低點數據，使用絕對座標系統中的自定義原點
        
        參數:
        frame_number (int): 帧编号
        time_sec (float): 时间（秒）
        lowest_point (tuple): 最低点坐标 (x, y)，相對於 ROI 的座標
        """
        if lowest_point is None or None in lowest_point:
            return
        
        # 將相對於 ROI 的座標轉換為絕對座標
        absolute_lowest = lowest_point
        if self.use_roi and self.roi and len(self.roi) == 4:
            x1, y1, x2, y2 = self.roi
            absolute_lowest = (lowest_point[0] + x1, lowest_point[1] + y1)
        
        lowest_x, lowest_y = absolute_lowest
        
        # 如果这是第一个样本，保存为参考点
        if not hasattr(self, 'reference_lowest_point') or self.reference_lowest_point is None:
            self.reference_lowest_point = absolute_lowest
        
        # 计算相对于参考点的偏移
        dx = lowest_x - self.reference_lowest_point[0]
        dy = lowest_y - self.reference_lowest_point[1]
        
        # 確保自定義座標原點被初始化，若未設置則使用預設值 (0, 0)
        if not hasattr(self, 'origin_x') or not hasattr(self, 'origin_y'):
            self.origin_x = 0
            self.origin_y = 0
            self.use_custom_origin = True
        
        # 計算相對於自定義原點的座標
        rel_x = lowest_x - self.origin_x
        rel_y = self.origin_y - lowest_y  # Y轴反转，向上为正
        
        # 确保lowest_point_data存在
        if not hasattr(self, 'lowest_point_data'):
            self.lowest_point_data = {
                'frame': [],
                'time': [],
                'x': [],
                'y': [],
                'rel_x': [],  # 相对于自定义原点的X坐标
                'rel_y': [],  # 相对于自定义原点的Y坐标（向上为正）
                'dx': [],
                'dy': []
            }
        
        # 添加数据
        self.lowest_point_data['frame'].append(frame_number)
        self.lowest_point_data['time'].append(time_sec)
        self.lowest_point_data['x'].append(lowest_x)
        self.lowest_point_data['y'].append(lowest_y)
        self.lowest_point_data['rel_x'].append(rel_x)
        self.lowest_point_data['rel_y'].append(rel_y)
        self.lowest_point_data['dx'].append(dx)
        self.lowest_point_data['dy'].append(dy)
    
    def track_lowest_point(self, contour, frame=None):
        """
        追踪轮廓中的最低点（Y坐标最大的点）
        
        参数:
        contour (numpy.ndarray): 轮廓点集
        frame (numpy.ndarray): 原始帧，用于可视化（可选）
        
        返回:
        tuple: (lowest_point_x, lowest_point_y)
        """
        if contour is None or len(contour) == 0:
            return None, None
        
        try:
            # 寻找轮廓中Y坐标最大的点（即最低点）
            lowest_point_idx = 0
            max_y = contour[0][0][1]  # 初始化为第一个点的Y坐标
            
            for i, point in enumerate(contour):
                y = point[0][1]
                if y > max_y:
                    max_y = y
                    lowest_point_idx = i
            
            # 获取最低点的坐标
            lowest_point = contour[lowest_point_idx][0]
            lowest_point_x, lowest_point_y = lowest_point
            
            # 如果提供了帧，在帧上绘制最低点
            if frame is not None:
                cv2.circle(frame, (lowest_point_x, lowest_point_y), 5, (0, 255, 255), -1)  # 黄色圆点
                cv2.putText(frame, "Lowest", (lowest_point_x + 5, lowest_point_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return lowest_point_x, lowest_point_y
        
        except Exception as e:
            print(f"追踪最低点时出错: {e}")
            return None, None

    def show_radial_analysis_window(self, original_frame, ring_position=None, contour=None, radial_points=None):
        """
        显示径向振动分析窗口，绘制当前检测到的径向点和轮廓
        
        参数:
        original_frame (numpy.ndarray): 原始视频帧
        ring_position (tuple): 圆环中心位置 (x, y)
        contour (numpy.ndarray): 圆环轮廓
        radial_points (list): 径向采样点列表
        """
        # 设定固定窗口大小
        window_width = 1024
        window_height = 768
        
        # 创建显示用的帧
        display_frame = original_frame.copy()
        
        # 检查是否启用了ROI裁剪
        roi_active = self.use_roi and self.roi is not None and len(self.roi) == 4
        roi_offset_x = 0
        roi_offset_y = 0
        
        if roi_active:
            # 获取ROI坐标
            x1, y1, x2, y2 = self.roi
            roi_offset_x = x1
            roi_offset_y = y1
            
            # 在原始帧上标记ROI区域
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, "ROI", (x1+5, y1+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 如果有检测到圆环，绘制轮廓和中心点
        if ring_position is not None:
            # 如果使用ROI，需要调整坐标
            if roi_active:
                # 调整圆心坐标
                adjusted_position = (int(ring_position[0] + roi_offset_x), int(ring_position[1] + roi_offset_y))
                
                # 绘制中心点
                cv2.circle(display_frame, adjusted_position, 3, (0, 0, 255), -1)
                
                # 如果有轮廓，绘制轮廓
                if contour is not None:
                    # 调整轮廓坐标
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += roi_offset_x
                    adjusted_contour[:, :, 1] += roi_offset_y
                    
                    # 绘制轮廓
                    cv2.drawContours(display_frame, [adjusted_contour], 0, (0, 255, 0), 1)
                
                # 如果有径向点，绘制径向点和从中心到径向点的线
                if radial_points and len(radial_points) > 0:
                    for point_info in radial_points:
                        if 'point' in point_info:
                            # 获取点坐标并调整
                            px, py = point_info['point']
                            adjusted_point = (int(px + roi_offset_x), int(py + roi_offset_y))
                            
                            # 获取角度和距离信息
                            angle = point_info.get('angle', 0)
                            distance = point_info.get('distance', 0)
                            
                            # 绘制径向点
                            cv2.circle(display_frame, adjusted_point, 1, (0, 255, 255), -1)
                            
                            # 绘制从中心到径向点的线
                            # cv2.line(display_frame, adjusted_position, adjusted_point, (0, 200, 200), 1)
                            
                            # 可选：在点旁边显示角度
                            # if angle % 45 == 0:  # 仅在特定角度显示数字，避免过于拥挤
                            #     cv2.putText(display_frame, f"{int(angle)}°", 
                            #             (adjusted_point[0] + 5, adjusted_point[1] + 5), 
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                # 无ROI的情况，直接绘制
                cv2.circle(display_frame, (int(ring_position[0]), int(ring_position[1])), 3, (0, 0, 255), -1)
                
                if contour is not None:
                    cv2.drawContours(display_frame, [contour], 0, (0, 255, 0), 2)
                    
                if radial_points and len(radial_points) > 0:
                    for point_info in radial_points:
                        if 'point' in point_info:
                            px, py = point_info['point']
                            angle = point_info.get('angle', 0)
                            distance = point_info.get('distance', 0)
                            
                            cv2.circle(display_frame, (int(px), int(py)), 2, (0, 255, 255), -1)
                            cv2.line(display_frame, 
                                    (int(ring_position[0]), int(ring_position[1])), 
                                    (int(px), int(py)), 
                                    (0, 200, 200), 1)
                            
                            if angle % 45 == 0:
                                cv2.putText(display_frame, f"{int(angle)}°", 
                                        (int(px) + 5, int(py) + 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 在顶部显示信息
        info_text = "Radial Vibration Analysis - "
        if radial_points:
            info_text += f"Detected {len(radial_points)} radial points"
        else:
            info_text += "No radial points detected"
        
        cv2.putText(display_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 计算视频与窗口的尺寸比例
        h, w = display_frame.shape[:2]
        
        # 缩放系数 - 可以调整这个值来改变视频在窗口中的大小
        scale_factor = 0.9  # 例如，0.9 表示视频会填充窗口的 90%
        
        # 计算保持原始比例下的显示尺寸
        scale = min(window_width / w, window_height / h) * scale_factor
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # 创建黑色背景画布
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # 计算置中位置
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2
        
        # 调整视频大小
        resized_frame = cv2.resize(display_frame, (new_width, new_height))
        
        # 将调整后的视频放在画布中心
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        # 显示径向分析数据（如果有）
        if self.is_recording and hasattr(self, 'vibration_samples') and len(self.vibration_samples) > 0:
            # 获取最近的样本
            recent_sample = self.vibration_samples[-1]
            
            # 如果有径向变形数据，显示统计信息
            if 'radial_deformations' in recent_sample and recent_sample['radial_deformations']:
                deformations = [d['deformation'] for d in recent_sample['radial_deformations']]
                avg_deform = np.mean(deformations)
                max_deform = np.max(deformations)
                min_deform = np.min(deformations)
                
                deform_text = [
                    f"average radial deformation: {avg_deform:.2f} pixel",
                    f"max radial deformation: {max_deform:.2f} pixel",
                    f"min radial deformation: {min_deform:.2f} pixel"
                ]
                
                # 显示统计信息
                y_pos = window_height - 90
                for text in deform_text:
                    cv2.putText(canvas, text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25
        
        # 创建固定大小的窗口
        cv2.namedWindow('Radial Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Radial Analysis', window_width, window_height)
        
        # 显示结果
        cv2.imshow('Radial Analysis', canvas)

    def calculate_radial_points_from_contour(self, frame, center, contour, num_points=None):
        """
        直接使用輪廓上的點作為徑向點，並按角度編號和採樣
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        center (tuple): 中心點座標
        contour (numpy.ndarray): 輪廓點集
        num_points (int): 要採樣的徑向點數量，預設為36
        
        返回:
        list: 徑向點列表
        """
        if num_points is None:
            num_points = self.num_radial_points

        if contour is None or len(contour) < 5:
            return []
            
        radial_points = []
        center_x, center_y = center
        
        # 將輪廓轉換為二維點列表
        contour_points = contour.reshape(-1, 2)
        
        # 計算每個點的角度和距離
        point_data = []
        for point in contour_points:
            x, y = point
            
            # 計算相對於中心的偏移
            dx = x - center_x
            dy = y - center_y
            
            # 計算角度 (0-360度)
            angle = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
            
            # 計算距離
            distance = np.sqrt(dx*dx + dy*dy)
            
            point_data.append({
                'point': (int(x), int(y)),
                'angle': angle,
                'distance': distance
            })
        
        # 按角度排序
        point_data.sort(key=lambda p: p['angle'])
        
        # 確保採樣36個點
        if len(point_data) > num_points:
            # 如果點數太多，均勻採樣
            step = len(point_data) // num_points
            sampled_points = point_data[::step][:num_points]
        elif len(point_data) < num_points:
            # 如果點數不足，插值
            while len(point_data) < num_points:
                # 找到角度間隔最大的兩個點之間進行插值
                max_gap_index = 0
                max_gap = 0
                for i in range(len(point_data)):
                    next_index = (i + 1) % len(point_data)
                    gap = (point_data[next_index]['angle'] - point_data[i]['angle'] + 360) % 360
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_index = i
                
                # 在最大間隔處插入新點
                next_index = (max_gap_index + 1) % len(point_data)
                p1 = point_data[max_gap_index]
                p2 = point_data[next_index]
                
                # 線性插值
                interp_angle = (p1['angle'] + p2['angle']) / 2
                interp_dist = (p1['distance'] + p2['distance']) / 2
                
                # 計算新點座標
                interp_angle_rad = np.deg2rad(interp_angle)
                new_x = int(center_x + interp_dist * np.cos(interp_angle_rad))
                new_y = int(center_y + interp_dist * np.sin(interp_angle_rad))
                
                point_data.insert(next_index, {
                    'point': (new_x, new_y),
                    'angle': interp_angle,
                    'distance': interp_dist,
                    'interpolated': True
                })
            
            sampled_points = point_data
        else:
            sampled_points = point_data
        
        # 確保最終返回的是36個點
        radial_points = sampled_points[:num_points]
        
        # 調試可視化（如果需要）
        if self.debug:
            debug_img = frame.copy()
            
            # 繪製輪廓
            cv2.drawContours(debug_img, [contour], 0, (0, 255, 0), 2)
            
            # 繪製中心點
            cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
            
            # 繪製每個徑向點並標註角度
            for i, point_info in enumerate(radial_points):
                point = point_info['point']
                angle = point_info['angle']
                
                # 繪製徑向點
                cv2.circle(debug_img, point, 2, 
                        (0, 255, 255) if point_info.get('interpolated', False) else (0, 0, 255), 
                        -1)
                
                # 繪製從中心到點的線
                cv2.line(debug_img, (int(center_x), int(center_y)), point, (255, 0, 0), 1)
                
                # 每隔一定數量標註角度
                if i % 6 == 0:  # 每6個點標註一次
                    cv2.putText(debug_img, f"{angle:.1f}°", 
                        (point[0] + 5, point[1] + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 顯示調試圖像
            cv2.namedWindow('Contour Points', cv2.WINDOW_NORMAL)
            cv2.imshow('Contour Points', debug_img)
            cv2.waitKey(1)
        
        return radial_points
    
    def show_binary_view_window(self, binary_img):
        """
        显示增强版的二值化视图窗口，使用与追踪状态相同的窗口大小和布局
        
        参数:
        binary_img (numpy.ndarray): 二值化图像
        """
        # 设定固定窗口大小
        window_width = 1024
        window_height = 768
        
        # 确保binary_img是3通道彩色图像
        if len(binary_img.shape) == 2:
            display_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = binary_img.copy()
        

        
        # 计算影像与窗口的尺寸比例
        h, w = display_img.shape[:2]
        
        # 缩放系数 - 使用与追踪状态窗口相同的缩放
        scale_factor = 0.9  # 例如，0.9 表示影片会填充窗口的 90%
        
        # 计算保持原始比例下的显示尺寸
        scale = min(window_width / w, window_height / h) * scale_factor
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # 创建黑色背景画布
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # 计算置中位置
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2
        
        # 调整影像大小
        resized_img = cv2.resize(display_img, (new_width, new_height))
        
        # 将调整后的影像放在画布中心
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
        
        # 添加标题和说明
        cv2.putText(canvas, "Binary View", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Binary View', canvas)
        
    def select_best_gradient_layer(self, contours_info, previous_position=None):
        """
        從多層漸層輪廓中選擇最佳的一層用於追蹤 - 優化版
        
        參數:
        contours_info (list): 輪廓信息列表
        previous_position (tuple): 上一幀檢測到的位置
        
        返回:
        dict: 選中的輪廓信息
        """
        if not contours_info:
            return None
        
        # 按面積排序 (從大到小)，這樣外圈會在前面
        sorted_contours = sorted(contours_info, key=lambda x: x['area'], reverse=True)
        
        # 對於細環模式，直接選擇最大面積的輪廓
        if self.thin_ring_mode:
            # 如果有上一幀位置，檢查最大的輪廓是否合理
            if previous_position and sorted_contours:
                # 計算最大輪廓中心到上一幀位置的距離
                largest_contour = sorted_contours[0]
                center = largest_contour['center']
                dist = np.sqrt((center[0] - previous_position[0])**2 + 
                            (center[1] - previous_position[1])**2)
                
                # 如果距離過大，查看其他大輪廓
                if dist > 50 and len(sorted_contours) > 1:  # 50像素閾值
                    # 查看次大的輪廓
                    for c in sorted_contours[1:min(3, len(sorted_contours))]:
                        center = c['center']
                        new_dist = np.sqrt((center[0] - previous_position[0])**2 + 
                                        (center[1] - previous_position[1])**2)
                        if new_dist < dist and c['area'] > sorted_contours[0]['area'] * 0.7:
                            # 找到一個更接近且面積足夠大的輪廓
                            return c
                
            # 返回最大面積的輪廓
            return sorted_contours[0] if sorted_contours else None
        
        # 非細環模式下的邏輯 (保持原有代碼但簡化)
        # 如果有上一幀位置，計算與各輪廓的距離
        if previous_position:
            # 只計算最大的幾個輪廓到上一幀位置的距離
            candidate_contours = sorted_contours[:min(5, len(sorted_contours))]
            for c in candidate_contours:
                center = c['center']
                dist = np.sqrt((center[0] - previous_position[0])**2 + 
                            (center[1] - previous_position[1])**2)
                c['prev_distance'] = dist
            
            # 在面積較大的輪廓中尋找距離適中的
            for c in candidate_contours:
                if c.get('prev_distance', float('inf')) < 50:  # 50像素閾值
                    return c
                    
        # 如果沒有找到合適的輪廓，使用最大面積的輪廓
        return sorted_contours[0] if sorted_contours else None

    def filter_ring_contour(self, frame, previous_position=None):
        """
        過濾方法，專門針對黑背景上的白色圓環輪廓，優化版本
        
        參數:
        frame (numpy.ndarray): 輸入幀
        previous_position (tuple): 上一幀的位置
        
        返回:
        tuple: (position, angle, contour, binary_img, success, inner_diameter, outer_diameter)
        """
        # 轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用自適應直方圖均衡化以增強對比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 使用自適應閾值 - 對細環更有效
        adaptive_threshold = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, -2  # 負C值有助於黑色背景上的白色
        )
        
        # 創建二值圖像用於顯示
        binary_img = cv2.cvtColor(adaptive_threshold, cv2.COLOR_GRAY2BGR)
        
        # 只使用一種輪廓檢索模式
        contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按圓形度和面積過濾輪廓
        valid_contours = []
        min_area = max(10, self.detection_area_min * 0.3)  # 降低閾值以捕獲細環
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 跳過太小的
            if area < min_area:
                continue
                
            # 計算圓形度，以鎖定圓形
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 對細環使用寬鬆的圓形度範圍
            if 0.5 < circularity < 1.3:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                    
                    valid_contours.append({
                        'contour': contour,
                        'area': area,
                        'circularity': circularity,
                        'center': center
                    })
        
        # 只在需要時繪製輪廓 (節省計算時間)
        if self.debug:
            # 用不同顏色繪製最大的3個輪廓
            for i, contour_info in enumerate(sorted(valid_contours, key=lambda x: x['area'], reverse=True)[:3]):
                color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == 1 else (255, 0, 0)
                cv2.drawContours(binary_img, [contour_info['contour']], 0, color, 1)
        
        # 按面積排序並選擇最大的輪廓作為外圈
        if valid_contours:
            # 首先按面積大小排序 (從大到小)
            valid_contours.sort(key=lambda x: x['area'], reverse=True)
            
            # 選擇最大的輪廓，但如果有上一幀位置則優先考慮連續性
            selected_contour_info = valid_contours[0]  # 默認選最大的
            
            if previous_position and len(valid_contours) > 1:
                # 計算最大輪廓中心到上一幀位置的距離
                center = selected_contour_info['center']
                distance = np.sqrt((center[0] - previous_position[0])**2 + 
                                (center[1] - previous_position[1])**2)
                
                # 如果最大輪廓距離過大，查看次大的輪廓
                if distance > 50:  # 50像素閾值
                    for c in valid_contours[1:min(3, len(valid_contours))]:
                        center = c['center']
                        new_dist = np.sqrt((center[0] - previous_position[0])**2 + 
                                        (center[1] - previous_position[1])**2)
                        # 面積至少為最大輪廓的70%且距離更近
                        if new_dist < distance and c['area'] > valid_contours[0]['area'] * 0.7:
                            selected_contour_info = c
                            break
            
            # 獲取選中的輪廓和中心點
            selected_contour = selected_contour_info['contour']
            center = selected_contour_info['center']
            
            # 高亮顯示選定的輪廓
            cv2.drawContours(binary_img, [selected_contour], 0, (0, 255, 255), 2)
            cv2.circle(binary_img, center, 3, (0, 0, 255), -1)
            
            # 計算角度和直徑
            angle = 0
            try:
                if len(selected_contour) >= 5:
                    ellipse = cv2.fitEllipse(selected_contour)
                    angle = ellipse[2]
                    
                    # 從橢圓獲取直徑
                    axes = ellipse[1]
                    outer_diameter = max(axes)
                    # 對於細環，內徑估計更接近外徑
                    inner_diameter = outer_diameter * 0.9 if self.thin_ring_mode else outer_diameter * 0.8
                    
                    # 繪製橢圓
                    cv2.ellipse(binary_img, ellipse, (255, 0, 255), 1)
                else:
                    # 從面積估計
                    radius = np.sqrt(selected_contour_info['area'] / np.pi)
                    outer_diameter = radius * 2
                    inner_diameter = outer_diameter * 0.9 if self.thin_ring_mode else outer_diameter * 0.8
            except:
                # 備用方法
                radius = np.sqrt(selected_contour_info['area'] / np.pi)
                outer_diameter = radius * 2
                inner_diameter = outer_diameter * 0.9 if self.thin_ring_mode else outer_diameter * 0.8
            
            return center, angle, selected_contour, binary_img, True, inner_diameter, outer_diameter
        
        # 如果沒有找到有效輪廓，嘗試霍夫圓檢測作為最後手段
        circles = cv2.HoughCircles(
            enhanced, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=50, 
            param2=30, 
            minRadius=10, 
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # 選擇最大的圓 (假設是外圈)
            largest_circle = None
            largest_radius = 0
            
            for circle in circles[0, :]:
                x, y, r = circle
                if r > largest_radius:
                    largest_radius = r
                    largest_circle = circle
            
            if largest_circle is not None:
                x, y, r = largest_circle
                
                # 繪製圓
                cv2.circle(binary_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(binary_img, (x, y), 2, (0, 0, 255), 3)
                
                # 創建圓的合成輪廓
                contour_points = []
                for angle in np.linspace(0, 2*np.pi, 36):
                    px = int(x + r * np.cos(angle))
                    py = int(y + r * np.sin(angle))
                    contour_points.append([[px, py]])
                
                synthetic_contour = np.array(contour_points, dtype=np.int32)
                
                # 返回圓
                return (x, y), 0, synthetic_contour, binary_img, True, r*0.9, r*1.2
        
        # 如果到這裡，沒有找到圓環
        return None, 0, None, binary_img, False, 0, 0

    def detect_all_contours(self, frame, previous_position=None):
        """
        檢測幀中的所有環形輪廓，並優先選擇外圈
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        previous_position (tuple): 上一幀檢測到的位置 (可選)
        
        返回:
        tuple: (center, angle, contours, binary_img, success, contour_info_list, selected_contour)
            contour_info_list 是包含每個輪廓詳細信息的列表
            selected_contour 是選中的輪廓
        """
        # 將圖像轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用高斯模糊減少噪聲
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 嘗試多種二值化方法以獲得最佳結果
        binary_methods = [
            # 自適應閾值 - 高斯
            {
                'func': cv2.adaptiveThreshold,
                'params': {
                    'maxValue': 255,
                    'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    'thresholdType': cv2.THRESH_BINARY_INV,
                    'blockSize': 10,
                    'C': 1.5
                }
            },
            # 自適應閾值 - 均值
            {
                'func': cv2.adaptiveThreshold,
                'params': {
                    'maxValue': 255,
                    'adaptiveMethod': cv2.ADAPTIVE_THRESH_MEAN_C,
                    'thresholdType': cv2.THRESH_BINARY_INV,
                    'blockSize': 15,
                    'C': 5
                }
            },
            # Otsu閾值
            {
                'func': lambda img, **kwargs: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            }
        ]
        
        best_contours = []
        best_binary = None
        best_score = 0
        
        # 嘗試每一種二值化方法
        for method in binary_methods:
            try:
                # 應用二值化
                if 'params' in method:
                    thresh = method['func'](blurred, **method['params'])
                else:
                    thresh = method['func'](blurred)
                
                # 應用形態學操作以改善輪廓檢測
                kernel = np.ones((3, 3), np.uint8)
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # 創建二值化的彩色圖像用於顯示
                binary_img = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
                
                # 查找所有輪廓
                contours, hierarchy = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                # 過濾和評估輪廓
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # 過濾過小的輪廓
                    if area > self.detection_area_min:
                        # 計算圓形度
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        # 選擇接近圓形的輪廓
                        if 0.4 < circularity < 1.0:
                            # 計算質心
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                center = (cx, cy)
                                
                                # 保存這個輪廓的詳細信息
                                valid_contours.append({
                                    'contour': contour,
                                    'area': area,
                                    'circularity': circularity,
                                    'center': center
                                })
                
                # 檢查是否找到足夠的輪廓
                if len(valid_contours) >= 3:  # 至少希望找到3個輪廓
                    # 計算此結果的得分
                    method_score = len(valid_contours) * 0.5  # 輪廓數量增加分數
                    
                    # 根據圓形度增加分數
                    avg_circularity = sum(c['circularity'] for c in valid_contours) / len(valid_contours)
                    method_score += avg_circularity * 10
                    
                    # 如果有前一幀位置，檢查是否有輪廓接近該位置
                    if previous_position:
                        min_dist = float('inf')
                        for c in valid_contours:
                            dist = np.sqrt((c['center'][0] - previous_position[0])**2 + 
                                        (c['center'][1] - previous_position[1])**2)
                            min_dist = min(min_dist, dist)
                        
                        # 接近前一位置的結果得分更高
                        proximity_score = max(0, 1 - min_dist / 100)  # 100px為參考距離
                        method_score += proximity_score * 5
                    
                    # 更新最佳結果
                    if method_score > best_score:
                        best_score = method_score
                        best_contours = valid_contours
                        best_binary = binary_img
            except Exception as e:
                if self.debug:
                    print(f"二值化方法出錯: {e}")
        
        # 如果沒有找到有效輪廓，返回失敗
        if not best_contours:
            # 返回空的二值化圖像或原始幀的灰度版本
            if best_binary is None:
                best_binary = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return None, None, [], best_binary, False, [], None
        
        # 檢查輪廓的中心點是否形成一個集群
        # 這對於確保所有輪廓都屬於同一個環很重要
        all_centers = np.array([c['center'] for c in best_contours])
        
        # 計算所有中心點的均值
        mean_center = np.mean(all_centers, axis=0)
        
        # 計算每個中心點到均值的距離
        distances = np.sqrt(np.sum((all_centers - mean_center) ** 2, axis=1))
        
        # 找出距離較近的輪廓（可能屬於同一環）
        # 使用距離閾值，可以根據環的預期大小調整
        max_dist_threshold = max(frame.shape) * 0.1  # 幀尺寸的10%作為閾值
        coherent_indices = np.where(distances < max_dist_threshold)[0]
        
        # 如果沒有足夠的一致輪廓，嘗試調整閾值
        if len(coherent_indices) < 3:
            max_dist_threshold = max(frame.shape) * 0.2  # 增加閾值
            coherent_indices = np.where(distances < max_dist_threshold)[0]
        
        # 篩選一致的輪廓
        coherent_contours = [best_contours[i] for i in coherent_indices]
        
        # 根據面積排序輪廓 (從大到小 = 從外到內)
        sorted_contours = sorted(coherent_contours, key=lambda x: x['area'], reverse=True)
        
        if len(sorted_contours) > 0:
            # 選擇最外層輪廓 (面積最大的)
            selected_contour_info = sorted_contours[0]
            selected_contour = selected_contour_info['contour']
        else:
            selected_contour = None
        
        # 準備返回的結果
        center = mean_center
        angle = None  # 可以從擬合橢圓獲取，暫不設置
        
        # 在二值化圖像上標記所有有效輪廓
        for i, contour_info in enumerate(sorted_contours):
            contour = contour_info['contour']
            
            # 使用不同顏色標記不同輪廓
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 綠、藍、紅、黃
            color = colors[i % len(colors)]
            
            # 畫輪廓
            cv2.drawContours(best_binary, [contour], 0, color, 2)
            
            # 標記編號
            cx, cy = contour_info['center']
            cv2.putText(best_binary, str(i+1), (int(cx), int(cy)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 標記整體中心
        cv2.circle(best_binary, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)
        
        # 嘗試為選定的輪廓計算角度
        if selected_contour is not None and len(selected_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(selected_contour)
                angle = ellipse[2]
                # 在二值化圖像上畫出擬合的橢圓
                cv2.ellipse(best_binary, ellipse, (255, 255, 255), 1)
            except Exception as e:
                if self.debug:
                    print(f"擬合橢圓出錯: {e}")
        
        return center, angle, sorted_contours, best_binary, True, sorted_contours, selected_contour
  
    def process_vibration(self, start_frame=None, end_frame=None):
        """
        處理影片並分析圓環振動，包含細環處理優化
        
        參數:
        start_frame (int): 起始幀，None表示從頭開始
        end_frame (int): 結束幀，None表示處理到結尾
        """
        # 創建其他顯示窗口
        cv2.namedWindow('Binary View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Binary View', 1024, 768)
        
        cv2.namedWindow('Tracking Status', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracking Status', 1024, 768)

        cv2.namedWindow('Radial Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Radial Analysis', 1024, 768)

        # 初始化視頻播放控制
        self.is_paused = False
        delay = int(1000 / (self.fps * self.play_speed))  # 每幀延遲時間（毫秒）
        
        # 初始化snapshot功能(預設關閉)
        if not hasattr(self, 'snapshot_enabled'):
            self.snapshot_enabled = False
        snapshots_saved = 0
        snapshots_dir = os.path.join(self.output_dir, 'contour_snapshots')
        
        # 設置起始幀
        if start_frame is not None and 0 <= start_frame < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
        else:
            frame_count = 0
        
        # 設置結束幀
        if end_frame is not None and end_frame <= self.total_frames:
            total_frames_to_process = end_frame - frame_count
        else:
            total_frames_to_process = self.total_frames - frame_count

        print(f"\n開始處理視頻: {self.video_path}")
        print(f"處理幀範圍: {frame_count} 到 {frame_count + total_frames_to_process - 1}")
        print("\n控制: 空格=暫停/播放 | s=減速 | f=加速 | b=切換二值化視圖 | r=開始/停止記錄 | t=切換細環模式 | c=切換輪廓快照 | v=生成快照視頻 | a=分析 | d=徑向分析 | q或ESC=退出\n")
        
        # 初始化變數
        last_ring_position = None
        last_ring_time = None
        current_velocity = [0, 0]
        velocity_history = []
        use_high_speed_mode = False
        speed_threshold = 5.0
        frames_processed = 0
        lowest_point = None
        self.reference_radial_distances = {}
        contour_selection_index = 2

        selected_contour = None
        ring_angle = 0
        radial_points = []

        # 初始化細環檢測模式（可以從命令行參數設置）
        if not hasattr(self, 'thin_ring_mode'):
            self.thin_ring_mode = False
        
        # 初始化自動檢測環類型（可以從命令行參數設置）
        if not hasattr(self, 'auto_detect_ring_type'):
            self.auto_detect_ring_type = False

        # 處理循環
        while frames_processed < total_frames_to_process:
            roi_active = self.use_roi and self.roi is not None and len(self.roi) == 4
            roi_offset_x = 0
            roi_offset_y = 0

            if roi_active:
                # 獲取ROI坐標
                x1, y1, x2, y2 = self.roi
                roi_offset_x = x1
                roi_offset_y = y1
                
            # 處理暫停
            if self.is_paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):  # 空格键
                    self.is_paused = False
                    print("继续播放")
                elif key == ord('n'):  # 'n'键：下一帧
                    pass  # 继续执行处理下一帧
                elif key == ord('l'):  # 'l'键：切换追踪模式
                    self.tracking_mode_only = not self.tracking_mode_only
                    print(f"显示模式: {'仅追踪模式' if self.tracking_mode_only else '完整分析模式'}")
                elif key == ord('r'):  # 'r'键：开始/停止记录
                    if self.is_recording:
                        self.stop_vibration_recording()
                    else:
                        self.start_vibration_recording()
                elif key == ord('a'):  # 'a'键：分析振动
                    if len(self.vibration_samples) > 10:
                        print("正在分析振动数据...")
                        fig = self.analyze_and_plot_vibration()
                        plt.show(block=False)
                elif key == ord('d'):  # 'd'键：径向分析
                    if len(self.vibration_samples) > 10:
                        print("正在分析径向振动...")
                        fig = self.plot_radial_vibration()
                        plt.show(block=False)
                elif key == ord('v'):  # 'v'键：切换径向分析窗口显示
                    self.show_radial_window = not self.show_radial_window if hasattr(self, 'show_radial_window') else False
                    print(f"径向分析窗口: {'显示' if self.show_radial_window else '隐藏'}")
                    if not self.show_radial_window:
                        cv2.destroyWindow('Radial Analysis')
                elif key == ord('c'):  # 'c'键：切换轮廓快照功能
                    self.toggle_contour_snapshot()
                    if self.snapshot_enabled:
                        print(f"輪廓快照開始保存到: {snapshots_dir}")
                    else:
                        print(f"已保存 {snapshots_saved} 張輪廓快照")
                elif key == ord('v'):  # 'v'键：从现有快照生成视频
                    if snapshots_saved > 0:
                        print("\n正在从现有快照生成延时摄影视频...")
                        video_path = self.create_contour_timelapse()
                        if video_path:
                            print(f"延时摄影视频已生成: {video_path}")
                elif key == ord('q') or key == 27:  # 'q'键或ESC：退出
                    break
                else:
                    continue  # 维持暂停状态
            
            # 讀取下一幀
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 增加幀計數
            frames_processed += 1
            frame_count += 1
            
            # 處理幀
            try:
                # 儲存原始幀
                original_frame = frame.copy()
                
                # 應用 ROI 裁剪 (如果已啟用)
                if self.use_roi:
                    frame = self.apply_roi_to_frame(frame)
                
                # 決定使用哪種檢測方法
                use_thin_ring_detection = self.thin_ring_mode
                
                # 如果啟用自動檢測，則自動判斷環的類型
                if self.auto_detect_ring_type:
                    ring_type = self.determine_ring_type(frame)
                    use_thin_ring_detection = (ring_type == 'thin')
                
                # 計算當前速度並決定是否使用高速模式
                current_speed = np.sqrt(current_velocity[0]**2 + current_velocity[1]**2)
                use_high_speed_mode = current_speed > speed_threshold


                # 決定使用哪種檢測方法
                if use_thin_ring_detection:
                    # 方法1: 使用新的增強版細環檢測方法
                    center, angle, selected_contour, binary_img, success, inner_diameter, outer_diameter = self.filter_ring_contour(frame, previous_position=last_ring_position)
                    
                    # 如果新方法失敗，嘗試使用原有的漸層層級方法作為備用
                    if not success:
                        center, angle, selected_contour, binary_img, success, inner_diameter, outer_diameter = self.enhanced_thin_ring_detection(frame, previous_position=last_ring_position)
                        # 獲取所有潛在的輪廓
                        all_contours_info = self.detect_thin_ring_contours(frame, previous_position=last_ring_position) 
                        
                        # 選擇最佳的漸層層級 - 現在會優選外圈
                        best_contour_info = self.select_best_gradient_layer(all_contours_info, previous_position=last_ring_position) 
                        
                        if best_contour_info:
                            center = best_contour_info['center'] 
                            contour = best_contour_info['contour']
                            selected_contour = contour
                            
                            # 嘗試擬合橢圓以獲取角度和內外徑
                            try:
                                if len(contour) >= 5:
                                    ellipse = cv2.fitEllipse(contour)
                                    angle = ellipse[2]
                                    # 估計內外徑 - 維持比例關係
                                    diameter = max(ellipse[1])
                                    outer_diameter = diameter  # 這是外圈直徑
                                    inner_diameter = diameter * 0.8  # 估計
                                    outer_diameter = diameter * 1.2
                                
                                # 創建二值化圖像用於顯示
                                binary_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                                cv2.drawContours(binary_img, [contour], 0, (0, 255, 0), 2)
                                cv2.circle(binary_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                                
                                success = True
                            except Exception as e:
                                if self.debug:
                                    print(f"細環處理出錯: {e}")
                                success = False
                    
                    # 如果檢測成功，計算徑向點
                    if success and selected_contour is not None:
                        radial_points = self.calculate_radial_points_for_thin_rings(frame, center, selected_contour)
                else:
                    # 使用標準檢測方法
                    center, angle, contours, binary_img, success, all_contours_info, selected_contour = self.detect_all_contours(frame, previous_position=last_ring_position)
                    
                    # 如果檢測成功，使用標準徑向點計算
                    if success and all_contours_info:
                        # 選擇指定的輪廓 (確保索引有效)
                        valid_index = min(contour_selection_index, len(all_contours_info) - 1)
                        selected_info = all_contours_info[valid_index]
                        
                        # 獲取所選輪廓的信息
                        selected_contour = selected_info['contour']
                        selected_center = selected_info['center']
                        
                        # 應用卡爾曼濾波器平滑位置
                        smooth_position = self.update_kalman_filter(selected_center)
                        center = smooth_position
                        
                        # 新的內外徑計算代碼
                        if len(all_contours_info) >= 2:
                            # 使用最內層和最外層輪廓計算直徑
                            inner_info = all_contours_info[0]
                            outer_info = all_contours_info[-1]
                            
                            inner_diameter = self.estimate_contour_diameter(inner_info['contour'])
                            outer_diameter = self.estimate_contour_diameter(outer_info['contour'])
                        else:
                            # 如果只有一個輪廓，使用其大小的估計
                            avg_diameter = self.estimate_contour_diameter(selected_contour)
                            inner_diameter = avg_diameter * 0.8
                            outer_diameter = avg_diameter * 1.2
                        
                        # 標準徑向點計算
                        radial_points = self.calculate_radial_points_from_contour(frame, center, selected_contour)
                # 顯示追蹤狀態窗口
                self.show_tracking_window(
                    original_frame, 
                    center if success else None,
                    selected_contour if success else None,
                    highspeed_mode=use_high_speed_mode,
                    lowest_point=lowest_point
                )
                
                # 顯示徑向分析窗口
                self.show_radial_analysis_window(
                    original_frame, 
                    center if success else None,
                    selected_contour if success else None,
                    radial_points if success else None
                )
                
                # 顯示二值化圖像
                if self.show_binary and binary_img is not None:
                    self.show_binary_view_window(binary_img)
                

                # 確保檢測成功且center不為None
                if success and center is not None:
                    # 計算實際幀編號和時間（秒）
                    actual_frame_number = frame_count
                    time_sec = actual_frame_number / self.fps
                    current_time = time_sec
                    
                    # 追蹤輪廓的最低點
                    lowest_point = None
                    if selected_contour is not None:
                        lowest_point = self.track_lowest_point(selected_contour, binary_img)
                        self.update_lowest_point_data(actual_frame_number, time_sec, lowest_point)
                    
                    # 儲存輪廓快照（如果啟用）
                    if self.snapshot_enabled and selected_contour is not None:
                        # 保存對齊的輪廓快照
                        snapshot_path = self.save_aligned_contour_snapshot(
                            original_frame, 
                            selected_contour,
                            center if not roi_active else (center[0] + roi_offset_x, center[1] + roi_offset_y),
                            actual_frame_number
                        )
                        if snapshot_path:
                            snapshots_saved += 1
                            if snapshots_saved % 50 == 0:
                                print(f"已保存 {snapshots_saved} 張輪廓快照")
                    
                    # 計算速度和更新卡爾曼濾波器
                    if last_ring_position is not None and last_ring_time is not None:
                        # 計算時間差
                        dt = current_time - last_ring_time
                        if dt > 0:
                            # 計算速度 (像素/秒)
                            vx = (center[0] - last_ring_position[0]) / dt
                            vy = (center[1] - last_ring_position[1]) / dt
                            
                            # 轉換為像素/幀
                            vx_frame = vx / self.fps
                            vy_frame = vy / self.fps
                            
                            # 使用指數移動平均更新速度估計
                            alpha = 0.3
                            current_velocity[0] = alpha * vx_frame + (1 - alpha) * current_velocity[0]
                            current_velocity[1] = alpha * vy_frame + (1 - alpha) * current_velocity[1]
                            
                            # 保存速度歷史
                            velocity_history.append(current_velocity.copy())
                            if len(velocity_history) > 10:
                                velocity_history.pop(0)
                    
                    # 如果正在記錄，添加振動樣本
                    if self.is_recording:
                        self.add_radial_vibration_sample(
                            actual_frame_number, 
                            time_sec, 
                            center, 
                            inner_diameter, 
                            outer_diameter,
                            radial_points
                        )
                    
                    # 計算相對坐標
                    absolute_position = center
                    if self.use_roi and self.roi is not None and len(self.roi) == 4:
                        x1, y1, x2, y2 = self.roi
                        absolute_position = (center[0] + x1, center[1] + y1)
                    
                    rel_x = absolute_position[0] - self.origin_x
                    rel_y = self.origin_y - absolute_position[1]
                    
                    # 存儲數據
                    self.data['frame'].append(actual_frame_number)
                    self.data['time'].append(time_sec)
                    self.data['x'].append(center[0])
                    self.data['y'].append(center[1])
                    self.data['angle'].append(angle if angle is not None else 0)
                    self.data['inner_diameter'].append(inner_diameter)
                    self.data['outer_diameter'].append(outer_diameter)
                    self.data['ring_thickness'].append((outer_diameter - inner_diameter) / 2)
                    self.data['rel_x'].append(rel_x) 
                    self.data['rel_y'].append(rel_y)
                    
                    # 添加到追蹤歷史
                    self.tracking_history.append(center)
                    if len(self.tracking_history) > self.max_history_length:
                        self.tracking_history.pop(0)
                    
                    # 執行實時 FFT 分析
                    if self.is_recording and len(self.vibration_plot_data['times']) >= self.fft_window_size:
                        self.perform_realtime_fft()
                    
                    # 更新上一幀位置和時間
                    last_ring_position = center
                    last_ring_time = current_time
                    
                    # 在終端顯示更新資訊（每 10 幀更新一次）
                    if frames_processed % 10 == 0:
                        progress = (frames_processed / total_frames_to_process) * 100
                        mode_name = "細環" if use_thin_ring_detection else "標準"
                        print(f"\r處理進度: {progress:.1f}% (幀 {actual_frame_number}/{self.total_frames}) - 檢測模式: {mode_name}", end="")
                
                else:
                    # 檢測失敗，重置卡爾曼濾波器
                    self.reset_kalman_filter()
                    current_velocity[0] *= 0.8
                    current_velocity[1] *= 0.8
                
                # 按鍵控制
                delay = int(1000 / (self.fps * self.play_speed)) if not self.is_paused else 0
                delay = max(1, delay)
                key = cv2.waitKey(delay) & 0xFF
                
                # 處理控制命令
                if key == ord(' '):  # 空格键：暂停/播放
                    self.is_paused = not self.is_paused
                    print(f"{'暂停' if self.is_paused else '播放'}")
                elif key == ord('s'):  # 's'键：减慢速度
                    self.play_speed = max(0.1, self.play_speed - 0.1)
                    print(f"播放速度: {self.play_speed:.1f}x")
                elif key == ord('f'):  # 'f'键：加快速度
                    self.play_speed = min(5.0, self.play_speed + 0.1)
                    print(f"播放速度: {self.play_speed:.1f}x")
                elif key == ord('b'):  # 'b'键：切换二值化视图显示
                    self.show_binary = not self.show_binary
                    print(f"二值化视图: {'开启' if self.show_binary else '关闭'}")
                elif key == ord('/'):  # '/'键：切换细环检测模式
                    self.thin_ring_mode = not self.thin_ring_mode
                    print(f"细环检测模式: {'开启' if self.thin_ring_mode else '关闭'}")
                    # 重置卡尔曼滤波器，因为检测方法改变
                    self.reset_kalman_filter()
                elif key == ord('l'):  # 'l'键：切换追踪显示模式
                    self.tracking_mode_only = not self.tracking_mode_only
                    print(f"显示模式: {'仅追踪模式' if self.tracking_mode_only else '完整分析模式'}")
                elif key == ord('r'):  # 'r'键：开始/停止记录
                    if self.is_recording:
                        self.stop_vibration_recording()
                    else:
                        self.start_vibration_recording()
                elif key == ord('c'):  # 'c'键：切换轮廓快照功能
                    self.toggle_contour_snapshot()
                    if self.snapshot_enabled:
                        print(f"輪廓快照開始保存到: {snapshots_dir}")
                    else:
                        print(f"已保存 {snapshots_saved} 張輪廓快照")
                elif key == ord('v'):  # 'v'键：从现有快照生成视频
                    if snapshots_saved > 0:
                        print("\n正在从现有快照生成延时摄影视频...")
                        video_path = self.create_contour_timelapse()
                        if video_path:
                            print(f"延时摄影视频已生成: {video_path}")
                elif key == ord('a'):  # 'a'键：分析振动
                    if len(self.vibration_samples) > 10:
                        print("\n正在分析振动数据...")
                        fig = self.analyze_and_plot_vibration()
                        plt.show(block=False)
                elif key == ord('d'):  # 'd'键：分析径向振动
                    if len(self.vibration_samples) > 10:
                        print("\n正在分析径向振动...")
                        fig = self.plot_radial_vibration()
                        plt.show(block=False)
                elif key == ord('t'):  # 't'键：显示轨迹
                    if len(self.vibration_samples) > 10:
                        print("\n绘制振动轨迹...")
                        fig = self.plot_vibration_trajectory()
                        plt.show(block=False)
                elif key == ord('q') or key == 27:  # 'q'键或ESC：退出
                    break
                    
            except Exception as e:
                print(f"\n處理幀 {frame_count} 時出現錯誤: {e}")
                traceback.print_exc()

        # 處理完成後，自動保存數據，不詢問
        if len(self.vibration_samples) > 0:
            print("\n\n處理完成！")
            print(f"共處理 {frames_processed} 幀，記錄了 {len(self.vibration_samples)} 個振動樣本")
            
            # 自動保存數據，不詢問
            print("正在自動保存數據...")
            
            # 保存振動數據
            csv_path = self.save_vibration_data()
            if hasattr(self, 'lowest_point_data') and len(self.lowest_point_data['frame']) > 0:
                lowest_point_csv = self.save_lowest_point_data()
                print(f"最低點數據已保存至: {lowest_point_csv}")
            
            # 分析並保存分析結果
            print("生成並保存振動分析...")
            try:
                fig = self.analyze_and_plot_vibration()
                if fig:
                    # 保存振動分析圖表
                    self.save_vibration_analysis(fig)
                    
                    # 顯示分析結果
                    plt.show(block=False)
            except Exception as e:
                print(f"分析過程出錯: {e}")
                traceback.print_exc()
            
            # 保存徑向振動分析
            try:
                radial_fig = self.plot_radial_vibration()
                if radial_fig:
                    radial_path = os.path.join(self.output_dir, f"radial_vibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    radial_fig.savefig(radial_path, dpi=300, bbox_inches='tight')
                    print(f"徑向振動分析已保存至: {radial_path}")
                    plt.show(block=False)
            except Exception as e:
                print(f"徑向分析出錯: {e}")
                traceback.print_exc()
            
            # 保存振動軌跡
            try:
                traj_fig = self.plot_vibration_trajectory()
                if traj_fig:
                    traj_path = os.path.join(self.output_dir, f"vibration_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    traj_fig.savefig(traj_path, dpi=300, bbox_inches='tight')
                    print(f"振動軌跡已保存至: {traj_path}")
                    plt.show(block=False)
            except Exception as e:
                print(f"軌跡繪製出錯: {e}")
                traceback.print_exc()
            
            # 如果有保存輪廓快照，自動生成視頻
            if snapshots_saved > 0:
                print(f"從 {snapshots_saved} 張輪廓快照生成視頻...")
                try:
                    video_path = self.create_contour_timelapse()
                    if video_path:
                        print(f"輪廓快照視頻已生成: {video_path}")
                except Exception as e:
                    print(f"生成輪廓快照視頻時出錯: {e}")
                    traceback.print_exc()

        # 釋放資源
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 確保窗口關閉
       
    def enhanced_ring_detection(self, frame, previous_position=None):
        """
        針對細圓環優化的檢測方法
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        previous_position (tuple): 上一幀檢測到的位置
        
        返回:
        tuple: (position, angle, contour, binary_img, success, inner_diameter, outer_diameter)
        """
        # 轉為灰度圖像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用自適應直方圖均衡化以增強對比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 使用高斯模糊減少噪聲（較小的核以保留細節）
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 創建組合二值化結果的容器
        binary_combined = np.zeros_like(gray)
        
        # 方法1: 自適應閾值 - 對細輪廓更靈敏
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 9, 2
        )
        binary_combined = cv2.bitwise_or(binary_combined, binary_adaptive)
        
        # 方法2: Canny邊緣檢測 - 對細邊緣檢測效果好
        binary_canny = cv2.Canny(blurred, 50, 150)
        binary_combined = cv2.bitwise_or(binary_combined, binary_canny)
        
        # 方法3: Otsu 二值化
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_combined = cv2.bitwise_or(binary_combined, binary_otsu)
        
        # 形態學操作以連接斷裂的邊緣
        kernel = np.ones((2, 2), np.uint8)  # 較小的核以保留細節
        binary = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, kernel)
        
        # 創建二值化的彩色圖像用於顯示
        binary_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 查找所有輪廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 在二值化圖像上繪製所有輪廓（淡藍色）
        cv2.drawContours(binary_img, contours, -1, (255, 200, 0), 1)
        
        # 過濾和評估輪廓 - 對細輪廓降低面積要求
        valid_contours = []
        min_area = max(10, self.detection_area_min * 0.5)  # 降低最小面積要求
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 過濾過小的輪廓，但閾值較低
            if area > min_area:
                # 計算圓形度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # 對於細輪廓，放寬圓形度條件
                if 0.4 < circularity < 1.0:
                    # 計算質心
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = (cx, cy)
                        
                        # 保存這個輪廓
                        valid_contours.append({
                            'contour': contour,
                            'area': area,
                            'circularity': circularity,
                            'center': center
                        })
        
        # 如果有效輪廓太少，返回失敗
        if len(valid_contours) < 2:
            return None, 0, None, binary_img, False, 0, 0
        
        # 根據面積排序輪廓，從小到大
        sorted_contours = sorted(valid_contours, key=lambda x: x['area'])
        
        # 尋找內圈和外圈的最佳組合
        best_ring_pair = None
        best_pair_score = 0
        
        # 限制搜索範圍，減少計算量
        max_contours_to_check = min(15, len(valid_contours))  # 增加檢查的輪廓數
        
        for i in range(max_contours_to_check - 1):
            for j in range(i + 1, max_contours_to_check):
                contour_i = sorted_contours[i]
                contour_j = sorted_contours[j]
                
                # 確保兩個輪廓有明顯的面積差異（一個是內環，一個是外環）
                # 對於細環，降低面積比例要求
                if contour_j['area'] < contour_i['area'] * 1.1:  # 從1.2降到1.1
                    continue
                
                # 檢查兩個輪廓的中心是否足夠接近
                center_i = contour_i['center']
                center_j = contour_j['center']
                center_distance = np.sqrt((center_i[0] - center_j[0])**2 + 
                                        (center_i[1] - center_j[1])**2)
                
                # 放寬中心距離閾值
                center_distance_threshold = min(contour_i['area'], contour_j['area']) ** 0.5 * 0.5  # 從0.3增加到0.5
                
                if center_distance > center_distance_threshold:
                    continue
                
                # 計算得分：考慮圓形度、面積比例和中心距離
                circularity_score = (contour_i['circularity'] + contour_j['circularity']) / 2
                
                # 理想的面積比例（外圈面積/內圈面積）應該在某個範圍內
                # 對於細環，面積比例通常較小
                area_ratio = contour_j['area'] / contour_i['area']
                area_ratio_score = 1.0 - min(abs(area_ratio - 1.5) / 1.5, 1.0)  # 調整預期比例從2.0到1.5
                
                # 中心距離得分，距離越小越好
                center_distance_score = 1.0 - center_distance / center_distance_threshold
                
                # 綜合得分
                pair_score = circularity_score * 0.5 + area_ratio_score * 0.3 + center_distance_score * 0.2
                
                # 如果有上一幀位置，考慮與預測位置的距離
                if previous_position:
                    pair_center = ((center_i[0] + center_j[0]) / 2, (center_i[1] + center_j[1]) / 2)
                    prev_distance = np.sqrt((pair_center[0] - previous_position[0])**2 + 
                                        (pair_center[1] - previous_position[1])**2)
                    
                    # 速度越大，距離閾值越大
                    # 預設一個合理的速度值
                    estimated_velocity = 5.0  # 如果有實際速度估計可以替換
                    distance_threshold = max(50, estimated_velocity * 2)
                    
                    prev_distance_score = max(0, 1 - prev_distance / distance_threshold)
                    
                    # 根據距離調整得分，距離越近得分越高
                    pair_score = pair_score * 0.7 + prev_distance_score * 0.3
                
                # 更新最佳組合
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_ring_pair = (contour_i, contour_j)
        
        # 如果找到合適的內外環組合
        if best_ring_pair:
            inner_contour_info = best_ring_pair[0]
            outer_contour_info = best_ring_pair[1]
            
            inner_contour = inner_contour_info['contour']
            outer_contour = outer_contour_info['contour']
            
            # 計算內外橢圓
            if len(inner_contour) >= 5 and len(outer_contour) >= 5:
                try:
                    inner_ellipse = cv2.fitEllipse(inner_contour)
                    outer_ellipse = cv2.fitEllipse(outer_contour)
                    
                    # 計算環的中心為內外橢圓中心的平均值
                    inner_center = inner_ellipse[0]
                    outer_center = outer_ellipse[0]
                    ring_center = ((inner_center[0] + outer_center[0]) / 2, 
                                (inner_center[1] + outer_center[1]) / 2)
                    
                    # 計算環的朝向（使用外橢圓的角度）
                    ring_angle = outer_ellipse[2]
                    
                    # 在二值化图像上標記選中的內外輪廓
                    cv2.drawContours(binary_img, [inner_contour], 0, (0, 255, 0), 1)  # 內圈綠色
                    cv2.drawContours(binary_img, [outer_contour], 0, (0, 0, 255), 1)  # 外圈紅色
                    
                    # 在二值化图像上绘製內外橢圓
                    cv2.ellipse(binary_img, inner_ellipse, (0, 255, 0), 1)  # 內圓綠色
                    cv2.ellipse(binary_img, outer_ellipse, (0, 0, 255), 1)  # 外圓紅色
                    
                    # 標記中心點
                    cv2.circle(binary_img, (int(ring_center[0]), int(ring_center[1])), 2, (255, 0, 0), -1)
                    
                    # 計算內外徑
                    inner_diameter = max(inner_ellipse[1])
                    outer_diameter = max(outer_ellipse[1])
                    
                    # 計算更準確的輪廓
                    refined_outer_contour = self.refine_contour_for_thin_rings(frame, outer_contour, ring_center)
                    if refined_outer_contour is not None:
                        outer_contour = refined_outer_contour
                    
                    return ring_center, ring_angle, outer_contour, binary_img, True, inner_diameter, outer_diameter
                    
                except Exception as e:
                    if self.debug:
                        print(f"計算橢圓時出錯: {e}")
        
        # 如果無法找到合適的輪廓，嘗試使用Hough變換檢測圓
        circles = cv2.HoughCircles(
            enhanced, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            if len(circles[0]) > 0:
                # 選擇最佳圓
                best_circle = circles[0][0]
                x, y, r = best_circle
                
                # 創建一個圓形輪廓
                contour_points = []
                for angle in np.linspace(0, 2*np.pi, 36):
                    px = int(x + r * np.cos(angle))
                    py = int(y + r * np.sin(angle))
                    contour_points.append([[px, py]])
                
                synthetic_contour = np.array(contour_points, dtype=np.int32)
                
                # 在二值化圖像上繪製偵測到的圓
                cv2.circle(binary_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(binary_img, (x, y), 2, (0, 0, 255), 3)
                
                return (x, y), 0, synthetic_contour, binary_img, True, r*0.8, r*1.2
        
        # 如果無法找到合適的輪廓
        return None, 0, None, binary_img, False, 0, 0

    def refine_contour_for_thin_rings(self, frame, contour, center):
        """
        對細圓環輪廓進行精細化處理
        
        參數:
        frame (numpy.ndarray): 原始圖像幀
        contour (numpy.ndarray): 原始輪廓
        center (tuple): 中心點坐標
        
        返回:
        numpy.ndarray: 優化後的輪廓
        """
        try:
            # 轉為灰度圖像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 創建輪廓掩碼
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # 對掩碼區域應用Canny邊緣檢測
            edges = cv2.Canny(gray, 50, 150)
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            # 在邊緣上查找輪廓
            refined_contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(refined_contours) > 0:
                # 選擇最大的輪廓
                largest_contour = max(refined_contours, key=cv2.contourArea)
                
                # 確認輪廓與原始輪廓的中心大致相同
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    refined_center = (cx, cy)
                    
                    # 計算中心點距離
                    distance = np.sqrt((refined_center[0] - center[0])**2 + (refined_center[1] - center[1])**2)
                    
                    # 如果中心點接近，返回優化後的輪廓
                    if distance < 20:  # 可以調整此閾值
                        return largest_contour
            
            # 如果沒有找到合適的優化輪廓，返回原始輪廓
            return contour
        
        except Exception as e:
            if self.debug:
                print(f"輪廓精細化處理時出錯: {e}")
            return contour
  
    def update_vibration_plot_data(self, sample):
        """
        更新振動繪圖數據，增加徑向振動
        
        參數:
        sample (dict): 振動樣本
        """
        # 添加時間和位置數據
        self.vibration_plot_data['times'].append(sample['time'])
        self.vibration_plot_data['x_pos'].append(sample['x'])
        self.vibration_plot_data['y_pos'].append(sample['y'])
        
        # 添加振動數據（相對於參考位置的偏移）
        self.vibration_plot_data['x_vibration'].append(sample['dx'])
        self.vibration_plot_data['y_vibration'].append(sample['dy'])
        
        # 計算並添加徑向振動數據
        if self.reference_position is not None:
            # 計算當前點到參考點的距離
            dx = sample['x'] - self.reference_position[0]
            dy = sample['y'] - self.reference_position[1]
            current_radius = np.sqrt(dx*dx + dy*dy)
            
            # 計算參考半徑 (第一次記錄的半徑)
            if 'reference_radius' not in self.vibration_plot_data:
                self.vibration_plot_data['reference_radius'] = current_radius
                self.vibration_plot_data['radial_vibration'] = [0.0]  # 初始點的徑向振動為0
            else:
                # 計算徑向振動 (當前半徑 - 參考半徑)
                radial_vibration = current_radius - self.vibration_plot_data['reference_radius']
                self.vibration_plot_data['radial_vibration'].append(radial_vibration)
        
        # 如果有徑向變形數據，則計算平均徑向變形
        if 'radial_deformations' in sample and sample['radial_deformations']:
            # 計算所有角度的平均徑向變形
            deformations = [d['deformation'] for d in sample['radial_deformations']]
            avg_deformation = np.mean(deformations)
            
            # 添加到振動數據
            if 'avg_radial_deformation' not in self.vibration_plot_data:
                self.vibration_plot_data['avg_radial_deformation'] = []
            self.vibration_plot_data['avg_radial_deformation'].append(avg_deformation)
        
        # 限制數據長度，保持與 FFT 窗口大小一致
        max_length = max(self.fft_window_size, 200)  # 至少保留200個點
        if len(self.vibration_plot_data['times']) > max_length:
            self.vibration_plot_data['times'].pop(0)
            self.vibration_plot_data['x_pos'].pop(0)
            self.vibration_plot_data['y_pos'].pop(0)
            self.vibration_plot_data['x_vibration'].pop(0)
            self.vibration_plot_data['y_vibration'].pop(0)
            
            # 也移除徑向振動數據
            if 'radial_vibration' in self.vibration_plot_data and len(self.vibration_plot_data['radial_vibration']) > max_length:
                self.vibration_plot_data['radial_vibration'].pop(0)
            
            # 移除平均徑向變形數據
            if 'avg_radial_deformation' in self.vibration_plot_data and len(self.vibration_plot_data['avg_radial_deformation']) > max_length:
                self.vibration_plot_data['avg_radial_deformation'].pop(0)

    def save_aligned_contour_snapshot(self, frame, contour, center, frame_number, save_dir=None):
        """
        保存對齊的輪廓快照，以圓心為中心
        
        參數:
        frame (numpy.ndarray): 原始幀
        contour (numpy.ndarray): 輪廓點集
        center (tuple): 圓心坐標 (x, y)
        frame_number (int): 幀編號（用於命名）
        save_dir (str): 保存目錄，如果為None則使用預設輸出目錄下的snapshots子目錄
        
        返回:
        str: 保存的圖像路徑
        """
        try:
            import os
            import cv2
            import numpy as np
            
            # 設置保存目錄
            if save_dir is None:
                save_dir = os.path.join(self.output_dir, 'contour_snapshots')
            
            # 確保目錄存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 確定ROI的大小（根據輪廓尺寸自動調整或使用固定尺寸）
            if contour is not None and len(contour) > 0:
                # 計算輪廓的邊界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 計算包含輪廓的正方形區域的邊長（取長寬的最大值並增加一定邊距）
                box_size = max(w, h) + 40  # 增加邊距
                
                # 確保尺寸是偶數，便於對齊中心點
                if box_size % 2 != 0:
                    box_size += 1
            else:
                # 默認使用固定大小
                box_size = 200  # 可以根據需要調整
            
            # 創建輸出圖像（正方形，黑色背景）
            output_size = (box_size, box_size)
            output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            
            # 計算源圖像上的ROI區域（以圓心為中心）
            center_x, center_y = int(center[0]), int(center[1])
            half_size = box_size // 2
            
            # 計算ROI的左上角和右下角坐標
            roi_x1 = center_x - half_size
            roi_y1 = center_y - half_size
            roi_x2 = center_x + half_size
            roi_y2 = center_y + half_size
            
            # 檢查ROI是否超出圖像邊界，並進行調整
            src_x1 = max(0, roi_x1)
            src_y1 = max(0, roi_y1)
            src_x2 = min(frame.shape[1], roi_x2)
            src_y2 = min(frame.shape[0], roi_y2)
            
            # 計算目標圖像上的對應區域
            dst_x1 = half_size - (center_x - src_x1)
            dst_y1 = half_size - (center_y - src_y1)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            # 複製ROI到輸出圖像
            if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 > dst_x1 and dst_y2 > dst_y1:
                output_image[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
            
            # 在輸出圖像上繪製中心點和輪廓
            center_in_output = (half_size, half_size)
            cv2.circle(output_image, center_in_output, 3, (0, 0, 255), -1)  # 紅色中心點
            
            # 如果有輪廓，將其轉換到輸出圖像坐標系並繪製
            if contour is not None and len(contour) > 0:
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] = contour[:, :, 0] - roi_x1 + dst_x1
                adjusted_contour[:, :, 1] = contour[:, :, 1] - roi_y1 + dst_y1
                cv2.drawContours(output_image, [adjusted_contour], 0, (0, 255, 0), 2)  # 綠色輪廓
            
            # 添加幀編號和時間信息
            # time_sec = frame_number / self.fps if hasattr(self, 'fps') else frame_number
            # cv2.putText(output_image, f"Frame: {frame_number}", (10, 20), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(output_image, f"Time: {time_sec:.3f}s", (10, 40), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 保存圖像
            filename = f"contour_{frame_number:06d}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, output_image)
            
            # 如果是第一幀，保存設置信息
            if frame_number == 0 or not os.path.exists(os.path.join(save_dir, 'snapshot_info.txt')):
                with open(os.path.join(save_dir, 'snapshot_info.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"視頻文件: {self.video_path}\n")
                    f.write(f"快照大小: {box_size}x{box_size} 像素\n")
                    f.write(f"FPS: {self.fps if hasattr(self, 'fps') else 'N/A'}\n")
                    f.write(f"起始幀: {frame_number}\n")
                    f.write("注意: 所有圖像均以輪廓中心對齊\n")
                    f.write("紅點: 圓心位置\n")
                    f.write("綠線: 檢測到的輪廓\n")
            
            return filepath
            
        except Exception as e:
            import traceback
            print(f"保存輪廓快照時出錯: {e}")
            traceback.print_exc()
            return None

    def create_contour_timelapse(self, snapshot_dir=None, output_file=None, fps=10):
        """
        從保存的輪廓快照創建延時攝影視頻
        
        參數:
        snapshot_dir (str): 快照目錄，默認為輸出目錄下的contour_snapshots
        output_file (str): 輸出視頻文件路徑，默認為輸出目錄下的contour_timelapse.mp4
        fps (int): 輸出視頻的幀率
        
        返回:
        str: 輸出視頻的路徑
        """
        try:
            import os
            import cv2
            import glob
            
            # 設置目錄和輸出文件
            if snapshot_dir is None:
                snapshot_dir = os.path.join(self.output_dir, 'contour_snapshots')
            
            if output_file is None:
                output_file = os.path.join(self.output_dir, 'contour_timelapse.mp4')
            
            # 檢查快照目錄是否存在
            if not os.path.exists(snapshot_dir):
                print(f"錯誤: 快照目錄不存在: {snapshot_dir}")
                return None
            
            # 獲取所有PNG圖像文件並按名稱排序
            image_files = sorted(glob.glob(os.path.join(snapshot_dir, 'contour_*.png')))
            
            if not image_files:
                print(f"錯誤: 在 {snapshot_dir} 中沒有找到輪廓快照")
                return None
            
            # 讀取第一張圖像以獲取尺寸
            first_image = cv2.imread(image_files[0])
            height, width, channels = first_image.shape
            
            # 創建視頻寫入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # 處理所有圖像
            for image_file in image_files:
                img = cv2.imread(image_file)
                video_writer.write(img)
            
            # 釋放視頻寫入器
            video_writer.release()
            
            print(f"延時攝影視頻已創建: {output_file}")
            return output_file
            
        except Exception as e:
            import traceback
            print(f"創建延時攝影視頻時出錯: {e}")
            traceback.print_exc()
            return None

    def toggle_contour_snapshot(self, event=None):
        """
        切換是否保存輪廓快照的功能
        可通過按下'c'鍵觸發
        """
        # 初始化snapshot_enabled屬性（如果尚未設置）
        if not hasattr(self, 'snapshot_enabled'):
            self.snapshot_enabled = False
            
        # 切換狀態
        self.snapshot_enabled = not self.snapshot_enabled
        
        # 輸出當前狀態
        mode_name = "已啟用" if self.snapshot_enabled else "已禁用"
        print(f"輪廓快照功能: {mode_name}")
        
        # 如果啟用了快照功能，確保快照目錄存在
        if self.snapshot_enabled:
            import os
            snapshot_dir = os.path.join(self.output_dir, 'contour_snapshots')
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f"快照將保存到: {snapshot_dir}")
    
    def detect_all_contours(self, frame, previous_position=None):
        """
        檢測幀中的所有環形輪廓，並按從內到外排序
        
        參數:
        frame (numpy.ndarray): 輸入圖像幀
        previous_position (tuple): 上一幀檢測到的位置 (可選)
        
        返回:
        tuple: (center, angle, contours, binary_img, success, contour_info_list, selected_contour)
            contour_info_list 是包含每個輪廓詳細信息的列表
            selected_contour 是選中的輪廓
        """
        # 將圖像轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 應用高斯模糊減少噪聲
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 嘗試多種二值化方法以獲得最佳結果
        binary_methods = [
            # 自適應閾值 - 高斯
            {
                'func': cv2.adaptiveThreshold,
                'params': {
                    'maxValue': 255,
                    'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    'thresholdType': cv2.THRESH_BINARY_INV,
                    'blockSize': 10,
                    'C': 1.5
                }
            },
            # 自適應閾值 - 均值
            {
                'func': cv2.adaptiveThreshold,
                'params': {
                    'maxValue': 255,
                    'adaptiveMethod': cv2.ADAPTIVE_THRESH_MEAN_C,
                    'thresholdType': cv2.THRESH_BINARY_INV,
                    'blockSize': 15,
                    'C': 5
                }
            },
            # Otsu閾值
            {
                'func': lambda img, **kwargs: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            }
        ]
        
        best_contours = []
        best_binary = None
        best_score = 0
        
        # 嘗試每一種二值化方法
        for method in binary_methods:
            try:
                # 應用二值化
                if 'params' in method:
                    thresh = method['func'](blurred, **method['params'])
                else:
                    thresh = method['func'](blurred)
                
                # 應用形態學操作以改善輪廓檢測
                kernel = np.ones((3, 3), np.uint8)
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # 創建二值化的彩色圖像用於顯示
                binary_img = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
                
                # 查找所有輪廓
                contours, hierarchy = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                # 過濾和評估輪廓
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # 過濾過小的輪廓
                    if area > self.detection_area_min:
                        # 計算圓形度
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        # 選擇接近圓形的輪廓
                        if 0.4 < circularity < 1.0:
                            # 計算質心
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                center = (cx, cy)
                                
                                # 保存這個輪廓的詳細信息
                                valid_contours.append({
                                    'contour': contour,
                                    'area': area,
                                    'circularity': circularity,
                                    'center': center
                                })
                
                # 檢查是否找到足夠的輪廓
                if len(valid_contours) >= 3:  # 至少希望找到3個輪廓
                    # 計算此結果的得分
                    method_score = len(valid_contours) * 0.5  # 輪廓數量增加分數
                    
                    # 根據圓形度增加分數
                    avg_circularity = sum(c['circularity'] for c in valid_contours) / len(valid_contours)
                    method_score += avg_circularity * 10
                    
                    # 如果有前一幀位置，檢查是否有輪廓接近該位置
                    if previous_position:
                        min_dist = float('inf')
                        for c in valid_contours:
                            dist = np.sqrt((c['center'][0] - previous_position[0])**2 + 
                                        (c['center'][1] - previous_position[1])**2)
                            min_dist = min(min_dist, dist)
                        
                        # 接近前一位置的結果得分更高
                        proximity_score = max(0, 1 - min_dist / 100)  # 100px為參考距離
                        method_score += proximity_score * 5
                    
                    # 更新最佳結果
                    if method_score > best_score:
                        best_score = method_score
                        best_contours = valid_contours
                        best_binary = binary_img
            except Exception as e:
                if self.debug:
                    print(f"二值化方法出錯: {e}")
        
        # 如果沒有找到有效輪廓，返回失敗
        if not best_contours:
            # 返回空的二值化圖像或原始幀的灰度版本
            if best_binary is None:
                best_binary = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return None, None, [], best_binary, False, [], None
        
        # 檢查輪廓的中心點是否形成一個集群
        # 這對於確保所有輪廓都屬於同一個環很重要
        all_centers = np.array([c['center'] for c in best_contours])
        
        # 計算所有中心點的均值
        mean_center = np.mean(all_centers, axis=0)
        
        # 計算每個中心點到均值的距離
        distances = np.sqrt(np.sum((all_centers - mean_center) ** 2, axis=1))
        
        # 找出距離較近的輪廓（可能屬於同一環）
        # 使用距離閾值，可以根據環的預期大小調整
        max_dist_threshold = max(frame.shape) * 0.1  # 幀尺寸的10%作為閾值
        coherent_indices = np.where(distances < max_dist_threshold)[0]
        
        # 如果沒有足夠的一致輪廓，嘗試調整閾值
        if len(coherent_indices) < 3:
            max_dist_threshold = max(frame.shape) * 0.2  # 增加閾值
            coherent_indices = np.where(distances < max_dist_threshold)[0]
        
        # 篩選一致的輪廓
        coherent_contours = [best_contours[i] for i in coherent_indices]
        
        # 根據面積排序輪廓 (從小到大 = 從內到外)
        sorted_contours = sorted(coherent_contours, key=lambda x: x['area'])
        if len(sorted_contours) > 0:
            selected_contour = sorted_contours[-1]['contour']  # 取最大的輪廓
        # 準備返回的結果
        center = mean_center
        angle = None  # 可以從擬合橢圓獲取，暫不設置
        
        # 在二值化圖像上標記所有有效輪廓
        for i, contour_info in enumerate(sorted_contours):
            contour = contour_info['contour']
            
            # 使用不同顏色標記不同輪廓
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 綠、藍、紅、黃
            color = colors[i % len(colors)]
            
            # 畫輪廓
            cv2.drawContours(best_binary, [contour], 0, color, 2)
            
            # 標記編號
            cx, cy = contour_info['center']
            cv2.putText(best_binary, str(i+1), (int(cx), int(cy)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 標記整體中心
        cv2.circle(best_binary, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)
        
        # 選擇第三個輪廓（如果存在）
        selected_contour = None
        if len(sorted_contours) > 2:
            selected_contour = sorted_contours[2]['contour']
            # 如果需要角度，為第三個輪廓擬合橢圓以獲取角度
            if len(selected_contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(selected_contour)
                    angle = ellipse[2]
                    # 在二值化圖像上畫出擬合的橢圓
                    cv2.ellipse(best_binary, ellipse, (255, 255, 255), 1)
                except Exception as e:
                    if self.debug:
                        print(f"擬合橢圓出錯: {e}")
        
        return center, angle, sorted_contours, best_binary, True, sorted_contours, selected_contour

    def estimate_contour_diameter(self, contour):
        """
        估計輪廓的直徑
        
        參數:
        contour (numpy.ndarray): 輪廓點集
        
        返回:
        float: 估計直徑
        """
        # 嘗試擬合橢圓
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                # 使用長軸作為直徑
                diameter = max(ellipse[1])
                return diameter
            except Exception:
                pass  # 繼續使用下面的方法
        
        # 如果無法擬合橢圓，使用最大距離估計
        # 計算質心
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = np.array([cx, cy])
        
        # 計算所有點到中心的最大距離
        max_dist = 0
        for point in contour:
            point = point[0]  # 獲取實際坐標
            dist = np.sqrt(np.sum((point - center) ** 2))
            max_dist = max(max_dist, dist)
        
        # 直徑是最大距離的2倍
        diameter = max_dist * 2
        
        return diameter

    def calculate_radial_points(self, frame, center, contour, num_points=36):
        """
        計算射線與輪廓的精確交點
        """
        if contour is None or len(contour) < 5:
            return []
            
        radial_points = []
        center_x, center_y = center
        
        # 將輪廓轉換為一系列線段
        segments = []
        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i + 1) % len(contour)][0]
            segments.append((p1, p2))
        
        # 對每個角度進行採樣
        for i in range(num_points):
            angle_deg = i * (360 / num_points)
            angle_rad = np.deg2rad(angle_deg)
            
            # 計算從中心點出發的射線方向
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            
            # 射線的遠端點 (確保足夠遠)
            far_x = center_x + dx * max(frame.shape)
            far_y = center_y + dy * max(frame.shape)
            
            # 尋找射線與輪廓線段的交點
            closest_intersection = None
            min_distance = float('inf')
            
            for segment in segments:
                # 線段端點
                (x1, y1), (x2, y2) = segment
                
                # 計算射線與線段的交點
                # 使用參數方程: p1 + t1(p2-p1) = p3 + t2(p4-p3)
                # 其中p1,p2是線段端點，p3,p4是射線端點
                
                # 線段向量
                sx = x2 - x1
                sy = y2 - y1
                
                # 射線向量
                rx = far_x - center_x
                ry = far_y - center_y
                
                # 行列式 (用於判斷平行)
                det = sx * ry - sy * rx
                
                if abs(det) < 1e-8:  # 平行或共線
                    continue
                    
                # 計算交點參數
                t1 = ((center_x - x1) * ry - (center_y - y1) * rx) / det
                t2 = (sx * (center_y - y1) - sy * (center_x - x1)) / det
                
                # 檢查交點是否在線段和射線上
                if 0 <= t1 <= 1 and t2 >= 0:
                    # 計算交點坐標
                    ix = x1 + t1 * sx
                    iy = y1 + t1 * sy
                    
                    # 計算到中心的距離
                    dist = np.sqrt((ix - center_x)**2 + (iy - center_y)**2)
                    
                    # 更新最近的交點
                    if dist < min_distance:
                        min_distance = dist
                        closest_intersection = (int(ix), int(iy))
            
            # 如果找到交點，添加到徑向點列表
            if closest_intersection is not None:
                radial_points.append({
                    'point': closest_intersection,
                    'angle': angle_deg,
                    'distance': min_distance
                })
        
        return radial_points    
    
    def adjust_coordinates_for_roi(self, point, to_absolute=True):
        """
        根據ROI設置調整點坐標
        
        參數:
        point (tuple): 原始坐標點 (x, y)
        to_absolute (bool): True表示從ROI相對坐標轉換為絕對坐標，False表示相反
        
        返回:
        tuple: 調整後的坐標點
        """
        if not self.use_roi or self.roi is None or len(self.roi) != 4:
            return point  # 如果沒有設置ROI，直接返回原始點
        
        x, y = point
        x1, y1, x2, y2 = self.roi
        
        if to_absolute:
            # 從ROI相對坐標轉換為絕對坐標
            return (x + x1, y + y1)
        else:
            # 從絕對坐標轉換為ROI相對坐標
            return (x - x1, y - y1)

def curve_fit(func, xdata, ydata, p0=None):
    """
    簡化版的曲線擬合，用於振動相位分析
    
    參數:
    func (callable): 擬合函數
    xdata (numpy.ndarray): x數據
    ydata (numpy.ndarray): y數據
    p0 (list): 初始參數猜測
    
    返回:
    tuple: (最佳參數, 殘差平方和)
    """
    from scipy.optimize import minimize
    
    if p0 is None:
        # 根據數據特性生成初始猜測
        amplitude = np.max(np.abs(ydata))
        p0 = [amplitude, 0]
    
    # 定義殘差函數
    def residual(params):
        return np.sum((func(xdata, *params) - ydata) ** 2)
    
    # 最小化殘差平方和
    result = minimize(residual, p0)
    
    # 返回最優參數和殘差
    return result.x, result.fun

def select_frame_range(cap, fps, total_frames):
    """
    互動式選擇視頻幀範圍的獨立函數
    
    參數:
    cap (cv2.VideoCapture): 視頻捕獲物件
    fps (float): 幀率
    total_frames (int): 總幀數
    
    返回:
    tuple: (起始幀, 結束幀)
    """
    # 創建窗口
    cv2.namedWindow('Frame Selection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame Selection', 1024, 768)
    
    # 初始化變數
    start_frame = 0
    end_frame = total_frames - 1
    current_position = 0
    is_selecting_start = True  # True: 選擇起始幀, False: 選擇結束幀
    
    print("\n開始選擇視頻幀範圍")
    print(f"總幀數: {total_frames}, FPS: {fps:.2f}")
    print("操作說明:")
    print("  - 左/右方向鍵: 前後移動")
    print("  - PgUp/PgDn: 快速前後移動")
    print("  - 1-9: 跳轉到視頻的 10%-90% 位置")
    print("  - R: 設置當前幀為起始幀")
    print("  - E: 設置當前幀為結束幀")
    print("  - Enter: 確認選擇")
    print("  - ESC: 取消選擇")
    
    while True:
        # 設置當前幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        ret, frame = cap.read()
        if not ret:
            print("無法讀取視頻幀")
            break
        
        # 創建顯示幀
        display_frame = frame.copy()
        
        # 顯示當前幀信息
        current_time = current_position / fps
        # cv2.putText(display_frame, f"幀: {current_position}/{total_frames-1} (時間: {current_time:.2f}秒)", 
                # (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示選擇的起始和結束幀
        # cv2.putText(display_frame, f"起始幀: {start_frame} (時間: {start_frame/fps:.2f}秒)", 
        #         (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
        #         (0, 255, 0) if is_selecting_start else (255, 255, 255), 2)
        
        # cv2.putText(display_frame, f"結束幀: {end_frame} (時間: {end_frame/fps:.2f}秒)", 
        #         (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
        #         (0, 255, 0) if not is_selecting_start else (255, 255, 255), 2)
        
        # 顯示進度條
        progress_width = display_frame.shape[1] - 40
        progress_height = 20
        progress_y = display_frame.shape[0] - 50
        
        # 繪製進度條背景
        cv2.rectangle(display_frame, (20, progress_y), (20 + progress_width, progress_y + progress_height), 
                    (100, 100, 100), -1)
        
        # 計算位置
        start_pos = int(20 + (start_frame / total_frames) * progress_width)
        end_pos = int(20 + (end_frame / total_frames) * progress_width)
        current_pos = int(20 + (current_position / total_frames) * progress_width)
        
        # 繪製選擇範圍
        cv2.rectangle(display_frame, (start_pos, progress_y), (end_pos, progress_y + progress_height), 
                    (0, 200, 0), -1)
        
        # 繪製當前位置指示器
        cv2.line(display_frame, (current_pos, progress_y - 10), (current_pos, progress_y + progress_height + 10), 
                (0, 0, 255), 2)
        
        # 添加操作提示
        # cv2.putText(display_frame, "R: 設置起始幀  E: 設置結束幀  Enter: 確認  ESC: 取消", 
        #         (20, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 顯示幀
        cv2.imshow('Frame Selection', display_frame)
        
        # 處理鍵盤輸入
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            print("取消幀範圍選擇")
            cv2.destroyAllWindows()
            return None, None
            
        elif key == 13:  # Enter
            if start_frame <= end_frame:
                print(f"確認幀範圍: {start_frame} 到 {end_frame}")
                cv2.destroyAllWindows()
                return start_frame, end_frame
            else:
                print("錯誤: 起始幀必須小於等於結束幀")
                
        elif key == ord('r') or key == ord('R'):  # 設置起始幀
            start_frame = current_position
            is_selecting_start = False  # 切換到選擇結束幀
            print(f"設置起始幀: {start_frame} (時間: {start_frame/fps:.2f}秒)")
            
        elif key == ord('e') or key == ord('E'):  # 設置結束幀
            end_frame = current_position
            is_selecting_start = True  # 切換到選擇起始幀
            print(f"設置結束幀: {end_frame} (時間: {end_frame/fps:.2f}秒)")
            
        # 導航
        elif key in [81, ord('a'), 97]:  # 左方向鍵或 'a'
            # 向後移動一幀
            current_position = max(0, current_position - 1)
            
        elif key in [83, ord('d'), 100]:  # 右方向鍵或 'd'
            # 向前移動一幀
            current_position = min(total_frames - 1, current_position + 1)
            
        # 快速導航
        elif key in [ord('s'), 2]:  # PgUp or 's'
            # 快速後退（移動一秒鐘的幀數）
            frames_per_second = int(fps)
            current_position = max(0, current_position - frames_per_second)
            
        elif key in [ord('w'), 3]:  # PgDn or 'w'
            # 快速前進（移動一秒鐘的幀數）
            frames_per_second = int(fps)
            current_position = min(total_frames - 1, current_position + frames_per_second)
        
        # 快速跳轉
        elif key == ord('1'):  # 跳轉到 10% 位置
            current_position = int(total_frames * 0.1)
            
        elif key == ord('2'):  # 跳轉到 20% 位置
            current_position = int(total_frames * 0.2)
            
        elif key == ord('3'):  # 跳轉到 30% 位置
            current_position = int(total_frames * 0.3)
            
        elif key == ord('4'):  # 跳轉到 40% 位置
            current_position = int(total_frames * 0.4)
            
        elif key == ord('5'):  # 跳轉到 50% 位置
            current_position = int(total_frames * 0.5)
            
        elif key == ord('6'):  # 跳轉到 60% 位置
            current_position = int(total_frames * 0.6)
            
        elif key == ord('7'):  # 跳轉到 70% 位置
            current_position = int(total_frames * 0.7)
            
        elif key == ord('8'):  # 跳轉到 80% 位置
            current_position = int(total_frames * 0.8)
            
        elif key == ord('9'):  # 跳轉到 90% 位置
            current_position = int(total_frames * 0.9)
            
        elif key == ord('0'):  # 跳轉到開始
            current_position = 0
            
        elif key == ord('-'):  # 跳轉到結尾
            current_position = total_frames - 1
    
    # 如果執行到這裡，表示用戶關閉了窗口
    cv2.destroyAllWindows()
    return None, None

def save_settings(self, filename=None):
    """
    將所有設定保存到JSON文件中
    
    參數:
    filename (str): 文件名，如果為None，則使用當前時間作為文件名
    
    返回:
    str: 保存的文件路徑
    """
    import json
    from datetime import datetime
    
    # 如果沒有提供文件名，使用時間戳生成
    if filename is None:
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'vibration_settings_{current_datetime}.json'
        
    # 確保文件有.json後綴
    if not filename.endswith('.json'):
        filename += '.json'
        
    # 完整的文件路徑
    filepath = os.path.join(self.output_dir, filename)
    
    # 收集所有設定
    settings = {
        'video_path': self.video_path,
        'pixel_to_cm': self.pixel_to_cm,
        'detection_area_min': self.detection_area_min,
        'play_speed': self.play_speed,
        'use_custom_origin': self.use_custom_origin,
        'origin_x': self.origin_x,
        'origin_y': self.origin_y,
        'use_roi': self.use_roi,
        'apply_perspective': hasattr(self, 'apply_perspective') and getattr(self, 'apply_perspective', False),
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存ROI如果有使用
    if self.use_roi and hasattr(self, 'roi') and self.roi is not None:
        settings['roi'] = list(self.roi)
    
    # 保存最大歷史長度和振動樣本數
    settings['max_history_length'] = self.max_history_length
    settings['max_vibration_samples'] = self.max_vibration_samples
    
    # 保存透視矩陣如果有使用
    if hasattr(self, 'perspective_matrix') and self.perspective_matrix is not None and getattr(self, 'apply_perspective', False):
        # numpy陣列需要特殊處理才能保存到JSON
        settings['perspective_matrix'] = self.perspective_matrix.tolist()
        settings['warped_width'] = self.warped_width
        settings['warped_height'] = self.warped_height
    
    # 寫入JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
        
    print(f"設定已保存到: {filepath}")
    return filepath

def load_settings(self, filepath):
    """
    從JSON文件加載設定
    
    參數:
    filepath (str): 設定文件的完整路徑
    
    返回:
    bool: 是否成功加載設定
    """
    try:
        import json
        import numpy as np
        
        # 檢查文件是否存在
        if not os.path.exists(filepath):
            print(f"錯誤: 設定文件不存在: {filepath}")
            return False
            
        # 讀取JSON文件
        with open(filepath, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        # 應用基本設定
        if 'pixel_to_cm' in settings:
            self.pixel_to_cm = settings['pixel_to_cm']
            print(f"已加載比例尺: 1像素 = {self.pixel_to_cm:.4f} cm")
            
        if 'detection_area_min' in settings:
            self.detection_area_min = settings['detection_area_min']
            
        if 'play_speed' in settings:
            self.play_speed = settings['play_speed']
            
        if 'max_history_length' in settings:
            self.max_history_length = settings['max_history_length']
            
        if 'max_vibration_samples' in settings:
            self.max_vibration_samples = settings['max_vibration_samples']
            
        # 應用自定義座標原點
        if 'use_custom_origin' in settings and settings['use_custom_origin']:
            self.use_custom_origin = True
            self.origin_x = settings['origin_x']
            self.origin_y = settings['origin_y']
            print(f"已加載自定義座標原點: ({self.origin_x}, {self.origin_y})")
        
        # 應用ROI設定
        if 'use_roi' in settings and settings['use_roi'] and 'roi' in settings:
            self.use_roi = True
            self.roi = settings['roi']
            print(f"已加載ROI: {self.roi}")
        
        # 應用透視變換設定
        if 'apply_perspective' in settings and settings['apply_perspective'] and 'perspective_matrix' in settings:
            self.apply_perspective = True
            # 將列表轉換回numpy陣列
            self.perspective_matrix = np.array(settings['perspective_matrix'], dtype=np.float32)
            self.warped_width = settings['warped_width']
            self.warped_height = settings['warped_height']
            print(f"已加載透視變換矩陣和尺寸設定: {self.warped_width}x{self.warped_height}")
        
        print(f"成功從 {filepath} 加載設定")
        # 寫入加載日志
        log_file = os.path.join(self.output_dir, 'settings_load_log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{current_time}] 加載設定文件: {filepath}\n")
            
        return True
        
    except Exception as e:
        import traceback
        print(f"加載設定時發生錯誤: {e}")
        traceback.print_exc()
        return False

def main():
    """主函數，處理命令行參數並運行程式"""
    parser = argparse.ArgumentParser(description='圓環振動模態分析程式')
    parser.add_argument('video', help='影片文件路徑')
    parser.add_argument('--output-dir', help='輸出目錄')
    parser.add_argument('--debug', action='store_true', help='開啟調試模式')
    parser.add_argument('--min-area', type=int, default=100, help='最小檢測面積 (像素)')
    parser.add_argument('--speed', type=float, default=1.0, help='初始播放速度 (0.1-5.0之間)')
    parser.add_argument('--calibrate', action='store_true', help='進行比例尺校正')
    parser.add_argument('--roi', action='store_true', help='選擇感興趣區域')
    parser.add_argument('--origin', action='store_true', help='設定自訂座標原點')  # 新增參數
    parser.add_argument('--frame-range', action='store_true', help='選擇處理幀範圍')
    parser.add_argument('--start-frame', type=int, help='起始幀編號')
    parser.add_argument('--end-frame', type=int, help='結束幀編號')
    parser.add_argument('--no-auto-record', action='store_true', help='禁用自動記錄模式')
    parser.add_argument('--snapshot', action='store_true', help='啟用輪廓快照功能（自動保存對齊的輪廓圖像）')
    parser.add_argument('--snapshot-fps', type=int, default=10, help='輪廓快照視頻的幀率 (預設: 10)')

    # 設定管理相關參數
    parser.add_argument('--apply-perspective', action='store_true', help='進行透視校正')
    parser.add_argument('--only-calibrate', action='store_true', help='僅進行校正，不執行振動分析')
    parser.add_argument('--save-settings', nargs='?', const='auto', help='保存設定到檔案 (不提供值則自動生成檔名)')
    parser.add_argument('--load-settings', help='從檔案載入設定')
    parser.add_argument('--save-after-calibration', action='store_true', help='完成校準後自動保存設定')
    parser.add_argument('--calibrate-all', action='store_true', help='連續執行所有校正 (透視、ROI、比例尺、原點)')
    parser.add_argument('--settings-only', action='store_true', help='僅調整和保存設定，不執行振動分析')
    
    # 細環追蹤相關參數
    parser.add_argument('--thin-ring', action='store_true', help='啟用細環專用分析模式')
    parser.add_argument('--auto-detect-ring', action='store_true', help='自動偵測環的類型並選擇合適的分析方法')
    parser.add_argument('--min-thin-area', type=int, default=10, help='細環模式的最小檢測面積 (像素)')
    parser.add_argument('--thin-ring-threshold', type=float, default=1.5, help='判定為細環的面積比例閾值')
    parser.add_argument('--multi-layer', action='store_true', help='針對多層漸層進行優化')
    parser.add_argument('--enhanced-edges', action='store_true', help='使用增強的邊緣檢測')
    
    parser.add_argument('--radial-points', type=int, default=24, help='徑向追蹤點數量 (預設: 24)')

    # 在初始化分析器後設置


    args = parser.parse_args()
    
    
    try:
        # 初始化分析器
        analyzer = RingVibrationAnalyzer(
            args.video,
            output_dir=args.output_dir,
            debug=args.debug,
            detection_area_min=args.min_area,
            play_speed=args.speed,
            auto_record=not args.no_auto_record
        )
        
        # 將save_settings和load_settings方法添加到分析器
        analyzer.save_settings = save_settings.__get__(analyzer, RingVibrationAnalyzer)
        analyzer.load_settings = load_settings.__get__(analyzer, RingVibrationAnalyzer)
        

        analyzer.snapshot_enabled = args.snapshot
        if args.snapshot:
            print(f"已啟用輪廓快照功能，將保存對齊的輪廓圖像")
        # 設定透視校正屬性
        analyzer.apply_perspective = args.apply_perspective
        analyzer.num_radial_points = args.radial_points
        # 設置細環相關參數
        analyzer.thin_ring_mode = args.thin_ring
        analyzer.auto_detect_ring_type = args.auto_detect_ring
        
        # 如果指定了細環的最小面積和閾值，設置細環參數
        if args.thin_ring:
            if not hasattr(analyzer, 'thin_ring_params'):
                analyzer.thin_ring_params = {}
                
            analyzer.thin_ring_params.update({
                'min_area': args.min_thin_area,
                'area_ratio_target': args.thin_ring_threshold,
                'multi_layer': args.multi_layer,
                'enhanced_edges': args.enhanced_edges
            })
        
        # 如果指定了加載設定文件，則首先加載設定
        if args.load_settings:
            print(f"正在從 {args.load_settings} 加載設定...")
            if analyzer.load_settings(args.load_settings):
                print("設定加載成功")
            else:
                print("設定加載失敗，將使用命令行參數和默認值")
        
        # 記錄是否執行了任何校正，用於決定是否自動保存
        did_calibration = False
        
        # 檢查是否要執行所有校正
        if args.calibrate_all:
            args.apply_perspective = True
            args.roi = True
            args.calibrate = True
            args.origin = True
        
        # 是否在校正後自動保存設定
        auto_save = args.save_after_calibration or args.save_settings or args.settings_only
        
        # 進行透視校正
        if args.apply_perspective:
            print("開始透視校正...")
            perspective_matrix = analyzer.calibrate_perspective()
            if perspective_matrix is not None:
                print(f"透視校正完成")
                did_calibration = True
            else:
                print("透視校正未完成，將使用原始視頻")
        
        # 進行比例尺校正
        if args.calibrate:
            print("開始比例尺校正...")
            scale = analyzer.calibrate_scale()
            if scale:
                print(f"比例尺校正結果: 1像素 = {scale:.4f} cm")
                did_calibration = True
            else:
                print("比例尺校正未完成，將使用默認比例尺")
        
        # 選擇感興趣區域
        if args.roi:
            print("選擇感興趣區域...")
            roi = analyzer.select_roi()
            if roi:
                print(f"設定ROI: {roi}")
                did_calibration = True
            else:
                print("ROI選擇未完成，將使用完整視頻")
        
        # 設定自訂座標原點
        if args.origin:
            print("開始座標原點校正...")
            origin = analyzer.calibrate_origin()
            if origin:
                print(f"座標原點已設定為: ({analyzer.origin_x}, {analyzer.origin_y})")
                did_calibration = True
            else:
                print("座標原點校正未完成，將使用默認原點")
        
        # 校準完成後自動保存設定，或在設定模式下強制保存
        if (did_calibration and auto_save) or args.settings_only:
            print("正在保存所有校準設定...")
            filename = None
            if isinstance(args.save_settings, str) and args.save_settings != 'auto':
                filename = args.save_settings
                
            settings_file = analyzer.save_settings(filename)
            print(f"全部校準設定已保存到: {settings_file}")
        
        # 如果是僅校準模式或僅設定模式，則退出
        if args.only_calibrate or args.settings_only:
            if args.only_calibrate:
                print("僅校準模式，完成所有校準後退出")
            else:
                print("僅設定模式，保存設定後退出")
            analyzer.cap.release()
            cv2.destroyAllWindows()
            return
        
        # 選擇處理幀範圍
        start_frame = args.start_frame
        end_frame = args.end_frame
        
        if args.frame_range:
            print("選擇處理幀範圍...")
            selected_start, selected_end = select_frame_range(analyzer.cap, analyzer.fps, analyzer.total_frames)
            
            if selected_start is not None:
                start_frame = selected_start
            if selected_end is not None:
                end_frame = selected_end
        # 處理影片並分析振動
        analyzer.process_vibration(start_frame=start_frame, end_frame=end_frame)
    except KeyboardInterrupt:
        print("\n程序被使用者中斷")
        
        # 即使被中斷，仍嘗試保存已完成的設定
        if 'analyzer' in locals() and 'did_calibration' in locals() and did_calibration and auto_save:
            try:
                print("保存已完成的校準設定...")
                analyzer.save_settings()
                print("設定已保存")
            except Exception as e:
                print(f"保存設定時發生錯誤: {e}")
    except Exception as e:
        print(f"發生錯誤: {e}")
        traceback.print_exc()
    finally:
        # 確保資源被釋放
        if 'analyzer' in locals():
            try:
                analyzer.cap.release()
                cv2.destroyAllWindows()
            except:
                pass
if __name__ == "__main__":
    main()
