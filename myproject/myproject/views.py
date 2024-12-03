import os
import uuid
import cv2
import numpy as np
import tempfile
from django.conf import settings
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from django.core.exceptions import ValidationError

# 動画の初期化
def initialize_video_capture(video_file):
    # ユニークなファイル名をUUIDで生成
    unique_filename = str(uuid.uuid4()) + '.mp4'
    temp_file_path = os.path.join(tempfile.gettempdir(), unique_filename)
    
    # バイト列を一時ファイルに書き込む
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(video_file.read())

    # 動画キャプチャの初期化
    cap = cv2.VideoCapture(temp_file_path)
    ret, frame1 = cap.read()
    if not ret:
        print("動画の読み込みに失敗しました。")
        cap.release()
        os.remove(temp_file_path)  # 一時ファイルを削除
        return None, None, None
    return cap, cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), temp_file_path

# ヒストグラム平坦化
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# 平均フィルタの適用
def apply_average_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

# 光フロー計算
def calculate_optical_flow(prvs, next, roi):
    x, y, w, h = roi
    roi_prvs = prvs[y:y+h, x:x+w]
    roi_next = next[y:y+h, x:x+w]

    flow = cv2.calcOpticalFlowFarneback(roi_prvs, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx**2 + dy**2)

    return flow, mag

# 矢印描画
def draw_arrows(frame, flow, roi, mag, color=(0, 255, 0), scale=10, arrow_length=5, min_magnitude=1):
    x, y, w, h = roi
    for y_pos in range(0, h, scale):
        for x_pos in range(0, w, scale):
            if mag[y_pos, x_pos] > min_magnitude:
                dx, dy = flow[y_pos, x_pos]
                length = min(mag[y_pos, x_pos], arrow_length)
                end_x = int(x + x_pos + length * dx)
                end_y = int(y + y_pos + length * dy)

                thickness = 2
                cv2.arrowedLine(frame, (x + x_pos, y + y_pos), (end_x, end_y), color, thickness, tipLength=0.05)

# ホームページ
def index(request):
    return render(request, 'index.html')

# 動画アップロード処理
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # アップロードファイルの検証
        allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm']
        file_extension = os.path.splitext(video_file.name)[1].lower()

        if file_extension not in allowed_extensions:
            error_message = f"許可されていない拡張子です。アップロード可能な拡張子: {', '.join(allowed_extensions)}"
            print(error_message)  # コンソールに出力
            return render(request, 'error.html', {
                'error_message': error_message
            })
        
        # アップロードサイズ制限チェック (10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if video_file.size > max_size:
            error_message = "ファイルサイズが10MBを超えています。別の動画をアップロードしてください。"
            print(error_message)  # コンソールに出力
            return render(request, 'error.html', {
                'error_message': error_message
            })

        # 動画ファイルの初期化
        cap, prvs, temp_file_name = initialize_video_capture(video_file)
        if cap is None:
            error_message = "動画の読み込みに失敗しました。別の動画を試してください。"
            print(error_message)  # コンソールに出力
            return render(request, 'error.html', {
                'error_message': error_message
            })

        frame_count = 0
        total_top_95_percent_movement = 0.0
        pixel_to_distance = 0.0698  # 1ピクセルあたりの距離 (cm)
        roi = (200, 200, 100, 100)

        # 出力動画のユニークなファイル名を生成
        unique_output_filename = str(uuid.uuid4()) + '_output_video.avi'
        output_video_path = os.path.join(settings.MEDIA_ROOT, unique_output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            next_gray = histogram_equalization(next_gray)
            next_gray = apply_average_filter(next_gray)

            flow, mag_full = calculate_optical_flow(prvs, next_gray, roi)
            top_95_percent_movement_px = np.percentile(mag_full[mag_full > 1], 95)
            top_95_percent_movement_distance = top_95_percent_movement_px * pixel_to_distance
            total_top_95_percent_movement += top_95_percent_movement_distance

            frame_count += 1
            print(f"フレーム{frame_count}: 上位95%の移動量: {top_95_percent_movement_distance:.2f} ")

            draw_arrows(frame2, flow, roi, mag_full)

            frame2_with_roi = frame2.copy()
            cv2.rectangle(frame2_with_roi, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

            out.write(frame2_with_roi)

            prvs = next_gray

        print(f"ROIの上位95%の移動距離の合計: {total_top_95_percent_movement:.2f} ")

        cap.release()
        out.release()

        # 一時ファイルの削除
        os.remove(temp_file_name)

        # URL 生成してリダイレクト
        result_url = reverse('result') + f"?total_movement={total_top_95_percent_movement:.2f}&video_url={output_video_path}"
        return redirect(result_url)

    return render(request, 'error.html', {
        'error_message': "動画ファイルが送信されていません。"
    })

# 結果表示ビュー
def result(request):
    total_movement = request.GET.get('total_movement', '0')
    video_url = request.GET.get('video_url', '')
    
    # video_url を相対URLに変換
    if video_url.startswith(settings.MEDIA_ROOT):
        video_url = video_url.replace(settings.MEDIA_ROOT, settings.MEDIA_URL)
    
    return render(request, 'result.html', {
        'total_movement': total_movement,
        'video_url': video_url
    })

