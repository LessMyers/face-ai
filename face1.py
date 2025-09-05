import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import face_recognition
import cv2
import numpy as np

# 加载身份证正面照片
id_image = face_recognition.load_image_file("id_card_front.jpg")
id_face_encoding = face_recognition.face_encodings(id_image)[0]

# 打开视频文件或视频流
video_capture = cv2.VideoCapture("video.mp4")

# 获取视频的帧率（fps），并计算 5 秒钟应该提取的帧数
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_count = int(fps * 5)  # 获取 5 秒的视频帧数

# 初始化变量，保存最佳匹配分数和对应的帧
best_score = 0
best_frame_number = 0
best_frame = None

processed_frames = 0

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break  # 如果视频读取完毕，跳出循环

    # 每隔一定帧提取一帧，避免过多处理（5 秒内的视频）
    if processed_frames % int(fps) == 0:  # 每秒提取一帧
        # 转换帧为 RGB 格式（face_recognition 需要 RGB 格式）
        frame_rgb = frame[:, :, ::-1]
        
        # 获取当前帧中的所有人脸位置
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        
        for face_encoding in face_encodings:
            # 比较身份证照片与视频帧中的人脸相似度
            matches = face_recognition.compare_faces([id_face_encoding], face_encoding)
            
            # 计算人脸之间的距离，距离越小越相似
            face_distances = face_recognition.face_distance([id_face_encoding], face_encoding)
            similarity_score = 1 - face_distances[0]  # 将距离转换为相似度分数（1表示完全匹配）

            if similarity_score > best_score:  # 如果当前帧的相似度分数更高，更新最佳匹配
                best_score = similarity_score
                best_frame_number = processed_frames  # 保存帧的编号（或者你也可以保存时间戳）
                best_frame = frame  # 保存最佳匹配的帧
        
    processed_frames += 1
    
    # 假设处理 5 秒的视频，所以可以选择处理 5 秒后退出
    if processed_frames >= frame_count:
        break

# 释放视频资源
video_capture.release()

# 输出最佳匹配的结果
if best_frame is not None:
    print(f"最佳匹配的分数: {best_score:.2f} 在第 {best_frame_number} 帧")
    
    # 显示最佳匹配的帧
    cv2.imshow("Best Matching Frame", best_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("没有找到匹配的人脸。")
