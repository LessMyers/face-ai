import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import face_recognition
import numpy as np

# 初始化一个人脸库（字典），存储人脸名称和特征向量
face_database = {}  # 格式： {name: face_encoding}

# 计算欧几里得距离，用于判断两个特征向量之间的相似度
def euclidean_distance(enc1, enc2):
    return np.linalg.norm(enc1 - enc2)

# 查找最相似的三张人脸
def find_similar_faces(face_encoding, face_database, top_n=3, threshold=0.6):
    distances = []
    for name, encoding in face_database.items():
        distance = euclidean_distance(face_encoding, encoding)
        distances.append((name, distance))
    
    # 按照距离排序，距离越小越相似
    distances.sort(key=lambda x: x[1])
    
    # 返回最相似的 top_n 张人脸（如果相似度超过阈值）
    similar_faces = [face for face in distances[:top_n] if face[1] < threshold]
    return similar_faces

# 将新的面部特征加入到人脸库
def add_face_to_database(name, face_encoding, face_database):
    face_database[name] = face_encoding
    print(f"已将{name}加入人脸库！")

# 模拟从图片或视频提取一张新的面部特征
def process_new_face(image_path, name, face_database, threshold=0.6):
    # 加载并提取新的人脸特征
    new_image = face_recognition.load_image_file(image_path)
    new_face_encoding = face_recognition.face_encodings(new_image)[0]
    
    # 查找最相似的三张人脸
    similar_faces = find_similar_faces(new_face_encoding, face_database)
    
    if similar_faces:
        print("找到相似的人脸：")
        for face in similar_faces:
            print(f"人脸名称: {face[0]}, 相似度分数: {1 - face[1]:.2f}")
    else:
        print("没有找到相似的人脸，添加新的人脸到数据库中...")
        add_face_to_database(name, new_face_encoding, face_database)

# 示例：模拟从一张新图片中提取面部特征并与人脸库对比
process_new_face("new_face_image.jpg", "张三", face_database)
