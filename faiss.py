import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import face_recognition
import faiss
import numpy as np

# 假设我们已有的 10 万张人脸的编码，存储在一个列表中
# 我们用随机数据做示范，实际上这里应该是通过 face_recognition 提取的人脸特征
# 假设每张人脸特征是 128 维的
database = np.random.rand(100000, 128).astype('float32')  # 100000 张人脸的特征，128维

# 创建 Faiss 索引器，使用 L2 距离
index = faiss.IndexFlatL2(128)  # 128 是每个特征的维度
index.add(database)  # 将数据库中的人脸特征添加到索引中

# 假设我们现在有一个新的人脸特征进行查询
new_face_image = face_recognition.load_image_file("new_face.jpg")
new_face_encoding = face_recognition.face_encodings(new_face_image)[0].astype('float32')

# 使用 Faiss 查询最相似的 3 张人脸
k = 3  # 我们要找到最相似的 3 张人脸
distances, indices = index.search(np.array([new_face_encoding]), k)

# 输出最相似的 3 张人脸及其相似度（距离）
for i in range(k):
    print(f"匹配人脸 {indices[0][i]}，相似度分数：{1 - distances[0][i]:.2f}")
