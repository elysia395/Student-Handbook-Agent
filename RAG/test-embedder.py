# test_embedder.py
from RAG.Embedder import Embedder

print("开始加载模型...")
embedder = Embedder()  # 这里会触发下载
print("模型加载完成！")

# 测试一下
text = "Hello World"
embedding = embedder.model.encode(text)
print(f"向量维度: {embedding.shape}")
