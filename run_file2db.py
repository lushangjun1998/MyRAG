from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime

embedding_model = SentenceTransformer("./model_dir/BAAI/bge-large-zh-v1___5")
client = chromadb.PersistentClient("./chroma_db")

collection = client.get_or_create_collection(
    name="chroma_test",
    metadata={
        "介绍": "文本对象的向量数据库",
        "创建时间": datetime.now()
    }
)


def file2db(file_path):
    path_list = list(Path(file_path).glob("*.txt"))
    text_list = []
    for path in path_list:
        text = path.read_text(encoding="utf-8")
        text_list.append(text)

    embeddings = embedding_model.encode(text_list)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=text_list
    )

    print(f"向量数据库中的数据量: {collection.count()}")


if __name__ == '__main__':
    file2db("./docs")