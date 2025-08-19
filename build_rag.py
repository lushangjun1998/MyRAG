import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, models
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

# ————————  加载docs目录下所有txt + docx 文件  ————————
def load_documents_from_folder(folder_path: str):
    txt_loader = DirectoryLoader(
        folder_path, glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True
    )
    docx_loadr = DirectoryLoader(
        folder_path, glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True
    )
    txt_docs = txt_loader.load()
    docx_docs = docx_loadr.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    return splitter.split_documents(txt_docs + docx_docs)


# ————————  使用BGE本地模型构建Embedding  ————————
class MyEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_documents(self, texts):
        return self(texts)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()


def get_embedding_model(model_path: str):
    word_model = models.Transformer(model_path)
    pooling_model = models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_model, pooling_model])
    return MyEmbeddingFunction(model)


# ————————  构建向量数据库 / 加载已有数据库  ————————
def build_or_load_vectordb(docs, persist_directory="chroma_db", embedding=None):
    if os.path.exists(persist_directory):
        print("加载已有 ChromaDB 向量库...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print("构建新向量库并持久化...")
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()
    return vectordb


# ————————  构建 RetrievalQA 问答器  ————————
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    # llm = ChatOpenAI(
    #     model_name="AIx-codellm-completion",
    #     openai_api_base="http://10.8.4.23:36435/v1",
    #     openai_api_key="sk-"
    # )
    # 使用Ollama本地模型
    llm = ChatOllama(
        model="qwen3:8b",  # 本地Ollama部署的模型名称
        base_url="http://localhost:11434",  # Ollama默认端口
        temperature=0.7
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        # templates="""
        # 你是一位电池技术文档标准化专家，请根据提供的参考内容，将用户输入的技术描述转化为符合标准的专业表述。
        # 
        # 参考内容：
        # {context}
        #
        # 用户输入：
        # {question}
        #
        # 要求：
        # 1. 保持原意的准确性
        # 2. 使用标准术语与标准表述
        # 3. 符合技术文档写作规范
        # 4. 输出格式：先给出标准化表述(不要给出来源的文档或标准)，然后简要说明修改原因
        # 5. 如果没有找到相关的参考内容，则你需要根据用户的输入给出相对应的修改建议。如果用户输入已经很标准了，则不需要给出修改建议。
        # """

        template="""
        你是一个严谨的RAG助手。请根据以下提供的上下文信息来回答问题。
        如果上下文信息不足以回答问题或者与上下文无关，请严格回复"对不起，无法回复相关问题", 不要回复其他任何内容
        上下文信息: {context}
        ----------------------
        问题: {question}
        """
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )


if __name__ == '__main__':
    doc_folder = "txt_file"
    embed_model_path = "model_dir/BAAI/bge-large-zh-v1___5"
    chroma_dir = "chroma_qa"

    embedding_model = get_embedding_model(embed_model_path)
    documents = load_documents_from_folder(doc_folder)
    vectordb = build_or_load_vectordb(documents, chroma_dir, embedding_model)
    qa = build_qa_chain(vectordb)

    while True:
        q = input("输入你的问题: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = qa.invoke(q)['result']
        print(f"回答: {ans}\n")
