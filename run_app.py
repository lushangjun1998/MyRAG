import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from build_rag import get_embedding_model, load_documents_from_folder, build_qa_chain, build_or_load_vectordb
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import markdown

app = FastAPI(
    title="RAG",
    description="电池策略LLM(内测版)",
    version="1.0.0"
)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 添加CORES配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源（生产环境应限制为具体域名）
    allow_credentials=True,
    allow_methods=["*"], # 允许所有方法（GET/POST等）
    allow_headers=["*"], # 允许所有请求头
)


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    status: str


def init_rag_system():
    doc_folder = "txt_file"
    embed_model_path = "model_dir/BAAI/bge-large-zh-v1___5"
    chroma_dir = "chroma_qa"

    embedding_model = get_embedding_model(embed_model_path)
    if not os.path.exists(chroma_dir):
        documents = load_documents_from_folder(doc_folder)
    else:
        documents = None
    vectordb = build_or_load_vectordb(documents, chroma_dir, embedding_model)
    qa = build_qa_chain(vectordb)
    return qa


# 全局QA链
qa_chain = init_rag_system()

# 前端页面路由
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # result = await qa_chain.ainvoke(request.question)['result'] # TODO: 这里加入await和ainvoke会报错
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_chain.invoke(request.question)['result']
        )
        html_content = markdown.markdown(result)  # 转换为HTML
        return {
            "question": request.question,
            "answer": html_content,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时出错: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=8000)
