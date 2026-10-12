import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_pipeline import load_pdf, init_vector_db, get_retriever

load_dotenv()
app = FastAPI(title="PDF RAG DeepSeek")
# 新增temperature=0.2，越低越严谨，减少编造
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)

# 优化后的提示词，强制只能用文档内容回答
def format_docs(docs):
    return "\n\n【文档片段】\n".join([d.page_content for d in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是文档问答助手，**只能使用下方【文档片段】中的信息回答用户问题**。
1. 如果文档片段没有对应答案，直接回复：文档中没有找到相关内容，不要编造任何信息。
2. 禁止使用文档以外的知识进行回答。
3. 回答尽量简洁，优先引用原文要点。
4. 不要自己额外拓展无关内容。

【文档片段】
{context}
"""),
    ("human", "{question}"),
])

retriever = get_retriever()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    save_path = os.path.join("uploads", file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    docs = load_pdf(save_path)
    init_vector_db(docs)
    return {"msg": "PDF上传并向量化完成，可以开始提问"}


@app.post("/ask")
async def ask(question: str):
    answer = rag_chain.invoke(question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
