
##pip install langchain langchain-openai pypdf faiss-cpu python-dotenv


import os
from dotenv import load_dotenv

# 1. LangChain 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def setup_rag_chain(pdf_path: str):
    """
    PDF 파일로부터 RAG 체인을 생성합니다.
    이 함수가 RAG의 모든 핵심 단계를 포함합니다.
    """
    
    # --- 1. Load (로드) ---
    # PyPDFLoader를 사용해 PDF 파일 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ PDF 로드 완료. 총 {len(documents)} 페이지.")

    # --- 2. Split (분할) ---
    # 문서를 의미 있는 단위(Chunk)로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"✅ 문서 분할 완료. 총 {len(splits)} 조각(chunks).")

    # --- 3. Embed & Store (임베딩 & 저장) ---
    # 텍스트 조각을 벡터로 변환(임베딩)하고 FAISS 벡터 스토어에 저장
    print("⏳ 임베딩 및 벡터 스토어 생성 중...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("✅ 벡터 스토어 생성 완료.")

    # --- 4. Retrieve & Generate (검색 & 생성) ---
    # (1) LLM 모델 정의
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # (2) Retriever (검색기) 정의
    # vectorstore.as_retriever() : 질문과 유사한 벡터를 DB에서 검색하는 객체
    retriever = vectorstore.as_retriever()

    # (3) RAG Chain (RetrievalQA) 생성
    # RetrievalQA 체인은 '검색기'와 'LLM'을 결합하여 RAG를 완성합니다.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # "stuff": 검색된 문서를 모두 프롬프트에 넣는 방식
        retriever=retriever,
        return_source_documents=True # 답변의 근거가 된 원본 문서도 반환
    )
    
    print("✅ RAG 체인 생성 완료.")
    return qa_chain

def main():
    # API 키 로드
    load_dotenv()
    
    # 처리할 PDF 파일 경로
    pdf_file_path = "sample.pdf" # 여기에 준비한 PDF 파일 이름을 넣으세요.

    if not os.path.exists(pdf_file_path):
        print(f"오류: '{pdf_file_path}' 파일을 찾을 수 없습니다.")
        return

    # 1. PDF 파일로 RAG 체인 생성
    qa_chain = setup_rag_chain(pdf_file_path)

    # 2. 터미널에서 무한 루프로 질문 받기
    print("\n--- PDF 챗봇 시작 (종료: 'exit') ---")
    while True:
        query = input("\n[질문] ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue
            
        # 3. RAG 체인 실행 (질문)
        print("⏳ 답변 생성 중...")
        try:
            result = qa_chain.invoke({"query": query})
            
            # 결과 출력
            print("\n[답변]")
            print(result["result"])
            
            # (선택) 근거가 된 문서 조각(Source) 출력
            print("\n[근거 문서 (Sources)]")
            for i, doc in enumerate(result["source_documents"]):
                print(f"--- (Source {i+1}) ---")
                print(f"내용: {doc.page_content[:150]}...")
                print(f"출처: {doc.metadata.get('page', 'N/A')} 페이지\n")
                
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()