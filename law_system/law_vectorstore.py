"""
law_vectorstore.py - 부동산 법령 벡터스토어 관리 시스템
1. 초기: 법령 PDF들을 학습하여 벡터스토어 생성
2. 이후: 학습된 벡터스토어를 로드하여 검색 기능 제공
3. 다른 AI 모델들의 기반 라이브러리 역할
4. 임베딩 모델 캐싱으로 로딩 속도 대폭 개선
"""

import os
import json
import pdfplumber
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from config.logger_config import get_logger
logger = get_logger(__name__)

# SQLite3 버전 문제 해결을 위한 monkey patch
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # pysqlite3가 없으면 기본 sqlite3 사용
    pass

# LangChain 관련 imports (최신 버전)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 환경변수 로드
load_dotenv()

# --- Path Setup ---
try:
    # 현재 파일의 디렉토리 (C:\LLM\law_system)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리 (C:\LLM)
    PROJECT_ROOT = os.path.dirname(current_dir)
except NameError:
    # 대화형 환경 등에서 __file__이 정의되지 않은 경우를 대비
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 기본 데이터 및 벡터스토어 경로를 절대 경로로 설정
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "law_docs")
DEFAULT_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")
# --- End Path Setup ---

_cached_embeddings = None

def get_cached_embeddings():
    """캐시된 임베딩 모델 반환 (최초 1회만 로딩)"""
    global _cached_embeddings
    
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    return _cached_embeddings

class LawVectorStore:
    """부동산 법령 학습 및 검색 시스템"""

    def __init__(self, persist_directory: str = "vectorstore", collection_name: str = "rental_law"):
        """
        초기화

        Args:
            persist_directory: 벡터 DB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings_model = None
        self.vectorstore = None

        self._setup_embeddings()

    def _setup_embeddings(self):
        """임베딩 모델 설정 (캐시된 버전 사용)"""
        self.embeddings_model = get_cached_embeddings()

    def extract_center_text_from_pdf(self, file_path: str) -> List[Document]:
        """
        PDF에서 중앙 텍스트를 추출하여 Document 객체로 변환

        Args:
            file_path: PDF 파일 경로

        Returns:
            Document 객체 리스트
        """
        documents = []
        filename = os.path.basename(file_path)
        logger.info(f"📄 PDF 파일 처리 시작: {filename}")
        
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"   - 총 페이지 수: {len(pdf.pages)}")
            for page_num, page in enumerate(pdf.pages):
                page_width = page.width
                page_height = page.height

                words = page.extract_words()
                if not words:
                    continue

                # 여백 기준 정의
                left = page_width * 0.05
                right = page_width * 0.95
                top_margin = 60
                bottom_margin = page_height - 60

                center_words = [
                    word['text']
                    for word in words
                    if left <= word['x0'] <= right and top_margin <= word['top'] <= bottom_margin
                ]

                if center_words:
                    joined_text = " ".join(center_words)

                    # 조항 번호 추출
                    article_match = re.search(
                        r"(제\d+조(\의\d+)?(?:의\d+)?)(?=\s|\()", joined_text)
                    article_title = article_match.group(
                        1) if article_match else None

                    documents.append(Document(
                        page_content=joined_text,
                        metadata={
                            "source": f"{file_path} - Page {page.page_number + 1}",
                            "file_path": file_path,
                            "page_number": page.page_number + 1,
                            "article": article_title,
                            "law": self._extract_law_name(file_path)
                        }
                    ))
                    
                    if article_title:
                        logger.info(f"   - 페이지 {page_num + 1}: {article_title} 조항 추출됨")
                    else:
                        logger.info(f"   - 페이지 {page_num + 1}: 텍스트 추출됨 (조항 번호 없음)")
        
        logger.info(f"   ✅ {filename}에서 총 {len(documents)}개 문서 추출 완료")
        return documents

    def _extract_law_name(self, file_path: str) -> str:
        """파일명에서 법령명 추출 (확장자 제거)"""
        filename = os.path.basename(file_path)
        return filename.replace(".pdf", "")

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        디렉토리 내 모든 PDF 파일에서 문서를 로드

        Args:
            directory_path: PDF 파일들이 있는 디렉토리 경로

        Returns:
            Document 객체 리스트
        """
        all_documents = []

        if not os.path.exists(directory_path):
            logger.warning(f"⚠️ 디렉토리가 존재하지 않습니다: {directory_path}")
            return all_documents

        logger.info(f"📁 디렉토리 스캔 시작: {directory_path}")
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        logger.info(f"   - 발견된 PDF 파일 수: {len(pdf_files)}개")
        
        for idx, filename in enumerate(pdf_files, 1):
            logger.info(f"\n[{idx}/{len(pdf_files)}] 처리 중...")
            file_path = os.path.join(directory_path, filename)
            documents = self.extract_center_text_from_pdf(file_path)
            all_documents.extend(documents)

        logger.info(f"\n✅ 전체 문서 로드 완료: 총 {len(all_documents)}개 문서")
        return all_documents

    def create_vectorstore(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 50):
        """
        벡터 저장소 생성

        Args:
            documents: Document 객체 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침 크기
        """
        if not documents:
            logger.warning("⚠️ 문서가 없어 벡터스토어를 생성할 수 없습니다.")
            return

        logger.info("\n🔨 벡터스토어 생성 시작")
        logger.info(f"   - 입력 문서 수: {len(documents)}개")
        logger.info(f"   - 청크 크기: {chunk_size}")
        logger.info(f"   - 청크 겹침: {chunk_overlap}")

        # 저장 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"   - 저장 경로: {self.persist_directory}")

        # 문서 분할
        logger.info(f"\n📝 문서 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_documents = text_splitter.split_documents(documents)
        logger.info(f"   - 분할된 청크 수: {len(split_documents)}개")

        # 벡터 저장소 생성
        logger.info(f"\n🚀 벡터 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
        self.vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embeddings_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        logger.info(f"   ✅ 벡터스토어 생성 완료!")

        # 처리 상태 기록
        processed_states = {}
        file_counts = {}
        file_chunk_counts = {}

        # 파일별 문서 수와 청크 수 계산
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            if file_path:
                filename = os.path.basename(file_path)
                file_counts[filename] = file_counts.get(filename, 0) + 1

        for chunk in split_documents:
            file_path = chunk.metadata.get('file_path', '')
            if file_path:
                filename = os.path.basename(file_path)
                file_chunk_counts[filename] = file_chunk_counts.get(
                    filename, 0) + 1

        # 상태 저장
        logger.info(f"\n📊 파일별 처리 통계:")
        for filename, count in file_counts.items():
            chunk_count = file_chunk_counts.get(filename, 0)
            processed_states[filename] = {
                "vectorstore_processed": True,
                "document_count": count,
                "chunk_count": chunk_count
            }
            logger.info(f"   - {filename}: {count}개 문서 → {chunk_count}개 청크")

        self._save_processed_states(processed_states)
        logger.info(f"\n✅ 처리 상태 저장 완료")

    def load_existing_vectorstore(self) -> bool:
        """
        기존 벡터 저장소 로드

        Returns:
            로드 성공 여부
        """
        logger.info(f"\n🔍 기존 벡터스토어 확인 중...")
        logger.info(f"   - 확인 경로: {self.persist_directory}")
        
        if os.path.exists(self.persist_directory):
            try:
                logger.info(f"   - 벡터스토어 디렉토리 발견!")
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                
                # 저장된 문서 수 확인
                try:
                    doc_count = self.vectorstore._collection.count()
                    logger.info(f"   ✅ 벡터스토어 로드 성공! (저장된 문서: {doc_count}개)")
                except:
                    logger.info(f"   ✅ 벡터스토어 로드 성공!")
                
                return True
            except Exception as e:
                logger.error(f"   ❌ 벡터스토어 로드 실패: {e}")
                return False
        else:
            logger.info(f"   - 기존 벡터스토어가 없습니다. 새로 생성 필요.")
            return False

    def search_relevant_law(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        관련 법령 조항 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            검색된 법령 정보 리스트
        """
        if not self.vectorstore:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")

        results = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": result.page_content,
                "metadata": result.metadata,
                "law_name": result.metadata.get("law", ""),
                "article": result.metadata.get("article", ""),
                "page_number": result.metadata.get("page_number", "")
            }
            for result in results
        ]

    def search_by_article(self, article_number: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 조항 번호로 검색

        Args:
            article_number: 조항 번호 (예: "제6조", "제7조의2")
            k: 반환할 문서 수

        Returns:
            해당 조항 정보
        """
        if not self.vectorstore:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")

        results = self.vectorstore.similarity_search(article_number, k=k)

        # 조항 번호가 정확히 매칭되는 것들 우선 반환
        exact_matches = [
            result for result in results
            if result.metadata.get("article") == article_number
        ]

        if exact_matches:
            results = exact_matches

        return [
            {
                "content": result.page_content,
                "metadata": result.metadata,
                "law_name": result.metadata.get("law", ""),
                "article": result.metadata.get("article", ""),
                "page_number": result.metadata.get("page_number", "")
            }
            for result in results
        ]

    def search_by_keywords(self, keywords: List[str], k: int = 4) -> List[Dict[str, Any]]:
        """
        키워드 리스트로 검색

        Args:
            keywords: 키워드 리스트
            k: 반환할 문서 수

        Returns:
            검색된 법령 정보 리스트
        """
        query = " ".join(keywords)
        return self.search_relevant_law(query, k)

    def _get_processed_states_path(self) -> str:
        """처리 상태 JSON 파일 경로 반환"""
        return os.path.join(self.persist_directory, "processed_states.json")

    def _load_processed_states(self) -> dict:
        """처리 상태 JSON 파일 로드"""
        states_path = self._get_processed_states_path()

        if os.path.exists(states_path):
            try:
                with open(states_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                return {}
        return {}

    def _save_processed_states(self, states: dict):
        """처리 상태 JSON 파일 저장"""
        states_path = self._get_processed_states_path()

        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(states_path), exist_ok=True)

        try:
            with open(states_path, "w", encoding="utf-8") as f:
                json.dump(states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

    def add_new_documents(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 50):
        """
        새로 추가된 PDF 파일들만 벡터스토어에 추가

        Args:
            directory_path: PDF 파일들이 있는 디렉토리 경로
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침 크기
        """
        if not self.vectorstore:
            logger.warning("⚠️ 벡터스토어가 로드되지 않아 새 문서를 추가할 수 없습니다.")
            return

        if not os.path.exists(directory_path):
            logger.warning(f"⚠️ 디렉토리가 존재하지 않습니다: {directory_path}")
            return

        logger.info(f"\n🔄 새로운 PDF 파일 확인 중...")
        logger.info(f"   - 디렉토리: {directory_path}")

        # 모든 PDF 파일 찾기
        all_pdf_files = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                all_pdf_files.append(filename)

        if not all_pdf_files:
            logger.info("   - PDF 파일이 없습니다.")
            return

        logger.info(f"   - 전체 PDF 파일: {len(all_pdf_files)}개")

        # 처리 상태 로드
        processed_states = self._load_processed_states()

        # 새로운 파일들 찾기
        new_files = [
            filename for filename in all_pdf_files
            if filename not in processed_states or
            not processed_states[filename].get("vectorstore_processed", False)
        ]

        if not new_files:
            logger.info("   ✅ 모든 파일이 이미 처리되었습니다.")
            return

        logger.info(f"   - 새로운 파일: {len(new_files)}개")
        for f in new_files:
            logger.info(f"     • {f}")

        # 새 파일들 처리
        for idx, filename in enumerate(new_files, 1):
            logger.info(f"\n[{idx}/{len(new_files)}] 새 파일 추가 중...")
            file_path = os.path.join(directory_path, filename)

            try:
                # PDF에서 문서 추출
                documents = self.extract_center_text_from_pdf(file_path)

                if documents:
                    # 문서 분할
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    split_documents = text_splitter.split_documents(documents)

                    # 벡터스토어에 추가
                    logger.info(f"   - 벡터스토어에 {len(split_documents)}개 청크 추가 중...")
                    self.vectorstore.add_documents(split_documents)

                    # 처리 상태 업데이트
                    if filename not in processed_states:
                        processed_states[filename] = {}
                    processed_states[filename]["vectorstore_processed"] = True
                    processed_states[filename]["document_count"] = len(
                        documents)
                    processed_states[filename]["chunk_count"] = len(
                        split_documents)

                    # 상태 저장
                    self._save_processed_states(processed_states)
                    logger.info(f"   ✅ {filename} 추가 완료!")

            except Exception as e:
                logger.error(f"   ❌ {filename} 처리 실패: {e}")

        logger.info(f"\n✅ 새 문서 추가 완료!")

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        LangChain Retriever 객체 반환

        Args:
            search_kwargs: 검색 관련 추가 인자

        Returns:
            VectorStoreRetriever 객체
        """
        if not self.vectorstore:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")

        if search_kwargs is None:
            search_kwargs = {"k": 4}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


# 전역 인스턴스 (싱글톤 패턴)
_law_retriever = None


def initialize_law_vectorstore(data_directory: str = DEFAULT_DATA_DIR,
                               persist_directory: str = DEFAULT_PERSIST_DIR,
                               force_recreate: bool = False) -> Optional[LawVectorStore]:
    """
    법령 시스템 초기화

    Args:
        data_directory: PDF 파일들이 있는 디렉토리 (기본값: 절대 경로)
        persist_directory: 벡터 DB 저장 디렉토리 (기본값: 절대 경로)
        force_recreate: 기존 벡터스토어가 있어도 새로 생성할지 여부

    Returns:
        초기화된 LawVectorStore 인스턴스 또는 None
    """
    global _law_retriever

    logger.info("=" * 60)
    logger.info("🚀 법령 벡터스토어 초기화 시작")
    logger.info("=" * 60)

    # LawVectorStore 인스턴스 생성 시 절대 경로 사용
    if _law_retriever is None:
        logger.info(f"📍 새 인스턴스 생성")
        logger.info(f"   - 데이터 디렉토리: {data_directory}")
        logger.info(f"   - 저장 디렉토리: {persist_directory}")
        _law_retriever = LawVectorStore(persist_directory=persist_directory)

    # 강제 재생성 모드
    if force_recreate:
        logger.info(f"🔄 강제 재생성 모드 활성화")

    # 기존 벡터스토어 확인
    if not force_recreate and _law_retriever.load_existing_vectorstore():
        # 추가: 이미 로드된 경우에도 새 문서가 있는지 확인하고 추가
        _law_retriever.add_new_documents(data_directory)
        logger.info("=" * 60)
        logger.info("✅ 법령 벡터스토어 초기화 완료 (기존 스토어 사용)")
        logger.info("=" * 60)
        return _law_retriever

    # 새로 학습
    logger.info(f"\n📚 새로운 벡터스토어 생성 중...")
    documents = _law_retriever.load_documents_from_directory(data_directory)

    if documents:
        _law_retriever.create_vectorstore(documents)
        logger.info("=" * 60)
        logger.info("✅ 법령 벡터스토어 초기화 완료 (새로 생성)")
        logger.info("=" * 60)
        return _law_retriever
    else:
        # 벡터스토어를 새로 생성하지 못했더라도, 기존 것이 있는지 다시 한번 확인
        if _law_retriever.load_existing_vectorstore():
            logger.info("=" * 60)
            logger.info("✅ 법령 벡터스토어 초기화 완료 (기존 스토어 사용)")
            logger.info("=" * 60)
            return _law_retriever
        
        logger.error("=" * 60)
        logger.error("❌ 법령 벡터스토어 초기화 실패!")
        logger.error("=" * 60)
        return None


def get_law_vectorstore() -> Optional[LawVectorStore]:
    """
    법령 벡터스토어 반환

    Returns:
        LawVectorStore 인스턴스 또는 None
    """
    global _law_retriever

    if _law_retriever is None:
        # 초기화 함수 호출 시 인자 없이 호출하여 기본 절대 경로 사용
        initialize_law_vectorstore()

    return _law_retriever


# 간편 사용 함수들
def search_law(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """간편 검색 함수"""
    vectorstore = get_law_vectorstore()
    if vectorstore:
        return vectorstore.search_relevant_law(query, k)
    return []


def get_retriever(search_kwargs: Optional[Dict[str, Any]] = None):
    """간편 Retriever 반환 함수"""
    vectorstore = get_law_vectorstore()
    if vectorstore:
        return vectorstore.get_retriever(search_kwargs)
    return None


# 사용 예제
if __name__ == "__main__":
    print("=== 법령 벡터스토어 시스템 ===")

    # 이제 data_directory와 persist_directory를 명시적으로 전달할 필요가 없습니다.
    # 함수에 설정된 기본 절대 경로를 사용합니다.
    system = initialize_law_vectorstore(force_recreate=False)

    if system:
        # 검색 테스트
        results = system.search_relevant_law("임대차 계약 해지", k=2)
        for i, result in enumerate(results):
            print(f"{i+1}. {result['article']} ({result['law_name']})")
            print(f"내용: {result['content'][:100]}...")

        print(f"\nOK 벡터스토어 준비 완료 (경로: {system.persist_directory})")
    else:
        print(f"ERROR 벡터스토어 초기화 실패 (경로 확인: {DEFAULT_PERSIST_DIR})")
