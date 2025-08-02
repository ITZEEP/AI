# law_system 패키지 초기화
from .law_vectorstore import LawVectorStore, get_law_vectorstore, search_law

__all__ = ['LawVectorStore', 'get_law_vectorstore', 'search_law']
