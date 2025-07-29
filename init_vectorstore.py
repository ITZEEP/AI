#!/usr/bin/env python3
"""
벡터스토어 초기화 스크립트
법령 PDF 파일들을 학습하여 벡터스토어를 생성합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "law_system"))

def create_sample_law_structure(data_dir):
    """샘플 법령 문서 구조 생성"""
    print("📁 Creating sample law document structure...")
    
    # Just print instructions, no file creation needed
    print("\n📋 To enable legal analysis features:")
    print("1. Add PDF law documents to:")
    print(f"   {data_dir}")
    print("2. Restart the service:")
    print("   docker-compose restart")
    print("3. Vectorstore will be initialized automatically")

def main():
    print("=== 벡터스토어 초기화 시작 ===")
    
    try:
        # law_vectorstore 모듈 import
        from law_system.law_vectorstore import initialize_law_vectorstore
        
        print("OK law_vectorstore module import success")
        
        # 데이터 디렉토리 경로
        data_dir = current_dir / "data" / "law_docs"
        vectorstore_dir = current_dir / "data" / "vectorstore"
        
        print(f"Law documents directory: {data_dir}")
        print(f"Vectorstore directory: {vectorstore_dir}")
        
        # 벡터스토어 디렉토리 생성
        try:
            vectorstore_dir.mkdir(parents=True, exist_ok=True)
            print(f"OK Vectorstore directory created/verified: {vectorstore_dir}")
        except Exception as e:
            print(f"ERROR Vectorstore directory creation failed: {e}")
            return False
        
        # 법령 문서 파일 확인
        if not data_dir.exists():
            print(f"ERROR 법령 문서 디렉토리가 없습니다: {data_dir}")
            print("Creating directory...")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"Found PDF files: {len(pdf_files)}")
        
        if not pdf_files:
            print("ERROR PDF files not found")
            print("Creating sample law document structure...")
            create_sample_law_structure(data_dir)
            return False
        
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        # 벡터스토어 초기화
        print("\nInitializing vectorstore...")
        vectorstore = initialize_law_vectorstore(
            data_directory=str(data_dir),
            persist_directory=str(vectorstore_dir),
            force_recreate=False  # 기존 벡터스토어가 있으면 재사용
        )
        
        if vectorstore:
            print("OK 벡터스토어 초기화 성공!")
            
            # 테스트 검색
            print("\nTesting search...")
            results = vectorstore.search_relevant_law("임대차 계약", k=2)
            
            if results:
                print(f"OK 검색 결과 {len(results)}개 발견:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('law_name', '법령명 미상')} - {result.get('article', '')}")
                    print(f"     {result.get('content', '')[:100]}...")
            else:
                print("WARNING 검색 결과가 없습니다")
            
            return True
        else:
            print("ERROR 벡터스토어 초기화 실패")
            return False
            
    except ImportError as e:
        print(f"ERROR 모듈 import 실패: {e}")
        print("필요한 패키지를 설치해주세요:")
        print("pip install langchain-text-splitters langchain-huggingface langchain-chroma pdfplumber chromadb")
        print("\n패키지 설치 명령:")
        print("pip install --upgrade pip")
        print("pip install 'langchain-text-splitters>=0.0.1'")
        print("pip install 'langchain-huggingface>=0.0.3'")  
        print("pip install 'langchain-chroma>=0.1.0'")
        print("pip install 'sentence-transformers>=2.2.2'")
        return False
    except Exception as e:
        print(f"ERROR 벡터스토어 초기화 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: Vectorstore initialization completed!")
        print("Now restart the service and the warning messages will disappear.")
    else:
        print("\nERROR: Vectorstore initialization failed.")
        sys.exit(1)