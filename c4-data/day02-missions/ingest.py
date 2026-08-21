"""학습 문서를 청킹하고 Chroma에 저장한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from study_core import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, PROJECT_DIR


DATA_DIR = PROJECT_DIR / "data"


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"doc_id": path.stem, "source": path.name},
            )
        )
    if not documents:
        raise RuntimeError(f"학습 문서가 없습니다: {DATA_DIR}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=400,
        chunk_overlap=80,
    )

    staged: list[Document] = []
    for document in documents:
        pieces = header_splitter.split_text(document.page_content)
        for piece in pieces:
            piece.metadata = {**document.metadata, **piece.metadata}
        staged.extend(pieces)

    chunks = size_splitter.split_documents(staged)
    sequence: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        number = sequence.get(doc_id, 0)
        chunk.metadata["chunk_id"] = f"{doc_id}-{number:04d}"
        sequence[doc_id] = number + 1
    return chunks


def build_index(rebuild: bool = False) -> int:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    current_count = store._collection.count()
    if current_count and not rebuild:
        print(f"기존 색인 {current_count:,}개를 사용합니다. 다시 만들려면 --rebuild를 지정하세요.")
        return current_count
    if current_count:
        store.delete_collection()

    chunks = split_documents(load_documents())
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"학습 문서 {len(chunks):,}개 조각을 {CHROMA_DIR}에 저장했습니다.")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 학습 자료 색인")
    parser.add_argument("--rebuild", action="store_true", help="기존 색인을 지우고 다시 생성")
    args = parser.parse_args()
    load_dotenv(PROJECT_DIR / ".env")
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()

