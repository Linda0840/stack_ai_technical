import uuid


class IngestionService:
    def __init__(self, vector_store: dict):
        self._vector_store = vector_store

    async def ingest_files(self, files: list) -> tuple[list[str], int]:
        document_ids: list[str] = []
        total_chunks = 0

        for upload_file in files:
            doc_id = str(uuid.uuid4())
            raw_bytes = await upload_file.read()
            text = self._extract_text(raw_bytes, upload_file.filename)

            self._vector_store[doc_id] = {
                "filename": upload_file.filename,
                "text_preview": text[:200],
            }

            document_ids.append(doc_id)
            total_chunks += 1

        return document_ids, total_chunks

    def _extract_text(self, raw_bytes: bytes, filename: str) -> str:
        return f"[stub] extracted text from {filename}"

