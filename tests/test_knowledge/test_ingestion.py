import pytest
from src.knowledge.ingestion import DocumentIngestor


class TestDocumentIngestor:
    def test_unsupported_file_type(self, tmp_path):
        ingestor = DocumentIngestor()
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestor.ingest_file(str(test_file))

    def test_nonexistent_file(self):
        ingestor = DocumentIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest_file("/nonexistent/file.txt")

    def test_ingest_txt_file(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. This is a test file with multiple sentences.")
        chunks = ingestor.ingest_file(str(test_file))
        assert len(chunks) > 0
        assert all("content" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_ingest_md_file(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        test_file = tmp_path / "test.md"
        test_file.write_text("# Title\n\nSome markdown content here for testing.")
        chunks = ingestor.ingest_file(str(test_file))
        assert len(chunks) > 0

    def test_ingest_py_file(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        test_file = tmp_path / "test.py"
        test_file.write_text("# Python file\ndef hello():\n    print('hello')")
        chunks = ingestor.ingest_file(str(test_file))
        assert len(chunks) > 0

    def test_chunk_overlap(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=20, chunk_overlap=10)
        test_file = tmp_path / "test.txt"
        test_file.write_text("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")
        chunks = ingestor.ingest_file(str(test_file))
        assert len(chunks) >= 2

    def test_ingest_directory(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        # Create multiple supported files
        (tmp_path / "a.txt").write_text("File A content here.")
        (tmp_path / "b.md").write_text("File B content here.")
        (tmp_path / "c.py").write_text("File C content here.")
        (tmp_path / "d.xyz").write_text("Unsupported file.")

        chunks = ingestor.ingest_directory(str(tmp_path))
        # Should have chunks from the 3 supported files
        assert len(chunks) >= 3

    def test_chunk_has_required_fields(self, tmp_path):
        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for field validation.")
        chunks = ingestor.ingest_file(str(test_file))
        for chunk in chunks:
            assert "id" in chunk
            assert "content" in chunk
            assert "chunk_index" in chunk
            assert "content_hash" in chunk
            assert "metadata" in chunk
