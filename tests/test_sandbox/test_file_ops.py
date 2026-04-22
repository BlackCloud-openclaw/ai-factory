import pytest
from src.execution.file_ops import FileOperations


class TestFileOperations:
    def test_write_and_read(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("test.txt", "hello world")
        content = ops.read_file("test.txt")
        assert content == "hello world"

    def test_append_file(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("test.txt", "first line\n")
        ops.append_file("test.txt", "second line\n")
        content = ops.read_file("test.txt")
        assert "first line" in content
        assert "second line" in content

    def test_delete_file(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("test.txt", "to delete")
        assert ops.delete_file("test.txt") is True
        assert ops.delete_file("test.txt") is False  # already deleted

    def test_create_temp_file(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        file_path = ops.create_temp_file("# test\ndef hello(): pass", suffix=".py")
        assert file_path.endswith(".py")
        content = ops.read_file(file_path)
        assert "def hello()" in content

    def test_get_file_info(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("test.txt", "info test")
        info = ops.get_file_info("test.txt")
        assert info["name"] == "test.txt"
        assert info["size"] > 0
        assert info["is_file"] is True

    def test_path_traversal_protection(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Access denied"):
            ops.read_file("../../etc/passwd")

    def test_ensure_directory(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        dir_path = ops.ensure_directory("subdir/nested")
        assert str(tmp_path / "subdir" / "nested") == dir_path

    def test_list_files(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("a.txt", "a")
        ops.write_file("b.py", "b")
        ops.write_file("c.md", "c")
        files = ops.list_files(pattern="*.txt")
        assert any("a.txt" in f for f in files)

    def test_copy_file(self, tmp_path):
        ops = FileOperations(base_dir=str(tmp_path))
        ops.write_file("source.txt", "copy me")
        ops.copy_file("source.txt", "dest.txt")
        assert ops.read_file("dest.txt") == "copy me"
