"""Regression: ``read_local_path_to_plain_text`` always returns a 5-tuple."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.local_path_text import read_local_path_to_plain_text

if TYPE_CHECKING:
    from pathlib import Path


def _unpack_five(res: tuple[str, str | None, str | None, list[str], str | None]) -> None:
    content, _err, enc, image_paths, markdown_path = res
    assert isinstance(content, str)
    assert enc is None or isinstance(enc, str)
    assert isinstance(image_paths, list)
    assert markdown_path is None or isinstance(markdown_path, str)


def test_missing_file_five_tuple(tmp_path: Path) -> None:
    res = read_local_path_to_plain_text(str(tmp_path / "missing.txt"))
    assert len(res) == 5
    _unpack_five(res)


def test_plain_text_file_five_tuple(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hi", encoding="utf-8")
    res = read_local_path_to_plain_text(str(f))
    assert len(res) == 5
    _unpack_five(res)
    assert res[0] == "hi"
    assert res[1] is None


def test_opaque_binary_five_tuple(tmp_path: Path) -> None:
    f = tmp_path / "opaque.bin"
    f.write_bytes(b"\x00\xffNOT_FTYP_MARK")
    res = read_local_path_to_plain_text(str(f))
    assert len(res) == 5
    _unpack_five(res)


def test_ftyp_signature_bin_five_tuple(tmp_path: Path) -> None:
    f = tmp_path / "videoish.bin"
    f.write_bytes(b"\x00\x00\x00\x20ftyp" + b"\x00" * 20)
    res = read_local_path_to_plain_text(str(f))
    assert len(res) == 5
    _unpack_five(res)
