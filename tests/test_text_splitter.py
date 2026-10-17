import pytest

from pdf_rag_agent.text_splitter import split_text


def test_empty_text_returns_empty_list():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_returns_one_chunk():
    assert split_text("Hello RAG", chunk_size=20, chunk_overlap=5,) == ["Hello RAG"]


def test_long_text_is_split_into_chunks():
    text = "abcdefghijklmnopqrst"

    chunks = split_text(text, chunk_size=8, chunk_overlap=2)

    assert chunks == ["abcdefgh", "ghijklmn", "mnopqrst"]


def test_chunks_respect_maximum_size():
    text = "A" * 100
    chunks = split_text(text, chunk_size=20, chunk_overlap=5)

    assert all(len(chunk) <= 20 for chunk in chunks)


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=10, chunk_overlap=10)


def test_negative_overlap_raises_error():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_overlap=-1)


def test_non_string_input_raises_error():
    with pytest.raises(TypeError):
        split_text(123)