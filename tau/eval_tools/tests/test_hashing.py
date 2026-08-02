from tau.eval_tools.hashing import hash_text, hash_texts, normalize_text


def test_normalize_text_strips_and_normalizes_line_endings():
    assert normalize_text("  hello\r\nworld  \r\n") == "hello\nworld"


def test_normalize_text_preserves_internal_whitespace_and_case():
    assert normalize_text("Solve X + 1 = 2") == "Solve X + 1 = 2"


def test_hash_text_is_deterministic():
    assert hash_text("What is 2 + 2?") == hash_text("What is 2 + 2?")


def test_hash_text_ignores_incidental_whitespace_and_line_endings():
    assert hash_text("What is 2 + 2?\n") == hash_text("What is 2 + 2?\r\n")
    assert hash_text("  What is 2 + 2?  ") == hash_text("What is 2 + 2?")


def test_hash_text_differs_for_different_content():
    assert hash_text("What is 2 + 2?") != hash_text("What is 3 + 3?")


def test_hash_text_is_sha256_hex():
    digest = hash_text("anything")
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_hash_texts_preserves_order():
    hashes = hash_texts(["a", "b", "a"])
    assert hashes[0] == hashes[2]
    assert hashes[0] != hashes[1]
    assert len(hashes) == 3
