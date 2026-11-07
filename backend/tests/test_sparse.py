"""BM25 sparse vectors for keyword search."""

from voice_agent.knowledge.sparse import encode_document, encode_query, stem, term_id, tokenize


def test_tokenize_drops_stop_words_and_endings():
    assert tokenize("How do I add my newborn baby to the ration card?") == [
        "add",
        "newborn",
        "baby",
        "ration",
        "card",
    ]
    assert stem("cards") == stem("card") == "card"
    assert stem("changes") == stem("changed") == stem("changing") == stem("change")
    assert stem("families") == stem("family")
    assert stem("address") == stem("addresses")
    assert tokenize("") == []


def test_term_ids_are_stable_32_bit():
    assert term_id("card") == term_id("card") < 2**32
    assert term_id("card") != term_id("cart")


def test_document_weights_saturate_with_frequency():
    once = encode_document("ration shop")
    many = encode_document("ration " * 50 + "shop")
    w_once = dict(zip(once.indices, once.values, strict=True))[term_id("ration")]
    w_many = dict(zip(many.indices, many.values, strict=True))[term_id("ration")]
    assert w_once < w_many < 2.2  # bounded by k1 + 1
    assert once.indices == sorted(once.indices)


def test_query_vector_is_one_per_distinct_term():
    q = encode_query("the card card shop")
    assert q.indices == sorted({term_id("card"), term_id("shop")}) and q.values == [1.0, 1.0]
    assert encode_query("the and of").indices == []
