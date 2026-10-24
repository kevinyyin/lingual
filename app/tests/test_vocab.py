from server.vocab import load_vocab


def test_active_order_stable():
    assert [s.gloss for s in load_vocab().active] == [s.gloss for s in load_vocab().active]


def test_no_duplicate_glosses():
    glosses = [s.gloss for s in load_vocab().active]
    assert len(glosses) == len(set(glosses))


def test_size_in_range():
    assert 75 <= len(load_vocab(extended=None).active) <= 160  # the curated, hand-checked file


def test_inactive_excluded(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "signs:\n"
        "  - {gloss: HELLO, english: hello, category: social, asl_citizen: HELLO}\n"
        "  - {gloss: BYE, english: bye, category: social, asl_citizen: BYE, active: false}\n"
    )
    v = load_vocab(p)
    assert [s.gloss for s in v.active] == ["HELLO"]
    assert v.glosses == {"HELLO"}
    assert v.get("BYE") is not None and v.index("HELLO") == 0
    assert v.prompt_table() == "HELLO — hello"


def test_yaml_boolean_words_stay_glosses():
    v = load_vocab()
    assert {"YES", "NO", "OK"} <= v.glosses
    assert v.get("YES").asl_citizen == "YES"


def test_extended_vocab_merges_after_curated(tmp_path):
    base = tmp_path / "vocab.yaml"
    base.write_text("signs:\n  - {gloss: HELLO, english: hello, category: social, asl_citizen: HELLO}\n", encoding="utf-8")
    ext = tmp_path / "vocab_extended.yaml"
    ext.write_text("signs:\n  - {gloss: HELLO, english: hi there, category: x, asl_citizen: HELLO, curated: false}\n"
                   "  - {gloss: TREE, english: tree, category: noun, asl_citizen: TREE, curated: false}\n", encoding="utf-8")
    v = load_vocab(base, extended=ext)
    assert [s.gloss for s in v.active] == ["HELLO", "TREE"]
    assert v.get("HELLO").english == "hello" and v.get("HELLO").curated  # the curated entry wins
    assert not v.get("TREE").curated
    assert [s.gloss for s in load_vocab(base).active] == ["HELLO"]  # custom files never pick up the global extension
