from training.eval import grid_accuracy, latency_stats, spell_accuracy


def line(direction, stamps, picks=0):
    return {"direction": direction, "stamps": stamps, "result": {"picks": picks}, "error": None}


def test_latency_uses_pick_time_when_present():
    lines = [
        line("sign2speech", {"hands_rest": 0, "picked": 5000, "first_audio": 6200}, picks=1),
        line("sign2speech", {"hands_rest": 0, "picked": 10, "first_audio": 1500}, picks=0),
        line("speech2sign", {"speech_end": 0, "planned": 900, "avatar_start": 950}),
    ]
    s = latency_stats(lines)
    assert sorted(s["sign2speech"]) == [1200, 1500]
    assert s["speech2sign"] == [950]


def test_latency_skips_turns_without_audio():
    s = latency_stats([line("sign2speech", {"hands_rest": 0, "picked": 10})])
    assert s["sign2speech"] == []


def test_grid_accuracy_counts():
    rows = [("yn", "happy", "yn"), ("yn", "sad", "yn"), ("wh", "angry", "wh"), ("statement", "happy", "yn")]
    overall, by_emotion = grid_accuracy(rows)
    assert overall == 0.75
    assert by_emotion["happy"] == 0.5 and by_emotion["sad"] == 1.0


def test_spell_accuracy():
    assert spell_accuracy([("ANA", "ANA"), ("PRIYA", "PRIA"), ("SAM", "SAM")]) == 2 / 3
