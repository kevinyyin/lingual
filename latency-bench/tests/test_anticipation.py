"""Longest-common-prefix logic for the sampled anticipation strategy, and stable_ms commits."""

import asyncio

from bench.providers.cascade import Incremental, longest_common_prefix, strip_committed


def test_lcp_basic_and_word_level():
    t = ["Tomorrow morning I have to take my daughter to school",
         "Tomorrow morning I have to take my son",
         "tomorrow morning, I have to go"]
    # comparison ignores case and punctuation; words come from the first text
    assert longest_common_prefix(t) == "Tomorrow morning I have to"


def test_lcp_does_not_split_words():
    # "to" vs "today": a character prefix would wrongly keep "to"
    assert longest_common_prefix(["I go to work", "I go today"]) == "I go"


def test_lcp_edge_cases():
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix(["Only one candidate here"]) == "Only one candidate here"
    assert longest_common_prefix(["Yes I can", "No I can"]) == ""
    assert longest_common_prefix(["Same words.", "Same words"]) == "Same words."
    assert longest_common_prefix(["Short", "Short and longer"]) == "Short"


def test_strip_committed():
    assert strip_committed("Tomorrow morning I have to take", "Tomorrow morning") == "I have to take"
    assert strip_committed("Tomorrow morning", "Tomorrow morning I have") == ""     # nothing new
    assert strip_committed("Yesterday I went", "Tomorrow morning") == ""            # diverged: commit nothing


class _Host:
    def is_same_language(self, _):
        return False


def test_stable_ms_commits_words_after_they_stood_long_enough():
    inc = Incremental(_Host(), {"incremental": {"stable_ms": 300}})
    ms = 1_000_000
    assert inc.stable_prefix(0, ["hola", "que"], 0) == 0
    assert inc.stable_prefix(0, ["hola", "que", "tal"], 200 * ms) == 0        # nothing has stood 300 ms yet
    assert inc.stable_prefix(0, ["hola", "que", "tal"], 320 * ms) == 2        # hola, que since t=0
    # a revision at position 1 restarts the clock there and after it
    assert inc.stable_prefix(0, ["hola", "qué", "tal"], 400 * ms) == 1
    assert inc.stable_prefix(0, ["hola", "qué", "tal"], 710 * ms) == 3        # all stood >= 300 ms since 400


def test_round1_rule_unchanged_without_stable_ms():
    inc = Incremental(_Host(), {"incremental": {"stable_updates": 2}})
    assert inc.stable_prefix(0, ["a", "b", "c"], 0) == 0
    assert inc.stable_prefix(0, ["a", "b", "x"], 1) == 2
