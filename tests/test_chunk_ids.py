"""Chunk ids are a stable, addressable contract — eval ground truth and
citations reference chunks by id directly, so the format and the specific
`vacation-policy#entitlement#0` id must hold exactly.
"""

from data.generate import read_policies

from support_agent.chunking import StructuralChunker

CHUNKER = StructuralChunker()


def _chunk_ids(policy_id: str) -> list[str]:
    text = read_policies()[policy_id].text
    return [c.id for c in CHUNKER.chunk(policy_id, text)]


def test_vacation_policy_entitlement_chunk_id_matches_the_spec():
    assert "vacation-policy#entitlement#0" in _chunk_ids("vacation-policy")


def test_chunk_id_format_across_the_whole_corpus():
    for policy_id, policy in read_policies().items():
        for chunk in CHUNKER.chunk(policy_id, policy.text):
            prefix, slug, n = chunk.id.split("#")
            assert prefix == policy_id
            assert slug == slug.lower()
            assert " " not in slug
            assert n.isdigit()


def test_multi_word_heading_slugifies_with_hyphens():
    ids = _chunk_ids("vacation-policy")

    assert "vacation-policy#part-time-and-mid-year-changes#0" in ids


def test_chunk_ids_are_unique_within_a_policy():
    ids = _chunk_ids("vacation-policy")

    assert len(ids) == len(set(ids))


def test_chunk_content_is_prefixed_with_policy_title_as_context():
    text = read_policies()["vacation-policy"].text
    chunks = CHUNKER.chunk("vacation-policy", text)

    assert all(c.content.startswith("Vacation Policy — ") for c in chunks)


def test_chunk_meta_tags_the_chunker_strategy():
    text = read_policies()["vacation-policy"].text
    chunks = CHUNKER.chunk("vacation-policy", text)

    assert all(c.meta["chunker"] == "structural" for c in chunks)


def test_chunking_is_deterministic():
    text = read_policies()["vacation-policy"].text

    assert CHUNKER.chunk("vacation-policy", text) == CHUNKER.chunk("vacation-policy", text)


def test_section_at_or_under_budget_is_a_single_chunk():
    text = "# Title\n\n## Heading\n\n" + " ".join(["word"] * 500)

    chunks = CHUNKER.chunk("policy", text)

    assert len(chunks) == 1
    assert chunks[0].id == "policy#heading#0"


def test_section_over_budget_sub_splits_with_overlap():
    text = "# Title\n\n## Heading\n\n" + " ".join(f"w{i}" for i in range(900))

    chunks = CHUNKER.chunk("policy", text)

    assert len(chunks) == 2
    assert [c.id for c in chunks] == ["policy#heading#0", "policy#heading#1"]
    # last 50 words of the first window reappear at the start of the second
    # (each chunk's content is prefixed with 3 context tokens: "Title — Heading")
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    assert first_words[-50:] == second_words[3:53]


# the complete, pinned set of ids the whole corpus must produce. Eval
# scenarios and citations reference these directly, so a heading edit that
# silently changes an id has to fail a test here, not surface later as a
# broken citation.
EXPECTED_CHUNK_IDS = {
    "vacation-policy#entitlement#0",
    "vacation-policy#part-time-and-mid-year-changes#0",
    "vacation-policy#booking-and-approval#0",
    "vacation-policy#carryover#0",
    "sick-leave-policy#notification-and-certification#0",
    "sick-leave-policy#pay-continuation#0",
    "sick-leave-policy#sickness-during-vacation#0",
    "parental-leave-policy#entitlement-and-notice#0",
    "parental-leave-policy#part-time-during-leave#0",
    "parental-leave-policy#eligibility-and-benefit-interactions#0",
    "expense-policy#receipts-and-submission#0",
    "expense-policy#late-submissions#0",
    "it-access-policy#vpn-access#0",
    "it-access-policy#password-self-service#0",
    "it-access-policy#software-licenses#0",
    "it-access-policy#security-incidents#0",
}


def test_the_full_chunk_id_set_is_pinned_as_ground_truth():
    all_ids = {
        chunk.id
        for policy_id, policy in read_policies().items()
        for chunk in CHUNKER.chunk(policy_id, policy.text)
    }

    assert all_ids == EXPECTED_CHUNK_IDS
