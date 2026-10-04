"""The policy corpus version is stored on the chunks, and `get_policy_version` reads it
from there. These tests need the database (`make db-up`) and are skipped without one.
The `seeded_db` fixture (tests/conftest.py) loads the generated chunks, which all come
from policy files with version 2026-09.1.
"""

import psycopg
import pytest

from support_agent.tools.policy_search import get_policy_version


def test_the_version_comes_from_the_seeded_chunks(seeded_db):
    assert get_policy_version() == "2026-09.1"


def test_there_is_no_version_when_there_are_no_chunks(clean_db):
    assert get_policy_version() is None


def test_the_version_follows_the_database_not_the_policy_files(seeded_db):
    # The chunks are re-seeded from a newer policy version. The files on disk still say
    # 2026-09.1, but the answer must name what the database holds.
    seeded_db.execute("UPDATE policy_chunks SET policy_version = '2026-10.2'")

    assert get_policy_version() == "2026-10.2"


def test_a_partly_refreshed_corpus_names_every_version_it_holds(seeded_db):
    # one policy was re-seeded and the others were not
    seeded_db.execute(
        "UPDATE policy_chunks SET policy_version = '2026-10.2' WHERE policy = 'vacation-policy'"
    )

    assert get_policy_version() == "2026-09.1, 2026-10.2"


def test_chunks_from_another_chunker_are_ignored(seeded_db):
    # a chunking experiment stored one chunk under another strategy, with another version.
    # search only reads the structural chunks, so that version was not used.
    seeded_db.execute(
        "UPDATE policy_chunks SET chunker = 'fixed', policy_version = '2026-10.2'"
        " WHERE id = 'vacation-policy#entitlement#0'"
    )

    assert get_policy_version() == "2026-09.1"


def test_a_chunk_cannot_be_stored_without_a_version(clean_db):
    with pytest.raises(psycopg.errors.NotNullViolation):
        clean_db.execute(
            "INSERT INTO policy_chunks (id, policy, heading, content, domain)"
            " VALUES ('p#h#0', 'p', 'h', 'text', 'hr')"
        )
