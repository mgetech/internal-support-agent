"""The employee selector in the UI is a copy of the seeded employees. This test fails
when the seed changes and the copy does not (make db-up).
"""

from __future__ import annotations

from ui.app import EMPLOYEES


def test_ui_roster_matches_the_seeded_employees(seeded_db):
    rows = seeded_db.execute("SELECT id, name, role FROM employees ORDER BY id").fetchall()

    assert {id: (name, role) for id, name, role in rows} == EMPLOYEES
