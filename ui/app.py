"""Streamlit review surface. It talks to the REST API and nothing else. The employee
selector is the login stub made visible: it sets the X-Employee-Id header.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# the model can take a while to answer
CHAT_TIMEOUT_SECONDS = 120

# The selector stands in for an identity provider; a real deployment has no employee list.
EMPLOYEES = {
    "emp_001": ("Anna Keller", "employee"),
    "emp_002": ("Jonas Vogel", "employee"),
    "emp_003": ("Mia Schuster", "employee"),
    "emp_004": ("Paul Brandt", "employee"),
    "emp_005": ("Laura Fink", "hr_partner"),
    "emp_006": ("Elif Yildiz", "employee"),
    "emp_007": ("Tobias Wagner", "employee"),
    "emp_008": ("Sara Nowak", "employee"),
    "emp_009": ("David Hoffmann", "employee"),
    "emp_010": ("Nora Bergmann", "manager"),
    "emp_011": ("Felix Lorenz", "employee"),
    "emp_012": ("Katrin Sommer", "employee"),
}

OUTCOME_BADGES = {
    "resolve": ("Resolved", "green"),
    "propose_action": ("Waiting for approval", "blue"),
    "escalate": ("Escalated to a human", "orange"),
    "refuse_with_citation": ("Refused", "red"),
}


def get_error_message(error: httpx.HTTPError) -> str:
    """A short text for a failed API call."""
    if isinstance(error, httpx.HTTPStatusError):
        try:
            detail = error.response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = error.response.text
        return f"{error.response.status_code}: {detail}"
    return f"Cannot reach the API at {API_URL}."


def create_chat(employee_id: str, message: str) -> dict[str, Any]:
    """Send one message to the agent as the selected employee."""
    response = httpx.post(
        f"{API_URL}/chat",
        json={"message": message},
        headers={"X-Employee-Id": employee_id},
        timeout=CHAT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def show_outcome_badge(outcome: str) -> None:
    label, color = OUTCOME_BADGES.get(outcome, (outcome, "gray"))
    st.badge(label, color=color)


def show_chat_tab(employee_id: str) -> None:
    # every employee has their own conversation
    history = st.session_state.setdefault("history", {}).setdefault(employee_id, [])

    for item in history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item["role"] == "assistant":
                show_outcome_badge(item["outcome"])

    question = st.chat_input("Ask about HR or IT")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply = create_chat(employee_id, question)
        except httpx.HTTPError as error:
            st.error(get_error_message(error))
            return
        st.write(reply["answer"])
        show_outcome_badge(reply["outcome"])
    history.append(
        {
            "role": "assistant",
            "content": reply["answer"],
            "outcome": reply["outcome"],
            "request_id": reply["request_id"],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Support agent", layout="centered")
    st.title("Support agent")

    with st.sidebar:
        employee_id = st.selectbox(
            "Signed in as",
            options=list(EMPLOYEES),
            format_func=lambda id: f"{EMPLOYEES[id][0]} ({EMPLOYEES[id][1]})",
        )
        st.caption(f"API: {API_URL}")

    (chat_tab,) = st.tabs(["Chat"])
    with chat_tab:
        show_chat_tab(employee_id)


# streamlit runs the script as __main__; a test can import it without drawing the page
if __name__ == "__main__":
    main()
