"""Streamlit review surface. It talks to the REST API and nothing else. The employee
selector is the login stub made visible: it sets the X-Employee-Id header.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# the model can take a while to answer
CHAT_TIMEOUT_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 10

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

ACTION_STATUS_BADGES = {
    "pending_approval": ("Waiting for approval", "blue"),
    "approved": ("Approved", "green"),
    "executed": ("Approved and done", "green"),
    "rejected": ("Rejected", "red"),
    "expired": ("Expired", "gray"),
}

# Only an HR partner can read Decision Records, so only they get 'Why this answer?'.
HR_PARTNER_ROLE = "hr_partner"

# the same pattern as the citation check in guardrails/verifier.py
CITATION_PATTERN = re.compile(r"\[chunk:([^\]\s]+)\]")

# Shown above every tab. It does not name a contact: the system has no contact data.
AI_NOTICE = (
    "You are talking to an AI assistant, not a person. It can make mistakes, so open the "
    "numbered sources in an answer and check important answers with your HR or IT team. "
    "A human is always reachable: you can contact your HR or IT team directly at any time. "
    "Anything that changes data is decided by a person."
)

HANDOFF_NOTES = {
    "escalate": "The assistant cannot decide this. A person needs to handle it, so please "
    "contact your HR or IT team directly.",
    "propose_action": "A person decides on this request. You can follow it in 'My requests'.",
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


def get_approvals(employee_id: str) -> list[dict[str, Any]]:
    """The actions waiting for approval, as the selected employee."""
    response = httpx.get(
        f"{API_URL}/approvals",
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_my_actions(employee_id: str) -> list[dict[str, Any]]:
    """The selected employee's own actions, in any status."""
    response = httpx.get(
        f"{API_URL}/me/actions",
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600, max_entries=200, show_spinner=False)
def get_policy_chunk(employee_id: str, chunk_id: str) -> dict[str, Any]:
    """One policy section by its chunk id. The id has a '#', so it is encoded for the path."""
    response = httpx.get(
        f"{API_URL}/policy-chunks/{quote(chunk_id, safe='')}",
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def create_feedback(
    employee_id: str, request_id: str, rating: str, comment: str | None
) -> dict[str, Any]:
    """Rate the answer to a request, as the employee who made it."""
    body = {"rating": rating, **({"comment": comment} if comment else {})}
    response = httpx.post(
        f"{API_URL}/feedback/{request_id}",
        json=body,
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def create_approval(employee_id: str, action_id: int, decision: str) -> dict[str, Any]:
    """Approve or reject an action as the selected employee."""
    response = httpx.post(
        f"{API_URL}/approvals/{action_id}",
        json={"decision": decision},
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_decision(employee_id: str, request_id: str) -> dict[str, Any]:
    """The full Decision Record of a request, as the selected employee."""
    response = httpx.get(
        f"{API_URL}/decisions/{request_id}",
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_decisions(employee_id: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
    """The newest Decision Records as a summary list. Filters that are None are left out."""
    response = httpx.get(
        f"{API_URL}/decisions",
        params={name: value for name, value in filters.items() if value is not None},
        headers={"X-Employee-Id": employee_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def format_time(timestamp: str) -> str:
    """An ISO timestamp from the API, in the local time zone."""
    return datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")


def show_outcome_badge(outcome: str) -> None:
    label, color = OUTCOME_BADGES.get(outcome, (outcome, "gray"))
    st.badge(label, color=color)


def show_answer(employee_id: str, text: str) -> None:
    """The answer, with each [chunk:<id>] turned into a number. Each number has a button
    under the answer that opens the text of that policy section.
    """
    chunk_ids = list(dict.fromkeys(CITATION_PATTERN.findall(text)))
    numbers = {chunk_id: number for number, chunk_id in enumerate(chunk_ids, start=1)}
    st.write(CITATION_PATTERN.sub(lambda match: f"**[{numbers[match[1]]}]**", text))

    if not chunk_ids:
        return
    with st.container(horizontal=True):
        for chunk_id, number in numbers.items():
            try:
                chunk = get_policy_chunk(employee_id, chunk_id)
            except httpx.HTTPError as error:
                with st.popover(f"[{number}] {chunk_id}"):
                    st.error(get_error_message(error))
                continue
            with st.popover(f"[{number}] {chunk['policy']} · {chunk['heading']}"):
                st.markdown(f"**{chunk['heading']}**")
                st.markdown(chunk["content"])
                st.caption(f"{chunk['id']} · policy version {chunk['policy_version']}")


def show_handoff_note(outcome: str) -> None:
    """Tell the employee when a person has to take over or decide."""
    if outcome in HANDOFF_NOTES:
        st.caption(f":material/support_agent: {HANDOFF_NOTES[outcome]}")


def show_feedback(employee_id: str, item: dict[str, Any]) -> None:
    """Thumbs up or down with an optional comment, once per reply."""
    if "feedback" in item:
        st.caption(f"Thanks, your {item['feedback']} vote was saved.")
        return

    request_id = item["request_id"]
    with st.popover("Rate this answer", icon=":material/thumbs_up_down:"):
        with st.form(f"feedback_{request_id}", border=False):
            # the widget gives 0 for thumbs down and 1 for thumbs up
            thumb = st.feedback("thumbs", key=f"thumb_{request_id}")
            comment = st.text_input("Comment (optional)", max_chars=1000, key=f"note_{request_id}")
            sent = st.form_submit_button("Send")
        if not sent:
            return
        if thumb is None:
            st.warning("Choose thumbs up or thumbs down first.")
            return
        rating = "up" if thumb == 1 else "down"
        try:
            create_feedback(employee_id, request_id, rating, comment.strip() or None)
        except httpx.HTTPError as error:
            st.error(get_error_message(error))
            return
    item["feedback"] = rating
    st.rerun()


def show_decision_record(record: dict[str, Any]) -> None:
    """Why the agent did what it did: the summary, the cost, the citations, the versions
    and the evidence, all from the saved record.
    """
    show_outcome_badge(record["outcome"])
    st.write(record["summary"])

    with st.container(horizontal=True):
        st.metric("Tokens", record["total_tokens"], border=True)
        st.metric("Cost (EUR)", f"{record['total_cost_eur']:.6f}", border=True)
        st.metric("Time (ms)", record["latency_ms"], border=True)

    st.markdown("**Citations**")
    if record["citations"]:
        st.code("\n".join(record["citations"]), language=None)
    else:
        st.caption("None")

    st.markdown("**Versions**")
    st.json(
        {"policy": record["policy_version"], "prompts": record["prompt_versions"]}, expanded=False
    )

    if record["model_calls"]:
        st.markdown("**Model calls**")
        st.dataframe(record["model_calls"], hide_index=True)

    st.markdown("**Evidence**")
    st.json(record["evidence"], expanded=False)
    st.caption(f"Request {record['request_id']} · trace {record['trace_id'] or 'none'}")


def show_why_expander(employee_id: str, request_id: str) -> None:
    # the record is read when the expander opens, then kept for this session
    why = st.expander("Why this answer?", key=f"why_{request_id}", on_change="rerun")
    if not why.open:
        return
    records = st.session_state.setdefault("records", {})
    with why:
        if request_id not in records:
            try:
                records[request_id] = get_decision(employee_id, request_id)
            except httpx.HTTPError as error:
                st.error(get_error_message(error))
                return
        show_decision_record(records[request_id])


def show_chat_tab(employee_id: str) -> None:
    # every employee has their own conversation
    history = st.session_state.setdefault("history", {}).setdefault(employee_id, [])

    for item in history:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.write(item["content"])
                continue
            show_answer(employee_id, item["content"])
            show_outcome_badge(item["outcome"])
            show_handoff_note(item["outcome"])
            if EMPLOYEES[employee_id][1] == HR_PARTNER_ROLE:
                show_why_expander(employee_id, item["request_id"])
            show_feedback(employee_id, item)

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
    history.append(
        {
            "role": "assistant",
            "content": reply["answer"],
            "outcome": reply["outcome"],
            "request_id": reply["request_id"],
        }
    )
    # draw the whole conversation again, so the new reply has its expander too
    st.rerun()


def show_requests_tab(employee_id: str) -> None:
    st.button("Refresh", icon=":material/refresh:", key="refresh_requests")

    try:
        actions = get_my_actions(employee_id)
    except httpx.HTTPError as error:
        st.error(get_error_message(error))
        return
    if not actions:
        st.info("You have no requests yet. Ask the agent for leave or a ticket in the chat.")
        return

    for action in actions:
        label, color = ACTION_STATUS_BADGES.get(action["status"], (action["status"], "gray"))
        with st.container(border=True):
            st.markdown(f"**#{action['id']} · {action['tool']}**")
            st.badge(label, color=color)
            st.caption(f"Requested at {format_time(action['created_at'])}")
            if action["decided_by"]:
                approver_name = EMPLOYEES.get(action["decided_by"], ("unknown",))[0]
                decided_at = format_time(action["decided_at"])
                st.caption(f"Decided by {approver_name} ({action['decided_by']}) at {decided_at}")
            st.json(action["payload"])


def show_approvals_tab(employee_id: str) -> None:
    # the result of the last decision, kept over the rerun that refreshes the list
    result = st.session_state.pop("approval_result", None)
    if result:
        kind, text = result
        (st.success if kind == "success" else st.error)(text)

    st.button("Refresh", icon=":material/refresh:")

    try:
        actions = get_approvals(employee_id)
    except httpx.HTTPError as error:
        st.error(get_error_message(error))
        return
    if not actions:
        st.info("Nothing is waiting for approval.")
        return

    for action in actions:
        action_id = action["id"]
        requester_name = EMPLOYEES.get(action["employee_id"], ("unknown",))[0]
        created_at = format_time(action["created_at"])
        with st.container(border=True):
            st.markdown(f"**#{action_id} · {action['tool']}**")
            st.caption(f"Requested by {requester_name} ({action['employee_id']}) at {created_at}")
            st.json(action["payload"])
            # the buttons are always on, so the API's refusals can be seen. A decision on
            # an own request is refused, and so is any decision from a non-approver
            with st.container(horizontal=True):
                approve = st.button(
                    "Approve", key=f"approve_{action_id}", type="primary", icon=":material/check:"
                )
                reject = st.button("Reject", key=f"reject_{action_id}", icon=":material/close:")
            if not action["can_decide"]:
                st.caption("This is your own request. The API refuses your decision.")

        if not (approve or reject):
            continue
        try:
            decision = create_approval(employee_id, action_id, "approve" if approve else "reject")
            st.session_state["approval_result"] = (
                "success",
                f"Action {action_id} is {decision['status']}.",
            )
        except httpx.HTTPError as error:
            st.session_state["approval_result"] = ("error", get_error_message(error))
        st.rerun()


def show_records_tab(employee_id: str) -> None:
    st.caption(
        "Only an HR partner can read Decision Records; the API refuses everyone else. An HR "
        "partner can pick another employee, and every look at another employee's records is "
        "audited."
    )
    employee_ids = list(EMPLOYEES)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        outcome = st.selectbox("Outcome", ["All", *OUTCOME_BADGES])
        subject_id = st.selectbox(
            "Employee",
            employee_ids,
            index=employee_ids.index(employee_id),
            format_func=lambda id: f"{EMPLOYEES[id][0]} ({id})",
        )
        from_date = st.date_input("From", value=None, format="YYYY-MM-DD")
        to_date = st.date_input("To", value=None, format="YYYY-MM-DD")

    filters = {
        "outcome": None if outcome == "All" else outcome,
        "subject_employee_id": None if subject_id == employee_id else subject_id,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
    }
    try:
        records = get_decisions(employee_id, filters)
    except httpx.HTTPError as error:
        st.error(get_error_message(error))
        return
    if not records:
        st.info("No records match.")
        return

    st.caption(f"The newest {len(records)} records, the newest first. Select a row to see why.")
    rows = [{**record, "created_at": format_time(record["created_at"])} for record in records]
    selection = st.dataframe(
        rows,
        hide_index=True,
        key="records_table",
        on_select="rerun",
        selection_mode="single-row",
        column_order=[
            "created_at",
            "employee_id",
            "outcome",
            "summary",
            "total_tokens",
            "total_cost_eur",
        ],
    )
    if not selection.selection.rows:
        return

    request_id = records[selection.selection.rows[0]]["request_id"]
    try:
        record = get_decision(employee_id, request_id)
    except httpx.HTTPError as error:
        st.error(get_error_message(error))
        return
    with st.container(border=True):
        show_decision_record(record)


def main() -> None:
    st.set_page_config(page_title="Support agent", layout="wide")
    st.title("Support agent")
    st.info(AI_NOTICE, icon=":material/smart_toy:")

    with st.sidebar:
        employee_id = st.selectbox(
            "Signed in as",
            options=list(EMPLOYEES),
            format_func=lambda id: f"{EMPLOYEES[id][0]} ({EMPLOYEES[id][1]})",
        )
        st.caption(f"API: {API_URL}")

    # only the open tab runs, so the chat does not fetch the queue on every message
    chat_tab, requests_tab, approvals_tab, records_tab = st.tabs(
        ["Chat", "My requests", "Approvals", "Decision records"],
        key="active_tab",
        on_change="rerun",
    )
    if chat_tab.open:
        with chat_tab:
            show_chat_tab(employee_id)
    if requests_tab.open:
        with requests_tab:
            show_requests_tab(employee_id)
    if approvals_tab.open:
        with approvals_tab:
            show_approvals_tab(employee_id)
    if records_tab.open:
        with records_tab:
            show_records_tab(employee_id)


# streamlit runs the script as __main__; a test can import it without drawing the page
if __name__ == "__main__":
    main()
