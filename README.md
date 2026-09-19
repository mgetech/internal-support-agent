# internal-support-agent

An internal employee support agent for HR and IT questions — the AskHR /
internal-helpdesk class of enterprise agent, built from scratch. A LangGraph agent
resolves requests like leave balances, leave bookings, and IT outage checks with typed
tools over Postgres, and terminates every request in exactly one of four outcomes:
`resolve`, `propose_action`, `escalate`, `refuse_with_citation`. Every request produces
a persisted Decision Record explaining why.

**What works today:** nothing yet.
