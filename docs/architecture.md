# Architecture

This file explains how the agent is built and why. It describes what exists today. What
is planned is in the roadmap in the README.

## How a request flows

```mermaid
flowchart LR
    A["Request"] --> B["1. classify"]
    B --> C["2. agent"]
    C -- tool calls --> T["3. tools"]
    T -- results --> C
    T -.-> K[("Policy retrieval - RAG")] & H[("Records and outages")]
    C -- draft answer --> V["4. verify citations"]
    V -- bad citation: retry once --> C
    V -- passes --> F["5. finalize"]
    F --> D[("Decision Record + audit event")]

     A:::Ash
     B:::Sky
     C:::Sky
     T:::Peach
     K:::Aqua
     H:::Aqua
     V:::Peach
     F:::Pine
     D:::Aqua
    classDef Aqua stroke-width:1px, stroke-dasharray:none, stroke:#46EDC8, fill:#DEFFF8, color:#378E7A
    classDef Sky stroke-width:1px, stroke-dasharray:none, stroke:#374D7C, fill:#E2EBFF, color:#374D7C
    classDef Rose stroke-width:1px, stroke-dasharray:none, stroke:#FF5978, fill:#FFDFE5, color:#8E2236
    classDef Pine stroke-width:1px, stroke-dasharray:none, stroke:#254336, fill:#27654A, color:#FFFFFF
    classDef Peach stroke-width:1px, stroke-dasharray:none, stroke:#FBB35A, fill:#FFEFDB, color:#8F632D
    classDef Ash stroke-width:1px, stroke-dasharray:none, stroke:#999999, fill:#EEEEEE, color:#000000
```

The dotted lines show where the data comes from. The retrieval (RAG) is the
`search_policies` tool. The chunk ids it returns are the only ids an answer may cite.
Every path ends in `finalize`, which saves the Decision Record.

A request ends in `escalate` at these points:

| Step | Reason |
|---|---|
| classify | the request is outside HR and IT, or the classification failed |
| agent | no model answered, or the model sent tool arguments that were not valid JSON |
| agent | the next tool calls would go over the tool limit |
| verify | the answer cited a chunk this request did not retrieve, a second time |

The `refuse` node then writes the answer in the language of the request.

Every request ends in exactly one outcome:

| Outcome | When |
|---|---|
| `resolve` | the answer passed the citation check and no action was proposed |
| `propose_action` | the answer passed the check and the request proposed at least one action |
| `escalate` | a person has to take over: out of scope, a model failure, the tool limit, or a failed check |
| `refuse_with_citation` | part of the outcome list, but nothing sets it yet |

The rule is to escalate instead of guessing. Whenever the agent cannot continue safely,
it hands the request to a person.

## Data

All data is synthetic and comes from one deterministic generator (`data/generate.py`):
12 employees with leave balances, two IT outages and five HR and IT policies. The same
run always gives the same data. Some edge cases are planted on purpose, such as an
employee with no days left.

A heading-aware chunker splits each policy into sections. A chunk id has the form
`<policy>#<heading-slug>#<n>`, for example `vacation-policy#entitlement#0`. Citations and
the evaluation refer to chunks by this id, so the format is stable. Each section is
embedded with `text-embedding-3-small` (1536 dimensions) and stored with pgvector.

## Identity

The logged-in employee is bound once per request into a `RequestContext`. Tools and the
audit log read the employee and the request id from there. No tool takes an employee id
as an argument, so the model cannot ask for another employee's data. A test checks this
for every tool schema, and another test checks that no tool runs without a bound request.

## Tools

| Tool | What it does |
|---|---|
| `get_leave_balance` | the employee's own vacation days |
| `get_known_outages` | IT outages that are still open |
| `search_policies` | policy sections, found by meaning and by keywords |
| `submit_leave_request` | proposes a leave request |
| `create_ticket` | proposes an IT ticket |

The queries are typed and use parameters. The model never writes SQL.

**Writes are proposals.** The two write tools check the rules in code and then add a row
to `pending_actions`. They do not book leave or open a ticket. A person with the right
role approves the row later, in a separate flow. Sending the same proposal twice returns
the first one.

**Search.** `search_policies` joins two rankings with reciprocal rank fusion: vector
similarity (pgvector, cosine) and Postgres full-text search. It returns the top five
sections.

## The graph

The graph is a LangGraph `StateGraph`. The state holds the messages, the classification,
the chunk ids that the request retrieved, the ids of the proposed actions, the number of
tool calls used, the outcome and the evidence.

| Node | What it does |
|---|---|
| `classify` | one call to the small model with a strict JSON schema: family (`hr`, `it`, `other`), risk (`read`, `write`, `out_of_scope`) and language (`en`, `de`) |
| `agent` | one model call with the tools. The reply may ask for tool calls |
| `tools` | runs the requested calls with LangGraph's `ToolNode`, adds the chunk ids of each search to the citation whitelist, and counts the calls |
| `tool_limit_reached` | ends the request when the limit is reached |
| `verify` | the deterministic citation check on the draft answer |
| `refuse` | writes the answer for every escalation, in the language of the request |
| `finalize` | builds and saves the Decision Record. It runs on every path |

**Why classify is a separate step.**
- A request that is out of scope escalates before the agent runs. No tool is offered
  and none is called.
- The answer must fit a strict schema. An answer that does not fit escalates, so the
  graph never guesses a value.
- The language comes from the classification, so the refusal text is chosen without
  another model call.
- It uses the small model. A request that is clearly out of scope costs one small call
  and no agent loop.

The cost is one more model call on every request. A wrong `out_of_scope` escalates a
request that the agent could have answered. The classification is also a model answer,
so this check is not fully deterministic.

**Tool limit.** A request may use at most 8 tool calls (`MAX_TOOL_CALLS_PER_REQUEST`).
If the calls used plus the calls just requested would go over the limit, none of the new
calls run, and the request escalates. The calls of one reply are never cut to fit.

**Evidence.** Every step adds items to the request's evidence list: the classification,
each tool result, each model call and fallback, each check result, each retry and each
escalation with its reason. The list is the source for the Decision Record.

**Design choices**
- A node returns only the fields it changes. Lists such as the evidence and the chunk ids
  add up across nodes. Values such as the outcome and the tool counter are replaced.
- A proposal is a row in the database, and the graph ends after it. It does not pause for
  approval. The approver is another person who may answer much later, and the approval
  must be in the audit log. This is why the graph has no checkpointer and no
  `interrupt()`.
- Each request starts with one user message. The graph keeps nothing between requests.
- The model retry is our own code and not LangGraph's `RetryPolicy`. See the model client
  below.

## Citation check

Every `[chunk:<id>]` in the draft answer must be a chunk that this request retrieved. The
check is plain code, and no model judges it.

- If the answer passes, the request ends in `resolve` or `propose_action`.
- If it fails for the first time, the objection is saved in the state. The agent runs
  again and sees the objection as a system message. This is a loop in the graph, because
  no error happened and the second try needs a different input.
- If it fails a second time, the request ends in `escalate`.

## Decision Records

Every request that reaches the end of the graph saves one row in `decision_records`:

- the outcome and one sentence on why. The sentence is built from the evidence with fixed
  wording, so no model writes it
- the evidence in order, and the citations of the final answer
- the policy version, the prompt versions and every check result
- each model call with tokens and cost in euros, the totals, and the time taken
- the trace id, which is empty because tracing is not built yet

The record and the final `decision_recorded` audit event are saved in one transaction.
The request id is unique, so a request can have only one record.

**Policy version.** Each policy section is stored with the version from its policy file's
frontmatter. A record names the version that the database holds, not the version of the
files on disk. If sections of several versions are stored, the record names all of them.

## Orphan check

A proposal is saved when the tool runs. The record is saved later, in `finalize`. If the
process crashes between the two, the proposal has no record.

`flag_orphaned_actions` finds these rows and writes an `orphan_flagged` audit event for
each, with the actor `system`. It does not change the action.

- **Grace period.** Only actions older than `ORPHAN_GRACE_MINUTES` (default 10) are
  checked, so a request that is still running is never flagged.
- **No duplicates.** An action that already has an event is skipped, so running the check
  at every start adds nothing new. The check uses the action id, because one request can
  propose several actions.
- **Concurrent starts.** A transaction-level advisory lock makes two processes that start
  together flag each action only once.

We did not use one transaction for the whole request. It would stay open during the model
calls, hold locks, and roll back the audit trail of a crashed request. Short transactions
and a later check cost less. Nothing calls the check at startup yet, because there is no
API.

## Audit log

`audit_log` is append-only. Every tool call, proposal, model call and decision is written
with the request id and the actor: `agent`, `employee:<id>`, `approver:<id>` or `system`.
The request id and the employee come from the request context, never from an argument.

## Model client and cost

All model calls go through one client that uses the OpenAI SDK and the Responses API with
Azure OpenAI.

- **Retry.** The client retries connection errors and the status codes 408, 409, 429 and
  5xx, up to two more times, with a delay that doubles. The SDK's own retries are turned
  off, so the limit is in one place.
- **Fallback.** If every try on the main model fails, the client tries the fallback model
  when one is set. If that fails too, it raises an error, and the graph escalates.
- **Evidence.** Every try, every switch to the fallback and the final failure is an
  evidence item, so the fallback chain can be read from the record alone.
- **Cost.** Each successful call gets its token counts and its cost in euros from a price
  table in the settings. Creating the client fails if a model has no price.

We wrote this retry and did not use LangGraph's `RetryPolicy`. It would run the whole node
again. It could not switch to a fallback model or record each try.

## Prompts

The agent prompt and the classifier prompt are YAML files in `prompts/`, each with an id
and a version. A test stores a checksum of every prompt, and it fails if the text changes
and the version does not.

## Tests

`make test` needs no model keys.

- **Unit tests** need no database. The model client, the cost, the verifier, the nodes and
  the state are tested with stand-ins.
- **Component tests** run the real graph with a scripted model and fake tools. The
  Decision Record is collected in a list, so there is no database.
- **Integration tests** use the compose Postgres (`make db-up`). They cover the tools, the
  schema, the audit log, saving Decision Records, the policy version and the orphan check.
  A comment marks them, and they are skipped when no database is reachable.
- **Security tests** run for every tool: no identity argument, no run without a bound
  request, and write tools only add to the approval queue.

## Known limits

- A policy claim that has no citation at all passes the check. Claim extraction is planned.
- The trace id is always empty until tracing is built.
- `refuse_with_citation` is never set.
- If saving the record fails in `finalize`, the request fails.
- The policy version is read once at `finalize`. Re-seeding at that moment could make the
  record name the new version.
- No test runs the real graph with the real database from start to end, and no test calls
  a live model.
- There is no API yet, so nothing calls the orphan check at startup.
