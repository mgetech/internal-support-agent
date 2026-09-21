-- Schema for internal-support-agent. Applied on first container start via
-- docker-entrypoint-initdb.d, and by any test fixture that provisions a
-- database from scratch. Tables land here as later commits add them.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE employees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    employment_type TEXT NOT NULL CHECK (employment_type IN ('full_time', 'part_time')),
    weekly_hours NUMERIC NOT NULL,
    country TEXT NOT NULL,
    hired_at DATE NOT NULL
);

CREATE TABLE leave_balances (
    employee_id TEXT NOT NULL REFERENCES employees (id),
    year INT NOT NULL,
    entitlement_days NUMERIC NOT NULL,
    taken_days NUMERIC NOT NULL DEFAULT 0,
    pending_days NUMERIC NOT NULL DEFAULT 0,
    PRIMARY KEY (employee_id, year)
);

CREATE TABLE leave_requests (
    id SERIAL PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees (id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    days NUMERIC NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted'
);

CREATE TABLE tickets (
    id SERIAL PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees (id),
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_by TEXT NOT NULL
);

CREATE TABLE known_outages (
    id SERIAL PRIMARY KEY,
    system TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('investigating', 'identified', 'resolved')),
    started_at TIMESTAMPTZ NOT NULL,
    note TEXT
);

-- id format: "<policy>#<heading-slug>#<n>" — stable and ground-truth-addressable;
-- citations and eval scenarios reference these ids directly.
CREATE TABLE policy_chunks (
    id TEXT PRIMARY KEY,
    policy TEXT NOT NULL,
    heading TEXT NOT NULL,
    content TEXT NOT NULL,
    domain TEXT NOT NULL CHECK (domain IN ('hr', 'it')),
    audience TEXT NOT NULL DEFAULT 'all',
    lang TEXT NOT NULL DEFAULT 'en',
    effective_date DATE,
    chunker TEXT NOT NULL DEFAULT 'structural',
    embedding vector(1536),
    ts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX policy_chunks_ts_idx ON policy_chunks USING GIN (ts);
CREATE INDEX policy_chunks_embedding_idx ON policy_chunks USING hnsw (embedding vector_cosine_ops);

-- the approval queue: a gated write stops here and waits for a human decision.
CREATE TABLE pending_actions (
    id SERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    employee_id TEXT NOT NULL REFERENCES employees (id),
    tool TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_approval'
        CHECK (status IN ('pending_approval', 'approved', 'rejected', 'executed', 'expired')),
    decided_by TEXT,
    decided_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- append-only trail: every tool call, proposal, approval, execution and model
-- call, with the actor that performed it. actor: agent | employee:<id> |
-- approver:<id> | system. event: tool_call | action_proposed |
-- action_approved | action_rejected | action_executed | decision_recorded |
-- feedback_received | model_call.
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    payload JSONB NOT NULL,
    at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per request: the persisted "why" — outcome, evidence trail,
-- citations, versions and cost, readable without re-running anything.
CREATE TABLE decision_records (
    request_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees (id),
    channel TEXT NOT NULL CHECK (channel IN ('rest', 'mcp')),
    outcome TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence JSONB NOT NULL,
    citations TEXT[] NOT NULL DEFAULT '{}',
    policy_version TEXT,
    prompt_versions JSONB,
    verifier JSONB,
    model_calls JSONB,
    total_tokens INT NOT NULL DEFAULT 0,
    total_cost_eur NUMERIC NOT NULL DEFAULT 0,
    latency_ms INT,
    trace_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES decision_records (request_id),
    employee_id TEXT NOT NULL REFERENCES employees (id),
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    triage_status TEXT NOT NULL DEFAULT 'new'
        CHECK (triage_status IN ('new', 'reviewed', 'scenario_created', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- rolled up from decision_records; not written to directly by requests.
CREATE TABLE usage_daily (
    day DATE NOT NULL,
    employee_id TEXT NOT NULL REFERENCES employees (id),
    requests INT NOT NULL DEFAULT 0,
    total_tokens INT NOT NULL DEFAULT 0,
    total_cost_eur NUMERIC NOT NULL DEFAULT 0,
    PRIMARY KEY (day, employee_id)
);
