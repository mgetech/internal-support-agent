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
