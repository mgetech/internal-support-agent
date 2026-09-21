-- Schema for internal-support-agent. Applied on first container start via
-- docker-entrypoint-initdb.d, and by any test fixture that provisions a
-- database from scratch. Tables land here as later commits add them.

CREATE EXTENSION IF NOT EXISTS vector;
