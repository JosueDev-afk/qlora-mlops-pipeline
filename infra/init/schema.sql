-- Operational (OLTP) schema. English identifiers by project convention.

CREATE TABLE company (
    id      UUID PRIMARY KEY,
    name    TEXT NOT NULL,
    sector  TEXT
);

CREATE TABLE contact (
    id                 UUID PRIMARY KEY,
    company_id         UUID REFERENCES company(id),
    name               TEXT,
    phone              TEXT,
    email              TEXT,
    validation_status  JSONB NOT NULL DEFAULT '{}',  -- per field: validated | unvalidated | pending
    version            INT  NOT NULL DEFAULT 1,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE appointment (
    id                UUID PRIMARY KEY,
    contact_id        UUID REFERENCES contact(id),
    scheduled_at      TIMESTAMPTZ NOT NULL,
    location          TEXT,
    status            TEXT NOT NULL,                 -- pending | confirmed | rescheduled | cancelled
    rescheduled_from  UUID REFERENCES appointment(id)
);

CREATE TABLE call (
    id              UUID PRIMARY KEY,
    flow            TEXT NOT NULL,                   -- validate_contact | confirm_appointment
    contact_id      UUID REFERENCES contact(id),
    appointment_id  UUID REFERENCES appointment(id),
    model_version   TEXT NOT NULL,                   -- lineage reaches production
    prompt_hash     TEXT NOT NULL,
    outcome         TEXT,                            -- completed | rejected | partial | failed
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ
);

CREATE TABLE turn (
    id                 UUID PRIMARY KEY,
    call_id            UUID REFERENCES call(id) ON DELETE CASCADE,
    idx                INT  NOT NULL,
    speaker            TEXT NOT NULL,                -- agent | person
    transcript         TEXT,
    asr_confidence     REAL,
    structured_output  JSONB,
    graph_node         TEXT,
    latency_ms         INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (call_id, idx)
);

CREATE INDEX turn_call_idx      ON turn (call_id, idx);
CREATE INDEX call_model_version ON call (model_version, prompt_hash);
