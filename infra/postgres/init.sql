CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE text_chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT        NOT NULL,
    source      TEXT        NOT NULL,
    chunk_index INTEGER     NOT NULL,
    content     TEXT        NOT NULL,
    embedding   vector(1536) NOT NULL,
    tsv         TSVECTOR    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata    JSONB       NOT NULL DEFAULT '{}'
);

CREATE TABLE image_assets (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT         NOT NULL,
    source    TEXT         NOT NULL,
    image_url TEXT         NOT NULL,
    caption   TEXT,
    embedding vector(768)  NOT NULL,
    metadata  JSONB        NOT NULL DEFAULT '{}'
);

CREATE INDEX text_chunks_embedding_idx
    ON text_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX text_chunks_tsv_idx
    ON text_chunks
    USING gin (tsv);

CREATE INDEX text_chunks_doc_id_idx
    ON text_chunks (doc_id);

CREATE INDEX image_assets_embedding_idx
    ON image_assets
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX image_assets_doc_id_idx
    ON image_assets (doc_id);
