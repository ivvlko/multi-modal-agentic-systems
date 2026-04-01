#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "http://localhost:8001")
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

DB_DSN = "postgresql://{user}:{password}@{host}:{port}/{db}".format(
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=os.environ.get("POSTGRES_PORT", "5432"),
    db=os.environ["POSTGRES_DB"],
)


def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(".", start + CHUNK_SIZE // 2, end)
            if boundary > 0:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    response = await client.post("/embed/text", json={"texts": texts})
    response.raise_for_status()
    return response.json()["embeddings"]


async def embed_images(client: httpx.AsyncClient, urls: list[str]) -> list[list[float]]:
    response = await client.post("/embed/image", json={"image_urls": urls})
    response.raise_for_status()
    return response.json()["embeddings"]


async def seed_articles(pool: asyncpg.Pool, client: httpx.AsyncClient, articles_dir: Path) -> None:
    txt_files = sorted(articles_dir.glob("*.txt"))
    if not txt_files:
        print("  no .txt files found in", articles_dir)
        return

    for txt_file in txt_files:
        doc_id = txt_file.stem
        content = txt_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(content)
        print(f"  {txt_file.name}: {len(chunks)} chunk(s)")

        embeddings = await embed_texts(client, chunks)

        async with pool.acquire() as conn:
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                await conn.execute(
                    """
                    INSERT INTO text_chunks (doc_id, source, chunk_index, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    doc_id,
                    txt_file.name,
                    idx,
                    chunk,
                    str(embedding),
                    "{}",
                )
        print(f"  ✓ {txt_file.name}")


async def seed_images(pool: asyncpg.Pool, client: httpx.AsyncClient, images_file: Path) -> None:
    images: list[dict] = json.loads(images_file.read_text(encoding="utf-8"))

    for img in images:
        print(f"  {img['doc_id']} — {img['image_url'][:60]}...")
        try:
            embeddings = await embed_images(client, [img["image_url"]])
        except httpx.HTTPStatusError as exc:
            print(f"  ✗ skipped ({exc.response.status_code})")
            continue
        except httpx.HTTPError as exc:
            print(f"  ✗ skipped ({exc})")
            continue

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO image_assets (doc_id, source, image_url, caption, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                ON CONFLICT DO NOTHING
                """,
                img["doc_id"],
                img["source"],
                img["image_url"],
                img.get("caption"),
                str(embeddings[0]),
                json.dumps(img.get("metadata", {})),
            )
        print(f"  ✓ {img['doc_id']}")


async def main() -> None:
    data_dir = Path(__file__).parent / "example_data"

    print("Checking embedder health...")
    async with httpx.AsyncClient(base_url=EMBEDDER_URL, timeout=120.0) as client:
        try:
            resp = await client.get("/health")
            resp.raise_for_status()
        except Exception as exc:
            print(f"Embedder not reachable at {EMBEDDER_URL}: {exc}")
            sys.exit(1)
        print(f"  embedder OK ({EMBEDDER_URL})")

        print("\nConnecting to database...")
        pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=3)
        print(f"  database OK\n")

        print("Seeding text articles...")
        await seed_articles(pool, client, data_dir / "articles")

        print("\nSeeding images...")
        await seed_images(pool, client, data_dir / "images.json")

        await pool.close()

    print("\nSeed complete.")


if __name__ == "__main__":
    asyncio.run(main())
