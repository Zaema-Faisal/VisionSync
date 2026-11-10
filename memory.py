"""VisionSync Memory Engine v1.0.
Keeps the existing `memories` table intact and adds `memory_items` for
explicit, editable, local memories. No cloud calls are made by this module.
"""
import sqlite3
import re
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DATABASE = str(BASE_DIR / "vision_memory.db")

def _conn():
    c = sqlite3.connect(DATABASE, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=10000")
    return c

def create_database():
    with _conn() as c:
        # Backwards compatible with the user's original memory.py
        c.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT, information TEXT, timestamp TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE COLLATE NOCASE,
            information TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'personal',
            updated_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _key(value):
    v = re.sub(r"\s+", " ", str(value).strip().lower())
    v = re.sub(r"[^a-z0-9 ]", "", v)
    return v[:120]

def save_memory(user, information):
    """Legacy API retained. No change to existing callers."""
    info = str(information or "").strip()
    if not info:
        raise ValueError("Memory text cannot be blank.")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO memories(user,information,timestamp) VALUES(?,?,?)",
            (str(user), info, datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
        return cur.lastrowid

def remember(information, category="personal", key=None):
    """Save an explicit fact; exact duplicate or key overwrites instead of duplicating."""
    info = re.sub(r"\s+", " ", str(information or "")).strip().rstrip(".")
    if not info or len(info) > 1000:
        raise ValueError("Memory must contain 1 to 1000 characters.")
    identity = _key(key or info)
    if not identity:
        raise ValueError("Memory key cannot be blank.")
    stamp = _now()
    with _conn() as c:
        c.execute("""
            INSERT INTO memory_items(key, information, category, updated_at, created_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                information=excluded.information,
                category=excluded.category,
                updated_at=excluded.updated_at
        """, (identity, info, str(category), stamp, stamp))
        row = c.execute("SELECT id FROM memory_items WHERE key = ?", (identity,)).fetchone()
        return row["id"]

def list_memories(limit=50):
    with _conn() as c:
        rows = c.execute("""
            SELECT id,key,information,category,updated_at
            FROM memory_items ORDER BY updated_at DESC,id DESC LIMIT ?
        """, (min(max(int(limit),1),200),)).fetchall()
        return [dict(row) for row in rows]

def search_memories(query, limit=8):
    """
    Retrieve relevant memories with lightweight lexical ranking.

    Searches both the editable memory_items table and the legacy memories
    table. This keeps retrieval local and requires no additional packages.
    """
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "how", "i", "if", "in", "into", "is", "it", "me", "my",
        "of", "on", "or", "our", "please", "should", "so", "that", "the",
        "their", "them", "there", "these", "they", "this", "to", "us",
        "was", "we", "what", "when", "where", "which", "who", "why",
        "will", "with", "would", "you", "your", "user", "assistant"
    }

    def tokens(value):
        result = set()
        for word in re.findall(r"[a-z0-9]+", str(value or "").lower()):
            if len(word) <= 1 or word in stopwords:
                continue
            # Small, dependency-free normalisation for common English endings.
            if len(word) > 5 and word.endswith("ing"):
                word = word[:-3]
            elif len(word) > 4 and word.endswith("ed"):
                word = word[:-2]
            elif len(word) > 4 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            result.add(word)
        return result

    query_text = re.sub(r"\s+", " ", str(query or "")).strip().lower()
    query_terms = tokens(query_text)
    if not query_terms:
        return []

    try:
        max_results = min(max(int(limit), 1), 20)
    except (TypeError, ValueError):
        max_results = 8

    candidates = list_memories(200)

    # Keep older records searchable without altering the legacy table.
    with _conn() as c:
        legacy_rows = c.execute("""
            SELECT id, information, timestamp
            FROM memories
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    for row in legacy_rows:
        info = str(row["information"] or "").strip()
        if info:
            candidates.append({
                "id": row["id"],
                "key": "",
                "information": info,
                "category": "legacy",
                "updated_at": row["timestamp"] or "",
                "legacy": True,
            })

    ranked = []
    for row in candidates:
        info = str(row.get("information", ""))
        searchable = (
            f"{info} {row.get('key', '')} {row.get('category', '')}"
        ).lower()
        overlap = query_terms & tokens(searchable)
        if not overlap:
            continue

        coverage = len(overlap) / max(len(query_terms), 1)
        score = (len(overlap) * 2.0) + coverage
        normalized_info = re.sub(r"\s+", " ", info.lower())
        if len(query_text) >= 5 and query_text in normalized_info:
            score += 4.0

        # Avoid weak single-word matches for longer questions.
        minimum_overlap = 1 if len(query_terms) <= 4 else 2
        if len(overlap) < minimum_overlap and coverage < 0.5:
            continue

        result = dict(row)
        result["relevance"] = round(score, 3)
        ranked.append(result)

    ranked.sort(
        key=lambda item: (
            item["relevance"],
            str(item.get("updated_at", "")),
            int(item.get("id", 0))
        ),
        reverse=True
    )
    return ranked[:max_results]

def forget_memory(memory_id):
    with _conn() as c:
        cur = c.execute("DELETE FROM memory_items WHERE id=?", (int(memory_id),))
        return cur.rowcount > 0

def forget_matching(query):
    """Delete only when there is one unambiguous exact/substring match."""
    q = _key(query)
    if not q:
        return ("missing", None)
    rows = list_memories(200)
    matches = [x for x in rows if q == _key(x["information"]) or q == x["key"]]
    if not matches:
        matches = [x for x in rows if q in _key(x["information"])]
    if len(matches) != 1:
        return ("ambiguous" if matches else "missing", matches)
    match = matches[0]
    forget_memory(match["id"])
    return ("deleted", match)

def get_memory():
    """Legacy-facing string. Reads old and new tables; stays bounded."""
    with _conn() as c:
        old = c.execute(
            "SELECT information FROM memories ORDER BY id DESC LIMIT 8"
        ).fetchall()
    new = list_memories(20)
    text = [x["information"] for x in new]
    for row in old:
        if row["information"] not in text:
            text.append(row["information"])
    return "\n".join(text[:25])

create_database()
