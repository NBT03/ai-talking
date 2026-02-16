import json
from pathlib import Path

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

def load_memory(session_id):
    path = MEMORY_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["messages"]

def save_memory(session_id, messages):
    path = MEMORY_DIR / f"{session_id}.json"
    data = {
        "conversation_id": session_id,
        "messages": messages
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
