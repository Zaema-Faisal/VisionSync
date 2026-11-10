from pathlib import Path
from datetime import datetime, date
from collections import deque
from threading import Lock
import traceback
import threading
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ai_router import ask_ai
from reminder_manager import add_reminder, list_reminders, delete_reminder, due_reminders, recent_tasks
from notes import (
    add_note,
    get_pending_notes,
    get_all_notes,
    complete_note,
    reopen_note,
    delete_note_by_id,
)
from voice_test import (
    start_voice_service,
    stop_voice_service,
    get_voice_status,
)

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = BASE_DIR / "visionsync_chatbot_interface.html"

app = FastAPI(title="VisionSync AI", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ReminderCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    due_at: str

class TaskCreate(BaseModel):
    note: str = Field(min_length=1, max_length=500)
    note_date: str | None = None

_voice_events = deque(maxlen=300)
_event_lock = Lock()

def publish_voice_event(event):
    with _event_lock:
        _voice_events.append({
            **event,
            "timestamp": datetime.now().isoformat()
        })


@app.on_event("startup")
def start_voice_automatically():
    """Start the local microphone/wake-word service when VisionSync starts."""
    try:
        result = start_voice_service(publish_voice_event)
        print("[VisionSync] Hands-free wake-word service:", result.get("message", result))
    except Exception as error:
        # Keep the dashboard/API available even if the microphone is unavailable.
        print("[VisionSync] Could not auto-start voice service:", repr(error))

def _task_dict(task, include_status=False):
    # pending rows: (id, note, created_at)
    if include_status:
        task_id, note, task_date, created_at, status = task
        return {
            "id": task_id,
            "note": note,
            "date": task_date,
            "created_at": created_at,
            "status": status,
        }

    task_id, note, created_at = task
    return {
        "id": task_id,
        "note": note,
        "created_at": created_at,
        "date": date.today().isoformat(),
        "status": "pending",
    }

@app.get("/")
def home():
    return {
        "application": "VisionSync",
        "status": "online",
        "dashboard": "/dashboard"
    }

@app.get("/dashboard")
def dashboard():
    if not DASHBOARD_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="VisionSync dashboard HTML not found."
        )
    return FileResponse(DASHBOARD_FILE, media_type="text/html")

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "assistant": "Vision",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        print(f"\n[API received]: {message}")
        response = ask_ai(message)
        if not response:
            raise RuntimeError("AI returned an empty response.")
        print(f"[Vision replied]: {response}")
        return {"success": True, "response": response}
    except Exception as error:
        print("[API ERROR]", repr(error))
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Vision encountered an internal error."
        )

# -------------------- TASK API --------------------

@app.get("/api/tasks")
def list_tasks(view: str = Query(default="pending", pattern="^(pending|all)$")):
    try:
        # Seven calendar days: today plus previous six days.
        tasks = recent_tasks(days=7, pending_only=(view == "pending"))

        return {
            "success": True,
            "view": view,
            "count": len(tasks),
            "tasks": tasks,
        }
    except Exception as error:
        print("[TASK LIST ERROR]", repr(error))
        raise HTTPException(status_code=500, detail="Could not load tasks.")

@app.post("/api/tasks")
def create_task(request: TaskCreate):
    note = request.note.strip()
    if not note:
        raise HTTPException(status_code=400, detail="Task cannot be empty.")

    try:
        task_id = add_note(note, request.note_date)
        return {
            "success": True,
            "message": "Task saved.",
            "task": {
                "id": task_id,
                "note": note,
                "date": request.note_date or date.today().isoformat(),
                "status": "pending",
            }
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        print("[TASK CREATE ERROR]", repr(error))
        raise HTTPException(status_code=500, detail="Could not save task.")

@app.patch("/api/tasks/{task_id}/complete")
def mark_task_complete(task_id: int):
    try:
        if not complete_note(task_id):
            raise HTTPException(
                status_code=404,
                detail="Task not found or already completed."
            )
        return {"success": True, "message": "Task marked completed.", "id": task_id}
    except HTTPException:
        raise
    except Exception as error:
        print("[TASK COMPLETE ERROR]", repr(error))
        raise HTTPException(status_code=500, detail="Could not complete task.")

@app.patch("/api/tasks/{task_id}/reopen")
def reopen_task(task_id: int):
    try:
        if not reopen_note(task_id):
            raise HTTPException(
                status_code=404,
                detail="Task not found or already pending."
            )
        return {"success": True, "message": "Task moved back to pending.", "id": task_id}
    except HTTPException:
        raise
    except Exception as error:
        print("[TASK REOPEN ERROR]", repr(error))
        raise HTTPException(status_code=500, detail="Could not reopen task.")

@app.delete("/api/tasks/{task_id}")
def remove_task(task_id: int):
    try:
        if not delete_note_by_id(task_id):
            raise HTTPException(status_code=404, detail="Task not found.")
        return {"success": True, "message": "Task deleted.", "id": task_id}
    except HTTPException:
        raise
    except Exception as error:
        print("[TASK DELETE ERROR]", repr(error))
        raise HTTPException(status_code=500, detail="Could not delete task.")

# -------------------- VOICE API --------------------

@app.post("/api/voice/start")
def voice_start():
    result = start_voice_service(publish_voice_event)
    return {**result, "voice": get_voice_status()}

@app.post("/api/voice/stop")
def voice_stop():
    result = stop_voice_service()
    return {**result, "voice": get_voice_status()}

@app.get("/api/voice/status")
def voice_status():
    return get_voice_status()

@app.get("/api/voice/events")
def voice_events():
    with _event_lock:
        events = list(_voice_events)
        _voice_events.clear()
    return {"events": events}


@app.get("/api/tasks/archive")
def archived_tasks():
    """Recover older tasks without keeping them in the mirror's seven-day view."""
    from datetime import timedelta
    cutoff=(date.today()-timedelta(days=6)).isoformat()
    rows=get_all_notes()
    archived=[_task_dict(row, include_status=True) for row in rows if row[2]<cutoff]
    return {"success":True,"count":len(archived),"tasks":archived}

# -------------------- PHASE 5: REMINDERS --------------------
@app.get("/api/reminders")
def reminder_list():
    return {"success": True, "reminders": list_reminders()}

@app.post("/api/reminders")
def reminder_add(request: ReminderCreate):
    try:
        return {"success": True, "reminder": add_reminder(request.title, request.due_at)}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.delete("/api/reminders/{reminder_id}")
def reminder_remove(reminder_id: int):
    if not delete_reminder(reminder_id):
        raise HTTPException(status_code=404, detail="Reminder not found.")
    return {"success": True, "id": reminder_id}

_reminder_stop = threading.Event()
_reminder_thread = None

def _reminder_loop():
    while not _reminder_stop.is_set():
        try:
            for item in due_reminders():
                print("[REMINDER DUE]", item["id"], item["title"])
                publish_voice_event({"type":"reminder_due", "id":item["id"],
                                    "title":item["title"], "due_at":item["due_at"]})
        except Exception as exc:
            print("[REMINDER CHECK ERROR]", repr(exc))
        _reminder_stop.wait(5)

@app.on_event("startup")
def start_reminder_loop():
    global _reminder_thread
    _reminder_stop.clear()
    if _reminder_thread is None or not _reminder_thread.is_alive():
        _reminder_thread = threading.Thread(target=_reminder_loop, daemon=True, name="VisionReminders")
        _reminder_thread.start()

@app.on_event("shutdown")
def stop_reminder_loop():
    _reminder_stop.set()
