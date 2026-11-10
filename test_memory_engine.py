"""Run: python test_memory_engine.py (uses a throwaway test directory)."""
import tempfile
import shutil
import sys
from pathlib import Path

src = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    for name in ('memory.py', 'ai_router.py'):
        shutil.copyfile(src / name, folder / name)
    # Existing SQLite DB / schema is copied for migration testing, never modified.
    import sqlite3
    with sqlite3.connect(folder/'vision_memory.db') as c:
        c.execute('CREATE TABLE memories(id INTEGER PRIMARY KEY AUTOINCREMENT,user TEXT,information TEXT,timestamp TEXT)')
        c.execute('INSERT INTO memories(user,information,timestamp) VALUES(?,?,?)', ('Test','I used to like tea','2026-01-01'))
    (folder/'notes.py').write_text('''
from datetime import date
_tasks=[]
def add_note(text, note_date=None):
    i=len(_tasks)+1
    _tasks.append((i,text,'now','pending'))
    return i
def get_pending_notes(*args): return [(i,t,stamp) for i,t,stamp,status in _tasks if status=='pending']
def get_all_notes(*args): return [(i,t,date.today().isoformat(),stamp,status) for i,t,stamp,status in _tasks]
def complete_note(i):
    for n,r in enumerate(_tasks):
        if r[0]==i and r[3]=='pending': _tasks[n]=(r[0],r[1],r[2],'completed'); return True
    return False
def reopen_note(i): return False
def delete_note_by_id(i): return False
def find_notes(*args): return []
def clear_completed_notes(*args): return 0
''',encoding='utf-8')
    sys.path.insert(0,str(folder))
    import memory
    import ai_router
    assert 'I used to like tea' in memory.get_memory(), 'Legacy memory lost'
    assert 'saved' in ai_router.ask_ai('Remember that I prefer short answers').lower()
    assert len(memory.list_memories()) == 1
    assert len(memory.search_memories('short answers')) == 1
    assert 'saved' in ai_router.ask_ai('Remember that I prefer short answers').lower()
    assert len(memory.list_memories()) == 1, 'Duplicate memory'
    assert 'short answers' in ai_router.ask_ai('What do you remember about me?')
    assert 'Saved to today' in ai_router.ask_ai('Add another task, cook food'), 'Task regression'
    assert 'Cook food' in ai_router.ask_ai('Show my pending tasks for today')
    assert 'forgot' in ai_router.ask_ai('Forget I prefer short answers').lower()
    assert memory.list_memories() == []
    assert 'I used to like tea' in memory.get_memory(), 'Legacy content changed'
    print('PASS: legacy DB retained, save, recall, search, dedupe, forget and task commands.')
