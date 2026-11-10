import importlib.util, sys, types, tempfile, sqlite3, os
from pathlib import Path
from unittest.mock import Mock

base=Path(__file__).resolve().parent
sys.path.insert(0,str(base))
# No user DB is opened: fake modules + in-memory task records.
notes=types.ModuleType('notes')
rows=[]
def get_pending_notes(note_date=None):
    return [(x['id'],x['note'],'now') for x in rows if x['status']=='pending']
def add_note(note, date=None):
    n=len(rows)+1; rows.append({'id':n,'note':note,'status':'pending'}); return n
def get_all_notes(date=None):
    return [(x['id'],x['note'],'today','now',x['status']) for x in rows]
def complete_note(n):
    for x in rows:
        if x['id']==n and x['status']=='pending':x['status']='completed';return True
    return False
def delete_note_by_id(n):
    matches=[x for x in rows if x['id']==n]
    if not matches:return False
    rows.remove(matches[0]);return True
for name,func in [('add_note',add_note),('get_pending_notes',get_pending_notes),('get_all_notes',get_all_notes),('complete_note',complete_note),('delete_note_by_id',delete_note_by_id)]:setattr(notes,name,func)
notes.reopen_note=lambda n:False
notes.find_notes=lambda q:[]
notes.clear_completed_notes=lambda:0
sys.modules['notes']=notes
mem=types.ModuleType('memory')
mem.save_memory=lambda *args:None
mem.get_memory=lambda :''
mem.remember=lambda *args:None
mem.list_memories=lambda :[]
mem.forget_matching=lambda *args:('missing',None)
sys.modules['memory']=mem
ctx=types.ModuleType('context_manager')
class DummyContext:
    def handle_followup(self,*args):return None
    def observe(self,*args):pass
    def reset(self):pass
ctx.context=DummyContext()
sys.modules['context_manager']=ctx
spec=importlib.util.spec_from_file_location('patched_router',base/'ai_router.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
phrases=[
    ('I want to buy milk, add it as a task.','Buy milk'),
    ('I need to walk in the park. Add that as a task.','Walk in the park'),
    ('I want to cook food, add it as a task.','Cook food'),
    ('Add go to the mall as a task','Go to the mall'),
    ('Save cut vegetables to my tasks','Cut vegetables'),
]
for utterance,expected in phrases:
    res=r.handle_notes(utterance)
    assert len(rows)>0 and rows[-1]['note']==expected, (utterance,res,rows)
    assert 'saved' in res.lower() or 'already' in res.lower(), (utterance,res)
    print('PASS',utterance,'->',rows[-1]['note'])
before=len(rows)
r.handle_notes('I want to buy milk, add it as a task.')
assert len(rows)==before, 'Identical task should not be duplicated'
print('PASS duplicate prevention')
res=r.handle_notes('What are my tasks for today?')
assert 'Buy milk' in res and 'Go to the mall' in res
print('PASS fresh list')
res=r.handle_notes('I want to buy a milkshake.')
assert res is None and len(rows)==before
print('PASS normal statement does not create a task')
