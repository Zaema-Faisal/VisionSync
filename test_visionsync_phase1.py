"""Isolated task-command tests; uses a disposable SQLite database."""
import importlib, sys, tempfile, pathlib, shutil

ROOT = pathlib.Path(__file__).parent
ROUTER = ROOT / 'visionsync_phase1_router_fixed.py'
NOTES = ROOT / 'notes(4).py'
MEMORY = ROOT / 'memory(3).py'

with tempfile.TemporaryDirectory() as tmp:
    folder = pathlib.Path(tmp)
    shutil.copy2(ROUTER, folder / 'ai_router.py')
    shutil.copy2(NOTES, folder / 'notes.py')
    shutil.copy2(MEMORY, folder / 'memory.py')
    sys.path.insert(0, tmp)
    import notes, ai_router

    def test(command, task_name='Cut vegetables'):
        notes.delete_all_notes()
        task_id=notes.add_note(task_name)
        notes.add_note('Go to the park')
        response=ai_router.handle_notes(command)
        task=notes.get_note(task_id)
        assert task['status']=='completed', (command,response,task)
        assert 'completed' in response.lower(), (command,response)
        print('PASS',command,'=>',response)

    for command in (
        'Mark Cut vegetables as completed.',
        'Mark cut vegetables as done',
        'Complete cut vegetables',
        'Finish cutting vegetables',
        'I finished cutting vegetables',
        'Mark task cut vegetables completed',
        'Mark the cut vegetables task as complete',
    ):
        test(command)
    notes.delete_all_notes()
    notes.add_note('Cut vegetables')
    notes.add_note('Cut flowers')
    response=ai_router.handle_notes('Mark cut as completed')
    assert 'several' in response.lower(), response
    assert len(notes.get_pending_notes())==2
    print('PASS ambiguous task: no modification')
    notes.delete_all_notes()
    notes.add_note('Cut vegetables')
    response=ai_router.handle_notes('Mark washing dishes as completed')
    assert 'couldn\'t find' in response.lower(), response
    assert len(notes.get_pending_notes())==1
    print('PASS unknown task: no modification')
print('ALL TASK COMPLETION TESTS PASSED')
