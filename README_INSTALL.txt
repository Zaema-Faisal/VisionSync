VisionSync MagicMirror — Reminders UI

Replace ONLY visionsync_chatbot_interface.html in:
C:\Users\zaema\Desktop\VisionSync

Back up your existing file first. Do not replace Python files or databases.
Start: python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
Open: http://127.0.0.1:8000/dashboard and press Ctrl+F5.

The new Upcoming Reminders panel reads GET /api/reminders.
It shows pending reminders, their titles and local dates/times.
It refreshes every 10 seconds, after voice replies and on demand.
If the endpoint is unavailable, it displays an honest error, not fake reminders.
The page is compact and uses internal scrolling within the task and reminder sections.
