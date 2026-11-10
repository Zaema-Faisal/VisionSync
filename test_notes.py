from notes import add_note, get_today_notes


add_note("Finish the Vision smart mirror project.")

print("\nToday's notes:\n")

notes = get_today_notes()

for note in notes:
    print(note[0], "-", note[1])