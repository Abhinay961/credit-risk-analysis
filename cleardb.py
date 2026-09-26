import sqlite3
conn = sqlite3.connect("credit_history.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
DELETE FROM history;

""")
conn.commit()
print('cleared')