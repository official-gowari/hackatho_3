import sqlite3

conn = sqlite3.connect("placement_final.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM students")
count = cursor.fetchone()[0]

print("Total Students:", count)

conn.close()
