import sqlite3
import random

conn = sqlite3.connect('placement.db')
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS students;

CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cgpa REAL,
    internships INTEGER,
    projects INTEGER,
    communication INTEGER,
    placed INTEGER
);
""")

conn.commit()


for i in range(500):   # You can change to 300/1000

    cgpa = round(random.uniform(5.0, 9.9), 2)
    internships = random.randint(0, 4)
    projects = random.randint(0, 6)
    communication = random.randint(1, 10)

    score = 0

    if cgpa >= 7:
        score += 1

    if internships >= 1:
        score += 1

    if projects >= 2:
        score += 1

    if communication >= 6:
        score += 1

    if score >= 3:
        placed = 1
    else:
        placed = 0

    # Insert
    cur.execute("""
    INSERT INTO students (cgpa, internships, projects, communication, placed)
    VALUES (?, ?, ?, ?, ?)
    """, (cgpa, internships, projects, communication, placed))

    conn.commit()

    # Print (Optional)
    print("Student", i+1,
          "| CGPA:", cgpa,
          "| Internships:", internships,
          "| Projects:", projects,
          "| Communication:", communication,
          "| Placed:", "Yes" if placed else "No")


conn.close()

print("\nData Generation Completed!")