import random
import sqlite3
from datetime import datetime

DB_NAME = "placement_final.db"

def already_ran_today():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    today = datetime.now().date()

    cursor.execute("""
    SELECT COUNT(*) FROM students
    WHERE DATE(generated_at) = ?
    """, (today,))

    count = cursor.fetchone()[0]

    conn.close()

    return count > 0


# --- Feature Generator ---
def generate_student():
    cgpa = round(random.uniform(5.0, 9.8), 2)
    backlogs = random.choice([0, 1])
    internship = random.choice([0, 1, 2])
    projects = random.randint(0, 5)
    depth = random.choice([1, 2, 3])
    communication = random.randint(1, 10)
    problem_solving = random.randint(1, 10)

    return cgpa, backlogs, internship, projects, depth, communication, problem_solving


# --- Score Calculator ---
def calculate_score(cgpa, backlogs, internship, projects, depth, comm, prob):

    score = 0

    score += (cgpa / 10) * 15

    if backlogs == 1:
        score -= 20

    score += internship * 10
    score += projects * 2
    score += depth * 5
    score += comm * 1
    score += prob * 2

    score = max(0, min(100, score))

    return score


# --- Placement Logic ---
def assign_placement(score):

    if score >= 70:
        return 1
    elif 40 <= score < 70:
        return random.choice([0, 1])
    else:
        return 0


# --- Insert Into DB ---
def insert_student(data):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO students
    (cgpa, backlogs, internship_relevance, projects_count,
     project_depth, communication, problem_solving,
     readiness_score, placement, generated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()


# --- Batch Generator ---
def generate_batch():

    for _ in range(100):

        cgpa, backlogs, internship, projects, depth, comm, prob = generate_student()

        score = calculate_score(cgpa, backlogs, internship, projects, depth, comm, prob)
        placement = assign_placement(score)

        timestamp = datetime.now()

        insert_student((cgpa, backlogs, internship, projects, depth,
                        comm, prob, score, placement, timestamp))

if __name__ == "__main__":

    if already_ran_today():
        print("Data already generated today. Skipping...")
    else:
        generate_batch()
        print("100 students generated successfully")