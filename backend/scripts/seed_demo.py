"""Create demo accounts + a class with a small roster (idempotent).

    python scripts/seed_demo.py

Teacher: kulkarni@college.edu / demo1234      Student: rohan@students.papermind.local / demo1234
The frontend's one-click demo login (VITE_USE_MOCK=false) uses exactly these credentials.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.core.db import session_scope  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models import Class, Enrollment, User  # noqa: E402

PASSWORD = "demo1234"
ROSTER = [
    ("Rohan Mehta", "1NI22IS041", "rohan@students.papermind.local"),
    ("Ananya Rao", "1NI22IS008", None),
    ("Priya Nair", "1NI22IS057", None),
    ("Karthik Suresh", "1NI22IS062", None),
    ("Meera Iyer", "1NI22IS073", None),
]


def main() -> None:
    with session_scope() as db:
        teacher = db.scalar(select(User).where(User.email == "kulkarni@college.edu"))
        if teacher is None:
            teacher = User(name="Dr. S. Kulkarni", email="kulkarni@college.edu", role="teacher", password_hash=hash_password(PASSWORD))
            db.add(teacher)
            db.flush()
        klass = db.scalar(select(Class).where(Class.teacher_id == teacher.id, Class.sub_domain == "Object-Oriented Programming"))
        if klass is None:
            klass = Class(
                teacher_id=teacher.id, name="OOP with Java — 4th sem IS", subject="Object-Oriented Programming with Java",
                sub_domain="Object-Oriented Programming",
            )
            db.add(klass)
            db.flush()
        for name, roll, email in ROSTER:
            student = db.scalar(select(User).where(User.roll_number == roll))
            if student is None:
                student = User(
                    name=name, email=email or f"{roll.lower()}@students.papermind.local", role="student",
                    roll_number=roll, password_hash=hash_password(PASSWORD),
                )
                db.add(student)
                db.flush()
            if not db.scalar(select(Enrollment.id).where(Enrollment.student_id == student.id, Enrollment.class_id == klass.id)):
                db.add(Enrollment(student_id=student.id, class_id=klass.id))
        print(f"Seeded teacher {teacher.email}, class #{klass.id}, {len(ROSTER)} students (password: {PASSWORD})")


if __name__ == "__main__":
    main()
