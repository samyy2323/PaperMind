from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, owned_class, require_teacher
from app.core.security import hash_password
from app.models import Class, Enrollment, User
from app.schemas import ClassEnsure, ClassOut, RosterAdd, RosterPatch, RosterStudentOut

router = APIRouter(tags=["classes & roster"])


def get_or_create_class(db: Session, teacher: User, subject: str, sub_domain: str) -> Class:
    klass = db.scalar(
        select(Class).where(Class.teacher_id == teacher.id, Class.subject == subject, Class.sub_domain == sub_domain)
    )
    if klass is None:
        klass = Class(teacher_id=teacher.id, name=f"{subject} ({sub_domain})", subject=subject, sub_domain=sub_domain)
        db.add(klass)
        db.commit()
    return klass


@router.get("/classes", response_model=list[ClassOut])
def list_classes(db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    return db.scalars(select(Class).where(Class.teacher_id == teacher.id).order_by(Class.id)).all()


@router.post("/classes/ensure", response_model=ClassOut)
def ensure_class(body: ClassEnsure, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Idempotent: return the teacher's class for (subject, sub-domain), creating it if needed."""
    return get_or_create_class(db, teacher, body.subject.strip(), body.sub_domain.strip())


def _student_out(u: User) -> RosterStudentOut:
    return RosterStudentOut(id=u.id, name=u.name, roll_number=u.roll_number, email=u.email)


@router.get("/classes/{class_id}/roster", response_model=list[RosterStudentOut])
def get_roster(class_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    owned_class(db, class_id, teacher)
    students = db.scalars(
        select(User).join(Enrollment, Enrollment.student_id == User.id).where(Enrollment.class_id == class_id).order_by(User.roll_number)
    )
    return [_student_out(u) for u in students]


@router.post("/classes/{class_id}/roster", response_model=RosterStudentOut, status_code=status.HTTP_201_CREATED)
def add_student(class_id: int, body: RosterAdd, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    owned_class(db, class_id, teacher)
    roll = body.roll_number.strip()
    student = db.scalar(select(User).where(User.roll_number == roll))
    if student is not None and student.role != "student":
        raise HTTPException(status.HTTP_409_CONFLICT, "That roll number belongs to a non-student account")
    if student is None:
        email = (body.email or f"{roll.lower()}@students.papermind.local").strip().lower()
        if db.scalar(select(User.id).where(User.email == email)):
            raise HTTPException(status.HTTP_409_CONFLICT, "A user with that email already exists")
        student = User(
            name=body.name.strip(), email=email, role="student", roll_number=roll,
            password_hash=hash_password(body.password or roll),
        )
        db.add(student)
        db.flush()
    if db.scalar(select(Enrollment.id).where(Enrollment.student_id == student.id, Enrollment.class_id == class_id)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Student is already on this roster")
    db.add(Enrollment(student_id=student.id, class_id=class_id))
    db.commit()
    return _student_out(student)


def _enrolled_student(db: Session, class_id: int, student_id: int) -> User:
    student = db.scalar(
        select(User).join(Enrollment, Enrollment.student_id == User.id).where(Enrollment.class_id == class_id, User.id == student_id)
    )
    if student is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not on this roster")
    return student


@router.patch("/classes/{class_id}/roster/{student_id}", response_model=RosterStudentOut)
def edit_student(class_id: int, student_id: int, body: RosterPatch, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    owned_class(db, class_id, teacher)
    student = _enrolled_student(db, class_id, student_id)
    if body.name:
        student.name = body.name.strip()
    if body.roll_number and body.roll_number.strip() != student.roll_number:
        roll = body.roll_number.strip()
        if db.scalar(select(User.id).where(User.roll_number == roll)):
            raise HTTPException(status.HTTP_409_CONFLICT, "Roll number already in use")
        student.roll_number = roll
    db.commit()
    return _student_out(student)


@router.delete("/classes/{class_id}/roster/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_student(class_id: int, student_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Removes the student from this class roster (their account and past results are kept)."""
    owned_class(db, class_id, teacher)
    _enrolled_student(db, class_id, student_id)
    db.query(Enrollment).filter_by(class_id=class_id, student_id=student_id).delete()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
