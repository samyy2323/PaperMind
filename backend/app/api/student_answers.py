from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, owned_answer, require_teacher
from app.models import AnswerSheet, User
from app.schemas import OverrideIn
from app.services import results

router = APIRouter(prefix="/student-answers", tags=["review"])


def _answer_json(a, teacher: User) -> dict:
    return results.answer_out(a, for_student=False, names={teacher.id: teacher.name})


@router.patch("/{answer_id}/override")
def override_score(answer_id: int, body: OverrideIn, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Teacher replaces the AI score. A reason is mandatory and the score can't exceed the question's marks."""
    ans = owned_answer(db, answer_id, teacher)
    if not body.reason.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A reason is required when overriding a score")
    if body.score > ans.max_score:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Score must be between 0 and {ans.max_score:g}")
    ans.teacher_override_score, ans.teacher_override_reason = body.score, body.reason.strip()
    ans.overridden_by, ans.overridden_at = teacher.id, datetime.now(timezone.utc)
    results.recompute_status(db.get(AnswerSheet, ans.answer_sheet_id))
    db.commit()
    return _answer_json(ans, teacher)


@router.post("/{answer_id}/review")
def accept_ai_score(answer_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Teacher checked a flagged answer and accepts the AI score as-is."""
    ans = owned_answer(db, answer_id, teacher)
    ans.teacher_reviewed = True
    results.recompute_status(db.get(AnswerSheet, ans.answer_sheet_id))
    db.commit()
    return _answer_json(ans, teacher)
