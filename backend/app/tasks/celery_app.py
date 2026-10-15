from celery import Celery

from app.core.config import get_settings

_s = get_settings()

celery_app = Celery("papermind", broker=_s.REDIS_URL, backend=_s.REDIS_URL, include=["app.tasks.tasks"])
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # model inference is heavy: one sheet per worker slot at a time
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)
