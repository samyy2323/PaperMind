"""Celery task definitions. Tasks are chained; each returns the sheet id for the next one."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from celery import chain

from app.core.config import get_settings
from app.tasks import pipeline
from app.tasks.celery_app import celery_app

log = logging.getLogger(__name__)

# thread mode mimics a `--concurrency=1` Celery worker: one sheet at a time (OCR + LLM are memory hungry)
_dev_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")


@celery_app.task(name="pipeline.ocr")
def run_ocr(sheet_id: int, image_keys: list[str]) -> int:
    return pipeline.step_ocr(sheet_id, image_keys)


@celery_app.task(name="pipeline.correction")
def run_correction(sheet_id: int) -> int:
    return pipeline.step_correction(sheet_id)


@celery_app.task(name="pipeline.segmentation")
def run_segmentation(sheet_id: int) -> int:
    return pipeline.step_segmentation(sheet_id)


@celery_app.task(name="pipeline.grading")
def run_grading(sheet_id: int) -> int:
    return pipeline.step_grading(sheet_id)


@celery_app.task(name="pipeline.aggregate")
def run_aggregate(sheet_id: int) -> int:
    return pipeline.step_aggregate(sheet_id)


def build_chain(sheet_id: int, image_keys: list[str]):
    return chain(
        run_ocr.s(sheet_id, image_keys),
        run_correction.s(),
        run_segmentation.s(),
        run_grading.s(),
        run_aggregate.s(),
    )


def enqueue_pipeline(sheet_id: int, image_keys: list[str]) -> None:
    """Start grading for an answer sheet without blocking the request.

    PIPELINE_MODE=celery  -> publish the chain to Redis for a Celery worker (production)
    PIPELINE_MODE=thread  -> run the same chain in a background thread (local dev without Redis)
    PIPELINE_MODE=inline  -> run synchronously (tests)
    """
    mode = get_settings().PIPELINE_MODE
    sig = build_chain(sheet_id, image_keys)
    if mode == "celery":
        sig.apply_async()
    elif mode == "thread":
        _dev_executor.submit(sig.apply)
    elif mode == "inline":
        sig.apply()
    else:
        raise ValueError(f"Unknown PIPELINE_MODE: {mode}")
