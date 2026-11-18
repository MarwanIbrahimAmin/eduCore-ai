"""RAG-grounded assessment generation, split to stay within provider token limits."""

import json

from pydantic import ValidationError

from educore.teacher_ai.core.llm_client import (
    LLMRequestError,
    LLMTruncatedError,
    llm_client,
)
from .prompts import SYSTEM_PROMPT, build_assessment_prompt
from .schemas import AssessmentRequest, AssessmentResult, GeneratedAssessment
from .validator import validate_generated_assessment


class AssessmentGenerationError(Exception):
    """Raised when assessment generation or validation fails."""


def _one_batch(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
) -> AssessmentResult:
    """Generate one compact teacher-key batch, correcting malformed output once."""
    payload = request.model_dump(mode="json")
    source = (source_material or "")[:3000]
    prompt = build_assessment_prompt(payload, source_material=source)
    budget = min(max_tokens, 4096, max(1800, 900 + request.question_count * 650))
    last_error = None

    for attempt in range(2):
        try:
            raw_response = llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=budget,
            )
        except LLMTruncatedError:
            if attempt:
                raise
            source = source[:1000]
            prompt = build_assessment_prompt(payload, source_material=source) + (
                "\n\nCOMPACT RETRY: Keep each question, explanation, and marking point brief. "
                "Return all requested questions in one complete JSON object."
            )
            continue
        except LLMRequestError as exc:
            if exc.status_code != 413 or attempt:
                raise
            source = source[:1000]
            prompt = build_assessment_prompt(payload, source_material=source)
            budget = min(2048, budget)
            continue

        try:
            data = json.loads(raw_response)
            if not isinstance(data, dict):
                raise AssessmentGenerationError("The model response must be a JSON object.")
            teacher_data = data.get("teacher_version", data)
            teacher_version = AssessmentResult.model_validate(teacher_data)

            student_data = teacher_version.model_dump(mode="json")
            for question in student_data["questions"]:
                question["correct_answer"] = None
                question["explanation"] = ""
                question["marking_scheme"] = []
            result = GeneratedAssessment(
                student_version=AssessmentResult.model_validate(student_data),
                teacher_version=teacher_version,
            )
            errors = validate_generated_assessment(request, result)
            if errors:
                raise AssessmentGenerationError("; ".join(errors[:8]))
            for question in result.teacher_version.questions:
                if not question.correct_answer:
                    raise AssessmentGenerationError(
                        f"Teacher question {question.question_number} is missing its model answer."
                    )
            return teacher_version
        except (json.JSONDecodeError, ValidationError, AssessmentGenerationError) as exc:
            last_error = exc
            if attempt:
                break
            prompt += (
                "\n\nCORRECTION: Return one complete JSON object matching the schema. "
                "Keep explanations and marking points concise. Fix this issue: "
                + str(exc)[:1000]
            )

    raise AssessmentGenerationError(
        "The model could not produce a valid assessment after one correction."
    ) from last_error


def _split_request(request: AssessmentRequest) -> tuple[AssessmentRequest, AssessmentRequest]:
    left_count = request.question_count // 2
    right_count = request.question_count - left_count
    left_marks = round(request.total_marks * left_count / request.question_count, 3)
    right_marks = request.total_marks - left_marks
    left = request.model_copy(update={"question_count": left_count, "total_marks": left_marks})
    right = request.model_copy(update={"question_count": right_count, "total_marks": right_marks})
    return left, right


def _generate_portion(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
) -> AssessmentResult:
    """Generate a portion; if it still hits the cap, divide it into smaller calls."""
    try:
        return _one_batch(request, source_material, max_tokens)
    except LLMTruncatedError:
        if request.question_count <= 1:
            raise
    except LLMRequestError as exc:
        if exc.status_code != 413 or request.question_count <= 1:
            raise
    except AssessmentGenerationError:
        if request.question_count <= 1:
            raise

    left_request, right_request = _split_request(request)
    left = _generate_portion(left_request, source_material, max_tokens)
    right = _generate_portion(right_request, source_material, max_tokens)
    questions = [*left.questions, *right.questions]
    for number, question in enumerate(questions, start=1):
        question.question_number = number
    return AssessmentResult(
        title=left.title,
        instructions=left.instructions,
        questions=questions,
        total_marks=request.total_marks,
        duration_minutes=request.duration_minutes,
    )


def _generate_in_batches(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
) -> AssessmentResult:
    """Make predictable small calls so ordinary exams never hit the output cap."""
    batch_size = 4
    if request.question_count <= batch_size:
        return _generate_portion(request, source_material, max_tokens)

    portions = []
    remaining_marks = request.total_marks
    remaining_questions = request.question_count
    while remaining_questions:
        count = min(batch_size, remaining_questions)
        marks = (
            remaining_marks
            if count == remaining_questions
            else round(request.total_marks * count / request.question_count, 3)
        )
        portion_request = request.model_copy(update={
            "question_count": count,
            "total_marks": marks,
        })
        portions.append(_generate_portion(portion_request, source_material, max_tokens))
        remaining_marks -= marks
        remaining_questions -= count

    questions = [question for portion in portions for question in portion.questions]
    for number, question in enumerate(questions, start=1):
        question.question_number = number
    return AssessmentResult(
        title=portions[0].title,
        instructions=portions[0].instructions,
        questions=questions,
        total_marks=request.total_marks,
        duration_minutes=request.duration_minutes,
    )


def generate_assessment(
    request: AssessmentRequest,
    max_tokens: int = 4096,
    source_material: str = "",
) -> GeneratedAssessment:
    """Generate in small calls, then build the student copy without answers."""
    if not isinstance(request, AssessmentRequest):
        request = AssessmentRequest.model_validate(request)

    teacher_version = _generate_in_batches(request, source_material, max_tokens)
    student_data = teacher_version.model_dump(mode="json")
    for question in student_data["questions"]:
        question["correct_answer"] = None
        question["explanation"] = ""
        question["marking_scheme"] = []
    result = GeneratedAssessment(
        student_version=AssessmentResult.model_validate(student_data),
        teacher_version=teacher_version,
    )
    errors = validate_generated_assessment(request, result)
    if errors:
        raise AssessmentGenerationError("; ".join(errors[:8]))
    return result
