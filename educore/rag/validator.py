
"""
Extractive evidence validator for EduCore RAG.

The LLM selects sentence IDs instead of generating quotes.
Python extracts the actual evidence directly from sources.
"""

from __future__ import annotations

import json
import logging
import re

from educore.config import settings
from educore.provider import create_async_client

logger = logging.getLogger(__name__)


def extract_sentences(text: str) -> list[str]:
    """
    Split source text into sentences while preserving
    the original sentence content.
    """

    if not text or not text.strip():
        return []

    # Normalize line breaks but preserve sentence content.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n+\s*", " ", text).strip()

    # Split after sentence-ending punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text)

    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= 8
    ]


def validate_sentence_reference(
    source_number: int,
    sentence_number: int,
    sources: list,
    sentence_maps: dict[int, list[str]],
) -> str | None:
    """
    Return the exact original sentence for a valid reference.
    """

    if (
        not isinstance(source_number, int)
        or isinstance(source_number, bool)
        or not isinstance(sentence_number, int)
        or isinstance(sentence_number, bool)
    ):
        return None

    if source_number < 1 or source_number > len(sources):
        return None

    sentences = sentence_maps.get(source_number, [])

    if sentence_number < 1 or sentence_number > len(sentences):
        return None

    return sentences[sentence_number - 1]


class EvidenceValidator:
    """Validate evidence using source sentence references."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.OPENROUTER_CHAT_MODEL
        self.client = create_async_client()

    async def validate(
        self,
        question: str,
        sources: list,
    ) -> str:

        if not sources:
            logger.warning("No sources available")
            return "NOT_SUPPORTED"

        # 1. Extract sentences from each retrieved source.
        sentence_maps: dict[int, list[str]] = {}

        source_blocks = []

        for source_number, source in enumerate(
            sources,
            start=1,
        ):
            sentences = extract_sentences(source.content)

            sentence_maps[source_number] = sentences

            numbered_sentences = "\n".join(
                f"S{sentence_number}: {sentence}"
                for sentence_number, sentence
                in enumerate(sentences, start=1)
            )

            source_blocks.append(
                f"""
SOURCE {source_number}
{numbered_sentences}
"""
            )

        sources_text = "\n".join(source_blocks)

        # 2. Ask the model to select sentence IDs only.
        prompt = f"""
You are a strict evidence verification system.

Determine whether the supplied source sentences directly
support an answer to the user's question.

STRICT RULES:

1. Use only the supplied sentences.
2. Never use general knowledge.
3. Do not invent facts.
4. A topic mention alone is not sufficient evidence.
5. Select only sentences that directly support the answer.
6. A sentence ID must refer to an existing sentence.
7. Do not write or generate quotations.
8. Do not modify source text.
9. If essential information is missing, use PARTIALLY_SUPPORTED.
10. If no sentence supports the requested information,
    use NOT_SUPPORTED.

QUESTION:
{question}

SOURCE SENTENCES:
{sources_text}

Return valid JSON only:

{{
  "status": "SUPPORTED",
  "evidence": [
    {{
      "source_number": 1,
      "sentence_number": 2,
      "claim": "What this sentence directly supports"
    }}
  ]
}}

Allowed statuses:
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED

For NOT_SUPPORTED, evidence must be an empty list.

For SUPPORTED, the selected sentences must cover
all essential parts of the question.

For PARTIALLY_SUPPORTED, select only the supported parts.
"""

        try:
            request = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evidence verifier. "
                            "Select existing sentence IDs only. "
                            "Never generate quotations or use "
                            "external knowledge. Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0,
            }
            request["max_completion_tokens"] = 700
            request["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**request)

            content = (
                response.choices[0].message.content or ""
            ).strip()

            logger.debug("Evidence validator returned %d characters", len(content))

            if content.startswith("```"):
                content = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    content,
                )
                content = re.sub(
                    r"\s*```$",
                    "",
                    content,
                )

            result = json.loads(content)

            status = result.get("status")
            evidence = result.get("evidence")

            if status not in {
                "SUPPORTED",
                "PARTIALLY_SUPPORTED",
                "NOT_SUPPORTED",
            }:
                logger.warning("Invalid validator status")
                return "NOT_SUPPORTED"

            if not isinstance(evidence, list):
                logger.warning("Evidence must be a list")
                return "NOT_SUPPORTED"

            if status == "NOT_SUPPORTED":
                return "NOT_SUPPORTED"

            if not evidence:
                return "NOT_SUPPORTED"

            # 3. Independently validate all selected references.
            verified_evidence = []

            for item in evidence:

                if not isinstance(item, dict):
                    continue

                source_number = item.get("source_number")
                sentence_number = item.get("sentence_number")

                actual_sentence = validate_sentence_reference(
                    source_number=source_number,
                    sentence_number=sentence_number,
                    sources=sources,
                    sentence_maps=sentence_maps,
                )

                if actual_sentence is None:
                    logger.warning(
                        "INVALID EVIDENCE REFERENCE: %r",
                        item,
                    )
                    continue

                verified_evidence.append({
                    "source_number": source_number,
                    "sentence_number": sentence_number,
                    "claim": item.get("claim", ""),
                    "actual_evidence": actual_sentence,
                })

                logger.info(
                    "EVIDENCE VERIFIED | SOURCE=%d | SENTENCE=%d",
                    source_number,
                    sentence_number,
                )

                logger.info(
                    "ACTUAL SOURCE SENTENCE: %s",
                    actual_sentence,
                )

            logger.info(
                "VALIDATOR STATUS=%s | "
                "TOTAL_EVIDENCE=%d | VERIFIED=%d",
                status,
                len(evidence),
                len(verified_evidence),
            )

            if not verified_evidence:
                return "NOT_SUPPORTED"

            if len(verified_evidence) != len(evidence):
                return "NOT_SUPPORTED"

            if status == "SUPPORTED":
                return "SUPPORTED"

            if status == "PARTIALLY_SUPPORTED":
                return "PARTIALLY_SUPPORTED"

            return "NOT_SUPPORTED"

        except json.JSONDecodeError:
            logger.exception("Invalid JSON from evidence validator")
            return "NOT_SUPPORTED"

        except Exception:
            logger.exception("Evidence validation failed")
            return "NOT_SUPPORTED"
