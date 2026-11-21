# EduCore Full Testing Report

Generated: 2026-11-21T14:34:24.042758Z

Total: 24 | Passed: 24 | Failed: 0 | Info: 0

| Test | Status | Latency (ms) | Detail |
|------|--------|--------------|--------|
| Groq primary LLM | PASS | 1124.9 | Answer:  |
| Groq fallback LLM | PASS | 381.41 | Answer:  |
| Embedding model | PASS | 8160.95 | 384-dim, 2 texts, encode=57ms |
| Gemini image model | PASS | 1161.22 | Answer: None |
| Document upload | PASS | 172.5 | doc_id=8dbba4c3-235f-4c63-b0e1-cf36c5caf41b, status=pending |
| Document indexing | PASS | - | chunks=2 |
| RAG chat [ar/study] | PASS | 9447.75 | answer_len=288, citations=1 |
| RAG chat [en/study] | PASS | 1155.91 | answer_len=257, citations=1 |
| RAG chat [ar/study] | PASS | 1562.56 | answer_len=442, citations=1 |
| RAG chat [en/hint] | PASS | 1578.14 | answer_len=789, citations=1 |
| Document search | PASS | - | excerpts=2, doc=test_lecture.txt |
| Web search (disabled) | PASS | 42.47 | response_len=154, disabled_msg_present=True |
| Exam generation | PASS | - | topic=مقدمة في الذكاء الاصطناعي, exam_id=adfa9506-f51b-41b3-b5eb-15f2bceb2b4f, q |
| Exam answer key hidden | PASS | - | 0/3 questions expose an answer key |
| Exam submission | PASS | - | score=33.3%, correct=1/3, review_consistent=True |
| Study plan | PASS | - | topic=مقدمة في الذكاء الاصطناعي, hours=4, plan_days=7, summary=خطة أسبوعية لتقوي |
| Student dashboard | PASS | - | attempts=3, topics=12, course_topics=12 |
| Teacher assessment records | PASS | - | records=1 |
| Image analysis | PASS | 1793.8 | answer=أصفر |
| Wrong user (dashboard) | PASS | - | HTTP 404: {"detail":"Student or course was not found"} |
| Wrong user (teacher records) | PASS | - | HTTP 400: {"detail":"الحساب أو المقرر مش موجود."} |
| Wrong user (delete document) | PASS | - | HTTP 409: {"detail":"Not authorized to delete this document"} |
| Wrong user (upload document) | PASS | - | HTTP 400: {"detail":"Not authorized to access this project"} |
| Delete document | PASS | - | status_after_delete=HTTP 404 |

## Evaluation Entries (RAG Chat)

| Question | Mode | Answer Length | Citations | Arabic |
|----------|------|---------------|-----------|--------|
| ما هو المعالجة اللغوية الطبيعية؟ اشرحها ببساطة. | study | 288 | 1 | True |
| What is NLP and what are its key applications? | study | 257 | 1 | False |
| كمل الشرح عن تطبيقات NLP. | study | 442 | 1 | True |
| Explain machine learning algorithms briefly. | hint | 789 | 1 | False |

## Environment

- Python: `3.10.20`
- API: `http://127.0.0.1:8002/api`
- Run tag: `20261121143335`
- Chat model: `openai/gpt-oss-120b`
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- Vision model: `gemini-2.5-flash`
- Web search flags: OPENROUTER_ENABLE_WEB_SEARCH=`false`, GROQ_WEB_SEARCH_ENABLED=`false`

## Notes on test design

- Exam questions returned by the API intentionally omit `correct_index`; submission is validated against the score and review returned by the server.
- Web search is validated in its disabled-by-configuration state; no live external provider calls are made.
- Arabic content is uploaded as TXT due to the known Arabic PDF extraction defect.

## Residual risks

- Topic discovery and exam generation depend on LLM structured output and can hit provider rate limits.
- Arabic PDF extraction remains unverified/unfixed.
