import json
import logging
from dataclasses import dataclass

from cerebrum.application.service import Service
from cerebrum.cli.input_reader import InputReader
from cerebrum.cli.prompts import (
    REVIEW_JUDGE_SYSTEM_PROMPT,
    REVIEW_QUESTION_SYSTEM_PROMPT,
)
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.cli.views import print_section
from cerebrum.core.language_model import CallOptions, LanguageModel
from cerebrum.core.repository import ThoughtRecord

logger = logging.getLogger(__name__)

# Maximum number of attempts to convert LLM output to json.
MAX_JSON_ATTEMPTS = 3


@dataclass
class JudgeEvaluation:
    """
    Structured evaluation produced by the judge model.

    Represents the model’s assessment of a user's answer,
    including the qualitative verdict, suggested ideal
    response, and targeted feedback.
    """

    verdict: str
    ideal_answer: str
    feedback: str


class ThoughtReview:
    """
    High-level CLI tool for self-assessment of stored thoughts.

    It guides a user through structured recall and reflection,
    using LLM-generated questions and evaluations to reinforce
    understanding and surface gaps in comprehension.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        input_reader: InputReader,
        service: Service,
    ):
        """
        Initialize the ThoughtReview flow.

        Args:
                language_model:
                        Model used to generate questions and judge answers.
                input_reader:
                Provider for user input (text/voice), keeping raw I/O
                        out of review logic.
                service:
                        Application service used to fetch random thoughts from
                        the specified index.
        """
        self._model = language_model
        self._input_reader = input_reader
        self._service = service

    def run_random_review(self, index_id: str, limit: int) -> None:
        """
        Run the review loop on a random sample of thoughts.

        Args:
                index_id:
                        Identifier for the thought index to sample from.
                limit:
                        Maximum number of random thoughts to review.
        """
        thoughts = self._service.get_random_thoughts(index_id, limit)
        all_evaluations = []
        for i, thought in enumerate(thoughts):
            print(f"\n==== Review Thought {i + 1} ====")
            evaluations = self._review_loop(thought)
            all_evaluations += evaluations
        self._print_evaluation_stats(all_evaluations)

    def _review_loop(self, thought: ThoughtRecord) -> list[JudgeEvaluation]:
        """
        Run the end-to-end review flow for a single thought.

        This coordinates question generation, user interaction,
        and model-based judging to produce structured feedback
        on the thought.

        Args:
                thought:
                        The ThoughtRecord being reviewed.

        Returns:
                A list of JudgeEvaluation objects representing the
                successful evaluations generated during the review.
        """
        questions = self._produce_questions(thought)
        evaluations = []
        for i, question in enumerate(questions):
            print(f"\nQuestion {i + 1}: {question}")
            answer = self._input_reader.text(prompt="Your answer", allow_voice=True)
            evaluation = self._judge_loop(question, answer, thought)
            if not evaluation:
                print(
                    "\n(Judge failed to respond in valid format, skipping feedback.)\n",
                )
                continue
            self._print_evaluation(evaluation)
            evaluations.append(evaluation)

        self._print_thought(thought)
        return evaluations

    def _produce_questions(self, thought: ThoughtRecord) -> list[str]:
        """
        Generate review questions for a single thought via the LLM.

        The model is expected to return JSON with a top-level `questions`
        list.

        Args:
                thought:
                        The ThoughtRecord whose body is used to generate
                        review questions.

        Returns:
                A list of non-empty question strings. Returns
                an empty list if valid questions cannot be produced.
        """
        user_thought = f"Given thought: {thought.body}"
        messages = [
            {"role": "system", "content": REVIEW_QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_thought},
        ]

        last_raw = ""
        with typewriter_spinner(["Coming up with questions..."]):
            for _ in range(MAX_JSON_ATTEMPTS):
                raw = self._model.call(messages)
                last_raw = raw
                questions = self._parse_questions(raw)
                if questions:
                    return questions

        logger.error(
            "ReviewQuestions: model failed to return valid questions after %d attempts. last_raw=%r",
            MAX_JSON_ATTEMPTS,
            last_raw,
        )
        return []

    def _parse_questions(self, raw: str) -> list[str]:
        try:
            data = json.loads(raw)
            q = data.get("questions", [])
            if isinstance(q, list):
                questions = [
                    str(question).strip()
                    for question in q
                    if isinstance(question, str) and question.strip()
                ]
                return questions[:3]
        except Exception:
            return []

    def _judge_loop(
        self,
        question: str,
        answer: str,
        thought: ThoughtRecord,
    ) -> JudgeEvaluation | None:
        """
        Query the judge model for structured evaluation of a single answer.

        The judge receives the original thought, the question, and the user's
        answer, and is expected to return STRICT JSON matching JudgeEvaluation.
        On invalid JSON or missing fields,

        Args:
                question:
                        The question being answered.
                answer:
                        The user's free-form answer.
                thought:
                        The originating ThoughtRecord used as context.

        Returns:
                A JudgeEvaluation on success, or None if all attempts fail.
        """
        user_thought = (
            f"Thought: {thought.body}\n" f"Question: {question}\n" f"Answer: {answer}"
        )
        messages = [
            {"role": "system", "content": REVIEW_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_thought},
        ]
        call_options = CallOptions(format="json")

        last_raw = ""
        with typewriter_spinner(["Judging answer..."]):
            for _ in range(MAX_JSON_ATTEMPTS):
                raw = self._model.call(messages, call_options)
                last_raw = raw
                judgement = self._parse_judgement(raw)
                if judgement:
                    return judgement
                messages.append(
                    {
                        "role": "system",
                        "content": "Your last response was not valid JSON. "
                        "Respond again with STRICT JSON in the required format only.",
                    },
                )

        logger.error(
            "ReviewJudge: model failed to return valid judgement after %d attempts. last_raw=%r",
            MAX_JSON_ATTEMPTS,
            last_raw,
        )
        return None

    def _parse_judgement(self, raw: str) -> JudgeEvaluation | None:
        """
        Parse a raw judge model response into a JudgeEvaluation.

        Returns:
                A JudgeEvaluation if parsing and validation succeed,
                otherwise None.
        """
        try:
            data = json.loads(raw)
        except Exception:
            return None

        verdict = data.get("verdict")
        ideal = data.get("ideal_answer")
        feedback = data.get("feedback")

        if not isinstance(verdict, str):
            return None
        if not isinstance(ideal, str):
            return None
        if not isinstance(feedback, str):
            return None

        return JudgeEvaluation(
            verdict=verdict,
            ideal_answer=ideal,
            feedback=feedback,
        )

    def _print_evaluation(self, evaluation: JudgeEvaluation) -> None:
        text = (
            f"Verdict: {evaluation.verdict}\n"
            f"Feedback: {evaluation.feedback}\n"
            f"Ideal answer: {evaluation.ideal_answer}\n"
        )
        print(f"\n{text}")

    def _print_thought(self, thought):
        print_section("The Thought", thought.body)

    def _print_evaluation_stats(self, evaluations: list[JudgeEvaluation]) -> None:
        verdicts = {}
        for evaluation in evaluations:
            verdict = evaluation.verdict
            if verdict in verdicts:
                verdicts[verdict] += 1
            else:
                verdicts[verdict] = 1

        evaluation_stats = "\n".join(
            [f"{verdict}: {count}" for verdict, count in verdicts.items()],
        )
        print_section("Review Stats", evaluation_stats)
