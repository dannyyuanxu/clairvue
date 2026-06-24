"""Runs answer_claim() against data/processed/eval_questions.csv and scores results."""

EVAL_QUESTIONS_CSV_PATH = "data/processed/eval_questions.csv"


def load_eval_questions(path: str = EVAL_QUESTIONS_CSV_PATH):
    raise NotImplementedError


def run_evaluation(eval_questions) -> dict:
    raise NotImplementedError
