"""Fitting a running conversation into a model's context window."""


def trim(messages, budget, count_tokens):
    """Drop the oldest exchanges until the conversation fits the budget.

    Messages alternate user/assistant, so removing two at a time keeps the
    remainder starting on a user turn.
    """
    kept = list(messages)
    while len(kept) > 1 and count_tokens(kept) > budget:
        del kept[:2]
    return kept
