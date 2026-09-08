"""Shared limits and labels for independent object-category prompts."""

MAX_PROMPTS = 8
COLORS = (
    "#55e8ce",
    "#ffb86b",
    "#a99aff",
    "#ff7aab",
    "#70baff",
    "#e1df76",
    "#e09aff",
    "#9ce588",
)


def normalize_prompts(prompts):
    if not isinstance(prompts, list) or len(prompts) > MAX_PROMPTS:
        raise ValueError(
            f"prompts must be a list of at most {MAX_PROMPTS} objects to find"
        )
    result = []
    seen = set()
    for prompt in prompts:
        if (
            not isinstance(prompt, str)
            or len(prompt) > 256
            or any(ord(c) < 32 or ord(c) == 127 for c in prompt)
        ):
            raise ValueError(
                "each prompt must be text with at most 256 characters "
                "and no control characters"
            )
        prompt = prompt.strip()
        if prompt and prompt.casefold() not in seen:
            result.append(prompt)
            seen.add(prompt.casefold())
    return result
