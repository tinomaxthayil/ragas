from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"

JSON_FORMAT_QUESTION = HumanMessagePromptTemplate.from_template(
    """
Rewrite into a valid JSON

Input:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
    }}
    "hobbies": ["reading", "swimming", "cycling"]
}}
Ouput:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false,
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA"
    }},
    "hobbies": ["reading", "swimming", "cycling"]
}}


Input:
{{
    "statement": "The Earth is also known as "Terra" "
}}
Output:
{{
    "statement": "The Earth is also known as 'Terra'"
}}

Input:
{input}

Ouput:"""  # noqa: E501
)


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False


def generate_answer(llm, prompt, text):
    human_prompt = prompt.format(input=text)
    prompt = ChatPromptTemplate.from_messages([human_prompt])
    results = llm.generate(prompts=[prompt])
    return results.generations[0][0].text.strip()


def load_as_json(llm, text):
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError as err:
        retry = 3
        while retry>0:
            generated_text=generate_answer(llm, JSON_FORMAT_QUESTION, text)
            try:
                return json.loads(generated_text)
            except ValueError as e:
                retry-=1
        warnings.warn(f"Invalid json: {err}")

    return {}
