from __future__ import annotations

from jsonschema import Draft202012Validator
import jsonschema

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "minLength": 1},
                    "args": {"type": "object"},
                    "commitPoint": {"type": "boolean"},
                    "riskCost": {"type": "number", "minimum": 0}
                },
                "required": ["skill","args"]
            }
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "expected_post": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["goal","actions"]
}

validator = Draft202012Validator(PLAN_SCHEMA)

def validate_plan(plan: dict) -> list[str]:
    errors = []
    for e in validator.iter_errors(plan):
        errors.append(e.message)
    return errors
