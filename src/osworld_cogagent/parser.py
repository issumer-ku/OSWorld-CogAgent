"""Strict parser for CogAgent's official Grounded Operation action space."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParsedCogAgentAction:
    action: str
    grounded_operation: str
    pyautogui_code: list[str]


def _py_string(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _extract_field(response: str, field: str) -> str:
    match = re.search(
        rf"(?:^|\n)\s*{re.escape(field)}\s*:\s*(.*)",
        response or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def extract_grounded_operation(response: str) -> str:
    match = re.search(
        r"(?:^|\n)\s*Grounded\s+Operation\s*:\s*",
        response or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    tail = response[match.end() :].lstrip()
    terminal = re.match(r"END\b", tail, flags=re.IGNORECASE)
    if terminal:
        return "END"
    open_index = tail.find("(")
    if open_index <= 0:
        return tail.splitlines()[0].strip()

    quote: Optional[str] = None
    escaped = False
    depth = 0
    for index, char in enumerate(tail[open_index:], start=open_index):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return tail[: index + 1].strip()
    return tail.splitlines()[0].strip()


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name) and node.id in {"None", "True", "False"}:
        return {"None": None, "True": True, "False": False}[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return {
            "operation": node.func.id.upper(),
            "args": [_literal(arg) for arg in node.args],
            "kwargs": {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg},
        }
    raise ValueError(f"Unsupported CogAgent expression: {ast.dump(node)}")


def _normalize_integer_literals(text: str) -> str:
    """Normalize model-emitted decimal integers such as 057 to 57.

    Python 3 rejects decimal integer literals with leading zeroes. CogAgent
    occasionally pads normalized coordinates to three digits, so tokenize the
    expression and join only adjacent integer tokens. Quoted text and other
    literal types remain untouched.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        return text

    normalized: list[tokenize.TokenInfo] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.NUMBER and token.string.isdigit():
            digits = token.string
            end = token.end
            next_index = index + 1
            while next_index < len(tokens):
                following = tokens[next_index]
                if (
                    following.type != tokenize.NUMBER
                    or not following.string.isdigit()
                    or following.start != end
                ):
                    break
                digits += following.string
                end = following.end
                next_index += 1
            if next_index > index + 1:
                token = tokenize.TokenInfo(
                    token.type,
                    str(int(digits, 10)),
                    token.start,
                    end,
                    token.line,
                )
                index = next_index
                normalized.append(token)
                continue
        normalized.append(token)
        index += 1
    return tokenize.untokenize(normalized)


def parse_operation_expression(operation: str) -> dict[str, Any]:
    text = (operation or "").strip()
    if text.upper() == "END":
        return {"operation": "END", "args": [], "kwargs": {}}
    normalized_text = _normalize_integer_literals(text)
    try:
        tree = ast.parse(normalized_text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Grounded Operation syntax: {text}") from exc
    value = _literal(tree.body)
    if not isinstance(value, dict) or "operation" not in value:
        raise ValueError(f"Invalid Grounded Operation: {text}")
    return value


def _box_center(box: Any, width: int, height: int) -> tuple[int, int]:
    if isinstance(box, (list, tuple)) and len(box) == 1 and isinstance(box[0], (list, tuple)):
        box = box[0]
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not all(isinstance(value, (int, float)) for value in box)
    ):
        raise ValueError("CogAgent operation requires box=[[x1,y1,x2,y2]]")
    x1, y1, x2, y2 = (max(0.0, min(1000.0, float(value))) for value in box)
    return (
        min(width - 1, max(0, int(round(((x1 + x2) / 2.0) * width / 1000.0)))),
        min(height - 1, max(0, int(round(((y1 + y2) / 2.0) * height / 1000.0)))),
    )


def _normalize_key(key: Any) -> str:
    normalized = str(key or "").strip().lower().replace(" ", "_")
    aliases = {
        "return": "enter",
        "lcontrol": "ctrl",
        "rcontrol": "ctrl",
        "control": "ctrl",
        "left_control": "ctrl",
        "right_control": "ctrl",
        "lmenu": "alt",
        "rmenu": "alt",
        "left_alt": "alt",
        "right_alt": "alt",
        "lshift": "shift",
        "rshift": "shift",
        "left_shift": "shift",
        "right_shift": "shift",
        "up_arrow": "up",
        "down_arrow": "down",
        "left_arrow": "left",
        "right_arrow": "right",
        "page_up": "pageup",
        "page_down": "pagedown",
    }
    return aliases.get(normalized, normalized)


def _operation_to_code(parsed: dict[str, Any], width: int, height: int) -> list[str]:
    name = parsed["operation"].upper()
    kwargs = parsed.get("kwargs", {})
    if name == "END":
        return ["DONE"]
    if name in {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "HOVER"}:
        x, y = _box_center(kwargs.get("box"), width, height)
        function = {
            "CLICK": "click",
            "DOUBLE_CLICK": "doubleClick",
            "RIGHT_CLICK": "rightClick",
            "HOVER": "moveTo",
        }[name]
        return [f"pyautogui.{function}({x}, {y})"]
    if name == "TYPE":
        x, y = _box_center(kwargs.get("box"), width, height)
        text = kwargs.get("text")
        if text is None:
            raise ValueError("TYPE requires text")
        return [
            f"pyautogui.click({x}, {y})",
            'pyautogui.hotkey("ctrl", "a")',
            f"pyautogui.write({_py_string(text)}, interval=0.01)",
        ]
    if name in {"SCROLL_UP", "SCROLL_DOWN", "SCROLL_LEFT", "SCROLL_RIGHT"}:
        steps = max(1, min(abs(int(kwargs.get("step_count", 5))), 20))
        if name == "SCROLL_UP":
            return [f"pyautogui.scroll({steps})"]
        if name == "SCROLL_DOWN":
            return [f"pyautogui.scroll(-{steps})"]
        if name == "SCROLL_LEFT":
            return [f"pyautogui.hscroll(-{steps})"]
        return [f"pyautogui.hscroll({steps})"]
    if name == "KEY_PRESS":
        key = _normalize_key(kwargs.get("key"))
        if not key:
            raise ValueError("KEY_PRESS requires key")
        return [f"pyautogui.press({_py_string(key)})"]
    if name in {"KEY_DOWN", "KEY_UP"}:
        key = _normalize_key(kwargs.get("key"))
        if not key:
            raise ValueError(f"{name} requires key")
        function = "keyDown" if name == "KEY_DOWN" else "keyUp"
        return [f"pyautogui.{function}({_py_string(key)})"]
    if name == "GESTURE":
        actions = kwargs.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("GESTURE requires structured actions")
        code: list[str] = []
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("GESTURE actions must be structured operations")
            code.extend(_operation_to_code(action, width, height))
        return code
    if name == "LAUNCH":
        url = kwargs.get("url")
        app = kwargs.get("app")
        if url and str(url).strip().lower() != "none":
            return [
                'pyautogui.hotkey("ctrl", "l")',
                f"pyautogui.write({_py_string(url)}, interval=0.01)",
                'pyautogui.press("enter")',
            ]
        if app and str(app).strip().lower() != "none":
            return [
                'pyautogui.press("win")',
                f"pyautogui.write({_py_string(app)}, interval=0.01)",
                'pyautogui.press("enter")',
            ]
        raise ValueError("LAUNCH requires app or url")
    raise ValueError(f"Unsupported CogAgent operation: {name}")


def parse_cogagent_response(response: str, width: int, height: int) -> ParsedCogAgentAction:
    action = _extract_field(response, "Action")
    operation_text = extract_grounded_operation(response)
    if not operation_text:
        raise ValueError("Response is missing Grounded Operation")
    parsed = parse_operation_expression(operation_text)
    return ParsedCogAgentAction(
        action=action,
        grounded_operation=operation_text,
        pyautogui_code=_operation_to_code(parsed, width, height),
    )
