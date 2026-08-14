"""Render span-backed LaTeX as deterministic OCR-visible text.

The renderer is intentionally not a symbolic algebra system. It covers the
commands observed in RD-TableBench and preserves unknown command names instead
of silently deleting content.
"""

from __future__ import annotations

import html
import re

SUPERSCRIPT = str.maketrans(
    "0123456789+-=()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖ𐞥ʳˢᵗᵘᵛʷˣʸᶻᴬᴮꟳᴰᴱꟳᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾ𐞒ᴿˢᵀᵁⱽᵂˣʸᶻ",
)

SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "Gamma": "Γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "eta": "η",
    "theta": "θ",
    "Theta": "Θ",
    "mu": "μ",
    "pi": "π",
    "Pi": "Π",
    "rho": "ρ",
    "Sigma": "Σ",
    "tau": "τ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "chi": "χ",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "/",
    "cdot": ".",
    "cdots": "…",
    "ldots": "…",
    "sim": "~",
    "simeq": "≃",
    "approx": "≈",
    "equiv": "≡",
    "neq": "≠",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "in": "∈",
    "notin": "∉",
    "cup": "∪",
    "cap": "∩",
    "subset": "⊂",
    "supset": "⊃",
    "partial": "∂",
    "nabla": "∇",
    "infty": "∞",
    "sum": "Σ",
    "prod": "Π",
    "sqrt": "√",
    "circ": "°",
    "prime": "′",
    "odot": "⊙",
    "emptyset": "Ø",
    "rightarrow": "→",
    "leftarrow": "←",
    "to": "→",
    "%": "%",
    "&": "&",
    "#": "#",
    "_": "_",
    "$": "$",
    "{": "{",
    "}": "}",
}

SPACING_COMMANDS = {"quad", "qquad", "enspace", "hspace", "vspace"}
TEXT_COMMANDS = {
    "mathrm",
    "mathbf",
    "mathbb",
    "mathit",
    "mathsf",
    "mathtt",
    "text",
    "operatorname",
    "rm",
    "bf",
    "it",
}
ACCENT_COMMANDS = {"bar", "hat", "vec", "overline", "underline", "tilde"}
NO_OUTPUT_COMMANDS = {"left", "right", "displaystyle", "textstyle"}
UNICODE_FRACTIONS = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
    ("1", "5"): "⅕",
    ("2", "5"): "⅖",
    ("3", "5"): "⅗",
    ("4", "5"): "⅘",
    ("1", "6"): "⅙",
    ("5", "6"): "⅚",
    ("1", "8"): "⅛",
    ("3", "8"): "⅜",
    ("5", "8"): "⅝",
    ("7", "8"): "⅞",
}
PLAIN_UNDERSCORE = "\uE000"


def _protect_plain_underscores(source: str) -> str:
    output: list[str] = []
    in_math = False
    index = 0
    while index < len(source):
        if source.startswith("$$", index):
            in_math = not in_math
            output.append("$$")
            index += 2
            continue
        char = source[index]
        if char == "$":
            in_math = not in_math
        if char == "_" and not in_math and source[index + 1 :].lstrip()[:1] != "{":
            output.append(PLAIN_UNDERSCORE)
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _balanced_group(source: str, start: int) -> tuple[str, int]:
    if start >= len(source) or source[start] != "{":
        return "", start
    depth = 1
    index = start + 1
    while index < len(source) and depth:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
        index += 1
    content = source[start + 1 : index - 1] if depth == 0 else source[start + 1 :]
    return content, index


def _next_atom(source: str, start: int) -> tuple[str, int]:
    while start < len(source) and source[start].isspace():
        start += 1
    if start < len(source) and source[start] == "{":
        group, end = _balanced_group(source, start)
        return _render(group), end
    if start < len(source) and source[start] == "\\":
        return _render_command(source, start)
    return (source[start], start + 1) if start < len(source) else ("", start)


def _render_command(source: str, start: int) -> tuple[str, int]:
    index = start + 1
    if index >= len(source):
        return "", index
    if source[index] == "\\":
        return " ", index + 1
    if source[index].isalpha():
        end = index
        while end < len(source) and source[end].isalpha():
            end += 1
        command = source[index:end]
        while end < len(source) and source[end].isspace():
            end += 1
    else:
        command = source[index]
        end = index + 1

    if command in NO_OUTPUT_COMMANDS:
        return "", end
    if command in SPACING_COMMANDS or command in {",", ";", ":", "!", " "}:
        return " ", end
    if command == "frac":
        numerator, after_numerator = _next_atom(source, end)
        denominator, after_denominator = _next_atom(source, after_numerator)
        fraction = UNICODE_FRACTIONS.get((numerator, denominator))
        if fraction is not None:
            return fraction, after_denominator
        numerator = f"({numerator})" if re.search(r"[+\-=]", numerator) else numerator
        denominator = (
            f"({denominator})" if re.search(r"[+\-=]", denominator) else denominator
        )
        return f"{numerator}/{denominator}", after_denominator
    if command == "sqrt":
        value, after = _next_atom(source, end)
        return f"√{value}", after
    if command in TEXT_COMMANDS:
        value, after = _next_atom(source, end)
        return f" {value} ", after
    if command in ACCENT_COMMANDS:
        return _next_atom(source, end)
    if command in SYMBOLS:
        return SYMBOLS[command], end

    value, after = _next_atom(source, end)
    if after > end:
        return f"{command}{value}", after
    return command, end


def _render(source: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\\":
            value, index = _render_command(source, index)
            output.append(value)
            continue
        if char in "^_":
            value, index = _next_atom(source, index + 1)
            if char == "^":
                translated = value.translate(SUPERSCRIPT)
                output.append(translated if len(translated) == len(value) else f"^{value}")
            else:
                output.append(value)
            continue
        if char == "{":
            value, index = _balanced_group(source, index)
            output.append(_render(value))
            continue
        if char in "}$":
            index += 1
            continue
        if char == "~":
            output.append(" ")
            index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def latex_to_visible_text(value: str) -> str:
    """Convert mixed prose/LaTeX to deterministic visible text."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\\(", "$ ").replace("\\)", " $")
    text = text.replace("\\[", "$ ").replace("\\]", " $")
    text = _render(_protect_plain_underscores(text)).replace(PLAIN_UNDERSCORE, "_")
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+([,;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
