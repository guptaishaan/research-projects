#!/usr/bin/env python3
"""Regenerate submission titles, abstracts, and keywords from compiled papers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPERS = [
    ("clip-forgetting-forecasting", "Computer Vision", "2wvxlhkw"),
    ("eeg-controls-audit", "Biotechnology-A", "ocagun6c"),
    ("heart-disease-cross-hospital", "Biotechnology-B", "qvtmt7of"),
    ("bankruptcy-stress-test", "Financial Technology", "bihc5gy3"),
    ("exoplanet-model-comparison", "Aerospace Engineering", "izdqzr3x"),
    ("raddose-triage", "Experienced Track", "aiw35wob"),
]


def squash(value: str) -> str:
    # Poppler can expose the Libertine ``ff`` ligature through a private-use
    # codepoint; normalize it so portal-ready text remains plain Unicode.
    value = value.replace("\ue0e0", "f")
    replacements = {
        "𝑅 2": "R²",
        "𝐿2 -": "L₂-",
        "𝑛 =": "n =",
        "𝜇": "μ",
        "𝜌": "ρ",
        "Φ5": "Φ₅",
        "𝑅𝑝 /𝑅∗": "Rₚ/R*",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([,.;:])", r"\1", value)


def tex_field(tex: str, command: str) -> str:
    match = re.search(rf"\\{command}\{{(.*?)\}}", tex, re.S)
    if not match:
        return ""
    return squash(match.group(1).replace("--", "–").replace(r"\&", "&"))


def extract(name: str) -> tuple[str, str, str]:
    directory = ROOT / name
    pdf = directory / f"{name}.pdf"
    tex = (directory / "main.tex").read_text(encoding="utf-8")
    first_author = re.search(r"\\author\{([^}]+)\}", tex)
    if not first_author:
        raise RuntimeError(f"No author found in {directory / 'main.tex'}")

    title = tex_field(tex, "title")
    subtitle = tex_field(tex, "subtitle")
    if subtitle:
        separator = " " if title.endswith(("?", "!", ":")) else ": "
        title = f"{title}{separator}{subtitle}"
    keywords = tex_field(tex, "keywords")
    if not title or not keywords:
        raise RuntimeError(f"Could not extract title or keywords from {directory / 'main.tex'}")

    text = subprocess.check_output(
        # Long author blocks can push the ACM keyword line onto page two.
        ["pdftotext", "-f", "1", "-l", "2", str(pdf), "-"],
        text=True,
    )
    abstract_match = re.search(
        r"ishaangupta@stanford\.edu\s*(.*?)\s*CCS Concepts:", text, re.S
    )
    if not abstract_match:
        raise RuntimeError(f"Could not extract abstract from {pdf}")
    abstract = squash(abstract_match.group(1))

    return title, abstract, keywords


def main() -> None:
    sections = [
        "# ICAISE 2026 submission metadata",
        "",
        "Generated from the current compiled manuscripts. Student authors are listed",
        "alphabetically by surname, affiliated to Nexus Labs, and marked for equal contribution;",
        "Srikanth Samy and Ishaan Gupta close each author list.",
    ]
    for name, cluster, cluster_id in PAPERS:
        title, abstract, keywords = extract(name)
        sections.extend(
            [
                "",
                f"## {title}",
                "",
                f"- **Directory:** `papers/{name}/`",
                f"- **Cluster:** {cluster} (`{cluster_id}`)",
                f"- **Keywords:** {keywords}",
                "",
                "**Abstract**",
                "",
                abstract,
            ]
        )
    (ROOT / "SUBMISSION-METADATA.md").write_text(
        "\n".join(sections) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
