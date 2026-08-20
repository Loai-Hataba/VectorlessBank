"""
indexing/record_markdown.py

WHAT THIS FILE DOES
-------------------
Renders a list of Records as ONE markdown document that PageIndex can
index:

    # Campaigns                                     <- source
    ## Travel And Airlines                          <- LLM-assigned category
    ### Egypt Air July 2026 [campaign_egypt_air]    <- one record
    Campaign: Egypt Air ...                         <- display_text
    ## Home Appliances
    ### El Araby Installment [campaign_el_araby]
    ...

WHY IT EXISTS
-------------
PageIndex's md_to_tree() builds its hierarchy purely from markdown
heading depth -- '#' becomes a root, '##' its child, '###' its child.
There is no LLM reasoning about structure. So this file IS the tree
design: heading level 1 = source, level 2 = category, level 3 = record.

WHY THE RECORD ID IS IN THE HEADING
-----------------------------------
PageIndex nodes have no concept of a Record. They carry a title, a
node_id and some text -- nothing that points back at our data. But
TreeRetriever resolves answers through node.record_ids, so the link has
to survive the round trip through markdown.

Putting the ID in the heading as "[record_id]" solves that: it stays
human-readable for the traversal LLM, and pageindex_tree_builder.py can
parse it straight back off the title. No other mechanism in PageIndex
preserves an application identifier.

INPUTS  : source (str), list[Record], {record_id: category}
OUTPUTS : a markdown string / a written .md file path
"""

import re
from pathlib import Path

from config.settings import MARKDOWN_DIR


# Level 1 heading per source. Falls back to the raw source name.
SOURCE_TITLES = {
    "cards": "Credit Card Products",
    "offers": "Merchant Offers",
    "campaigns": "Campaigns",
}

UNCATEGORIZED = "Other"

# PageIndex treats these as headings, so a record's body text must never
# be allowed to look like one -- otherwise a fee description starting
# with '#' would silently become a node in the tree.
_HEADING_LINE = re.compile(r"^#{1,6}\s+\S")
_BOLD_HEADING_LINE = re.compile(r"^\*\*(.+?)\*\*\s*$")
_CODE_FENCE_LINE = re.compile(r"^```")


def build_markdown(
    source: str,
    records: list,
    categories: dict,
) -> str:
    """
    Render records as markdown, grouped under their category headings.
    """

    if not records:
        raise ValueError(
            f"Cannot build markdown for '{source}': no records."
        )

    source_title = SOURCE_TITLES.get(
        source,
        source.replace("_", " ").title(),
    )

    grouped: dict[str, list] = {}

    for record in records:

        category = categories.get(
            str(record.id),
            UNCATEGORIZED,
        ) or UNCATEGORIZED

        grouped.setdefault(category, []).append(record)

    lines = [
        f"# {source_title}",
        "",
    ]

    for category, category_records in grouped.items():

        lines.append(f"## {_escape_heading(category)}")
        lines.append("")

        for record in category_records:

            lines.append(
                f"### {_escape_heading(record.title)} "
                f"[{record.id}]"
            )

            lines.append("")

            lines.append(
                _escape_body(record.display_text)
            )

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    source: str,
    records: list,
    categories: dict,
    markdown_dir: Path = MARKDOWN_DIR,
) -> Path:
    """
    Render and save to <markdown_dir>/<source>.md, returning the path.

    md_to_tree() reads from a path, not a string, so the file is a
    required intermediate -- it is also the easiest way to eyeball what
    PageIndex actually saw when a tree looks wrong.
    """

    markdown_dir = Path(markdown_dir)

    markdown_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = markdown_dir / f"{source}.md"

    text = build_markdown(
        source=source,
        records=records,
        categories=categories,
    )

    with path.open("w", encoding="utf-8") as file:
        file.write(text)

    return path


def _escape_heading(text) -> str:
    """
    Flatten a title into something safe to sit on a heading line.

    Square brackets are stripped because the record ID is appended in
    brackets and the parser reads the LAST bracketed group.
    """

    text = str(text or "").replace("\n", " ")

    text = text.replace("[", "(").replace("]", ")")

    text = re.sub(r"\s+", " ", text).strip()

    text = text.lstrip("#").strip()

    return text or "Untitled"


def _escape_body(text) -> str:
    """
    Neutralize any line of record text that PageIndex would mistake for
    a heading or a code fence.

    A leading backslash stops the regex matching while staying readable
    to the traversal and generator LLMs.
    """

    escaped_lines = []

    for line in str(text or "").split("\n"):

        stripped = line.strip()

        if (
            _HEADING_LINE.match(stripped)
            or _BOLD_HEADING_LINE.match(stripped)
            or _CODE_FENCE_LINE.match(stripped)
        ):
            escaped_lines.append("\\" + line.lstrip())

        else:
            escaped_lines.append(line)

    return "\n".join(escaped_lines).strip()
