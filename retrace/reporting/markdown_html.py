"""A tiny, dependency-free Markdown -> HTML converter.

Supports exactly the subset the Erasure Report uses: ATX headings, GFM pipe
tables, unordered lists, fenced code blocks, blank-line paragraphs, ``**bold**``
and inline ``code``. Not a general Markdown implementation.
"""

from __future__ import annotations

import html
import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    text = html.escape(text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _CODE.sub(r"<code>\1</code>", text)
    return text


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{_inline(c)}</th>" for c in head]
    out += ["</tr></thead>", "<tbody>"]
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "".join(out)


def markdown_to_html(md: str) -> str:
    """Convert the report's Markdown subset to an HTML fragment."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_buf: list[str] = []
    para: list[str] = []
    list_buf: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def flush_list() -> None:
        if list_buf:
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in list_buf) + "</ul>")
            list_buf.clear()

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                code_buf.clear()
                in_code = False
            else:
                flush_para()
                flush_list()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not line.strip():
            flush_para()
            flush_list()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_para()
            flush_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if line.lstrip().startswith(("- ", "* ")):
            flush_para()
            list_buf.append(line.lstrip()[2:])
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and set(
            lines[i + 1].strip()
        ) <= set("|-: "):
            flush_para()
            flush_list()
            block = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            out.append(_table(block))
            i = j
            continue

        para.append(line.strip())
        i += 1

    flush_para()
    flush_list()
    return "\n".join(out)


_HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1.2rem; }}
  h1, h2, h3 {{ line-height: 1.25; }}
  h1 {{ font-size: 1.7rem; }} h2 {{ font-size: 1.3rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }}
  th, td {{ border: 1px solid #8884; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #8881; }}
  code {{ background: #8882; padding: .1rem .3rem; border-radius: 3px; }}
  pre {{ background: #8881; padding: .8rem; overflow-x: auto; border-radius: 6px; }}
  img {{ max-width: 100%; height: auto; }}
</style></head>
<body>
{body}
</body></html>
"""


def wrap_html(body_fragment: str, *, title: str) -> str:
    return _HTML_SHELL.format(title=html.escape(title), body=body_fragment)
