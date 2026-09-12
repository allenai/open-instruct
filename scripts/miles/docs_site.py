"""Keep repository-relative MILES links useful in both GitHub Markdown and MkDocs."""

import re
from pathlib import Path

LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^\s)]+)([^)]*\))")


def on_page_markdown(markdown, *, page, config, files):
    if not page.file.src_uri.startswith("miles/"):
        return markdown
    docs = Path(config["docs_dir"]).resolve()
    source = docs / page.file.src_uri
    repo = docs.parent

    def rewrite(match):
        destination = match[2]
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", destination) or destination.startswith(("/", "#")):
            return match[0]
        path, separator, anchor = destination.partition("#")
        target = (source.parent / path).resolve()
        if target.is_relative_to(docs) or not target.is_relative_to(repo):
            return match[0]
        url = config["repo_url"].rstrip("/") + "/blob/robertb/miles-olmo-core/" + str(target.relative_to(repo))
        return match[1] + url + (separator + anchor if separator else "") + match[3]

    return LINK.sub(rewrite, markdown)
