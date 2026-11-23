"""Adapt repository-relative MILES documentation links for the generated MkDocs site.
The hooks generate virtual configuration-reference pages and redirect links to
repository files, such as example configurations, to their GitHub locations. This
keeps the same Markdown sources useful both on GitHub and in the built documentation.
"""

import os
import re
import sys
from pathlib import Path

from mkdocs.structure.files import File

# MkDocs loads hooks by file path; its console entrypoint need not put the repo on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.miles import generate_docs  # noqa: E402

LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^\s)]+)([^)]*\))")


def on_files(files, config):
    """Build virtual reference pages without modifying the source checkout."""
    for path, content in generate_docs.render(os.environ.get("MILES_NATIVE_HELP")).items():
        uri = str(path.relative_to(generate_docs.ROOT / "docs"))
        existing = files.get_file_from_path(uri)
        if existing is not None:
            files.remove(existing)
        files.append(File.generated(config, uri, content=content))
    return files


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
        url = (
            config["repo_url"].rstrip("/")
            + "/blob/"
            + config.get("repo_branch", "main")
            + "/"
            + str(target.relative_to(repo))
        )
        return match[1] + url + (separator + anchor if separator else "") + match[3]

    return LINK.sub(rewrite, markdown)
