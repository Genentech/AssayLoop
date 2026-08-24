#!/usr/bin/env python
"""Stamp every local asset URL in the HTML with a content hash.

GitHub Pages serves ``assets/js/site.js`` with a long cache lifetime, and a
browser that already has a copy will keep using it across a deploy. The
symptoms are nasty because they are partial: the HTML is new, so the page looks
current, but the navigation, the charts, and the tooltips are last week's --
and every one of those is built by a cached script. Reloading does not
reliably help, because a plain reload revalidates the document and not always
its subresources.

So the URLs change when the bytes change. This rewrites

    <script src="assets/js/site.js">   ->   <script src="assets/js/site.js?v=a1b2c3d4">

for every stylesheet and script the pages reference, using a short hash of the
file's own contents. An unchanged file keeps its stamp, so a rebuild that
touched one script does not bust the cache for all of them.

The JSON in ``assets/data/`` never appears in the markup, so it gets a second
stamp of its own -- one hash over the whole directory, written onto the
``site.js`` tag as ``data-build``, which `site.js` reads back and appends to
every fetch. It cannot ride on `site.js`'s own hash: rebuilding the data does
not touch `site.js`, so that hash would not move and the browser would keep
serving the old table.

Run this after any edit to a script, stylesheet, or data file, and before
deploying::

    python docs/stamp_assets.py

It is idempotent: an existing ``?v=`` is replaced, not appended to.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent

#: Matches the src/href of a local css/js asset plus any stamp already on it.
ASSET_RE = re.compile(r'((?:src|href)=")(assets/(?:js|css)/[A-Za-z0-9_.-]+)(\?v=[^"]*)?(")')

#: Matches the site.js <script> tag, whose data-build carries the data stamp.
SITE_JS_RE = re.compile(r'(<script src="assets/js/site\.js[^"]*")(\s+data-build="[^"]*")?')


def stamp(path: pathlib.Path) -> str:
    """Eight hex characters of the file's content hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def data_stamp() -> str:
    """Eight hex characters over the contents of every file in assets/data/."""
    h = hashlib.sha256()
    for path in sorted((DOCS / "assets" / "data").rglob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:8]


def main() -> int:
    stamps: dict[str, str] = {}
    missing: list[str] = []
    changed: list[str] = []
    data = data_stamp()

    for html in sorted(DOCS.glob("*.html")):
        text = html.read_text()

        def sub(m: re.Match) -> str:
            rel = m.group(2)
            if rel not in stamps:
                target = DOCS / rel
                if not target.is_file():
                    # A reference to an asset that does not exist is a broken
                    # page, not something to paper over with a hash of nothing.
                    missing.append(f"{html.name} -> {rel}")
                    return m.group(0)
                stamps[rel] = stamp(target)
            return f"{m.group(1)}{rel}?v={stamps[rel]}{m.group(4)}"

        new = SITE_JS_RE.sub(rf'\1 data-build="{data}"', ASSET_RE.sub(sub, text))
        if new != text:
            html.write_text(new)
            changed.append(html.name)

    if missing:
        print("Referenced assets that do not exist:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    for rel, h in sorted(stamps.items()):
        print(f"  {h}  {rel}")
    print(f"  {data}  assets/data/ (data-build)")
    print(f"{len(stamps)} assets stamped; "
          f"{len(changed) or 'no'} page{'' if len(changed) == 1 else 's'} rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
