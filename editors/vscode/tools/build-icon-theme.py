#!/usr/bin/env python3
"""Regenerate `icons/stark-icon-theme.json` from the Seti theme VS Code ships.

A VS Code file icon theme replaces the user's whole icon set — there is no API for
contributing one icon to whatever theme they already use. To give `.stark` an icon
*without* blanking every other file, the theme has to be a copy of a complete base
theme with our one entry added. Seti is that base, because it is what a user sees
when they have never chosen a theme.

Usage:

    python3 tools/build-icon-theme.py [path/to/vs-seti-icon-theme.json]

The default location is the macOS application bundle. On Linux the Seti theme lives
under `/usr/share/code/resources/app/extensions/theme-seti/icons/`, on Windows under
`%LOCALAPPDATA%\\Programs\\Microsoft VS Code\\resources\\app\\extensions\\theme-seti\\icons\\`.

`seti.woff` must be copied next to the generated theme; the script does that too.
"""

import collections
import json
import shutil
import sys
from pathlib import Path

DEFAULT_SETI = Path(
    "/Applications/Visual Studio Code.app/Contents/Resources/app/extensions"
    "/theme-seti/icons/vs-seti-icon-theme.json"
)

STARK_ICON = "_stark"
STARK_LANGUAGE_ID = "stark"
STARK_EXTENSION = "stark"


def build(seti_path: Path, out_dir: Path) -> None:
    seti = json.loads(seti_path.read_text(), object_pairs_hook=collections.OrderedDict)

    theme = collections.OrderedDict()
    theme["information_for_contributors"] = [
        "GENERATED FILE -- do not hand-edit the Seti half.",
        "A VS Code file icon theme replaces the whole icon set; there is no way to contribute a single",
        "icon to the user's existing theme. So this theme is the built-in Seti theme with one entry",
        "added: `.stark` files and the `stark` language id map to `_stark` (./stark.svg). Every other",
        "file keeps the icon it has under Seti, which is VS Code's default theme.",
        "Seti half generated from " + seti["version"] + ", as shipped with VS Code, and seti.woff is",
        "copied from that extension unchanged. Seti-UI is MIT-licensed (jesseweed/seti-ui); icon fixes",
        "for anything other than .stark belong upstream there.",
        "To refresh against a newer VS Code, re-run tools/build-icon-theme.py.",
    ]
    theme["fonts"] = seti["fonts"]

    icons = collections.OrderedDict(seti["iconDefinitions"])
    icons[STARK_ICON] = collections.OrderedDict([("iconPath", "./stark.svg")])
    theme["iconDefinitions"] = icons

    theme["file"] = seti["file"]

    extensions = collections.OrderedDict(seti["fileExtensions"])
    extensions[STARK_EXTENSION] = STARK_ICON
    theme["fileExtensions"] = extensions

    theme["fileNames"] = seti["fileNames"]

    languages = collections.OrderedDict(seti["languageIds"])
    languages[STARK_LANGUAGE_ID] = STARK_ICON
    theme["languageIds"] = languages

    # The light variant only overrides the entries that need a lighter glyph. Ours is an
    # SVG that carries its own background, so it is the same icon in both.
    light = collections.OrderedDict(seti["light"])
    light["fileExtensions"] = collections.OrderedDict(light.get("fileExtensions", {}))
    light["fileExtensions"][STARK_EXTENSION] = STARK_ICON
    light["languageIds"] = collections.OrderedDict(light.get("languageIds", {}))
    light["languageIds"][STARK_LANGUAGE_ID] = STARK_ICON
    theme["light"] = light

    theme["version"] = seti["version"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stark-icon-theme.json").write_text(json.dumps(theme, indent=2) + "\n")
    shutil.copyfile(seti_path.parent / "seti.woff", out_dir / "seti.woff")

    print(
        f"wrote {out_dir / 'stark-icon-theme.json'}: "
        f"{len(icons)} icon definitions, {len(extensions)} file extensions, "
        f"{len(languages)} language ids"
    )


def main() -> int:
    seti_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SETI
    if not seti_path.is_file():
        print(f"Seti theme not found at {seti_path}", file=sys.stderr)
        print("Pass the path to vs-seti-icon-theme.json as an argument.", file=sys.stderr)
        return 1
    build(seti_path, Path(__file__).resolve().parent.parent / "icons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
