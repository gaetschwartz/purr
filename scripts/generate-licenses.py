# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Regenerate crates/purr-app/assets/licenses.json with cargo-about.

The file lists, per licence, every distinct licence text and the third-party
crates it applies to, for the app's Licenses page. Requires `cargo install
cargo-about --locked --features cli`. Configuration is about.toml.

Usage: uv run scripts/generate-licenses.py [--offline]
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "crates" / "purr-app" / "Cargo.toml"
CONFIG = ROOT / "about.toml"
OUTPUT = ROOT / "crates" / "purr-app" / "assets" / "licenses.json"
FEATURES = "static-ffmpeg vulkan cuda metal coreml"


class ReportCrate(TypedDict):
    name: str
    version: str
    source: str | None


class ReportUse(TypedDict):
    crate: ReportCrate


class ReportLicense(TypedDict):
    id: str
    name: str
    text: str
    used_by: list[ReportUse]


class Report(TypedDict):
    licenses: list[ReportLicense]


type Package = tuple[str, str]


@dataclass(slots=True)
class LicenceGroup:
    id: str
    name: str
    texts: dict[str, set[Package]] = field(default_factory=dict)

    def crate_count(self) -> int:
        return sum(len(packages) for packages in self.texts.values())


def run_cargo_about(offline: bool) -> Report:
    command = [
        "cargo",
        "about",
        "generate",
        "--format",
        "json",
        "--config",
        str(CONFIG),
        "--manifest-path",
        str(MANIFEST),
        "--features",
        FEATURES,
        "--locked",
    ]
    if offline:
        command.append("--offline")
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"cargo about failed with exit code {result.returncode}")
    report: Report = json.loads(result.stdout)
    return report


def group_by_licence(report: Report) -> list[LicenceGroup]:
    """Group by licence, then by distinct text, dropping the workspace's own crates."""
    groups: dict[str, LicenceGroup] = {}
    for entry in report["licenses"]:
        packages = {
            (used["crate"]["name"], used["crate"]["version"])
            for used in entry["used_by"]
            if used["crate"]["source"] is not None
        }
        if not packages:
            continue
        group = groups.setdefault(entry["id"], LicenceGroup(entry["id"], entry["name"]))
        group.texts.setdefault(entry["text"].strip(), set()).update(packages)
    return sorted(groups.values(), key=lambda group: (-group.crate_count(), group.id))


def render(groups: list[LicenceGroup]) -> str:
    licences = []
    for group in groups:
        notices = [
            {"text": text, "crates": [list(package) for package in sorted(packages)]}
            for text, packages in sorted(
                group.texts.items(), key=lambda item: min(item[1])
            )
        ]
        licences.append({"id": group.id, "name": group.name, "texts": notices})
    return json.dumps(licences, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="read licence files from the local registry only",
    )
    args = parser.parse_args()
    groups = group_by_licence(run_cargo_about(args.offline))
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
        file.write(render(groups))
    crates = sum(group.crate_count() for group in groups)
    print(f"{OUTPUT.relative_to(ROOT)}: {len(groups)} licences, {crates} crate entries")


if __name__ == "__main__":
    main()
