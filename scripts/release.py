# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit>=0.13"]
# ///
"""Cut a stable release: tag a commit as vX.Y.Z, then start the next development version.

develop always carries the version of the next release, so a tested canary commit usually
already has version X.Y.Z and is tagged as it is. When the version at that commit differs, a
`chore(release)` commit that sets it is made first. Afterwards develop moves on to X.(Y+1).0.
Nothing is pushed; the command to push is printed.

Usage: release.py X.Y.Z [--rev REV] [--dry-run]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.items import Table

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "Cargo.toml"
VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
BRANCH = "develop"
REMOTE = "origin"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=check, capture_output=True, text=True)


def out(*args: str) -> str:
    return run(*args).stdout.strip()


def fail(message: str) -> NoReturn:
    sys.exit(f"release: {message}")


def parse_version(version: str) -> tuple[int, int, int]:
    match = VERSION.match(version)
    if match is None:
        fail(f"{version!r} is not X.Y.Z")
    return int(match["major"]), int(match["minor"]), int(match["patch"])


def package_table(document: TOMLDocument) -> Table:
    workspace = document.get("workspace")
    package = workspace.get("package") if isinstance(workspace, Table) else None
    if not isinstance(package, Table):
        fail("Cargo.toml has no [workspace.package] table")
    return package


def workspace_version(text: str) -> str:
    return str(package_table(tomlkit.parse(text))["version"])


def with_version(text: str, version: str) -> str:
    document = tomlkit.parse(text)
    package_table(document)["version"] = version
    return tomlkit.dumps(document)


def set_version(version: str) -> None:
    MANIFEST.write_text(
        with_version(MANIFEST.read_text(encoding="utf-8"), version), encoding="utf-8", newline="\n"
    )
    run("cargo", "metadata", "--format-version", "1", "--offline")


def commit(message: str) -> None:
    run("git", "add", "Cargo.toml", "Cargo.lock")
    run("git", "commit", "-m", message)


def check_ready(tag: str, rev: str) -> None:
    if run("git", "status", "--porcelain").stdout.strip():
        fail("the working tree is not clean")
    if out("git", "branch", "--show-current") != BRANCH:
        fail(f"check out {BRANCH} first")
    run("git", "fetch", REMOTE, BRANCH, "--tags", "--quiet")
    if out("git", "rev-parse", "HEAD") != out("git", "rev-parse", f"{REMOTE}/{BRANCH}"):
        fail(f"{BRANCH} is not at {REMOTE}/{BRANCH}; pull or push first")
    if run("git", "merge-base", "--is-ancestor", rev, "HEAD", check=False).returncode != 0:
        fail(f"{rev} is not part of {BRANCH}")
    if out("git", "tag", "-l", tag):
        fail(f"the tag {tag} already exists")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="the release version, X.Y.Z")
    parser.add_argument("--rev", default="HEAD", help="the commit to release (default HEAD)")
    parser.add_argument("--dry-run", action="store_true", help="check and print the plan only")
    args = parser.parse_args()

    match = VERSION.match(args.version)
    if match is None:
        fail(f"{args.version!r} is not X.Y.Z")
    tag = f"v{args.version}"
    next_version = f"{match['major']}.{int(match['minor']) + 1}.0"
    check_ready(tag, args.rev)

    rev = out("git", "rev-parse", "--verify", f"{args.rev}^{{commit}}")
    current = workspace_version(out("git", "show", f"{rev}:Cargo.toml"))
    needs_commit = current != args.version
    short = out("git", "rev-parse", "--short=7", rev)
    print(f"release {tag} from {short} (version there: {current})")
    if needs_commit:
        print(f"  commit chore(release): {tag}")
    print(f"  annotated tag {tag}")
    develop_version = workspace_version(MANIFEST.read_text(encoding="utf-8"))
    bumps = parse_version(develop_version) <= parse_version(args.version)
    if bumps:
        print(f"  commit on {BRANCH} starting {next_version} (it carries {develop_version})")
    else:
        print(f"  {BRANCH} carries {develop_version}, past {args.version}, so its version stays")
    if args.dry_run:
        return

    if rev != out("git", "rev-parse", "HEAD"):
        run("git", "switch", "--detach", rev)
    try:
        if needs_commit:
            set_version(args.version)
            commit(f"chore(release): {tag}")
        run("git", "tag", "-a", tag, "-m", f"Purr {args.version}")
    finally:
        run("git", "switch", BRANCH)
    if bumps:
        set_version(next_version)
        commit(f"chore: start {next_version} development")
    print(f"\nPush with:\n  git push --atomic {REMOTE} {BRANCH} {tag}")


if __name__ == "__main__":
    main()
