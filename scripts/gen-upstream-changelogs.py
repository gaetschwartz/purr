# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate docs/changelogs/{whisper-rs,whisper-cpp,ggml}.md mechanically.

Old and new versions come from Cargo.lock at two git refs of this repo. The
vendored whisper.cpp and ggml revisions are derived from the upstream
repositories' own metadata (submodule pin, scripts/sync-ggml.last). Everything
written is either verbatim upstream data (GitHub release bodies, git subjects)
or a fixed-format rendering of `git log` / GitHub API JSON.

Usage: uv run scripts/gen-upstream-changelogs.py [--work-dir DIR] [--cargo-home DIR]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tomllib
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "changelogs"

WHISPER_RS_GIT = "https://codeberg.org/tazz4843/whisper-rs.git"
WHISPER_RS_WEB = "https://codeberg.org/tazz4843/whisper-rs"
WHISPER_RS_API = "https://codeberg.org/api/v1/repos/tazz4843/whisper-rs"
WHISPER_RS_GITHUB = "tazz4843/whisper-rs"
WHISPER_CPP = "ggml-org/whisper.cpp"
GGML = "ggml-org/ggml"

FIELD = "\x1f"
RECORD = "\x1e"
LOG_FORMAT = f"%H{FIELD}%ct{FIELD}%s{RECORD}"


class Commit(NamedTuple):
    sha: str
    timestamp: int
    subject: str


class Tag(NamedTuple):
    timestamp: int
    name: str
    commit: str


class Versions(NamedTuple):
    whisper_rs: str
    whisper_rs_sys: str


def run(args: Sequence[str | Path], cwd: Path | None = None) -> str:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{proc.stderr}")
    return proc.stdout


def git(repo: Path, *args: str) -> str:
    return run(["git", *args], cwd=repo)


def git_or_empty(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, encoding="utf-8", errors="replace", check=False)
    return proc.stdout if proc.returncode == 0 else ""


def last_sunday(year: int, month: int) -> datetime:
    day = datetime(year, month, 31, 1, tzinfo=UTC)
    return day - timedelta(days=(day.weekday() + 1) % 7)


def zurich(ts: float) -> datetime:
    """Europe/Zurich wall time from a unix timestamp using the EU DST rule (1996 onwards); the host has no tz database."""
    utc = datetime.fromtimestamp(ts, UTC)
    assert utc.year >= 1996
    summer = last_sunday(utc.year, 3) <= utc < last_sunday(utc.year, 10)
    return utc.astimezone(timezone(timedelta(hours=2 if summer else 1)))


def zone_name(moment: datetime) -> str:
    return "CEST" if moment.utcoffset() == timedelta(hours=2) else "CET"


def lock_versions(ref: str) -> Versions:
    lock = tomllib.loads(git(ROOT, "show", f"{ref}:Cargo.lock"))
    found: dict[str, str] = {}
    for pkg in lock["package"]:
        if pkg["name"] in ("whisper-rs", "whisper-rs-sys"):
            assert pkg["name"] not in found
            found[pkg["name"]] = pkg["version"]
    return Versions(found["whisper-rs"], found["whisper-rs-sys"])


def vcs_info(cargo_home: Path, crate: str, version: str) -> tuple[Path, str]:
    (crate_dir,) = glob.glob(str(cargo_home / "registry" / "src" / "*" / f"{crate}-{version}"))
    info = json.loads((Path(crate_dir) / ".cargo_vcs_info.json").read_text("utf-8"))
    return Path(crate_dir), info["git"]["sha1"]


def ensure_clone(url: str, dest: Path) -> Path:
    if dest.exists():
        git(dest, "fetch", "--tags", "--quiet", "origin")
    else:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, dest])
    return dest


def commits(repo: Path, rev_range: str, *paths: str) -> list[Commit]:
    out = git(repo, "log", "--no-merges", f"--format={LOG_FORMAT}", rev_range, *(["--", *paths] if paths else []))
    result = []
    for record in out.split(RECORD):
        if record.strip():
            sha, ts, subject = record.strip("\n").split(FIELD)
            result.append(Commit(sha, int(ts), subject))
    return result


def render_commits(entries: Sequence[Commit], url_base: str, heading: str) -> list[str]:
    by_day: dict[str, list[Commit]] = defaultdict(list)
    for entry in entries:
        by_day[zurich(entry.timestamp).date().isoformat()].append(entry)
    lines: list[str] = []
    for day in sorted(by_day, reverse=True):
        lines += [f"{heading} {day}", ""]
        lines += [f"- {c.subject} ([{c.sha[:10]}]({url_base}/{c.sha}))" for c in by_day[day]]
        lines.append("")
    return lines


def range_tags(repo: Path, old: str, new: str) -> list[Tag]:
    names = git(repo, "tag", "--list", "v*", "--contains", old, "--merged", new).split()
    tags = []
    for name in names:
        commit = git(repo, "rev-parse", f"{name}^{{commit}}").strip()
        if commit != old:
            tags.append(Tag(int(git(repo, "log", "-1", "--format=%ct", commit).strip()), name, commit))
    return sorted(tags)


def github_releases(slug: str) -> dict[str, dict[str, Any]]:
    pages = json.loads(run(["gh", "api", "--paginate", "--slurp", f"repos/{slug}/releases?per_page=100"]))
    return {r["tag_name"]: r for page in pages for r in page if not r["draft"]}


def render_releases(tags: Sequence[Tag], releases: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for tag in reversed(tags):
        release = releases.get(tag.name)
        lines += [f"# Release {tag.name}", ""]
        if release is None:
            lines += [f"No GitHub release exists for tag `{tag.name}` (tag commit `{tag.commit}`).", ""]
            continue
        published = zurich(datetime.fromisoformat(release["published_at"]).timestamp())
        lines += [
            f"- Title: {release['name']}",
            f"- URL: {release['html_url']}",
            f"- Tag commit: `{tag.commit}`",
            f"- Published: {published:%Y-%m-%d %H:%M} {zone_name(published)}",
            f"- Prerelease: {str(release['prerelease']).lower()}",
            "",
        ]
        body = (release["body"] or "").replace("\r\n", "\n").strip("\n")
        lines += [body or "(release body is empty)", "", "---", ""]
    return lines


def write(name: str, lines: Sequence[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip("\n") + "\n")
    print(f"{name}: {len(lines)} lines")


def vendored_matches(crate_dir: Path, repo: Path, pin: str, relpath: str) -> bool:
    vendored = run(["git", "hash-object", "--no-filters", crate_dir / "whisper.cpp" / relpath]).strip()
    return vendored == git(repo, "rev-parse", f"{pin}:{relpath}").strip()


def tag_names(tags: Sequence[Tag]) -> str:
    return ", ".join(t.name for t in tags) or "none"


def codeberg_count(kind: str) -> int:
    request = urllib.request.Request(f"{WHISPER_RS_API}/{kind}?limit=50", headers={"User-Agent": "changelog-script"})
    with urllib.request.urlopen(request) as response:
        return len(json.load(response))


def build_whisper_rs(repo: Path, old: str, new: str, lock_text: str, baseline_text: str, sha_text: str, generated: str) -> None:
    rng = f"{old}..{new}"
    sys_commits = commits(repo, rng, "sys")
    sys_shas = {c.sha for c in sys_commits}
    other = [c for c in commits(repo, rng) if c.sha not in sys_shas]
    url_base = f"{WHISPER_RS_WEB}/commit"
    archived = json.loads(run(["gh", "api", f"repos/{WHISPER_RS_GITHUB}", "--jq", "{archived}"]))["archived"]
    mirror_releases = len(json.loads(run(["gh", "api", f"repos/{WHISPER_RS_GITHUB}/releases"])))
    changelog_diff = git(repo, "diff", old, new, "--", "CHANGELOG.md", "sys/CHANGELOG.md")
    changelog_files = git(repo, "ls-tree", "--name-only", new, "CHANGELOG.md", "sys/CHANGELOG.md").split()
    first_line = git(repo, "show", f"{new}:CHANGELOG.md").splitlines()[0]

    lines = [
        "# whisper-rs and whisper-rs-sys changes",
        "",
        "Generated by `scripts/gen-upstream-changelogs.py`; do not edit by hand.",
        "",
        f"- Generated: {generated}",
        f"- Versions: {lock_text}",
        f"- How the versions were determined: {baseline_text}",
        f"- Source repository: {WHISPER_RS_WEB} (git). {sha_text}",
        f"- Old commit: `{old}`",
        f"- New commit: `{new}`",
        f"- Commit range: `{rng}`, merges excluded ({len(sys_commits) + len(other)} commits)",
        f"- Releases/tags on Codeberg: {codeberg_count('releases')} releases, {codeberg_count('tags')} tags returned by the API",
        f"- GitHub mirror {WHISPER_RS_GITHUB}: archived={str(archived).lower()}, {mirror_releases} releases",
        (
            f"- Changelog files at the new commit: {', '.join(changelog_files) or 'none'}; "
            f"first line of CHANGELOG.md: `{first_line}`"
        ),
        "- Diff of those changelog files between the two commits: "
        + ("see below" if changelog_diff.strip() else "empty, no entries were added"),
        "- Commit subjects are verbatim; dates are Europe/Zurich days.",
        "",
    ]
    if changelog_diff.strip():
        lines += ["# CHANGELOG diff", "", "```diff", changelog_diff.rstrip("\n"), "```", ""]
    lines += [f"# Commits touching `sys/` (whisper-rs-sys): {len(sys_commits)}", ""]
    lines += render_commits(sys_commits, url_base, "##")
    lines += [f"# Other commits (whisper-rs): {len(other)}", ""]
    lines += render_commits(other, url_base, "##")
    write("whisper-rs.md", lines)


def build_whisper_cpp(
    repo: Path,
    crate_dirs: tuple[Path, Path],
    rs_pins: tuple[str, str],
    pins: tuple[str, str],
    lock_text: str,
    baseline_text: str,
    generated: str,
) -> None:
    (old_dir, new_dir), (old_rs, new_rs), (old, new) = crate_dirs, rs_pins, pins
    rng = f"{old}..{new}"
    tags = range_tags(repo, old, new)
    releases = github_releases(WHISPER_CPP)
    url_base = f"https://github.com/{WHISPER_CPP}/commit"
    cmake_prefix = 'project("whisper.cpp" VERSION'

    def describe(*args: str) -> str:
        return git_or_empty(repo, "describe", "--tags", *args).strip() or "(none)"

    def cmake_line(rev: str) -> str:
        return "; ".join(
            f"`{line.strip()}`"
            for line in git(repo, "show", f"{rev}:CMakeLists.txt").splitlines()
            if line.startswith(cmake_prefix)
        )

    match_old = vendored_matches(old_dir, repo, old, "src/whisper.cpp")
    match_new = vendored_matches(new_dir, repo, new, "src/whisper.cpp")
    lines = [
        "# whisper.cpp changes",
        "",
        "Generated by `scripts/gen-upstream-changelogs.py`; do not edit by hand.",
        "",
        f"- Generated: {generated}",
        f"- whisper-rs versions: {lock_text}",
        f"- How the versions were determined: {baseline_text}",
        f"- Source repository: https://github.com/{WHISPER_CPP} (git history and GitHub Releases API)",
        (
            f"- Old whisper.cpp revision: `{old}`, the `sys/whisper.cpp` submodule pin of whisper-rs `{old_rs}` "
            "(`git rev-parse <commit>:sys/whisper.cpp` in the whisper-rs repository)"
        ),
        f"- New whisper.cpp revision: `{new}`, the `sys/whisper.cpp` submodule pin of whisper-rs `{new_rs}`",
        (
            f"- Old revision position: `git describe --contains` = {describe('--contains', old)}; nearest preceding tag = "
            f"{describe('--abbrev=0', '--match', 'v*', old)}; CMakeLists as vendored: {cmake_line(old)}"
        ),
        f"- New revision: exact tag = {describe('--exact-match', new)}; CMakeLists: {cmake_line(new)}",
        (
            "- Vendored sources in the published crates match these pins (blob of `src/whisper.cpp`): "
            f"old={str(match_old).lower()}, new={str(match_new).lower()}"
        ),
        f"- Commit range: `{rng}`, merges excluded ({len(commits(repo, rng))} commits)",
        (
            f"- Release notes: GitHub Releases bodies of the version tags contained in the range ({tag_names(tags)}), verbatim. "
            "A release's notes can describe changes made before the old revision when the old revision is not itself a tag."
        ),
        "- Commit subjects are verbatim; dates are Europe/Zurich days. Commits are grouped by the tag segment they fall in.",
        "",
    ]
    lines += render_releases(tags, releases)
    lines += ["# Commits", ""]
    segments: list[tuple[str, list[Commit]]] = []
    previous = old
    for tag in tags:
        segments.append((f"{previous[:10]}..{tag.name}", commits(repo, f"{previous}..{tag.commit}")))
        previous = tag.commit
    trailing = commits(repo, f"{previous}..{new}")
    if trailing:
        segments.append((f"{previous[:10]}..{new[:10]} (after the last tag)", trailing))
    for title, entries in reversed(segments):
        lines += [f"## {title}: {len(entries)} commits", ""]
        lines += render_commits(entries, url_base, "###")
    write("whisper-cpp.md", lines)


def build_ggml(
    wcpp_repo: Path,
    repo: Path,
    wcpp_pins: tuple[str, str],
    pins: tuple[str, str],
    lock_text: str,
    baseline_text: str,
    generated: str,
) -> None:
    (old_wcpp, new_wcpp), (old, new) = wcpp_pins, pins
    rng = f"{old}..{new}"
    tags = range_tags(repo, old, new)
    releases = github_releases(GGML)
    url_base = f"https://github.com/{GGML}/commit"
    entries = commits(repo, rng)
    after_last_tag = len(commits(repo, f"{tags[-1].commit}..{new}")) if tags else len(entries)

    def version_lines(rev: str) -> str:
        text = git(wcpp_repo, "show", f"{rev}:ggml/CMakeLists.txt")
        found = "; ".join(
            f"`{line.strip()}`" for line in text.splitlines() if line.startswith("set(GGML_VERSION_") and "BASE" not in line
        )
        return found or "(no GGML_VERSION_* lines)"

    def day(rev: str) -> str:
        return zurich(int(git(repo, "log", "-1", "--format=%ct", rev).strip())).date().isoformat()

    lines = [
        "# ggml changes",
        "",
        "Generated by `scripts/gen-upstream-changelogs.py`; do not edit by hand.",
        "",
        f"- Generated: {generated}",
        f"- whisper-rs versions: {lock_text}",
        f"- How the versions were determined: {baseline_text}",
        (
            f"- Source repository: https://github.com/{GGML} (git history and GitHub Releases API). "
            "This is the upstream repository that whisper.cpp vendors under `ggml/`; the ggml history inside whisper.cpp is not used."
        ),
        f"- Old ggml revision: `{old}` ({day(old)}), the content of `scripts/sync-ggml.last` at whisper.cpp `{old_wcpp}`",
        f"- New ggml revision: `{new}` ({day(new)}), the content of `scripts/sync-ggml.last` at whisper.cpp `{new_wcpp}`",
        (
            f"- `ggml/CMakeLists.txt` version lines at the old whisper.cpp revision: {version_lines(old_wcpp)}; "
            f"at the new one: {version_lines(new_wcpp)}"
        ),
        f"- Commit range: `{rng}`, merges excluded ({len(entries)} commits)",
        (
            f"- Release notes: GitHub Releases bodies of the version tags contained in the range ({tag_names(tags)}), verbatim. "
            f"{after_last_tag} commits come after the last tag of the range and are covered by no release."
        ),
        "- Commit subjects are verbatim; commits are grouped by Europe/Zurich day.",
        "",
    ]
    lines += render_releases(tags, releases)
    lines += [f"# Commits: {len(entries)}", ""]
    lines += render_commits(entries, url_base, "##")
    write("ggml.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default="cef2737^")
    parser.add_argument("--new-ref", default="HEAD")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "purr-upstream-changelogs")
    parser.add_argument("--cargo-home", type=Path, default=Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")))
    args = parser.parse_args()

    generated = f"{zurich(datetime.now(UTC).timestamp()):%Y-%m-%d %H:%M} (Europe/Zurich)"
    baseline_sha = git(ROOT, "rev-parse", args.baseline_ref).strip()
    old_ver, new_ver = lock_versions(args.baseline_ref), lock_versions(args.new_ref)
    old_dir, old_rs = vcs_info(args.cargo_home, "whisper-rs-sys", old_ver.whisper_rs_sys)
    new_dir, new_rs = vcs_info(args.cargo_home, "whisper-rs-sys", new_ver.whisper_rs_sys)
    assert vcs_info(args.cargo_home, "whisper-rs", old_ver.whisper_rs)[1] == old_rs
    assert vcs_info(args.cargo_home, "whisper-rs", new_ver.whisper_rs)[1] == new_rs

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rs_repo = ensure_clone(WHISPER_RS_GIT, args.work_dir / "whisper-rs")
    wcpp_repo = ensure_clone(f"https://github.com/{WHISPER_CPP}.git", args.work_dir / "whisper.cpp")
    ggml_repo = ensure_clone(f"https://github.com/{GGML}.git", args.work_dir / "ggml")

    old_wcpp = git(rs_repo, "rev-parse", f"{old_rs}:sys/whisper.cpp").strip()
    new_wcpp = git(rs_repo, "rev-parse", f"{new_rs}:sys/whisper.cpp").strip()
    old_ggml = git(wcpp_repo, "show", f"{old_wcpp}:scripts/sync-ggml.last").strip()
    new_ggml = git(wcpp_repo, "show", f"{new_wcpp}:scripts/sync-ggml.last").strip()

    lock_text = (
        f"whisper-rs {old_ver.whisper_rs} / whisper-rs-sys {old_ver.whisper_rs_sys} -> "
        f"whisper-rs {new_ver.whisper_rs} / whisper-rs-sys {new_ver.whisper_rs_sys}"
    )
    baseline_text = (
        f"Old = Cargo.lock at `{baseline_sha}` (`{args.baseline_ref}`, the parent of the first "
        f"dependency-upgrade commit of the upgrade session); new = Cargo.lock at `{args.new_ref}`, "
        "parsed with a TOML parser. Earlier lockfile versions in this repo's history "
        "(whisper-rs 0.14.3 / sys 0.13.0, whisper-rs 0.15.0 / sys 0.14.0) predate the upgrade "
        "session and are not covered."
    )
    sha_text = (
        "whisper-rs commits come from the `.cargo_vcs_info.json` of the published crates in the "
        "Cargo registry cache (whisper-rs and whisper-rs-sys of the same release share one commit)."
    )

    build_whisper_rs(rs_repo, old_rs, new_rs, lock_text, baseline_text, sha_text, generated)
    build_whisper_cpp(wcpp_repo, (old_dir, new_dir), (old_rs, new_rs), (old_wcpp, new_wcpp), lock_text, baseline_text, generated)
    build_ggml(wcpp_repo, ggml_repo, (old_wcpp, new_wcpp), (old_ggml, new_ggml), lock_text, baseline_text, generated)


if __name__ == "__main__":
    main()
