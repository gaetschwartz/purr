# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Render the Homebrew and Scoop package definitions for one Purr release.

Asset names mirror scripts/package-release-assets.sh. Hashes come from the
release's SHA256SUMS.txt. Every expected asset must be listed there, otherwise
nothing is written.

Usage: publish-packages.py --channel stable|canary --tag TAG --sums SHA256SUMS.txt
                           [--tap DIR] [--bucket DIR] [--repo OWNER/NAME]
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal

type Json = str | list[Json] | dict[str, Json]
type ChannelName = Literal["stable", "canary"]

TEMPLATES = Path(__file__).resolve().parent / "packaging"
SCHEMA = "https://raw.githubusercontent.com/ScoopInstaller/Scoop/master/schema.json"
LICENSE = "MIT"
DESCRIPTION = "Transcribe audio files and URLs on this computer"
APP_BUNDLE = "purr-app.app"
MAC = "macos-arm64-metal"
LINUX = "linux-x64-vulkan"
LINUX_CUDA = "linux-x64-all"
WINDOWS = "windows-x64-cuda-vulkan"
DEFAULT_NOTE = (
    "This build has no CUDA libraries: it uses Vulkan, then the CPU. "
    "For an NVIDIA GPU, install {cuda} instead (a much larger download)."
)
CUDA_NOTE = (
    "This build adds the CUDA libraries, so the download is large, and it needs an NVIDIA driver "
    "to use them. It uses an NVIDIA GPU when present, then Vulkan, then the CPU."
)
CANARY_SKIP = "Rolling canary, republished by CI on every build"
CANARY_VERSION = re.compile(r"^\d+\.\d+\.\d+-canary\.\d+$")
SUMS_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64}) [ *](?P<name>\S.*)$")


@dataclass(frozen=True, slots=True)
class Channel:
    name: ChannelName
    tag_prefix: str
    formula: str
    formula_class: str
    cask: str
    scoop_cli: str
    scoop_app: str
    formula_cuda: str
    formula_cuda_class: str
    scoop_cli_cuda: str
    scoop_app_cuda: str
    conflict_formula: str
    conflict_formula_cuda: str
    conflict_cask: str
    title_suffix: str
    description_note: str

    def tag(self, version: str) -> str:
        return f"{self.tag_prefix}{version}"

    def asset(self, version: str, kind: str, platform: str, ext: str) -> str:
        prefix = "purr" if self.name == "canary" else f"purr-{self.tag(version)}"
        return f"{prefix}-{kind}-{platform}{ext}"

    def cuda_pack(self, version: str) -> str:
        return self.asset(version, "runtime", "cuda-windows-x64", ".zip")


STABLE = Channel(
    name="stable",
    tag_prefix="v",
    formula="purr",
    formula_class="Purr",
    cask="purr-app",
    scoop_cli="purr",
    scoop_app="purr-app",
    formula_cuda="purr-cuda",
    formula_cuda_class="PurrCuda",
    scoop_cli_cuda="purr-cuda",
    scoop_app_cuda="purr-app-cuda",
    conflict_formula="purr-canary",
    conflict_formula_cuda="purr-canary-cuda",
    conflict_cask="purr-app-canary",
    title_suffix="",
    description_note="",
)
CANARY = Channel(
    name="canary",
    tag_prefix="canary-",
    formula="purr-canary",
    formula_class="PurrCanary",
    cask="purr-app-canary",
    scoop_cli="purr-canary",
    scoop_app="purr-app-canary",
    formula_cuda="purr-canary-cuda",
    formula_cuda_class="PurrCanaryCuda",
    scoop_cli_cuda="purr-canary-cuda",
    scoop_app_cuda="purr-app-canary-cuda",
    conflict_formula="purr",
    conflict_formula_cuda="purr-cuda",
    conflict_cask="purr-app",
    title_suffix=" Canary",
    description_note=", canary build",
)
CHANNELS: Mapping[str, Channel] = {c.name: c for c in (STABLE, CANARY)}


def parse_sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = SUMS_LINE.match(line)
        if m is None:
            sys.exit(f"{path}: not a sha256sum line: {line!r}")
        sums[m["name"]] = m["sha"]
    return sums


def version_of(channel: Channel, tag: str) -> str:
    if not tag.startswith(channel.tag_prefix) or len(tag) == len(channel.tag_prefix):
        sys.exit(f"tag {tag!r} does not match the {channel.name} channel")
    version = tag.removeprefix(channel.tag_prefix)
    if channel.name == "stable" and "-" in version:
        sys.exit(
            f"{tag!r} is a prerelease tag and is not published to the stable channel"
        )
    if channel.name == "canary" and CANARY_VERSION.match(version) is None:
        sys.exit(f"{tag!r} is not canary-X.Y.Z-canary.N")
    return version


def expected_assets(channel: Channel, version: str) -> list[str]:
    return [
        channel.asset(version, "cli", MAC, ".tar.gz"),
        channel.asset(version, "cli", LINUX, ".tar.gz"),
        channel.asset(version, "cli", LINUX_CUDA, ".tar.gz"),
        channel.asset(version, "desktop", MAC, ".dmg"),
        channel.asset(version, "cli", WINDOWS, ".zip"),
        channel.asset(version, "desktop", WINDOWS, "-portable.zip"),
        channel.cuda_pack(version),
    ]


def render_ruby(template: str, **fields: str) -> str:
    return Template((TEMPLATES / template).read_text(encoding="utf-8")).substitute(
        fields
    )


def livecheck(channel: Channel, url: str) -> str:
    if channel.name == "canary":
        return f'    skip "{CANARY_SKIP}"'
    return f"    url {url}\n    strategy :github_latest"


def release_url(repo: str, channel: Channel, version: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{channel.tag(version)}/{asset}"


def conflicts(channel: Channel, own: str) -> str:
    siblings = (channel.formula, channel.formula_cuda)
    names = (
        *(n for n in siblings if n != own),
        channel.conflict_formula,
        channel.conflict_formula_cuda,
    )
    return "\n".join(
        f'  conflicts_with "{name}", because: "both install a `purr` binary"'
        for name in names
    )


def formula(channel: Channel, repo: str, version: str, sums: Mapping[str, str]) -> str:
    macos = channel.asset(version, "cli", MAC, ".tar.gz")
    linux = channel.asset(version, "cli", LINUX, ".tar.gz")
    return render_ruby(
        "formula.rb.tmpl",
        class_name=channel.formula_class,
        repo=repo,
        version=version,
        macos_url=release_url(repo, channel, version, macos),
        macos_sha256=sums[macos],
        linux_url=release_url(repo, channel, version, linux),
        linux_sha256=sums[linux],
        livecheck=livecheck(channel, ":stable"),
        conflicts=conflicts(channel, channel.formula),
    )


def formula_cuda(
    channel: Channel, repo: str, version: str, sums: Mapping[str, str]
) -> str:
    linux = channel.asset(version, "cli", LINUX_CUDA, ".tar.gz")
    return render_ruby(
        "formula-cuda.rb.tmpl",
        class_name=channel.formula_cuda_class,
        repo=repo,
        version=version,
        linux_url=release_url(repo, channel, version, linux),
        linux_sha256=sums[linux],
        livecheck=livecheck(channel, ":stable"),
        conflicts=conflicts(channel, channel.formula_cuda),
    )


def cask(channel: Channel, repo: str, version: str, sums: Mapping[str, str]) -> str:
    asset = channel.asset(version, "desktop", MAC, ".dmg")
    owner = repo.split("/")[0]
    return render_ruby(
        "cask.rb.tmpl",
        token=channel.cask,
        repo=repo,
        url=release_url(
            repo,
            channel,
            "#{version}",
            channel.asset("#{version}", "desktop", MAC, ".dmg"),
        ),
        version=version,
        sha256=sums[asset],
        livecheck=livecheck(channel, ":url"),
        conflict=channel.conflict_cask,
        formula=f"{owner}/tap/{channel.formula}",
        app_bundle=APP_BUNDLE,
    )


def scoop_checkver(channel: Channel, repo: str) -> Json:
    if channel.name == "stable":
        return "github"
    return {
        "url": f"https://github.com/{repo}/releases",
        "regex": r"releases/tag/canary-(\d+\.\d+\.\d+-canary\.\d+)",
    }


def scoop_manifest(
    channel: Channel,
    repo: str,
    version: str,
    sums: Mapping[str, str],
    *,
    kind: str,
    ext: str,
    cuda: bool,
    description: str,
    extra: dict[str, Json],
) -> dict[str, Json]:
    assets = [channel.asset(version, kind, WINDOWS, ext)]
    template_assets = [channel.asset("$version", kind, WINDOWS, ext)]
    if cuda:
        assets.append(channel.cuda_pack(version))
        template_assets.append(channel.cuda_pack("$version"))
    urls: list[Json] = [release_url(repo, channel, version, name) for name in assets]
    hashes: list[Json] = [sums[name] for name in assets]
    template_urls: list[Json] = [
        release_url(repo, channel, "$version", name) for name in template_assets
    ]
    package = channel.scoop_cli_cuda if kind == "cli" else channel.scoop_app_cuda
    return {
        "$schema": SCHEMA,
        "version": version,
        "description": description,
        "homepage": f"https://github.com/{repo}",
        "license": LICENSE,
        "notes": CUDA_NOTE if cuda else DEFAULT_NOTE.format(cuda=package),
        "architecture": {
            "64bit": {
                "url": urls if cuda else urls[0],
                "hash": hashes if cuda else hashes[0],
            }
        },
        **extra,
        "checkver": scoop_checkver(channel, repo),
        "autoupdate": {
            "architecture": {
                "64bit": {"url": template_urls if cuda else template_urls[0]}
            },
            "hash": {"url": "$baseurl/SHA256SUMS.txt"},
        },
    }


def scoop_cli(
    channel: Channel, repo: str, version: str, sums: Mapping[str, str], *, cuda: bool
) -> dict[str, Json]:
    return scoop_manifest(
        channel,
        repo,
        version,
        sums,
        kind="cli",
        ext=".zip",
        cuda=cuda,
        description=f"{DESCRIPTION} (command-line tool{channel.description_note})",
        extra={"bin": "purr.exe"},
    )


def scoop_app(
    channel: Channel, repo: str, version: str, sums: Mapping[str, str], *, cuda: bool
) -> dict[str, Json]:
    return scoop_manifest(
        channel,
        repo,
        version,
        sums,
        kind="desktop",
        ext="-portable.zip",
        cuda=cuda,
        description=f"{DESCRIPTION} (desktop app{channel.description_note})",
        extra={"shortcuts": [["purr-app.exe", f"Purr{channel.title_suffix}"]]},
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--channel", choices=sorted(CHANNELS), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sums", type=Path, required=True)
    parser.add_argument("--tap", type=Path)
    parser.add_argument("--bucket", type=Path)
    parser.add_argument("--repo", default="gaetschwartz/purr")
    args = parser.parse_args()
    if args.tap is None and args.bucket is None:
        parser.error("at least one of --tap and --bucket is required")

    channel = CHANNELS[args.channel]
    version = version_of(channel, args.tag)
    sums = parse_sums(args.sums)
    missing = [name for name in expected_assets(channel, version) if name not in sums]
    if missing:
        sys.exit(f"{args.sums} does not list: {', '.join(missing)}")

    if args.tap is not None:
        write(
            args.tap / "Formula" / f"{channel.formula}.rb",
            formula(channel, args.repo, version, sums),
        )
        write(
            args.tap / "Formula" / f"{channel.formula_cuda}.rb",
            formula_cuda(channel, args.repo, version, sums),
        )
        write(
            args.tap / "Casks" / f"{channel.cask}.rb",
            cask(channel, args.repo, version, sums),
        )
    if args.bucket is not None:
        for name, manifest in (
            (
                channel.scoop_cli,
                scoop_cli(channel, args.repo, version, sums, cuda=False),
            ),
            (
                channel.scoop_app,
                scoop_app(channel, args.repo, version, sums, cuda=False),
            ),
            (
                channel.scoop_cli_cuda,
                scoop_cli(channel, args.repo, version, sums, cuda=True),
            ),
            (
                channel.scoop_app_cuda,
                scoop_app(channel, args.repo, version, sums, cuda=True),
            ),
        ):
            write(
                args.bucket / "bucket" / f"{name}.json",
                json.dumps(manifest, indent=4) + "\n",
            )


if __name__ == "__main__":
    main()
