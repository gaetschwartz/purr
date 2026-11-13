# Releasing

Purr ships on two channels. Both install the same command, `purr`.

| | canary | stable |
|---|---|---|
| Built from | every push to `develop` whose build jobs all pass | a `vX.Y.Z` tag |
| `purr --version` | `purr 0.3.0-canary.123 (abc1234 2026-11-13)` | `purr 0.3.0 (abc1234 2026-11-13)` |
| GitHub release | pre-release `canary-0.3.0-canary.123`, titled `Canary 0.3.0-canary.123 (abc1234)` | release `vX.Y.Z` (a tag with a `-` is marked pre-release) |
| Asset names | unversioned: `purr-cli-linux-x64-all.tar.gz` | `purr-vX.Y.Z-cli-linux-x64-all.tar.gz` |
| Homebrew (`gaetschwartz/tap`) | formula `purr-canary`, cask `purr-app-canary` | formula `purr`, cask `purr-app` |
| Scoop (`scoop-gaetschwartz`) | `purr-canary`, `purr-app-canary` | `purr`, `purr-app` |

A local build reports `purr 0.3.0-dev (abc1234 2026-11-13)`, with `abc1234-dirty` when the tree has uncommitted changes. Outside a git checkout the commit is left out.

## Versions

- One number for every crate: `[workspace.package] version` in the root `Cargo.toml`. It never contains a channel.
- `develop` carries the version of the next stable release. Right after `v0.2.0` is cut, `develop` is `0.3.0`, and its canaries are `0.3.0-canary.N`.
- The channel comes from the build environment: `PURR_CHANNEL` (`stable`, `canary`, or unset for a dev build) and, for a canary, `PURR_CANARY_NUMBER`. CI sets them in `build.yml`. A malformed value fails the compile. Git only supplies the commit and its day; nothing uses `git describe`.
- `N` is the `build.yml` run number, which only grows. Scoop and Homebrew both order `0.3.0-canary.9 < 0.3.0-canary.10 < 0.4.0-canary.1` by comparing the numeric parts as numbers, so a package manager sees every canary as an update. The release tag is `canary-<version>`, and the package versions are the same string.
- The release job reads the version from the built CLI (`purr --version`) and fails when it does not match the run number or the commit. The stable job fails when the CLI does not report the tag's version.

## Canary

1. A push to `develop` starts `build.yml`. A newer push cancels the older run, so only the newest commit publishes.
2. When every build job passes, `release (canary)` checks that `develop` has not moved, builds the changelog since the previous canary (or the last `v*` tag), and creates the release as a draft that targets the built commit. It publishes the draft once all assets are uploaded, so a cancelled run never leaves a half-filled public release.
3. It keeps the newest 5 `canary-*` releases and deletes older ones with their tags (`gh release delete --cleanup-tag`). Only names starting with `canary-` are ever deleted.
4. `publish packages (canary)` renders the Homebrew formula and cask and the Scoop manifests from the release's `SHA256SUMS.txt` and pushes them to the tap and bucket.

`ci.yml` is not a prerequisite: a canary can publish while its tests fail.

## Cutting a stable release

Pick a canary that has been tested, and note its commit. From a clean checkout of `develop` that is at `origin/develop`:

```
just release 0.3.0 --rev <commit>
git push --atomic origin develop v0.3.0
```

`--rev` defaults to `HEAD`. `scripts/release.py`:

1. Refuses unless the tree is clean, `develop` is checked out and equal to `origin/develop`, the commit is part of `develop`, and the tag is new.
2. When the version at the commit is not already `0.3.0`, makes a `chore(release): v0.3.0` commit on top of it that sets it and updates `Cargo.lock` (`cargo metadata` rewrites only the crate versions). A canary commit normally already carries it.
3. Creates the annotated tag `v0.3.0`.
4. When `develop` still carries `0.3.0`, adds a `chore: start 0.4.0 development` commit.
5. Prints the push command and stops; nothing is pushed.

`--dry-run` shows the plan. Only plain `X.Y.Z` versions are accepted. A pre-release tag such as `v0.3.0-rc.1` can be made by hand once the workspace version at that commit is set to `0.3.0-rc.1`, because the release job checks the CLI reports the tag's version. The workflow marks any tag with a `-` as a pre-release and does not publish it to the package managers.

Pushing the tag starts `build.yml` with `PURR_CHANNEL=stable`. `release (tag)` checks the CLI reports `0.3.0`, creates the GitHub release, and `publish packages (stable)` updates `purr` and `purr-app` in the tap and bucket.

A `workflow_dispatch` release workflow that creates the tag through the API would not start the build by itself, because pushes made with the default `GITHUB_TOKEN` do not trigger workflows. It would have to run `gh workflow run build.yml --ref v0.3.0` with `actions: write`, and `tagged-release` would have to accept that dispatch on a `v*` ref. The local command needs neither.
