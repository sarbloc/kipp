# Releasing

KIPP is published to PyPI as [`kipp-memory`](https://pypi.org/project/kipp-memory/).
Pushing a `v*` tag runs [`release.yml`](../.github/workflows/release.yml). The
workflow:

1. checks that the tag matches the version in `pyproject.toml`;
2. builds the sdist and wheel, then runs `twine check --strict`;
3. installs the wheel and runs the test suite against it;
4. publishes to PyPI with Trusted Publishing.

No PyPI token is stored anywhere.

## Cutting a release

1. Bump `version` in `pyproject.toml`.
2. In `CHANGELOG.md`, move the `[Unreleased]` notes under a new
   `## [X.Y.Z]` heading and update the compare links at the bottom.
3. Merge both changes to `main` in a PR.
4. Tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "kipp-memory X.Y.Z"
   git push origin vX.Y.Z
   ```

5. Watch the run: `gh run watch`. The publish job runs only after the build
   job passes.
6. Create the GitHub release from the tag, using the changelog entry:
   `gh release create vX.Y.Z --notes-file <(sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '1d;$d')`

A tag that doesn't match `pyproject.toml` fails the build job, and nothing is
published. PyPI never accepts the same version twice: if a release is broken,
bump the version and release again.

## One-time setup

- **GitHub:** the repo has an environment named `pypi`. Its deployment policy
  allows only `v*` tags.
- **PyPI:** under *Your account → Publishing → Add a new pending publisher*, add
  a GitHub publisher with these values:
  - PyPI project name: `kipp-memory`
  - Owner: `sarbloc`
  - Repository: `kipp`
  - Workflow: `release.yml`
  - Environment: `pypi`

  The first successful publish creates the project. After that, the publisher
  appears under the project's *Publishing* settings.
