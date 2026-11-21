"""--domain routing for the entity CLI verbs (issue #61).

store/delete/list/search/event used to touch only the shared domain, so a
dev or personal entity could be read with `kipp get --domain` but not
deleted, listed or searched from the CLI.
"""

from click.testing import CliRunner

from kipp.cli import main
from kipp.client import get_entity, get_unextracted_events


def run(*args):
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def store(entity_id, content, domain=None):
    args = ["store", "--type", entity_id.split(":")[0], "--id", entity_id, "--content", content]
    return run(*args, *(["--domain", domain] if domain else []))


def test_store_writes_to_the_given_domain_only(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")

    assert get_entity(cli_env, "project:alpha", domain="dev") is not None
    assert get_entity(cli_env, "project:alpha", domain="shared") is None


def test_store_defaults_to_shared(cli_env):
    store("project:alpha", "Alpha ships on Fridays.")

    assert get_entity(cli_env, "project:alpha", domain="shared") is not None
    assert get_entity(cli_env, "project:alpha", domain="dev") is None


def test_store_checks_parent_in_the_same_domain(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")

    out = run("store", "--type", "project", "--id", "project:alpha-ops",
              "--content", "Alpha ops runs the deploys.", "--parent", "project:alpha",
              "--domain", "dev")

    assert "Stored: project:alpha-ops" in out
    assert get_entity(cli_env, "project:alpha-ops", domain="dev").parent == "project:alpha"


def test_delete_removes_only_the_named_domain(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")
    store("project:alpha", "Alpha is a shared codename.")

    out = run("delete", "project:alpha", "--domain", "dev", "--yes")

    assert "Deleted: project:alpha (dev)" in out
    assert get_entity(cli_env, "project:alpha", domain="dev") is None
    assert get_entity(cli_env, "project:alpha", domain="shared") is not None


def test_delete_reports_missing_in_that_domain(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")

    result = CliRunner().invoke(main, ["delete", "project:alpha", "--yes"])

    assert result.exit_code == 1
    assert "Not found in shared: project:alpha" in result.output
    assert get_entity(cli_env, "project:alpha", domain="dev") is not None


def test_list_shows_only_the_given_domain(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")
    store("project:beta", "Beta is a shared thing.")

    dev = run("list", "--domain", "dev")
    shared = run("list")

    assert "project:alpha" in dev and "project:beta" not in dev
    assert "project:beta" in shared and "project:alpha" not in shared


def test_search_reads_the_given_domain(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="personal")

    assert "project:alpha" in run("search", "Alpha ships on Fridays.", "--domain", "personal")
    assert "project:alpha" not in run("search", "Alpha ships on Fridays.")


def test_event_logs_to_the_given_domain(cli_env):
    run("event", "Sarbloc moved the standup.", "--domain", "dev")

    assert len(get_unextracted_events(cli_env, domain="dev")) == 1
    assert get_unextracted_events(cli_env, domain="shared") == []


def test_domain_rejects_unknown_values(cli_env):
    result = CliRunner().invoke(main, ["list", "--domain", "work"])

    assert result.exit_code == 2
    assert "Invalid value for '--domain'" in result.output
