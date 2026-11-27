"""--domain routing for the entity CLI verbs (issue #61).

store/delete/list/search/event used to touch only the shared domain, so a
dev or personal entity could be read with `kipp get --domain` but not
deleted, listed or searched from the CLI.
"""

import json

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


def test_export_defaults_to_shared(cli_env):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")
    store("project:beta", "Beta is a shared thing.")

    records = json.loads(run("export"))

    assert [(r["domain"], r["id"]) for r in records] == [("shared", "project:beta")]


def test_export_all_then_import_restores_each_domain(cli_env, tmp_path):
    store("project:alpha", "Alpha ships on Fridays.", domain="dev")
    store("project:beta", "Beta is a shared thing.")
    store("person:carol", "Carol runs the allotment.", domain="personal")
    backup = tmp_path / "backup.json"
    backup.write_text(run("export", "--domain", "all"))
    for entity_id, domain in [("project:alpha", "dev"), ("project:beta", "shared"),
                              ("person:carol", "personal")]:
        run("delete", entity_id, "--domain", domain, "--yes")

    out = run("import", str(backup))

    assert "Imported 3 entities (shared: 1, dev: 1, personal: 1)." in out
    assert get_entity(cli_env, "project:alpha", domain="dev") is not None
    assert get_entity(cli_env, "project:alpha", domain="shared") is None
    assert get_entity(cli_env, "person:carol", domain="personal") is not None


def test_import_domain_filters_records(cli_env, tmp_path):
    backup = tmp_path / "backup.json"
    backup.write_text(json.dumps([
        {"domain": "dev", "id": "project:alpha", "type": "project",
         "facts": [{"text": "Alpha ships on Fridays.", "added": "2026-01-01", "source": "e"}]},
        {"id": "project:beta", "type": "project",
         "facts": [{"text": "Beta is a shared thing.", "added": "2026-01-01", "source": "e"}]},
    ]))

    out = run("import", str(backup), "--domain", "dev")

    assert "Imported 1 entities (dev: 1)." in out
    assert "Skipped 1 from other domains." in out
    assert get_entity(cli_env, "project:beta", domain="shared") is None


def test_import_rejects_unknown_domain_before_writing(cli_env, tmp_path):
    backup = tmp_path / "backup.json"
    backup.write_text(json.dumps([
        {"id": "project:beta", "type": "project",
         "facts": [{"text": "Beta is a shared thing.", "added": "2026-01-01", "source": "e"}]},
        {"domain": "work", "id": "project:alpha", "type": "project", "facts": []},
    ]))

    result = CliRunner().invoke(main, ["import", str(backup)])

    assert result.exit_code == 1
    assert "unknown domain 'work'" in result.output
    assert get_entity(cli_env, "project:beta", domain="shared") is None
