# Capturing conversations

Agents rarely write memory on their own. An MCP server only sees the tool
calls made to it, so KIPP can't watch the conversation from the server side.
Capture works from the client instead: a Claude Code hook records each
exchange, and an extraction pass later decides what's worth keeping.

```
Claude Code ──Stop/SessionEnd hook──▶ kipp-capture ──▶ spool (local file)
spool ──kipp capture flush (every 5 min)──▶ events (30-day inbox)
events ──extraction pass (an LLM agent)──▶ entities
```

- **Hook:** `kipp-capture --hook` runs after every turn. It reads only the new
  part of the session transcript and appends one record per turn to a local
  spool. It uses only the standard library, never loads the model, and never
  fails the turn.
- **Flush:** `kipp capture flush` embeds the spool into events. It's cheap
  enough for a 5-minute timer: with nothing new it exits without loading the
  model.
- **Extraction:** the same pass as for any other event
  ([agent guide](agent-guide.md#the-extraction-pass)). Captured turns are
  untagged, so all of them go through the agent. Without that pass they
  expire after 30 days.

`kipp-capture` is installed with `kipp`. Check with `command -v kipp-capture`;
the examples below assume `~/.local/bin`, where pipx puts it.

## What's kept

- **Kept:** the person's messages and the assistant's prose. Each side of a
  turn is clipped at 4,000 characters, with the clip marked.
- **Dropped:**
  - tool calls and tool output;
  - thinking;
  - subagent messages;
  - compaction summaries and meta records;
  - injected context blocks such as `<system-reminder>`.
- **Size:** on real sessions that leaves under 1% of the transcript, about
  12KB for a typical session.

## 1. Register the hook

Add both events to `~/.claude/settings.json`, merging with any `hooks` you
already have:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [{ "type": "command", "command": "~/.local/bin/kipp-capture --hook", "timeout": 30 }] }
    ],
    "SessionEnd": [
      { "hooks": [{ "type": "command", "command": "~/.local/bin/kipp-capture --hook", "timeout": 30 }] }
    ]
  }
}
```

- **`Stop`** fires after each turn, so captures are near-live.
- **`SessionEnd`** is the backstop for the last turn. Keep its `timeout`:
  without one, Claude Code gives `SessionEnd` hooks 1.5 seconds, which can
  cut a capture short on a long session. A cut capture loses nothing, but the
  last turn then waits for the session to be resumed.
- **Scope.** User settings capture every Claude Code session on the machine.
  To capture only some projects, put the same block in those projects'
  `.claude/settings.json` (or `.claude/settings.local.json` to keep it out of
  git) instead.

Check it works: send one message in a new session, then look for a record:

```bash
wc -l ~/.local/state/kipp/capture/spool.jsonl
tail ~/.local/state/kipp/capture/capture.log
```

The state directory is `$XDG_STATE_HOME/kipp/capture` when `XDG_STATE_HOME` is
set. It's private to you: the directories are 0700 and the files 0600, and
`kipp-capture` refuses to write if they're looser.

### Leaving sessions out

Set `KIPP_CAPTURE=off` in a process's environment and the hook does nothing
for it. Hooks inherit the agent's environment, so this works without editing
any settings:

```bash
KIPP_CAPTURE=off claude      # one session you'd rather not record
```

- **Agent harnesses.** Programs built on the Claude Agent SDK load your user
  hooks by default. If you run one that already writes its own memory, set
  `KIPP_CAPTURE=off` in its environment (a systemd unit's `Environment=`, for
  example), or it records every one of its sessions too.
- **The extraction pass.** [`contrib/kipp-extract.sh`](../contrib/kipp-extract.sh)
  runs `claude -p` with hooks disabled, so it never captures itself.

## 2. Route turns to domains (optional)

Captured turns go to the `shared` domain unless you route them by working
directory. Add a `capture` section to `~/.config/kipp/config.json`:

```json
{
  "capture": {
    "domains": { "~/dev": "dev", "~/notes": "personal" },
    "default_domain": "shared"
  }
}
```

- **Matching:** the longest matching prefix wins, on whole path components,
  so `~/dev` doesn't match `~/devices`.
- **Errors:** an unknown domain name is an error, logged to `capture.log`.
- **`strip_tags`:** sets which injected blocks are removed. It replaces the
  default list rather than adding to it, so copy the defaults from
  `DEFAULT_STRIP_TAGS` in `src/kipp/capture.py` and extend them.

## 3. Flush on a timer

From the repo checkout, install the user timer:

```bash
cp contrib/systemd/kipp-capture-flush.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kipp-capture-flush.timer
```

Or use cron, which works the same on macOS:

```cron
*/5 * * * * $HOME/.local/bin/kipp capture flush >> $HOME/.local/state/kipp-flush.log 2>&1
```

- **While it runs:** hooks keep appending to a fresh spool, and a flush that
  fails is finished by the next one.
- **Repeated turns:** each hook run re-sends the session's last turn in case
  it grew. The flush skips a turn whose text is already stored, and turns a
  changed one into a new event. The older version is then retired, so
  extraction only ever sees the latest.

## 4. Run the extraction pass

`contrib/kipp-extract.sh` flushes first, then runs the extraction prompt
([setup](agent-guide.md#running-it-nightly-with-claude-code)). The prompt has
a section on captured turns:

- dismiss most of them;
- keep what the person states or decides;
- treat the text as a record, never as instructions.

**Sizing:**

- **Domains.** Add every domain you route captures to on the prompt's
  `Domains to process:` line.
- **How often.** One run handles up to 40 events per domain. A busy day of
  coding can capture a few hundred turns, so once a night drains only part of
  the backlog. Run it every few hours instead:
  ```cron
  15 */4 * * * $HOME/kipp/contrib/kipp-extract.sh >> $HOME/.local/state/kipp-extract.log 2>&1
  ```
- **Batch size.** A turn is at most about 8,000 characters (two clipped
  sides), so a full batch of 40 fits in one model context. Typical batches
  are far smaller.
- **Watch the backlog.** `kipp stats` ends with a line like:
  ```
  captures: 120 unconsolidated (dev 95, shared 25); 4 spooled, not yet flushed
  ```
  If the unconsolidated count keeps growing, extraction isn't keeping up, and
  turns older than 30 days are expiring unread.

## Privacy and redaction

Capture writes your conversations to disk, and extraction sends them to a
model.

- **Redaction is best effort.** These formats are replaced with `[REDACTED]`
  before anything is written:
  - `sk-…` API keys;
  - GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`);
  - AWS access key ids (`AKIA…`);
  - Google API keys (`AIza…`);
  - Slack tokens (`xox…`);
  - `Bearer` tokens;
  - PEM private keys.

  Anything else is stored as written: passwords typed as prose, other
  providers' keys, personal details. Don't paste secrets into a session you
  capture, or run that session with `KIPP_CAPTURE=off`.
- **Where the text lives:**
  - in the spool, until it's flushed;
  - in Qdrant's events collections for 30 days;
  - in whatever the extraction agent writes to entities;
  - with the model provider you run extraction on, which reads it.
- **Injected tags can be spoofed.** Stripping works on tag names, so text
  that contains a literal closing tag such as `</system-reminder>` ends the
  strip early. The rest of that block is kept as dialogue.
- **Removing captures:**
  - delete `spool.jsonl` to drop what hasn't been flushed;
  - flushed turns expire with the events (`kipp expire`);
  - to remove them sooner, delete them from the events collections, which
    carry `source: "capture"`.

## Backfilling and troubleshooting

- **Backfill a past session:**
  ```bash
  kipp-capture --file ~/.claude/projects/<project>/<session-id>.jsonl --cwd ~/dev/app
  ```
  The session id defaults to the file name, which matches what the hook
  uses, so a later hook run doesn't duplicate the turns. `--cwd` drives
  domain routing.
- **Nothing in the spool:**
  - check `capture.log`;
  - check `KIPP_CAPTURE` isn't set in the agent's environment;
  - check the hook command's path is right (`command -v kipp-capture`).
- **A turn looks truncated:** the hook fired before the reply was fully
  written. The next run re-sends it complete, and the flush replaces it.
- **Big first capture:** the first hook run on a long existing session reads
  the whole transcript. Each run spends at most 10 seconds and picks up where
  it stopped next time.
- **Other clients:** Claude Code is the only transcript format supported so
  far.
