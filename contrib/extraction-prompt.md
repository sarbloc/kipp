You are running the periodic KIPP memory-extraction pass. You have the KIPP
memory tools (memory_extract, memory_events_unextracted, memory_search,
memory_get, memory_store, memory_event_resolve). Work through the domains
listed below, one at a time, and pass that `domain` on EVERY call, reads
included: the tools default to "shared", and a read in the wrong domain misses
the entity you meant to update.

Domains to process: shared

For each domain:

1. Call memory_extract({ domain }). It files every event that opens with a
   `[type:id]` tag onto that entity, if the entity exists. Note the counts.

2. Call memory_events_unextracted({ domain, limit: 40 }). These are the
   events nothing could file: untagged ones, and tagged ones whose entity does
   not exist yet.

3. Group the events by the subject they describe. For each group:
   a. Decide whether it is durable knowledge: a preference, a decision, a
      fact about a person, project or tool that will still be true next month.
      Small talk, status updates and one-off tasks are not.
   b. If durable, find the entity first: memory_search({ query, domain }) and,
      for a likely id, memory_get({ entity_id, domain }). Update the existing
      entity rather than creating a near-duplicate under another id.
   c. Write each fact with memory_store({ entity_type, entity_id, content,
      domain }):
      - entity_type is one of person | project | tool | preference | decision;
      - entity_id is a short, stable, lowercase slug (`alice`, `billing-api`);
      - content is ONE short, self-contained claim, ideally under 25 words,
        with the subject and key words first. Each fact is searched on its
        own, so split a paragraph into separate facts rather than storing
        it whole. memory_store returns a `warning` for any fact over 40
        words: re-store it as shorter facts.
      - If memory_store returns `oversized`, the entity is too broad. Put
        later facts on a topic entity named `<id>-<topic>` (e.g.
        `alice-health` for `person:alice`). memory_get on the entity lists
        its existing topic entities as `children`: reuse those. When you
        create a new one, pass `parent` with the entity's full id.
      - When a fact replaces an older one ("moved to Berlin" after "lives in
        London"), pass `supersedes` with the old fact's exact text.
   d. Call memory_event_resolve({ event_id, domain, created_entity_id }) for
      every event in the group, with the entity id you wrote to. Omit
      created_entity_id to dismiss an event that is not worth keeping.
      An unresolved event comes back on the next run.

4. If step 2 returned the full 40, the backlog is bigger than one batch. Stop
   anyway; the next run continues from there.

Finish with one line per domain: events filed by tag, entities created,
entities updated, events dismissed, events left.
