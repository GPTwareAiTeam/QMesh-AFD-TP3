# Agent instructions

## Committing
- Never commit notes, ledgers, runbooks, restore instructions, internal paths,
  hostnames, node names, usernames, emails, private IP addresses, internal
  ports, or credentials. This applies to *history* too: a file committed once
  stays readable forever, even after you delete it in a later commit.
- Internal artifacts (ledgers, GPU/queue notes, incident timelines, prod state
  dumps, `.env` files, `*.sqlite`, logs) belong in the private ops repo, never
  in a repo that is or could become public.
- Run the scan before every push and fix what it reports:
  `python3 .githooks/internal-scan.py --range origin/HEAD..HEAD`
  and `gitleaks git --config .gitleaks.toml --log-opts "--all" --redact .`
- If the scan reports a match you believe is a false positive, say so and stop.
  Do not add an allowlist entry, do not use `--no-verify`, and do not weaken a
  rule without the owner's approval.

## History
- Never force-push to rewrite history in a public repo without asking. Never
  delete a repo, change its visibility, or rewrite a branch you did not create.
- Never embed a credential in a remote URL, a git config, a workflow env, or a
  committed file. Use the environment or a secret store.

## Publishing
- Public repos are updated only through `publish_export.sh` (allowlist copy from
  the private working tree, scan, push). Do not work directly in a public clone.
## Agent rules (shared guardrails)

## Committing
- Never commit notes, ledgers, runbooks, restore instructions, internal paths,
  hostnames, node names, usernames, emails, private IP addresses, internal
  ports, or credentials. This applies to *history* too: a file committed once
  stays readable forever, even after you delete it in a later commit.
- Internal artifacts (ledgers, GPU/queue notes, incident timelines, prod state
  dumps, `.env` files, `*.sqlite`, logs) belong in the private ops repo, never
  in a repo that is or could become public.
- Run the scan before every push and fix what it reports:
  `python3 .githooks/internal-scan.py --range origin/HEAD..HEAD`
  and `gitleaks git --config .gitleaks.toml --log-opts "--all" --redact .`
- If the scan reports a match you believe is a false positive, say so and stop.
  Do not add an allowlist entry, do not use `--no-verify`, and do not weaken a
  rule without the owner's approval.

## History
- Never force-push to rewrite history in a public repo without asking. Never
  delete a repo, change its visibility, or rewrite a branch you did not create.
- Never embed a credential in a remote URL, a git config, a workflow env, or a
  committed file. Use the environment or a secret store.

## Publishing
- Public repos are updated only through `publish_export.sh` (allowlist copy from
  the private working tree, scan, push). Do not work directly in a public clone.
