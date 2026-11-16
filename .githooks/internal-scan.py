#!/usr/bin/env python3
"""internal-scan: denylist gate for paths and file contents.

Complements gitleaks (which handles credential regexes and skips binary files):
this tool enforces the *internal information* denylist -- ops notes, ledgers,
restore runbooks, production dumps, env files, logs, internal paths, hostnames,
usernames, private addresses and internal ports -- and it also reads binaries.

Modes
  internal-scan.py --files <path>...          pre-commit / staged files
  internal-scan.py --range <git-range>        commits being pushed or PR'd
  internal-scan.py --all                      every path ever committed (repo audit)

Exit code 1 on any CREDENTIAL or INTERNAL match. --warn-only for advisory runs.
"""
import argparse, os, re, subprocess, sys

# ------------------------------------------------------------------ denylist
USERS = r'(?:treadall|dcmaxhome|dcdexhome|dexhome|hwfloor)'

PATH_RULES = [
    ('INTERNAL', 'ops-notes-dir',   r'(^|/)(notes|runbooks?|ledgers)/|(^|/)ops/(deploy|network|hosts|router|runbook)/'),
    ('INTERNAL', 'ledger-file',     r'(?i)(^|/)[^/]*ledger[^/]*\.md$'),
    ('INTERNAL', 'restore-file',    r'(?i)(^|/)[^/]*restore[^/]*\.(md|txt)$'),
    ('INTERNAL', 'prod-dump',       r'(?i)(^|/)[^/]*prod[^/]*\.txt$'),
    ('INTERNAL', 'env-file',        r'(^|/)\.env([.-][a-z0-9]+)?$|(^|/)[^/]+\.env$'),
    ('NOISE',    'log-file',        r'(^|/)logs?/|\.log(\.[0-9a-z]+)?$'),
    ('CREDENTIAL','key-material',   r'(?i)(^|/)(\.ssh/|id_(rsa|ed25519)$|[^/]+\.(pem|key|p12|pfx)$|certs/[^/]+\.key$)'),
    ('CREDENTIAL','creds-file',     r'(?i)(^|/)[^/]*(creds|credentials|secrets?|passwords?)[^/]*\.(txt|json|ya?ml|env|md)$'),
    ('INTERNAL', 'db-dump',         r'(?i)\.(sqlite|sqlite3|dump|sql)$'),
    ('INTERNAL', 'internal-worktree', r'(?i)(^|/)(recipe-artifacts|p7-2gx[a-z0-9-]*|qmesh-forensic|kld-r5|projects-archive)(/|$)'),
]

CONTENT_RULES = [
    ('CREDENTIAL', 'github-token',    r'(gh[pousr]_[0-9A-Za-z]{36,}|github_pat_[0-9A-Za-z_]{40,})'),
    ('CREDENTIAL', 'private-key',     r'-----BEGIN [A-Z ]*PRIVATE KEY( BLOCK)?-----|openssh-key-v1'),
    ('CREDENTIAL', 'service-key',     r'(AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{35}|sk-(ant|proj|or-v1)[A-Za-z0-9]{20,}|xox[baprs]-[0-9A-Za-z-]{10,}|[0-9]{9,10}:AA[A-Za-z0-9_\-]{30,})'),
    ('CREDENTIAL', 'secret-assign',   r'(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?key|client[_-]?secret)\b\s*[:=]\s*["\x27]?[A-Za-z0-9/+._~\-]{20,}["\x27]?'),
    ('CREDENTIAL', 'cred-url',        r'[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@"\x27]{2,}:[^/\s:@"\x27]{4,}@[a-zA-Z0-9.\-]+'),
    ('CREDENTIAL', 'vpn-key',         r'(?im)^\s*(privatekey|presharedkey|psk)\s*=\s*[A-Za-z0-9+/=]{40,}'),
    ('INTERNAL',   'home-path',       r'/home/' + USERS + r'\b'),
    ('INTERNAL',   'data-path',       r'(?<![\w/])(?:/models/(?!REDACTED)[A-Za-z0-9]|/mnt/data[0-9]+\b|/mnt/str\b|/root/private\b|\.local/share/omp-dash\b|~/\.openclaw\b|/opt/vllm-moet\b)'),
    ('INTERNAL',   'hostname',        r'(?i)\b(gx10-[0-9a-z]+|seam-(p7|gx)(?!-REDACTED)[a-z0-9]*|expert-worker-gx[0-9]|gptware-vps|hwfloor-hf|sparkpoly)'),
    ('INTERNAL',   'node-name',       r'\b(GX[12]\b|P7\b)'),
    ('INTERNAL',   'private-ip',      r'(?<![\d.])(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3})(?![\d./-])'),
    ('INTERNAL',   'internal-port',   r':\b(864[0-9]|873[0-9]|874[0-9]|9119|9421|8998|8172|8120|6349|18735|47311|47351|29517)\b'),
    ('INTERNAL',   'internal-email',  r'(?i)\b(?:dcmaxhome|dcdexhome|' + USERS + r'|gptware)[a-z0-9._-]*@(?:users\.noreply\.github\.com|gmail\.com|gptware\.[a-z]+|proton(?:me)?\.[a-z]+)'),
    ('INTERNAL',   'rdma-or-netdev',  r'(?i)\b(roce(?!X-REDACTED)[a-z0-9]+|/dev/infiniband|allowedips|listen_port\s+[0-9]|wireguard)\b'),
    ('INTERNAL',   'ssh-material',    r'(ssh-(rsa|ed25519)\s+AAAA|authorized_keys|ecdsa-sha2-nistp)'),
]

PATH_RULES = [(s, n, re.compile(p)) for s, n, p in PATH_RULES]
CONTENT_RULES = [(s, n, re.compile(p)) for s, n, p in CONTENT_RULES]

SKIP_PATH = re.compile(r'(^|/)(node_modules|vendor|\.git)/|\.gitleaks\.toml$|(^|/)\.pre-commit-config\.yaml$|\.min\.(js|css)$|(^|/)(package-lock|uv)\.lock$|\.svg$|\.png$|\.jpg$|\.gif$|\.woff2?$|\.pdf$')
# Calibration corpora and vendored upstream sources legitimately contain code
# snippets that look like credentials; they are reviewed, not blocked.
SOFT_PATH = re.compile(r'standard_cal_data/|(^|/)vendor/|third_party/|(^|/)exllamav3/|(^|/)util/|(^|/)examples/|(^|/)eval/|(^|/)doc/')


PLACEHOLDER = re.compile(r'/path/to/|/your/|<[^>\n]{1,40}>|\bexample\.(?:com|org)\b|clientRegistration|context\.(?:next|getCurrent)Token|/redist/|\.nvidia\.com/|huggingface\.co')


def report(sev, rule, where, path, line, sample):
    print(f'{sev}\t{rule}\t{path}\t{where}\t{line}\t{sample[:120]}')
    return sev


def check_path(path):
    bad = 0
    for sev, rule, rx in PATH_RULES:
        if rx.search(path):
            bad += 1 if report(sev, rule, 'path', path, '', '') in ('CREDENTIAL', 'INTERNAL') else 0
    return bad


def check_text(path, text):
    bad = 0
    soft = bool(SOFT_PATH.search(path))
    for i, line in enumerate(text.splitlines(), 1):
        for sev, rule, rx in CONTENT_RULES:
            m = rx.search(line)
            if m:
                if soft and sev != 'CREDENTIAL':
                    continue
                if PLACEHOLDER.search(line):
                    continue
                if report(sev, rule, 'content', path, i, line.strip()) in ('CREDENTIAL', 'INTERNAL'):
                    bad += 1
    return bad


def check_blob(path, data):
    text = data.decode('utf-8', 'replace')
    if '\0' in text[:8192]:
        # binary: only credential patterns are meaningful, and only on the
        # NUL-free lines the decoder produced
        bad = 0
        for sev, rule, rx in CONTENT_RULES:
            if sev != 'CREDENTIAL':
                continue
            for i, line in enumerate(text.splitlines(), 1):
                m = rx.search(line)
                if m:
                    report(sev, rule, 'content(binary)', path, i, m.group()[:120])
                    bad += 1
                    break
        return bad
    return check_text(path, text)


def git(*args):
    return subprocess.run(['git', *args], capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--files', nargs='*', help='worktree paths (pre-commit mode)')
    g.add_argument('--range', help='git rev range, e.g. origin/main..HEAD')
    g.add_argument('--all', action='store_true', help='every path ever committed')
    ap.add_argument('--warn-only', action='store_true')
    a = ap.parse_args()

    bad = 0
    if a.files is not None:
        paths = a.files or [l.split('\t')[-1] for l in git('diff', '--cached', '--name-only', '-z').split(b'\0') if l]
        for p in paths:
            if SKIP_PATH.search(p):
                continue
            bad += check_path(p)
            if os.path.isfile(p):
                bad += check_blob(p, open(p, 'rb').read())
    elif a.range:
        rl = subprocess.run(['git', 'rev-list', '--no-merges', a.range], capture_output=True)
        if rl.returncode != 0:
            print(f"internal-scan: unusable --range {a.range!r}: "
                  f"{rl.stderr.decode('utf-8','replace').strip()[:200]}", file=sys.stderr)
            return 2
        revs = rl.stdout.split()
        for rev in revs:
            r = rev.decode()
            for name in git('show', '--format=', '--name-only', r).split():
                p = name.decode()
                if SKIP_PATH.search(p):
                    continue
                bad += check_path(p)
                blob = git('show', f'{r}:{p}')
                if blob:
                    bad += check_blob(p, blob)
    else:
        seen = set()
        rl = subprocess.run(['git', 'rev-list', '--objects', '--all'], capture_output=True)
        if rl.returncode != 0:
            print('internal-scan: rev-list --all failed: ' + rl.stderr.decode('utf-8', 'replace')[:200], file=sys.stderr)
            return 2
        for line in rl.stdout.splitlines():
            parts = line.split(b' ', 1)
            if len(parts) != 2 or not re.fullmatch(rb'[0-9a-f]{40}', parts[0]):
                continue
            oid, path = parts[0].decode(), parts[1].decode()
            if oid in seen or SKIP_PATH.search(path):
                continue
            seen.add(oid)
            bad += check_path(path)
            bad += check_blob(path, git('cat-file', 'blob', oid))
    print(f'internal-scan: {bad} blocking match(es)', file=sys.stderr)
    return 0 if (a.warn_only or bad == 0) else 1


if __name__ == '__main__':
    sys.exit(main())