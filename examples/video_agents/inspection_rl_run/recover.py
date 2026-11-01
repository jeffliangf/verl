"""Recover this experiment after classified transient failures, within its original budget."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from canary_ops import atomic_json, verify_checkpoint
from watchdog import MIN_FREE_BYTES, TERMINAL, scheduler


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=40, **kwargs).strip()


def decision(report, history, now):
    state = report['scheduler']['state']
    if state not in TERMINAL:
        return {'action': 'monitor', 'reason': 'Allocation is active or scheduler state is uncertain'}
    final = report.get('final_status')
    if final is None:
        return {'action': 'await_final', 'reason': 'Waiting for the watchdog to settle logs and checkpoints'}
    if str(final.get('job_id')) != str(report['job_id']):
        return {'action': 'await_final', 'reason': 'Final status belongs to another allocation'}
    if final.get('target_completed'):
        return {'action': 'complete', 'reason': 'Final checkpoint and evaluation are complete'}
    budget = report.get('allocation_budget') or {}
    remaining = budget.get('deadline_epoch', 0) - now
    if remaining <= 0:
        return {'action': 'budget_exhausted', 'reason': 'Original 72-hour deadline reached'}
    if remaining < 3 * 3600:
        return {'action': 'blocked', 'reason': 'Less than three hours remain for restart, useful work, and final save/evaluation'}
    if final.get('nonfinite_metric') or final.get('fatal_failure') or final.get('checkpoint_verification_error'):
        return {'action': 'blocked', 'reason': 'Numerical, code, or checkpoint integrity failure needs inspection'}
    if len(history) >= 6:
        return {'action': 'blocked', 'reason': 'Six automatic recoveries already attempted'}
    if history and now - history[-1]['submitted_epoch'] < 1800:
        return {'action': 'cooldown', 'reason': 'Waiting at least 30 minutes between recovery attempts'}
    checkpoint = final.get('checkpoint_step') or 0
    if sum(row.get('checkpoint_step', 0) == checkpoint for row in history) >= 2:
        return {'action': 'blocked', 'reason': 'Two recoveries without a newer checkpoint; inspect the repeated failure'}
    reason = final.get('stop_reason') or report.get('watchdog', {}).get('stop_reason') or ''
    if state in {'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'}:
        return {'action': 'recover', 'reason': f'Slurm {state}'}
    if state == 'CANCELLED':
        alert = report.get('watchdog_alert') or {}
        reason = alert.get('reason', '') if str(alert.get('job_id')) == str(report['job_id']) else ''
        if ('exceeded' in reason and 'minute deadline' in reason) or reason == 'No meaningful stage progress for 40 minutes':
            return {'action': 'recover', 'reason': reason}
        return {'action': 'stopped', 'reason': 'Cancellation was not a recoverable watchdog timeout'}
    if state == 'FAILED':
        text = report.get('failure_tail', '')
        fatal = ('CUDA out of memory', 'OutOfMemoryError', 'AssertionError', 'ValueError', 'SyntaxError', 'Nonrecoverable Gemini', 'HTTP 401', 'HTTP 403')
        transient = ('Connection reset', 'Connection refused', 'Connection timed out', 'Socket Timeout', 'NodeDiedError', 'node has been marked dead', 'Service failures exceeded', 'Gemini fallback fraction', 'required process')
        if not any(marker in text for marker in fatal) and any(marker in text for marker in transient):
            return {'action': 'recover', 'reason': 'Recorded transport, node, or provider outage'}
    return {'action': 'blocked', 'reason': f'Unclassified {state}; automatic recovery will not repeat a deterministic failure'}


def verify_launch(root, manifest):
    code = Path(manifest['code_checkout'])
    if command(['git', 'rev-parse', 'HEAD'], cwd=code) != manifest['code_commit']:
        raise RuntimeError('Code checkout no longer matches the recorded commit')
    if command(['git', 'status', '--porcelain'], cwd=code):
        raise RuntimeError('Code checkout is dirty')
    for name, digest in manifest['launcher_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Launch file changed: {name}')
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        raise RuntimeError('Less than 450 GiB free')
    marker = root / 'checkpoints/latest_checkpointed_iteration.txt'
    if marker.exists():
        step = int(marker.read_text())
        verify_checkpoint(root / 'checkpoints', step, manifest['nodes'] * manifest['gpus_per_node'])
        return step
    if any((root / 'checkpoints').glob('global_step_*')) or (root / 'checkpoints/verified.json').exists():
        raise RuntimeError('Checkpoint history exists without a complete latest marker')
    return 0


def submit(root, reason, *, automatic):
    """Submit held, record its identity, start monitoring, then release exactly that job."""
    manifest = read(root / 'launch.json')
    old = str(manifest['job_id'])
    if scheduler(old)['state'] not in TERMINAL:
        raise RuntimeError('Previous allocation has not terminated')
    checkpoint = verify_launch(root, manifest)
    budget = read(root / 'run-budget.json')
    minutes = int((budget['deadline_epoch'] - time.time()) // 60)
    if minutes <= 0:
        raise RuntimeError('Original run budget is exhausted')
    command(['systemctl', '--user', 'stop', manifest['watchdog_service']])
    archive = root / 'attempts' / old
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('launch.json', 'submission.json', 'watchdog-state.json', 'watchdog-alert.json', 'final-status.json', 'check-latest.json', 'restore-request.json'):
        path = root / name
        if path.exists() and not (archive / name).exists():
            shutil.copy2(path, archive / name)
    # Only allocation lifecycle records are reset; training/eval data and the
    # immutable run budget stay in place for the checkpoint callback to resume.
    for name in ('watchdog-state.json', 'watchdog-alert.json', 'final-status.json', 'monitor-warning.json'):
        (root / name).unlink(missing_ok=True)
    pending_path = root / 'recovery-pending.json'
    pending = read(pending_path)
    if pending is None:
        pending = {'old_job': old, 'token': f'grpo-recover-{uuid.uuid4().hex}', 'reason': reason, 'automatic': automatic, 'checkpoint_step': checkpoint}
        atomic_json(pending_path, pending)
    if pending['old_job'] != old:
        raise RuntimeError('Unresolved submission transaction belongs to another allocation')
    # A controller crash after sbatch must not create a second allocation.
    matches = []
    for line in command(['squeue', '-h', '-u', 'jeff', '-o', '%i|%k']).splitlines():
        job, comment = line.split('|', 1)
        if comment == pending['token']:
            matches.append(job)
    if len(matches) > 1:
        raise RuntimeError('Duplicate recovery submissions require inspection')
    job = matches[0] if matches else command([
        'sbatch', '--parsable', '--hold', f'--time={minutes}', f'--comment={pending["token"]}', str(root / 'train.sbatch'),
    ]).split(';')[0]
    details = command(['scontrol', 'show', 'job', job, '--oneliner']) + ' '
    expected = [f'JobId={job} ', f'JobName={manifest["job_name"]} ', f'UserId=jeff({os.getuid()}) ', f'Command={root / "train.sbatch"} ', 'JobState=PENDING ', 'Reason=JobHeldUser ']
    if not all(value in details for value in expected):
        raise RuntimeError('New held allocation identity does not match this experiment')
    now = time.time()
    pending.update(job_id=int(job), submitted_epoch=now, original_deadline_epoch=budget['deadline_epoch'])
    atomic_json(pending_path, pending)
    manifest.update(job_id=int(job), replaces_job_id=int(old), submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), recovery_reason=reason, recovery_checkpoint_step=checkpoint, slurm_time_limit_minutes=minutes)
    atomic_json(root / 'launch.json', manifest)
    atomic_json(root / 'submission.json', pending)
    history = read(root / 'recovery-history.json', [])
    if not any(row['job_id'] == int(job) for row in history):
        history.append(pending)
        atomic_json(root / 'recovery-history.json', history)
    command(['systemctl', '--user', 'restart', manifest['watchdog_service']])
    command(['systemctl', '--user', 'enable', '--now', manifest['check_timer']])
    command(['scontrol', 'release', job])
    pending_path.unlink()
    return {'action': 'submitted', **pending}


def recover(root, report, *, manual_reason=None):
    root = Path(root)
    with (root / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(root / 'launch.json')
        if manifest.get('controller_host') != os.uname().nodename:
            return {'action': 'blocked', 'reason': 'Recovery belongs on the recorded controller'}
        if (root / 'recovery-disabled.json').exists() and manual_reason is None:
            return {'action': 'disabled', 'reason': 'Recovery temporarily disabled for a planned deployment'}
        pending = read(root / 'recovery-pending.json')
        if pending and str(pending.get('job_id')) == str(manifest['job_id']):
            # Finish a transaction interrupted after launch.json was committed.
            status = scheduler(str(manifest['job_id']))
            history = read(root / 'recovery-history.json', [])
            if not any(row['job_id'] == manifest['job_id'] for row in history):
                atomic_json(root / 'recovery-history.json', [*history, pending])
            if status['state'] == 'PENDING' and 'JobHeldUser' in status.get('nodes_or_reason', ''):
                command(['systemctl', '--user', 'restart', manifest['watchdog_service']])
                command(['scontrol', 'release', str(manifest['job_id'])])
            (root / 'recovery-pending.json').unlink()
            return {'action': 'transaction_completed', 'job_id': manifest['job_id']}
        if pending:
            # A crash before committing launch.json may have removed the old
            # final-status file. Resume the held submission directly instead of
            # waiting another monitor cycle to recreate that terminal report.
            return submit(root, pending['reason'], automatic=pending['automatic'])
        if manual_reason is not None:
            return submit(root, manual_reason, automatic=False)
        report = {**report, 'watchdog_alert': read(root / 'watchdog-alert.json')}
        log = root / f'slurm-{manifest["job_id"]}.log'
        if log.exists():
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 131072))
                report['failure_tail'] = handle.read().decode(errors='replace')
        history = [row for row in read(root / 'recovery-history.json', []) if row.get('automatic')]
        choice = decision(report, history, time.time())
        if choice['action'] in ('monitor', 'await_final'):
            active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', manifest['watchdog_service']]).returncode == 0
            if not active:
                command(['systemctl', '--user', 'start', manifest['watchdog_service']])
        if choice['action'] == 'recover':
            choice = submit(root, choice['reason'], automatic=True)
        atomic_json(root / 'recovery-state.json', {**choice, 'checked_at_epoch': time.time()})
        return choice


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--manual-reason', required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.run_dir, {}, manual_reason=args.manual_reason)), flush=True)
