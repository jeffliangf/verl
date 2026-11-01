import json
import os
import time

import pytest

import recover
from canary_ops import atomic_json


def report(state='NODE_FAIL', **final):
    return {
        'job_id': 1, 'scheduler': {'state': state},
        'allocation_budget': {'deadline_epoch': 100000},
        'final_status': {'job_id': 1, 'checkpoint_step': 5, **final},
    }


@pytest.mark.parametrize('state', ['NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'])
def test_node_failures_recover(state):
    assert recover.decision(report(state), [], 100)['action'] == 'recover'


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'UNKNOWN', 'COMPLETING'])
def test_never_duplicate_live_or_uncertain_allocation(state):
    assert recover.decision(report(state), [], 100)['action'] == 'monitor'


@pytest.mark.parametrize('field', ['fatal_failure', 'nonfinite_metric', 'checkpoint_verification_error'])
def test_integrity_failures_block_recovery(field):
    assert recover.decision(report(**{field: {'problem': True}}), [], 100)['action'] == 'blocked'


def test_manual_cancel_stays_cancelled():
    assert recover.decision(report('CANCELLED'), [], 100)['action'] == 'stopped'
    r = report('CANCELLED')
    r['watchdog_alert'] = {'job_id': 1, 'reason': 'training exceeded 90 minute deadline'}
    assert recover.decision(r, [], 100)['action'] == 'recover'
    r['watchdog_alert']['job_id'] = 999
    assert recover.decision(r, [], 100)['action'] == 'stopped'


def test_no_restart_after_deadline_or_completion():
    assert recover.decision(report(target_completed=True), [], 100)['action'] == 'complete'
    assert recover.decision(report(), [], 100001)['action'] == 'budget_exhausted'
    assert recover.decision(report(), [], 99000)['action'] == 'blocked'


def test_restart_cooldown_and_no_progress_circuit_breaker():
    h = [{'submitted_epoch': 2000, 'checkpoint_step': 5}]
    assert recover.decision(report(), h, 3000)['action'] == 'cooldown'
    h.append({'submitted_epoch': 3000, 'checkpoint_step': 5})
    assert recover.decision(report(), h, 6000)['action'] == 'blocked'
    assert recover.decision(report(checkpoint_step=10), h, 6000)['action'] == 'recover'


@pytest.mark.parametrize('text,action', [
    ('Connection reset by peer', 'recover'),
    ('Gemini fallback fraction exceeded limit', 'recover'),
    ('Connection reset then ValueError', 'blocked'),
    ('CUDA out of memory', 'blocked'),
    ('Something unexpected', 'blocked'),
    ('Nonrecoverable Gemini HTTP 403', 'blocked'),
])
def test_only_classified_failures_recover(text, action):
    r = report('FAILED')
    r['failure_tail'] = text
    assert recover.decision(r, [], 100)['action'] == action


def test_submission_is_held_until_identity_and_monitor_are_recorded(tmp_path, monkeypatch):
    root = tmp_path
    now = time.time()
    manifest = {'job_id': 1, 'job_name': 'test-grpo', 'watchdog_service': 'watch.service', 'check_timer': 'check.timer'}
    atomic_json(root / 'launch.json', manifest)
    atomic_json(root / 'run-budget.json', {'deadline_epoch': now + 50000})
    atomic_json(root / 'final-status.json', {'old': True})
    monkeypatch.setattr(recover, 'scheduler', lambda job: {'state': 'FAILED'})
    monkeypatch.setattr(recover, 'verify_launch', lambda *args: 5)
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        if args[0] == 'sbatch':
            assert '--hold' in args and '--time=833' in args
            return '2'
        if args[:3] == ['scontrol', 'show', 'job']:
            return f'JobId=2 JobName=test-grpo UserId=jeff({os.getuid()}) Command={root / "train.sbatch"} JobState=PENDING Reason=JobHeldUser'
        if args[:2] == ['scontrol', 'release']:
            assert json.loads((root / 'launch.json').read_text())['job_id'] == 2
            assert ['systemctl', '--user', 'restart', 'watch.service'] in calls
            assert (root / 'recovery-pending.json').exists()
        return ''

    monkeypatch.setattr(recover, 'command', command)
    result = recover.submit(root, 'transient failure', automatic=True)
    assert result['checkpoint_step'] == 5
    assert result['original_deadline_epoch'] == now + 50000
    assert (root / 'attempts/1/final-status.json').exists()
    assert not (root / 'recovery-pending.json').exists()
    assert not (root / 'final-status.json').exists()
    assert len(json.loads((root / 'recovery-history.json').read_text())) == 1


def test_active_job_cannot_be_replaced(tmp_path, monkeypatch):
    atomic_json(tmp_path / 'launch.json', {'job_id': 1})
    monkeypatch.setattr(recover, 'scheduler', lambda job: {'state': 'RUNNING'})
    with pytest.raises(RuntimeError, match='not terminated'):
        recover.submit(tmp_path, 'test', automatic=True)


def test_interrupted_submission_resumes_without_a_terminal_report(tmp_path, monkeypatch):
    atomic_json(tmp_path / 'launch.json', {'job_id': 1, 'controller_host': os.uname().nodename})
    atomic_json(tmp_path / 'recovery-pending.json', {'old_job': '1', 'reason': 'node failure', 'automatic': True})
    calls = []
    monkeypatch.setattr(recover, 'submit', lambda root, reason, automatic: calls.append((reason, automatic)) or {'action': 'submitted'})
    assert recover.recover(tmp_path, {}) == {'action': 'submitted'}
    assert calls == [('node failure', True)]
