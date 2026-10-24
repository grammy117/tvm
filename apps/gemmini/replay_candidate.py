# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Remeasure one retained search ELF into a fresh, independently bound database.

The original failed measurement and database stay unchanged. No old cost or
placeholder cost is admitted. The sampled trace must replay to the exact
retained module, and only a complete new all-output pass produces a record.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import tvm
from tvm import meta_schedule as ms, tir
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR, open_gemmini_database
import measure
from tuning_library import publish_campaign_context, _successful_candidate
from graph_plans import module_digest


def clone_candidate(artifact_path, output, workload):
    """Copy exact guest bytes to a new evidence location without changing source."""
    artifact_path, output, workload = Path(artifact_path), Path(output), Path(workload)
    artifact = json.loads(artifact_path.read_text())
    for elf in artifact['elfs'].values():
        actual = measure.identity(elf['path'])
        if actual['sha256'] != elf['sha256'] or actual['bytes'] != elf['bytes']:
            raise ValueError('Retained executable bytes changed')
    shutil.copytree(artifact_path.parent, output, ignore=shutil.ignore_patterns(
        'measurement.json', 'rtl-timing.log', 'spike-valid.log', 'spike-negative.log', 'retries'))
    copied = dict(artifact, folder=str(output), workload=str(workload),
                  original_artifact=measure.identity(artifact_path))
    copied['elfs'] = {name: measure.identity(output / Path(item['path']).name)
                      for name, item in artifact['elfs'].items()}
    for name in artifact['elfs']:
        if copied['elfs'][name]['sha256'] != artifact['elfs'][name]['sha256']:
            raise ValueError('Copied executable differs from retained candidate')
    (output / 'artifact.json').write_text(json.dumps(copied, indent=2) + '\n')
    return copied


def commit_measurement(database_path, source_record, result):
    """Create a standard database and admit only the newly checked measurement."""
    database_path = Path(database_path)
    database_path.mkdir(parents=True, exist_ok=True)
    database = ms.database.JSONDatabase(work_dir=str(database_path))
    existing = database.get_all_tuning_records()
    if result['status'] != 'passed':
        if existing:
            raise ValueError('Failed replay cannot retain a successful database record')
        return database
    cycles = float(result['cycles'])
    if not 0 < cycles < 1e10:
        raise ValueError('Invalid successful replay cost')
    if existing:
        if len(existing) != 1 or not existing[0].run_secs or float(existing[0].run_secs[0]) != cycles:
            raise ValueError('Replay database contains unrelated records')
        if (not tvm.ir.structural_equal(existing[0].workload.mod, source_record.workload.mod)
                or tvm.ir.save_json(existing[0].trace.as_json()) != tvm.ir.save_json(source_record.trace.as_json())):
            raise ValueError('Replay database trace differs from retained source')
    else:
        database.commit_tuning_record(ms.database.TuningRecord(
            source_record.trace, database.commit_workload(source_record.workload.mod),
            [cycles], source_record.target, source_record.args_info))
    return database


def replay(source_root, artifact_path, output_root, *, max_cycles, rtl_seconds, finalize_only=False):
    started = time.monotonic()
    source_root, artifact_path = Path(source_root).resolve(strict=True), Path(artifact_path).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists() != finalize_only:
        raise ValueError('Replay requires a fresh output directory; --finalize-only requires existing retained evidence')
    if finalize_only and (output_root / 'database/measurement-context.json').exists():
        raise ValueError('Replay was already finalized; immutable context is preserved')
    if artifact_path.parent.parent != source_root / 'candidates':
        raise ValueError('Candidate is outside the source campaign')
    campaign = json.loads((source_root / 'campaign.json').read_text())
    artifact = json.loads(artifact_path.read_text())
    previous_path = artifact_path.parent / 'measurement.json'
    previous = json.loads(previous_path.read_text())
    if previous.get('status') != 'rtl_timeout_or_failure' or max_cycles <= previous.get('cycle_cap', 0):
        raise ValueError('Replay requires a failed capped candidate and larger total-cycle allowance')
    if rtl_seconds <= 0:
        raise ValueError('Replay wall limit must be positive')
    toolchain = measure.Toolchain.from_json(campaign['toolchain'])
    provenance = toolchain.provenance()
    if provenance['memory_model']['fingerprint'] != campaign['memory_model']['fingerprint']:
        raise ValueError('Replay memory model differs from the source campaign')
    if toolchain.timing_policy != 'fixed-warm':
        raise ValueError('Fresh measured library replay requires fixed-warm timing')
    caps = GemminiCapabilities()
    if campaign['capabilities'] != caps.fingerprint:
        raise ValueError('Source campaign capability mismatch')
    if measure.identity(source_root / 'program.json')['sha256'] != campaign['program_sha256']:
        raise ValueError('Source semantic program changed')
    if (measure.identity(campaign['compiler']['path'])['sha256'] != campaign['compiler']['sha256']
            or artifact['compiler']['sha256'] != campaign['compiler']['sha256']):
        raise ValueError('Source compiler identity changed')
    if artifact.get('memory_model', {}).get('fingerprint') != campaign['memory_model']['fingerprint']:
        raise ValueError('Source candidate memory identity mismatch')
    if measure.identity(artifact_path.parent / 'runner.c')['sha256'] != artifact['runner_sha256']:
        raise ValueError('Source runner changed')
    if measure.identity(artifact_path.parent / 'package/operators.ll')['sha256'] != artifact['program_sha256']:
        raise ValueError('Source LLVM changed')
    if previous.get('rtl_log') and measure.identity(previous['rtl_log']['path'])['sha256'] != previous['rtl_log']['sha256']:
        raise ValueError('Source failed console changed')
    prepared = tvm.ir.load_json((source_root / 'prepared.json').read_text())
    if (module_digest(prepared) != campaign['prepared_sha256']
            or json.loads((source_root / 'graph-plan.json').read_text())['prepared_ir_sha256'] != campaign['prepared_sha256']):
        raise ValueError('Source prepared graph identity mismatch')
    database = open_gemmini_database(source_root / 'database', prepared, capabilities=caps)
    trials = [json.loads(line) for line in (source_root / 'trials.jsonl').read_text().splitlines() if line]
    trial = next((row for row in trials if row.get('artifact') and Path(row['artifact']).resolve() == artifact_path), None)
    if trial is None or trial.get('cycles') is not None:
        raise ValueError('Source candidate is not a failed retained search trial')
    records = [record for record in database.get_all_tuning_records()
               if str(record.workload.mod['main'].attrs[WORKLOAD_ATTR]) == trial['workload']
               and json.loads(tvm.ir.save_json(record.trace.as_json())) == trial['trace']]
    if not records:
        raise ValueError('Exact source trace is absent from the standard database')
    record = records[0]
    if str(record.target) != str(caps.target()):
        raise ValueError('Source record target differs from the capability')
    source_workload = source_root / 'workloads' / trial['workload']
    if not tvm.ir.structural_equal(record.workload.mod, tvm.ir.load_json((source_workload / 'workload.json').read_text())):
        raise ValueError('Source workload differs from the retained trace')
    manifest = json.loads((source_workload / 'workload-manifest.json').read_text())
    for name in ('inputs', 'expected'):
        if measure.identity(source_workload / (name + '.bin'))['sha256'] != manifest[name + '_sha256']:
            raise ValueError('Source workload numerical bytes changed')
    scheduled_path = artifact_path.parent.with_suffix('.module.json')
    scheduled = tvm.ir.load_json(scheduled_path.read_text())
    schedule = tir.Schedule(record.workload.mod)
    record.trace.apply_to_schedule(schedule, remove_postproc=False)
    if not tvm.ir.structural_equal(schedule.mod, scheduled):
        raise ValueError('Source trace does not replay to the retained scheduled module')
    key = trial['workload']
    workload = output_root / 'workloads' / key
    candidate = output_root / 'candidates' / ('00000-' + key)
    if not finalize_only:
        output_root.mkdir(parents=True)
        for name in ('program.json', 'prepared.json', 'graph-plan.json', 'extraction.json'):
            shutil.copy2(source_root / name, output_root / name)
        input_bindings = campaign.get('input_bindings')
        if input_bindings is not None:
            actual = measure.identity(input_bindings['path'])
            if actual['sha256'] != input_bindings['sha256']:
                raise ValueError('Source operator input bindings changed')
            bindings_path = output_root / 'operator-input-bindings.json'
            shutil.copy2(input_bindings['path'], bindings_path)
            input_bindings = measure.identity(bindings_path)
        key = trial['workload']
        workload = output_root / 'workloads' / key
        workload.parent.mkdir()
        shutil.copytree(source_root / 'workloads' / key, workload)
        candidate = output_root / 'candidates' / ('00000-' + key)
        candidate.parent.mkdir()
        copied = clone_candidate(artifact_path, candidate, workload)
        shutil.copy2(scheduled_path, candidate.with_suffix('.module.json'))
        (output_root / 'hardware-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        source_evidence = {'schema': 'gemmini_exact_candidate_replay_v1',
            'selection': 'One retained sampled candidate; no new scheduling or historical cost import',
            'source_campaign': measure.identity(source_root / 'campaign.json'),
            'source_artifact': measure.identity(artifact_path), 'source_measurement': measure.identity(previous_path),
            'source_module': measure.identity(scheduled_path),
            'source_trials': measure.identity(source_root / 'trials.jsonl'),
            'source_baselines': measure.identity(source_root / 'tasks.json'),
            'source_database': [measure.identity(source_root / 'database' / name) for name in ('database_workload.json', 'database_tuning_record.json')],
            'source_rtl_log': previous.get('rtl_log'), 'source_valid_elf': artifact['elfs']['valid'],
            'replay_source': measure.identity(__file__), 'measurement_source': measure.identity(measure.__file__),
            'original_total_cycle_cap': previous['cycle_cap'], 'new_total_cycle_cap': max_cycles,
            'wall_limit_seconds': rtl_seconds, 'allowance_scope': 'Whole guest including all-output validation; model_run cost remains the second fenced interval'}
        (output_root / 'replay-provenance.json').write_text(json.dumps(source_evidence, indent=2) + '\n')
        result = measure.run_candidate(copied, toolchain, max_cycles=max_cycles, rtl_seconds=rtl_seconds)
        (candidate / 'measurement.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        source_evidence = json.loads((output_root / 'replay-provenance.json').read_text())
        for label, path in (('source_artifact', artifact_path), ('source_measurement', previous_path),
                            ('source_module', scheduled_path), ('source_campaign', source_root / 'campaign.json')):
            if measure.identity(path)['sha256'] != source_evidence[label]['sha256']:
                raise ValueError('Retained replay source identity changed: ' + label)
        if source_evidence['new_total_cycle_cap'] != max_cycles or source_evidence['wall_limit_seconds'] != rtl_seconds:
            raise ValueError('Finalization cannot change the already executed budget')
        copied = json.loads((candidate / 'artifact.json').read_text())
        result = json.loads((candidate / 'measurement.json').read_text())
        if copied['elfs']['valid']['sha256'] != artifact['elfs']['valid']['sha256']:
            raise ValueError('Retained measured ELF differs from source candidate')
        input_bindings = (measure.identity(output_root / 'operator-input-bindings.json')
                          if campaign.get('input_bindings') is not None else None)
        (output_root / 'finalization-provenance.json').write_text(json.dumps({
            'schema': 'gemmini_replay_finalization_v1', 'action': 'Validate and publish retained complete measurement; no execution',
            'measurement': measure.identity(candidate / 'measurement.json'),
            'original_replay_provenance': measure.identity(output_root / 'replay-provenance.json'),
            'finalizer_source': measure.identity(__file__)}, indent=2) + '\n')
    new_campaign = dict(campaign, strategy='retained_exact_candidate_replay', max_cycles=max_cycles,
                        rtl_seconds=rtl_seconds, toolchain=toolchain.as_json(),
                        wall_seconds=(result.get('spike_seconds', 0) + result.get('rtl_seconds', 0)
                                      if finalize_only else time.monotonic() - started),
                        wall_seconds_scope=('retained functional and RTL execution; setup not included'
                                            if finalize_only else 'complete replay helper'),
                        trials_per_task=0, trials_per_iteration=0, parallel=1,
                        retained_candidates_remeasured=1, newly_proposed_candidates=0,
                        input_bindings=input_bindings,
                        replay_provenance=measure.identity(output_root / 'replay-provenance.json'))
    if finalize_only:
        new_campaign['finalization_provenance'] = measure.identity(output_root / 'finalization-provenance.json')
    if result['status'] == 'passed':
        bindings = json.loads(Path(input_bindings['path']).read_text()) if input_bindings else None
        _successful_candidate(output_root, candidate / 'measurement.json', new_campaign, bindings)
    commit_measurement(output_root / 'database', record, result)
    row = dict(trial, artifact=str(candidate / 'artifact.json'), cycles=result.get('cycles'),
               status=result['status'], run_error=None if result['status'] == 'passed' else result['status'],
               post_search_retry=True, original_artifact=str(artifact_path), rtl_seconds=result.get('rtl_seconds'))
    (output_root / 'trials.jsonl').write_text(json.dumps(row) + '\n')
    (output_root / 'tasks.json').write_text(json.dumps({'tasks': [{'task': row['task'], 'workload': key,
        'scope': 'One explicit retained-candidate replay; source baselines retained in replay-provenance.json'}]}, indent=2) + '\n')
    (output_root / 'campaign.json').write_text(json.dumps(new_campaign, indent=2) + '\n')
    if result['status'] == 'passed':
        publish_campaign_context(output_root)
    return {'campaign': str(output_root), 'status': result['status'], 'cycles': result.get('cycles'),
            'artifact': str(candidate / 'artifact.json'), 'historical_costs_imported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-campaign', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-cycles', type=int, required=True)
    parser.add_argument('--rtl-seconds', type=int, required=True)
    parser.add_argument('--finalize-only', action='store_true', help='Publish an existing complete measurement without rerunning its ELF')
    args = parser.parse_args()
    result = replay(args.source_campaign, args.artifact, args.output_dir,
                    max_cycles=args.max_cycles, rtl_seconds=args.rtl_seconds, finalize_only=args.finalize_only)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 2)


if __name__ == '__main__':
    main()
