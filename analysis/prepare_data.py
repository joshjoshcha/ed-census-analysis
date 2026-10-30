"""Prepare patient-level measures and check the input data.

Run from the submission folder with the supplied data and reference folder in
that same folder. This creates output/encounters_derived.pkl for the analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'


def count_present(start, end, times):
    """At each time, count intervals started minus intervals already ended."""
    start_times = np.sort(start.dropna().astype('int64'))
    end_times = np.sort(end.dropna().astype('int64'))
    query_times = times.astype('int64')
    started = np.searchsorted(start_times, query_times, side='right')
    ended = np.searchsorted(end_times, query_times, side='right')
    return started - ended


def prepare_data():
    OUTPUT.mkdir(exist_ok=True)
    encounters = pd.read_csv(ROOT / 'encounters.csv')
    shifts = pd.read_csv(ROOT / 'physician_shifts.csv')

    # The supplied timestamps share a local clock. Don't convert time zones.
    for column in ['triage_time', 'md_assignment_time', 'discharge_order_time', 'departure_time']:
        encounters[column] = pd.to_datetime(encounters[column], format='%Y-%m-%dT%H:%M:%S')
    for column in ['shift_begin_time', 'shift_end_time']:
        shifts[column] = pd.to_datetime(shifts[column], format='%Y-%m-%dT%H:%M:%S')

    encounters = encounters.merge(
        shifts, on=['shift_id', 'phys_id'], how='left', validate='many_to_one'
    )
    encounters['assigned'] = encounters.md_assignment_time.notna()
    assigned = encounters.assigned

    # Match the reference: assigned patients count until their discharge order,
    # and unassigned patients count until physical departure.
    census_end = encounters.discharge_order_time.where(assigned, encounters.departure_time)
    encounters['census'] = count_present(
        encounters.triage_time, census_end, encounters.triage_time
    )
    # The reference counts simultaneous arrivals in encounter-ID order.
    ties = encounters.groupby('triage_time').encounter_id
    encounters['census'] -= ties.transform('size') - ties.rank(method='dense')
    encounters['census'] = encounters.census.astype(int)

    # Other measures answer different questions about where patients are.
    encounters['physical'] = count_present(
        encounters.triage_time, encounters.departure_time, encounters.triage_time
    ) - 1
    encounters['active_triage'] = count_present(
        encounters.loc[assigned, 'md_assignment_time'],
        encounters.loc[assigned, 'discharge_order_time'],
        encounters.triage_time,
    )
    queue_end = encounters.md_assignment_time.where(assigned, encounters.departure_time)
    encounters['queue_triage'] = count_present(
        encounters.triage_time, queue_end, encounters.triage_time
    ) - 1

    # Durations are in minutes.
    encounters['los'] = (encounters.discharge_order_time - encounters.md_assignment_time).dt.total_seconds() / 60
    encounters['wait'] = (encounters.md_assignment_time - encounters.triage_time).dt.total_seconds() / 60
    encounters['total'] = (encounters.discharge_order_time - encounters.triage_time).dt.total_seconds() / 60
    encounters['post_order'] = (encounters.departure_time - encounters.discharge_order_time).dt.total_seconds() / 60
    encounters['remaining'] = (encounters.shift_end_time - encounters.md_assignment_time).dt.total_seconds() / 60
    encounters['since_start'] = (encounters.md_assignment_time - encounters.shift_begin_time).dt.total_seconds() / 60
    encounters['after_end'] = (encounters.discharge_order_time - encounters.shift_end_time).dt.total_seconds() / 60

    encounters['staff_triage'] = count_present(
        shifts.shift_begin_time, shifts.shift_end_time, encounters.triage_time
    )
    encounters['active_assign'] = np.nan
    encounters['own_load'] = np.nan
    encounters['staff_assign'] = np.nan
    assigned_visits = encounters.loc[assigned]
    encounters.loc[assigned, 'active_assign'] = count_present(
        assigned_visits.md_assignment_time,
        assigned_visits.discharge_order_time,
        assigned_visits.md_assignment_time,
    ) - 1
    encounters.loc[assigned, 'staff_assign'] = count_present(
        shifts.shift_begin_time, shifts.shift_end_time, assigned_visits.md_assignment_time
    )

    # Workload counts the physician's other patients at the time of assignment.
    for physician, index in assigned_visits.groupby('phys_id').groups.items():
        visits = encounters.loc[index]
        encounters.loc[index, 'own_load'] = count_present(
            visits.md_assignment_time, visits.discharge_order_time, visits.md_assignment_time
        ) - 1

    encounters['hour'] = encounters.triage_time.dt.hour
    encounters['dow'] = encounters.triage_time.dt.day_of_week
    encounters['month'] = encounters.triage_time.dt.month
    encounters['year'] = encounters.triage_time.dt.year
    encounters['date'] = encounters.triage_time.dt.strftime('%Y-%m-%d')
    encounters['bin'] = pd.cut(
        encounters.census,
        [-1, 19, 29, 39, 49, 59, 69, 79, 999],
        labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
    )
    encounters.to_pickle(OUTPUT / 'encounters_derived.pkl')
    return encounters, shifts


def print_checks(encounters, shifts):
    print('DATES', encounters.triage_time.min(), encounters.triage_time.max(),
          'phys', encounters.phys_id.nunique())
    print('MISSING', encounters.isna().sum().to_string())
    print('ORDER', {
        'assign_before_triage': int((encounters.wait < 0).sum()),
        'nonpositive_los': int((encounters.los <= 0).sum()),
        'departure_before_order': int((encounters.post_order < 0).sum()),
        'before_shift': int((encounters.since_start < 0).sum()),
        'after_shift_assignment': int((encounters.remaining < 0).sum()),
    })
    columns = ['los', 'wait', 'remaining', 'since_start', 'after_end',
               'census', 'own_load', 'severity_score']
    print(encounters[columns].describe(percentiles=[.01, .1, .5, .9, .99]).to_string())
    print(encounters.groupby('bin', observed=True).agg(
        n=('assigned', 'size'), assigned=('assigned', 'mean'),
        severity=('severity_score', 'mean'), los=('los', 'mean'),
        wait=('wait', 'mean'), remaining=('remaining', 'mean'),
        load=('own_load', 'mean'), active=('active_triage', 'mean'),
        queue=('queue_triage', 'mean'), staff=('staff_triage', 'mean'),
    ).to_string())
    print('PHYS', encounters.groupby('phys_id').agg(
        n=('los', 'count'), los=('los', 'mean'), severity=('severity_score', 'mean'),
        census=('census', 'mean'),
    ).to_string())
    print('AFTER END COUNTS', encounters.after_end.round().value_counts().head(20).to_string())

    # These checks use the reference sample with positive assignment-to-order LOS.
    positive_los = encounters[encounters.los > 0].copy()
    print('after end', positive_los.after_end.quantile([0, .5, .9, .95, .99, .999, 1]).to_string())
    print('handoff candidates', positive_los[positive_los.after_end >= 120].after_end.value_counts().head(20).to_string())
    timing = positive_los.assign(rbin=pd.cut(
        positive_los.remaining, [0, 60, 120, 180, 240, 360, 540, 720, 1000, 2500]
    )).groupby('rbin', observed=True).agg(
        n=('los', 'size'), los=('los', 'mean'), sev=('severity_score', 'mean'),
        wait=('wait', 'mean'), over=('after_end', lambda x: (x >= 120).mean()),
    )
    print('timing', timing.to_string())
    severity = encounters.assign(sbin=pd.cut(
        encounters.severity_score, range(0, 101, 10), include_lowest=True
    )).groupby('sbin', observed=True).agg(
        n=('assigned', 'size'), assigned=('assigned', 'mean'),
        wait=('wait', 'mean'), los=('los', 'mean'),
    )
    print('SEVERITY', severity.to_string())
    print('HOURS', positive_los.groupby('hour').agg(
        n=('los', 'size'), census=('census', 'mean'), sev=('severity_score', 'mean'),
        rem=('remaining', 'mean'), los=('los', 'mean'),
        pre=('since_start', lambda x: (x < 0).mean()),
    ).to_string())
    print('support', positive_los.groupby('bin', observed=True).agg(
        before=('since_start', lambda x: (x < 0).mean()),
        end120=('after_end', lambda x: (x >= 120).mean()),
        sev=('severity_score', 'mean'),
    ).to_string())
    print('DUPS', encounters.encounter_id.duplicated().sum(),
          'shifts', shifts.shift_id.duplicated().sum())
    print('days', encounters.groupby('date').size().describe().to_string())


if __name__ == '__main__':
    encounters, shifts = prepare_data()
    print_checks(encounters, shifts)
