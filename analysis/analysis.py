"""Run the models, sensitivity analyses, figures, and validation checks."""

import os

import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'ed_mpl'))

import importlib.metadata as meta

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pyhdfe

from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / 'output'

all_encounters = pd.read_pickle(OUT / 'encounters_derived.pkl')

# Remove impossible patient event sequences from outcome sample; reference retained separately.
clean = (
    (all_encounters.los > 0)
    & (all_encounters.wait >= 0)
    & (all_encounters.post_order >= 0)
)
main_sample = all_encounters.loc[clean].copy().reset_index(drop=True)
main_sample['ym'] = main_sample.triage_time.dt.strftime('%Y-%m')
main_sample['how'] = main_sample.hour * 7 + main_sample.dow

# B contains the census-group indicators, with the under-20 group as the reference.
# S, T, and L below are spline controls for severity, shift time, and workload.
bins = pd.get_dummies(main_sample['bin'], dtype=float)

bn = list(bins.columns)

B = bins.iloc[:, 1:].to_numpy()

def spl(x, knots, scale=100):
    z = np.asarray(x) / scale
    return np.column_stack([z] + [np.maximum(z - k / scale, 0) for k in knots])

S = spl(main_sample.severity_score, [10, 20, 30, 40, 50, 60, 70, 80, 90], 100)

T = spl(main_sample.remaining, [60, 120, 180, 240, 360, 480, 540, 720, 900, 1080, 1440], 100)

L = spl(main_sample.own_load, [2, 4, 6, 8, 10, 15], 10)

def fit(data, y, x, fe=(), name='', save=True):
    # Put the outcome and controls together so fixed effects are removed from both.
    Z = np.column_stack([np.asarray(y), x]).astype(float)
    if fe:
        groups = [pd.factorize(data[c])[0] for c in fe]
        # Shift models use PyHDFE. Other models remove group means in turn.
        if 'shift_id' in fe:
            algorithm = pyhdfe.create(
                np.column_stack(groups),
                drop_singletons=False,
                compute_degrees=False,
                residualize_method='map',
                options={
                    'transform': 'kaczmarz',
                    'acceleration': 'gk',
                    'tol': 1e-7,
                    'iteration_limit': 10000,
                },
            )
            Z = algorithm.residualize(Z)
        else:
            reducers = [
                csr_matrix(
                    (np.ones(len(g)), (g, np.arange(len(g)))),
                    shape=(g.max() + 1, len(g)),
                )
                for g in groups
            ]
            counts = [np.bincount(g)[:, None] for g in groups]
            for iteration in range(1000):
                old = Z.copy() if iteration % 5 == 0 else None
                for g, reducer, nn in zip(groups, reducers, counts):
                    Z -= (reducer @ Z / nn)[g]
                if old is not None and np.max(np.abs(Z - old)) < 1e-07:
                    break
    else:
        Z -= Z.mean(axis=0)
    # After removing fixed effects, the outcome should have near-zero group means.
    if fe:
        balance = max(
            np.max(np.abs(np.bincount(g, weights=Z[:, 0]) / np.bincount(g)))
            for g in groups
        )
        print(name, 'max residual outcome group mean', balance, flush=True)
        assert balance < 0.0001, 'Fixed effects failed to converge'
    # Estimate the regression after removing the fixed effects.
    yy = Z[:, 0]
    xx = Z[:, 1:]
    inv = np.linalg.pinv(xx.T @ xx)
    beta = inv @ (xx.T @ yy)
    u = yy - xx @ beta
    # Cluster the standard errors by arrival date.
    g = pd.factorize(data.date)[0]
    scores = np.zeros((g.max() + 1, xx.shape[1]))
    np.add.at(scores, g, xx * u[:, None])
    sandwich = scores @ inv
    cov = sandwich.T @ sandwich
    if save:
        results = pd.DataFrame({
            'term': bn[1:],
            'coefficient': beta[:7],
            'se_date_cluster': np.sqrt(np.diag(cov))[:7],
        })
        results.to_csv(OUT / f'model_{name}.csv', index=False)
    print(
        name, 'n', len(yy), 'r2_within', 1 - u @ u / (yy @ yy),
        'coeff', np.round(beta[:min(len(beta), 15)], 3), flush=True,
    )
    return (beta, cov)


# Each stage starts from the same sample. Some stages change their local copy.
base_census, base_severity, base_timing, base_workload = B, S, T, L

def main_models():
    """Compare census groups and make the main figures."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    curves = {}

    for name, X, fe in [
        ('raw', B, ()),
        ('severity', np.c_[B, S], ()),
        ('physician', np.c_[B, S], ('phys_id',)),
        ('calendar', np.c_[B, S], ('phys_id', 'how', 'ym')),
        ('timing', np.c_[B, S, T], ('phys_id', 'how', 'ym')),
        ('operations', np.c_[B, S, T, L, d.staff_assign, d.active_assign / 10],
         ('phys_id', 'how', 'ym')),
        ('date', np.c_[B, S, T, L, d.staff_assign, d.active_assign / 10],
         ('phys_id', 'how', 'date')),
    ]:
        beta, cov = fit(d, d.los, X, fe, name)
        curves[name] = [0] + beta[:7].tolist()

    pd.DataFrame(curves, index=bn).to_csv(OUT / 'adjusted_contrasts.csv')

    # Determine whether observable composition varies with census beyond predictable arrival timing.
    for y in ['severity_score', 'remaining', 'since_start', 'wait']:
        fit(d, d[y], B, ('how', 'ym'), f'{y}_calendar')

    fit(d, (d.since_start < 0).astype(float), B, ('how', 'ym'), 'prestart_calendar')

    # Remaining-time and workload associations within physician/calendar, with and without shift FE.
    for name, fe in [('workload', ('phys_id', 'how', 'ym')), ('workload_shift', ('shift_id',))]:
        X = np.c_[d.own_load, S, T, d.active_assign / 10, d.staff_assign]
        beta, cov = fit(d, d.los, X, fe, name, False)
        pd.DataFrame({
            'term': ['own_load']
            + [f'severity_{i}' for i in range(S.shape[1])]
            + [f'timing_{i}' for i in range(T.shape[1])]
            + ['active_per10', 'staff'],
            'coef': beta,
            'se': np.sqrt(np.diag(cov)),
        }).to_csv(OUT / f'{name}.csv', index=False)

    # Summaries and alternate outcomes
    summ = d.groupby('bin', observed=True).agg(
        n=('los', 'size'),
        los=('los', 'mean'),
        severity=('severity_score', 'mean'),
        remaining=('remaining', 'mean'),
        own_load=('own_load', 'mean'),
        wait=('wait', 'mean'),
        total=('total', 'mean'),
        prestart=('since_start', lambda x: (x < 0).mean()),
    )

    summ.to_csv(OUT / 'census_summary.csv')

    print('CLEAN', len(d), 'removed', int((e.los > 0).sum() - len(d)), flush=True)

    print('PRESTART', d.groupby(d.since_start < 0).agg(n=('los', 'size'), los=('los', 'mean'), remaining=('remaining', 'mean'), sev=('severity_score', 'mean')).to_string(), flush=True)

    print('discharge before begin', int((d.discharge_order_time < d.shift_begin_time).sum()), flush=True)

    print('end violations', int((d.after_end > 180).sum()), flush=True)

    # Reproduce reference CSV exactly using the reference event-order census.
    r = e[e.los > 0].groupby('census').los.agg(['mean', 'std', 'size'])

    r = r[r['size'] >= 500]

    r.to_csv(OUT / 'reference_reproduced.csv')

    ref = pd.read_csv(ROOT / 'reference/census_los_reference_values.csv')

    print('REFCOL', ref.columns.tolist(), flush=True)

    print('REF', np.max(np.abs(r['mean'].values - ref['mean_los'].values)), np.array_equal(r['size'].values, ref.n_enc.values), flush=True)

    # Sensitivity comparisons on original bins.
    alt = pd.DataFrame({'reference': e[e.los > 0].groupby('bin', observed=True).los.mean(), 'clean': summ.los})

    alt['exclude_prestart'] = d[d.since_start >= 0].groupby('bin', observed=True).los.mean()

    d['scheduled_care'] = (d.discharge_order_time - d[['md_assignment_time', 'shift_begin_time']].max(axis=1)).dt.total_seconds() / 60

    alt['from_max_assignment_shiftstart'] = d.groupby('bin', observed=True).scheduled_care.mean()

    alt.to_csv(OUT / 'sensitivity.csv')


    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))

    for name, label in [('raw', 'Unadjusted'), ('severity', '+ Severity'), ('physician', '+ Physician effects'), ('calendar', '+ Calendar effects'), ('timing', '+ Shift time remaining')]:
        axs[0].plot(range(7), np.array(curves[name])[:7], marker='o', label=label)

    axs[0].set(xticks=range(7), xticklabels=bn[:7], xlabel='Whole-ED census at arrival', ylabel='LOS difference from census <20 (min)', title='Sequential adjustment')

    axs[0].legend(fontsize=8)

    for name in ['raw', 'timing', 'operations']:
        axs[1].plot(range(7), np.array(curves[name])[:7], marker='o', label={'raw': 'Unadjusted', 'timing': 'Patient, physician, calendar, shift timing', 'operations': '+ Own workload, active ED load, staffing'}[name])

    axs[1].set(xticks=range(7), xticklabels=bn[:7], xlabel='Whole-ED census at arrival', ylabel='LOS difference from census <20 (min)', title='Workload and staffing controls')

    axs[1].legend(fontsize=8)

    fig.text(0.01, 0.01, 'Clean assigned encounters. Models use splines. Results are descriptive. Census 80+ omitted from plot (n < 300).', fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(OUT / 'pi_result.png', dpi=180)

    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    axs[0, 0].plot(r.index, r['mean'], color='black')

    axs[0, 0].fill_between(r.index, r['mean'] - 1.96 * r['std'] / np.sqrt(r['size']), r['mean'] + 1.96 * r['std'] / np.sqrt(r['size']), color='0.85')

    axs[0, 0].set(xlabel='Reference arrival census', ylabel='Mean LOS (min)', title='Reference relationship (bins n >= 500)')

    h = d.groupby('hour').agg(census=('census', 'mean'), sev=('severity_score', 'mean'))

    axs[0, 1].plot(h.index, h.census, label='Census')

    axs[0, 1].plot(h.index, h.sev, label='Severity percentile')

    axs[0, 1].set(xlabel='Arrival hour', title='Predictable daily composition')

    axs[0, 1].legend()

    rbin = pd.cut(d.remaining, [0, 120, 240, 360, 540, 720, 900, 1200, 2500])

    g = d.groupby(rbin, observed=True).los.agg(['mean', 'size'])

    axs[1, 0].bar(range(len(g)), g['mean'], color='0.7', edgecolor='black', linewidth=0.6)

    axs[1, 0].set(xticks=range(len(g)), xticklabels=['0-2', '2-4', '4-6', '6-9', '9-12', '12-15', '15-20', '20+'], xlabel='Hours until recorded shift end at assignment', ylabel='Mean LOS (min)', title='Assignment timing and LOS')

    for col in ['los', 'wait', 'total']:
        axs[1, 1].plot(range(7), summ[col].iloc[:7], marker='o', label={'los': 'Assignment to order', 'wait': 'Triage to assignment', 'total': 'Triage to order'}[col])

    axs[1, 1].set(xticks=range(7), xticklabels=bn[:7], xlabel='Arrival census', ylabel='Minutes', title='Patient stages have different patterns')

    axs[1, 1].legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(OUT / 'mechanisms.png', dpi=180)

    plt.close(fig)


def sensitivity_checks():
    """Check alternate measures and selection into assignment."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    # Reuse the estimation routine and controls without running the main model sequence.

    def count(start, end, t):
        return np.searchsorted(np.sort(start.dropna().astype('int64')), t.astype('int64'), side='right') - np.searchsorted(np.sort(end.dropna().astype('int64')), t.astype('int64'), side='right')

    # Independent spot-check of sequential event census.
    rng = np.random.default_rng(981)

    ix = rng.choice(len(e), size=120, replace=False)

    end = e.discharge_order_time.where(e.assigned, e.departure_time)

    for i in ix:
        t = e.loc[i, 'triage_time']
        ident = e.loc[i, 'encounter_id']
        expected = ((e.triage_time < t) | (e.triage_time == t) & (e.encounter_id <= ident)).sum() - (end <= t).sum()
        assert expected == e.loc[i, 'census']

    print('120 independent census checks passed', flush=True)

    # Reconstruct censuses with valid intervals only. Exclude own encounter consistently.
    valid_census = end >= e.triage_time

    valid_active = (e.discharge_order_time > e.md_assignment_time) & e.assigned

    valid_queue = e.md_assignment_time.where(e.assigned, e.departure_time) >= e.triage_time

    q = e.loc[valid_active]

    for target in ['triage_time', 'md_assignment_time']:
        t = d[target]
        d['active_' + target] = count(q.md_assignment_time, q.discharge_order_time, t) - (target == 'md_assignment_time')

    d['census_valid'] = count(e.loc[valid_census, 'triage_time'], end[valid_census], d.triage_time) - 1

    valid_physical = e.departure_time >= e.triage_time

    d['physical_valid'] = count(e.loc[valid_physical, 'triage_time'], e.loc[valid_physical, 'departure_time'], d.triage_time) - 1

    # Own physician workload using valid assigned intervals.
    d['own_valid'] = 0

    for p, ids in d.groupby('phys_id').groups.items():
        z = q[q.phys_id == p]
        d.loc[ids, 'own_valid'] = count(z.md_assignment_time, z.discharge_order_time, d.loc[ids, 'md_assignment_time']) - 1

    print('census corrected delta', (d.census_valid - (d.census - 1)).describe().to_dict(), flush=True)

    # Current-physician treatment interval may not cover pre-shift care; restrict to within-shift assignments.
    full = d.copy()

    fullS = S

    fullT = T

    mask = d.since_start >= 0

    d = d.loc[mask].copy().reset_index(drop=True)

    S = fullS[mask]

    T = fullT[mask]

    for name, fe in [('onshift_phys', ('phys_id', 'how', 'ym')), ('onshift_shift', ('shift_id',))]:
        X = np.c_[d.own_valid, S, T, d.active_md_assignment_time / 10, d.staff_assign]
        b, cov = fit(d, d.los, X, fe, name, False)
        pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(cov))}).to_csv(OUT / f'{name}.csv', index=False)

    d = full

    S = fullS

    T = fullT

    # Stage-specific exposure after controls. Three exposures separate, no claim of causal identification.
    for exposure in ['census_valid', 'physical_valid', 'active_triage', 'active_md_assignment_time']:
        X = np.c_[d[exposure] / 10, S, T, d.own_valid / 10, d.staff_assign]
        b, cov = fit(d, d.los, X, ('phys_id', 'how', 'ym'), exposure, False)
        pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(cov))}).to_csv(OUT / f'exposure_{exposure}.csv', index=False)

    # Standardize observed assigned LOS to all-arrival severity x hour composition within each census bin.
    e['sdec'] = (e.severity_score // 10).clip(upper=9).astype(int)

    d['sdec'] = (d.severity_score // 10).clip(upper=9).astype(int)

    keys = ['bin', 'hour', 'sdec']

    alln = e.groupby(keys, observed=True).size().rename('n_all')

    validn = d.groupby(keys, observed=True).los.agg(['size', 'mean'])

    z = validn.join(alln)

    z['weighted_sum'] = z['mean'] * z.n_all

    out = z.groupby('bin', observed=True).agg(weighted_sum=('weighted_sum', 'sum'), n_all=('n_all', 'sum'))

    out['standardized_los'] = out.weighted_sum / out.n_all

    out['observed_los'] = d.groupby('bin', observed=True).los.mean()

    out.to_csv(OUT / 'selection_standardization.csv')

    print('selection standardization', out.to_string(), flush=True)

    # Marginal severity contrast before and after arrival calendar controls (same clean outcomes sample).
    print('CENSUS SIGNED differences all and assigned severity', e.groupby('bin', observed=True).severity_score.mean().to_dict(), flush=True)

    # Handoff-window pattern is suggestive, not observable handoffs.
    print('AFTER_END', d.after_end.quantile([0.95, 0.99, 0.999, 1]).to_dict(), '180plus', int((d.after_end > 180).sum()), flush=True)

    print('SHIFTLENGTHS', ((pd.read_csv(ROOT / 'physician_shifts.csv', parse_dates=['shift_end_time', 'shift_begin_time']).shift_end_time - pd.read_csv(ROOT / 'physician_shifts.csv', parse_dates=['shift_end_time', 'shift_begin_time']).shift_begin_time).dt.total_seconds() / 60).describe().to_dict(), flush=True)


def arrival_and_assignment_patterns():
    """Look at timing and composition after calendar controls."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    # Does recorded timing reflect predictable clock patterns only, or realized crowding too?
    for y in ['remaining', 'severity_score']:
        if y == 'remaining':
            X = np.c_[B, S]
        else:
            X = np.c_[B, T, d.own_load]
        b, c = fit(d, d[y], X, ('phys_id', 'how', 'ym'), f'routing_{y}')
        pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(c))}).to_csv(OUT / f'routing_full_{y}.csv', index=False)

    # All arrivals: selection probability conditional on severity/calendar. LPM is descriptive.
    d = e.copy().reset_index(drop=True)

    d['how'] = d.hour * 7 + d.dow

    d['ym'] = d.triage_time.dt.strftime('%Y-%m')

    B = pd.get_dummies(d.bin, dtype=float).iloc[:, 1:].to_numpy()

    S = spl(d.severity_score, list(range(10, 100, 10)), 100)

    # All-arrival composition is distinct from composition in the observed LOS sample.
    fit(d, d.severity_score, B, ('how', 'ym'), 'severity_all_arrivals_calendar')

    fit(d, d.assigned.astype(float), np.c_[B, S], ('how', 'ym'), 'assignment_probability')

    # Overlap and case-mix balance: every physician represented in each central census group?
    print('PHYS SUPPORT', d[d.assigned].groupby('bin', observed=True).phys_id.nunique().to_string(), flush=True)

    # Direct physician-composition index using a common global physician mean (descriptive, unadjusted).
    m = d.groupby('phys_id').los.mean()

    d['phys_mean'] = d.phys_id.map(m)

    print('PHYS MIX', d.groupby('bin', observed=True).phys_mean.mean().to_string(), flush=True)


def within_shift_models():
    """Compare workload within and between shifts."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    d = d[d.since_start >= 0].copy().reset_index(drop=True)

    d['remaining30'] = (d.remaining // 30).astype(int)

    S = spl(d.severity_score, list(range(10, 100, 10)))

    T = spl(d.remaining, [60, 120, 180, 240, 360, 480, 540, 720, 900, 1080, 1440])

    # Interactions permit severity effects and department conditions to vary with shift phase.
    for name, X, fe in [
        ('within_shift_flexible',
         np.c_[d.own_load, S, S * d.remaining.to_numpy()[:, None] / 540,
               d.active_assign / 10, d.staff_assign],
         ('shift_id', 'remaining30')),
        ('selection_within_shift',
         np.c_[d.own_load, T, d.active_assign / 10, d.staff_assign],
         ('shift_id',)),
    ]:
        b, c = fit(d, d.severity_score if name.startswith('selection') else d.los, X, fe, name, False)
        pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(c))}).to_csv(OUT / f'{name}.csv', index=False)

    # Within/between-shift workload decomposition holding physician identity constant.
    d['shift_mean_load'] = d.groupby('shift_id').own_load.transform('mean')

    d['within_load'] = d.own_load - d.shift_mean_load

    X = np.c_[d.within_load, d.shift_mean_load, S, T, d.active_assign / 10, d.staff_assign]

    b, c = fit(d, d.los, X, ('phys_id', 'how', 'ym'), 'within_between', False)

    pd.DataFrame({'coef': b, 'se': np.sqrt(np.diag(c))}).to_csv(OUT / 'within_between.csv', index=False)


def summarize_workload():
    """Collect workload results and make the comparison figure."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    specs = [
        ('Physician effects, pooling shifts', 'workload.csv', 395280),
        ('Within the same shift', 'workload_shift.csv', 395280),
        ('On-shift only, physician effects', 'onshift_phys.csv', 310435),
        ('On-shift only, shift effects', 'onshift_shift.csv', 310435),
        ('On-shift, flexible phase/severity', 'within_shift_flexible.csv', 310435),
    ]

    rows = []

    for label, file, n in specs:
        z = pd.read_csv(OUT / file).iloc[0]
        rows.append({
            'comparison': label,
            'n': n,
            'coefficient': z['coef'],
            'se_date_cluster': z['se'],
            'lo': z['coef'] - 1.96 * z['se'],
            'hi': z['coef'] + 1.96 * z['se'],
        })

    r = pd.DataFrame(rows)

    r.to_csv(OUT / 'workload_comparison.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    y = np.arange(len(r))[::-1]

    for i, row in r.iterrows():
        ax.errorbar(row.coefficient, y[i], xerr=1.96 * row.se_date_cluster, fmt='o', color='black', capsize=4)
        ax.text(row.coefficient + 0.4, y[i], f'{row.coefficient:+.2f}', va='center', fontsize=10)

    ax.axvline(0, color='0.6', lw=0.8)

    ax.set(yticks=y, yticklabels=r.comparison, xlim=(-7, 5), xlabel='LOS minutes per additional concurrent patient (95% date-cluster interval)', title='Physician workload and LOS')

    fig.text(0.02, 0.02, 'Severity, shift-phase, active ED load and scheduled staffing controls. Physician models add calendar effects.\nShift models replace physician/calendar effects with shift effects. Flexible model adds 30-minute phase effects and severity interactions.\nDescriptive comparisons: routing, reverse causality and unmeasured patient differences remain possible.', fontsize=8)

    fig.tight_layout(rect=[0, 0.15, 1, 1])

    fig.savefig(OUT / 'workload_comparison.png', dpi=180)

    plt.close(fig)

    pd.DataFrame({'package': ['numpy', 'pandas', 'matplotlib', 'scipy', 'pyhdfe'], 'version': [meta.version(x) for x in ['numpy', 'pandas', 'matplotlib', 'scipy', 'pyhdfe']]}).to_csv(OUT / 'runtime_versions.csv', index=False)

    print(r.to_string(index=False))


def validate_results():
    """Check the reference figure, sample, joins, and timestamps."""
    e = all_encounters.copy()
    d = main_sample.copy()
    B, S, T, L = base_census, base_severity, base_timing, base_workload

    r = pd.read_csv(OUT / 'reference_reproduced.csv')

    reference = pd.read_csv(ROOT / 'reference/census_los_reference_values.csv')

    assert len(r) == len(reference) == 63

    assert np.allclose(r['mean'], reference.mean_los, rtol=0, atol=1e-10)

    assert np.array_equal(r['size'], reference.n_enc)

    assert e.encounter_id.is_unique

    assert e.assigned.sum() == 396070 and len(e) == 410937

    assert len(d) == 395280

    assert e.loc[e.assigned, ['shift_begin_time', 'shift_end_time']].notna().all().all()

    # Starting/end-of-file boundaries should not generate the bin-level pattern.
    start = e.triage_time.min().normalize() + pd.Timedelta(days=2)

    stop = e.triage_time.max().normalize() - pd.Timedelta(days=2)

    interior = d[d.triage_time.between(start, stop)]

    base = d.groupby('bin', observed=True).los.mean()

    inside = interior.groupby('bin', observed=True).los.mean()

    print('Max absolute bin mean change excluding boundary days:', float((base - inside).abs().max()))

    # CSV source counts and parsing round trips.
    raw = pd.read_csv(ROOT / 'encounters.csv')

    for col in ['triage_time', 'md_assignment_time', 'discharge_order_time', 'departure_time']:
        parsed = pd.to_datetime(raw[col], format='%Y-%m-%dT%H:%M:%S')
        assert parsed.notna().sum() == raw[col].notna().sum()
        assert parsed.dropna().dt.strftime('%Y-%m-%dT%H:%M:%S').equals(raw.loc[parsed.notna(), col])

    print('Reference, sample, join, unique ID, timestamp round-trip and boundary checks passed.')


if __name__ == '__main__':
    main_models()
    sensitivity_checks()
    arrival_and_assignment_patterns()
    within_shift_models()
    summarize_workload()
    validate_results()
