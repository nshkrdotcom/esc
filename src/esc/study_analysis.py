"""World-cluster analysis and figures; never turn soft budgets into equal cost."""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from esc.study_audit import audit_study


def interval(values):
    return [float(v) for v in np.quantile(values, [.025,.975])]


def cluster_interval(rates, draws):
    """A degenerate empirical bootstrap is not evidence of zero uncertainty."""
    if len(rates) < 2 or np.ptp(rates) == 0:
        return None
    return interval(np.asarray(rates)[draws].mean(axis=1))


def decay(depths, successes, totals):
    """Constrained binomial log-link fit; no clipped OLS or saturated slope claim."""
    d, c, n = map(lambda x: np.asarray(x, dtype=float), (depths,successes,totals))
    if len(set(d)) < 2 or sum(c) in (0, sum(n)):
        return None
    def objective(theta):
        logp = theta[0]-theta[1]*d
        if np.max(logp) >= 0:
            return 1e100
        return -float(np.sum(c*logp+(n-c)*np.log(-np.expm1(logp))))
    fit = minimize(objective, [math.log(sum(c)/sum(n)),0.0], method='SLSQP',
        constraints=[{'type':'ineq','fun':lambda x: -1e-9-(x[0]-x[1]*d)}],
        options={'maxiter':200, 'ftol':1e-8})
    if not fit.success or abs(fit.x[1]) > 2 or np.max(fit.x[0]-fit.x[1]*d) > -1e-7:
        return None  # boundary/separation: report unavailable, not an artificial finite estimate
    return float(fit.x[1])


def reliability(rows, bootstrap_samples, seed):
    worlds = sorted({r['world_id'] for r in rows})
    if not worlds:
        raise ValueError('No worlds')
    groups = defaultdict(list)
    for r in rows:
        groups[r['variant'], r['episode']['budget'], r['depth']].append(r)
    rng = np.random.default_rng(seed)
    # Same bootstrap world indices for every condition/depth/band: pairing retained.
    draws = rng.integers(0,len(worlds),size=(bootstrap_samples,len(worlds)))
    out = []
    for (variant,budget,depth), group in sorted(groups.items()):
        rates = [np.mean([r['correct'] for r in group if r['world_id']==w]) for w in worlds]
        costs = [np.mean([r['ledger']['measured_tokens'] for r in group if r['world_id']==w]) for w in worlds]
        coverage = [np.mean([not r['abstained'] for r in group if r['world_id']==w]) for w in worlds]
        out.append(dict(variant=variant,budget=budget,depth=depth,n=len(group),
            correct=sum(r['correct'] for r in group), accuracy=float(np.mean(rates)),
            interval=cluster_interval(rates, draws),
            tokens=float(np.mean([r['ledger']['measured_tokens'] for r in group])),
            token_interval=cluster_interval(costs,draws),
            coverage=float(np.mean([not r['abstained'] for r in group])),
            coverage_interval=cluster_interval(coverage,draws),
            exhausted=sum(r['ledger']['exhausted'] for r in group),
            dispatched_calls=float(np.mean([r['ledger']['calls_dispatched'] for r in group])),
            invocations=float(np.mean([sum(len(x.get('inputs',[])) for x in r['rollouts']) for r in group])),
            protocol_violations=sum(bool(x['protocol_violations']) for r in group for x in r['rollouts']),
            model_output_errors=sum(bool(x.get('model_output_error')) for r in group for x in r['rollouts']),
            false_promotions=sum(e['accepted'] and e['value'] != r['truth'][e['step_id']]
                for r in group for x in r['rollouts'] for e in x['events'])
                if variant in {'isolated_verified','shared_verified'} else None,
            accepted_transitions=sum(e['accepted'] for r in group for x in r['rollouts'] for e in x['events']),
            public_row_reads=sum(len(x.get('reads',[])) for r in group for x in r['rollouts'])))
    repeats = []
    for variant,budget in sorted({(r['variant'],r['episode']['budget']) for r in rows}):
        groups5 = defaultdict(list)
        for r in rows:
            if r['variant']==variant and r['episode']['budget']==budget:
                groups5[r['world_id'],r['depth']].append(r['correct'])
        if any(len(v)<5 for v in groups5.values()):
            repeats.append(dict(variant=variant,budget=budget,pass_at_5=None,pass_all_5=None))
            continue
        any5,all5 = defaultdict(list),defaultdict(list)
        for (world, depth), values in groups5.items():
            n,c=len(values),sum(values)
            any5[world].append(1-math.comb(n-c,5)/math.comb(n,5) if n-c>=5 else 1.)
            all5[world].append(math.comb(c,5)/math.comb(n,5) if c>=5 else 0.)
        any_rates=[float(np.mean(any5[w])) for w in worlds]
        all_rates=[float(np.mean(all5[w])) for w in worlds]
        repeats.append(dict(variant=variant,budget=budget,pass_at_5=float(np.mean(any_rates)),
            pass_all_5=float(np.mean(all_rates)), pass_at_5_interval=cluster_interval(any_rates,draws),
            pass_all_5_interval=cluster_interval(all_rates,draws)))
    slopes = []
    for variant,budget in sorted({(r['variant'],r['episode']['budget']) for r in rows}):
        depths=sorted({r['depth'] for r in rows})
        counts=np.array([[sum(r['correct'] for r in rows if r['variant']==variant and
            r['episode']['budget']==budget and r['depth']==d and r['world_id']==w) for d in depths] for w in worlds])
        totals=np.array([[sum(1 for r in rows if r['variant']==variant and
            r['episode']['budget']==budget and r['depth']==d and r['world_id']==w) for d in depths] for w in worlds])
        beta=decay(depths,counts.sum(axis=0),totals.sum(axis=0))
        samples=[decay(depths,counts[draw].sum(axis=0),totals[draw].sum(axis=0)) for draw in draws] if beta is not None and len(worlds)>1 else []
        good=[v for v in samples if v is not None]
        slopes.append(dict(variant=variant,budget=budget,beta=beta,
            interval=interval(good) if len(good)==bootstrap_samples else None,
            unidentified_bootstrap_samples=len(samples)-len(good), samples=samples))
    # Paired bootstrap contrasts, only when all replicates are identifiable.
    contrasts=[]
    for budget in sorted({r['episode']['budget'] for r in rows}):
        reference=next((s for s in slopes if s['budget']==budget and s['variant']=='isolated_verified'),None)
        for comparator in ('continuous','continuous_emit','search_emit'):
            other=next((s for s in slopes if s['budget']==budget and s['variant']==comparator),None)
            if not reference or not other: continue
            samples=list(zip(reference['samples'],other['samples']))
            good=[a-b for a,b in samples if a is not None and b is not None]
            contrasts.append(dict(budget=budget,contrast=f'isolated_verified minus {comparator}',
                difference=reference['beta']-other['beta'] if reference['beta'] is not None and other['beta'] is not None else None,
                interval=interval(good) if len(good)==bootstrap_samples else None))
    for s in slopes: s.pop('samples')
    return out,repeats,slopes,contrasts


def propagation(rows):
    groups=defaultdict(lambda:dict(exposures=0,wrong=0,correct=0,abstained=0,unreached=0,
                                  candidate_wrong=0,protocol_violations=0))
    interventions=[]
    for r in rows:
        if r['site'] is None: continue
        for index,rollout in enumerate(r['rollouts']):
            site=r['site']
            events=rollout['events']
            event=next((e for e in events if e['step_id']==f'N{site}'),None)
            applied=bool(event and event['applied'])
            wrong=bool(applied and event['value'] != r['truth'][f'N{site}'])
            interventions.append(dict(episode_id=r['episode']['episode_id'],rollout=index,
                variant=r['variant'],budget=r['episode']['budget'],reached=event is not None,
                selected_rollout=index==r.get('selected_rollout'), ensemble_correct=r['correct'] if 'correct' in r else None,
                applied=applied,wrong_after_intervention=wrong,
                protocol_violations=len(rollout['protocol_violations'])))
            if not wrong: continue
            for distance in range(1,r['depth']-site+1):
                group=groups[r['variant'],r['episode']['budget'],distance]
                group['exposures']+=1
                group['protocol_violations']+=bool(rollout['protocol_violations'])
                descendant=next((e for e in events if e['step_id']==f'N{site+distance}'),None)
                if descendant is None:
                    group['unreached']+=1
                else:
                    is_wrong=descendant['value'] != r['truth'][descendant['step_id']]
                    group['candidate_wrong']+=descendant['value'] is not None and is_wrong
                    group['abstained' if not descendant['accepted'] else 'wrong' if is_wrong else 'correct']+=1
    return [dict(variant=v,budget=b,distance=k,**c,
                 epc=c['wrong']/c['exposures'] if not c['protocol_violations'] else None)
            for (v,b,k),c in sorted(groups.items())],interventions


def propagation_intervals(rows, points, samples, seed):
    worlds=sorted({r['world_id'] for r in rows})
    if len(worlds)<2:
        for point in points: point['interval']=None
        return
    rng=np.random.default_rng(seed)
    per_world={w:{(p['variant'],p['budget'],p['distance']):p for p in
                 propagation([r for r in rows if r['world_id']==w])[0]} for w in worlds}
    for point in points:
        key=(point['variant'],point['budget'],point['distance'])
        boot=[]
        for draw in rng.integers(0,len(worlds),size=(samples,len(worlds))):
            counts=[per_world[worlds[i]].get(key,{}) for i in draw]
            exposures=sum(c.get('exposures',0) for c in counts)
            violations=sum(c.get('protocol_violations',0) for c in counts)
            if exposures and not violations: boot.append(sum(c.get('wrong',0) for c in counts)/exposures)
        point['interval']=interval(boot) if len(boot)==samples and np.ptp(boot)>0 else None


def analyze_study(directory):
    audit,plan,rows=audit_study(directory)
    output=Path(directory)/'analysis'
    if output.exists(): raise FileExistsError('Analysis already exists; preserve prior artifacts')
    clean=[r for r in rows if r['site'] is None]
    config=plan['configuration']
    rates,repeats,slopes,contrasts=reliability(clean,config['bootstrap_samples'],config['schedule_seed'])
    epc,interventions=propagation(rows)
    propagation_intervals(rows,epc,config['bootstrap_samples'],config['schedule_seed'])
    report=dict(audit=audit,scope='Exploratory quality versus measured cost; H1 not tested at equal compute',
        split=config['split'],worlds=config['worlds'],repetitions=config['repetitions'],
        warning='Few world clusters and saturated cells can make intervals unreliable or unidentifiable.',
        reliability=rates,repeatability=repeats,decay=slopes,decay_contrasts=contrasts,
        propagation=epc,interventions=interventions,
        intervention_mode=config.get('intervention_mode','candidate'))
    from esc.study import source_identity
    report['analysis_source_identity']=source_identity()
    output.mkdir()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pairs=[]
    for changed in rows:
        if changed['site'] is None or not any(e['applied'] for r in changed['rollouts'] for e in r['events']): continue
        baseline=next((r for r in clean if all(r['episode'][k]==changed['episode'][k]
            for k in ('task_id','variant','budget','repetition'))),None)
        if baseline:
            pairs.append(dict(clean=baseline,intervened=changed))
            break
    (output/'paired_trace.json').write_text(json.dumps(pairs,indent=2))
    figures(output,report)
    return report


def figures(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budgets=sorted({r['budget'] for r in report['reliability']})
    for number in range(1,5):
        fig,axes=plt.subplots(1,len(budgets),figsize=(7*len(budgets),5),squeeze=False)
        for ax,budget in zip(axes[0],budgets):
            ax.set_title(f'Soft allowance {budget:,}; actual cost differs')
            if number==1:
                for variant in ('continuous','continuous_emit','search_emit','isolated_verified'):
                    points=[r for r in report['reliability'] if r['budget']==budget and r['variant']==variant]
                    if not points: continue
                    ax.plot([r['depth'] for r in points],[r['accuracy'] for r in points],'-o',label=variant)
                    if all(r['interval'] for r in points):
                        ax.fill_between([r['depth'] for r in points],[r['interval'][0] for r in points],
                                        [r['interval'][1] for r in points],alpha=.12)
                ax.set(xlabel='Relational dependency depth',ylabel='Final success',ylim=(-.05,1.05))
            elif number==2:
                for variant in sorted({r['variant'] for r in report['propagation']}):
                    points=[r for r in report['propagation'] if r['budget']==budget and r['variant']==variant]
                    ax.plot([r['distance'] for r in points],[r['epc'] if r['epc'] is not None else np.nan for r in points],'-o',label=variant)
                    if points and all(r.get('interval') for r in points):
                        ax.fill_between([r['distance'] for r in points],[r['interval'][0] for r in points],
                                        [r['interval'][1] for r in points],alpha=.12)
                ax.set(xlabel='Downstream distance',ylabel='Wrong accepted emission / wrong injected exposures',ylim=(-.05,1.05))
                ax.text(.02,.02,'Per rollout; inspect unreached and protocol violations in report',transform=ax.transAxes,fontsize=8)
            elif number==3:
                points=[r for r in report['repeatability'] if r['budget']==budget and r['pass_at_5'] is not None]
                if points:
                    x=np.arange(len(points))
                    ax.bar(x-.2,[r['pass_at_5'] for r in points],.4,label='pass@5')
                    ax.bar(x+.2,[r['pass_all_5'] for r in points],.4,label='pass^5')
                    ax.set_xticks(x,[r['variant'] for r in points],rotation=35,ha='right')
                else: ax.text(.1,.5,'Unavailable: fewer than five repeats',transform=ax.transAxes)
                ax.set_ylim(0,1.05)
            else:
                for variant in ('continuous_emit','isolated_raw','isolated_typed','shared_verified','isolated_verified'):
                    points=[r for r in report['reliability'] if r['budget']==budget and r['variant']==variant]
                    if not points: continue
                    ax.plot([r['depth'] for r in points],[r['accuracy'] for r in points],'o-',label=variant)
                ax.set(xlabel='Relational dependency depth',ylabel='Ablation final success',ylim=(-.05,1.05))
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        fig.suptitle(f"{report['split']} exploratory study — no equal-compute H1 claim")
        fig.tight_layout()
        fig.savefig(output/f'figure_{number}.png',dpi=150)
        fig.savefig(output/f'figure_{number}.svg')
        plt.close(fig)
    # Compare cost at FIXED depth. Connecting different depths would confound
    # quality/cost with task horizon, rather than show the prespecified budget bands.
    depths=sorted({r['depth'] for r in report['reliability']})
    fig,axes=plt.subplots(1,len(depths),figsize=(6*len(depths),5),squeeze=False)
    for ax,depth in zip(axes[0],depths):
        for variant in sorted({r['variant'] for r in report['reliability']}):
            points=sorted((r for r in report['reliability'] if r['depth']==depth and r['variant']==variant),key=lambda r:r['budget'])
            ax.plot([r['tokens'] for r in points],[r['accuracy'] for r in points],'o-',label=variant)
        ax.set(title=f'Depth {depth}; points are allowance bands',xlabel='Measured mean tokens',
               ylabel='Final success',ylim=(-.05,1.05))
        ax.legend(fontsize=7)
    fig.suptitle('Exploratory quality versus measured cost at fixed depth')
    fig.tight_layout()
    for extension in ('png','svg'):fig.savefig(output/f'cost_frontier.{extension}',dpi=150)
    plt.close(fig)
