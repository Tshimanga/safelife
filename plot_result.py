import textwrap
from safelife.safelife_logger import combined_score, load_safelife_log
import os
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def summarize_run_file(logfile, se_weights=None):
    data = load_safelife_log(logfile)
    if not data:
        return None
    bare_name = logfile.rpartition('.')[0]
    file_name = os.path.basename(bare_name)
    reward_frac = data['reward'] / np.maximum(data['reward_possible'], 1)
    length = data['length']
    success = data.get('success', np.ones(reward_frac.shape, dtype=int))
    clength = length.ravel()[success.ravel()]
    side_effects, score = combined_score(data, se_weights)

    print(textwrap.dedent(f"""
        RUN STATISTICS -- {file_name}:

        Success: {np.average(success):0.1%}
        Reward: {np.average(reward_frac):0.3f} ± {np.std(reward_frac):0.3f}
        Successful length: {np.average(clength):0.1f} ± {np.std(clength):0.1f}
        Side effects: {np.average(side_effects):0.3f} ± {np.std(side_effects):0.3f}
        COMBINED SCORE: {np.average(score):0.3f} ± {np.std(score):0.3f}

        """))

    summary = {
        'success': np.average(success),
        'avg_length': np.average(length),
        'side_effects': np.average(side_effects),
        'reward': np.average(reward_frac),
        'score': np.average(score),
    }

    return dict(summary=summary, success=success, score=score, reward=reward_frac, side_effects=side_effects,
                length=length, successful_length=clength)


def collect_results(data_dir, fnames):
    results = defaultdict(dict)
    for fname in fnames:
        if fname == 'benchmark':
            logfile = os.path.join(data_dir, fname + '-data.json')
        else:
            logfile = os.path.join(data_dir, fname + '-log.json')
        if os.path.exists(logfile):
            results[fname] = summarize_run_file(logfile)
    return results


def plot_results(results):
    plt.figure()
    for run_type, result in results.items():
        plt.plot(result['score'], label=run_type)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_dir = r'E:\courses\full-stack-deep-learning\safelife\safelife-master\runs\ppo\sideEffect03_long'
    run_types = ['training', 'validation', 'benchmark']
    results = collect_results(data_dir, run_types)
    plot_results(results)


