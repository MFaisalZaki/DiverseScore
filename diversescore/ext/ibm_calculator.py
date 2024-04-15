import pkg_resources
import tempfile
import subprocess
import sys
import os

from diversescore.utilities import dumpPlans

def IBMCalculator(domain, problem, planset, dumpplansetsize, selectplansetsize, metricslist, json_dumpfile, **kwargs):
    downward = pkg_resources.resource_filename(__name__, os.path.join("..", "..", "ibm-diversescore", "fast-downward.py"))
    computed_scores = {}
    with tempfile.TemporaryDirectory(**kwargs) as tempdir:
        # First dump the plans in the directory
        plansdumpdir = os.path.join(tempdir, "plans")
        os.makedirs(plansdumpdir, exist_ok=True)
        dumpPlans(planset, dumpplansetsize, plansdumpdir)

        # Now for every metric in the metric list, we need to have its own dir.
        for metric in metricslist:
            metricdir = os.path.join(tempdir, metric)
            os.makedirs(metricdir, exist_ok=True)
            
            all_metrics = None
            if '-' in metric:
                split_metrics = []
                for m in metric.split("-"):
                    split_metrics.append(f'compute_{m}_metric=true')
                all_metrics = ','.join(split_metrics)
            else:
                all_metrics = f'compute_{metric.lower()}_metric=true'

            dumpjsonplans = '' if json_dumpfile is None else f',dump_plans=true,json_file_to_dump={json_dumpfile}'

            score = f"subset({all_metrics},aggregator_metric=avg,plans_as_multisets=false,plans_subset_size={selectplansetsize},exact_method=false,similarity=false,reduce_labels=false{dumpjsonplans})"
            result = subprocess.check_output([sys.executable, 
                                downward, 
                                domain, 
                                problem,
                                "--diversity-score", score, 
                                "--internal-plan-files-path", plansdumpdir, 
                                "--internal-num-plans-to-read", str(dumpplansetsize if dumpplansetsize == selectplansetsize else len(planset)),
                                ], universal_newlines=True, cwd=metricdir)
            
            maxsum_score = -1
            for line in result.split("\n"):
                if "Score after clustering" in line:
                    _tmpline = line
                    _tmpline = _tmpline.replace("Score after clustering ", "").replace(",", "")
                    _tmpline = _tmpline.split(" ")
                    maxsum_score = float(_tmpline[0])
                    break
            computed_scores[metric] = maxsum_score
    return computed_scores