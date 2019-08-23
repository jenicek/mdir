#!/usr/bin/env python3
"""
Evaluate trained models by following a provided .yaml scenario

Needs yaml scenario with structure:

network:
  path: <network dir/file path, string>
  runtime: <runtime override, such as wrappers, dict>
validation: <validation section, dict>
data:
  test: <data parameters, dict>
"""

import os.path
import sys
import yaml

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import mdir
from daan.core.experiments import dict_deep_overlay

# Download necessary datasets

from cirtorch.utils.download import download_test
from cirtorch.utils.general import get_data_root

download_test(get_data_root())

# Parse arguments

scenarios = sys.argv[1:]
if len(scenarios) == 1 and not scenarios[0].endswith(".yml"):
    scenarios = ["eval.yml", "eval_%s.yml" % scenarios[0]]

# Parse input scenarios

scenario = {}
for params in scenarios:
    with open(params, 'r') as handle:
        scenario = dict_deep_overlay(scenario, yaml.safe_load(handle))
if not scenario:
    sys.stderr.write("Scenario needs to be specified\n")
    sys.exit(1)

# Execute

metadata, = mdir.stages.validate.validate(scenario, ())

# Pretty-print

scores = {
    "roxford5k/validation/score:ap_medium_avg.4": "roxford.5k medium",
    "rparis6k/validation/score:ap_medium_avg.4": "rparis.6k medium",
    "247tokyo1k/validation/score:ap_avg.4": "247tokyo.1k",
}
for heading, section in metadata.items():
    print("\n%s\n" % heading.capitalize())
    for key, value in section.items():
        if key in scores:
            print("    %-20s %s" % (scores[key], round(100*value, 2)))
    print()
