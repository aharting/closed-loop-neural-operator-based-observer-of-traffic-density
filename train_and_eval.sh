#!/bin/bash
#python orchestration/train_openloop_model.py --config configs/openloop.yaml
#python orchestration/evaluation_openloop.py --config configs/openloop.yaml --max_unravel 10

#python orchestration/train_closedloop_model.py --config configs/closedloop.yaml
#python orchestration/evaluation_closedloop.py --config configs/closedloop.yaml

python orchestration/evaluation_closedloop_report.py --config configs/closedloop_report.yaml
