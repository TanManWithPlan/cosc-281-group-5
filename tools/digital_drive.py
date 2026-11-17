"""Deterministic image-in-the-loop simulator; no ROS, robot, or paper required.

Flat ground, ideal velocity tracking, simplified camera geometry. Exercises real
perception, estimation and control. Not a model of tire slip or actuator dynamics.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot/src/team_navigation'))
from team_navigation.perception import Perception
from team_navigation.estimation import RoadEstimator
from team_navigation.control import RoadController

CASES = {
    'straight_from_left': dict(y=.10),
    'straight_from_right': dict(y=-.10),
    'gentle_s_curve': dict(amplitude=.08),
    'noisy_dim_curve': dict(amplitude=.08, noisy=True),
    'camera_freeze': dict(freeze=True),
    'tape_missing': dict(missing=True),
    'blue_boundary': dict(blue=True),
}


def run_case(name, seconds=80.):
    cfg = CASES[name]
    rng = np.random.default_rng(281)
    perception, estimator, controller = Perception(), RoadEstimator(), RoadController()
    dt, x, y, heading = .05, 0., cfg.get('y', 0.), 0.
    rows = []
    stamp, observation = None, None
    def road(s):
        return cfg.get('amplitude', 0.) * math.sin(2 * math.pi * s / 2.)
    for step in range(round(seconds/dt)):
        now = 1 + step * dt
        hazard = 10 <= now < 11
        freeze = cfg.get('freeze', False) and hazard
        if not freeze:
            frame = np.zeros((240,320,3), np.uint8)
            if not (cfg.get('missing', False) and hazard):
                points = []
                for row in range(100,240,3):
                    lookahead = .12 + .8*(1-row/240)
                    lateral = y + lookahead*math.sin(heading) - road(x+lookahead*math.cos(heading))
                    pixel = round(160 + 160*lateral/.35)
                    points.append((pixel,row))
                cv2.polylines(frame, [np.array(points,np.int32)], False, (0,255,255), 12)
            if cfg.get('blue', False) and hazard:
                frame[160:,:60] = (255,0,0)
            if cfg.get('noisy', False):
                noise = rng.normal(0,5,frame.shape)
                frame = np.clip(frame*.6+noise,0,255).astype(np.uint8)
            observation = perception.process(frame)
            stamp = now
        estimate = estimator.update(observation.error, observation.found, stamp, now)
        v, w = controller.compute(estimate, observation.found, observation.blue_warning)
        rows.append(dict(time_s=now, x_m=x, lateral_error_m=y-road(x),
                         heading_rad=heading, image_age_s=now-stamp,
                         found=int(observation.found), valid=int(estimate.valid),
                         forward_m_s=v, yaw_rad_s=w, hazard=int(hazard)))
        x += v * math.cos(heading)*dt
        y += v * math.sin(heading)*dt
        heading += w*dt
    errors = [abs(r['lateral_error_m']) for r in rows]
    unsafe = [r for r in rows if (r['image_age_s'] > .25 or
              not r['found'] or (cfg.get('blue') and r['hazard'])) and
              (r['forward_m_s'] != 0 or r['yaw_rad_s'] != 0)]
    metrics = dict(case=name, samples=len(rows), distance_m=x,
                   initial_error_m=errors[0], final_error_m=errors[-1],
                   max_error_m=max(errors), rms_error_m=math.sqrt(sum(e*e for e in errors)/len(errors)),
                   stopped_samples=sum(r['forward_m_s']==0 for r in rows),
                   unsafe_motion_samples=len(unsafe))
    assert not unsafe, metrics
    assert errors[-1] < .04, metrics
    assert max(errors) < .16, metrics
    return metrics, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'simulation-results')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in CASES:
        metrics, rows=run_case(name)
        results.append(metrics)
        with (args.output/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
