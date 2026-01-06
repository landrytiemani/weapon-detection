#!/usr/bin/env python3
"""
RQ3: Temporal Tracking (FULLY INTEGRATED)
==========================================
Tests ByteTrack integration using your SingleExperiment class.

Usage:
    python run_rq3_tracking.py --config config.yaml
    python run_rq3_tracking.py --config config.yaml --experiment frame_gap
"""

import os
import json
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, asdict
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main_perclass import SingleExperiment


@dataclass
class TrackingResult:
    config_name: str
    use_tracker: bool
    frame_gap: int
    mAP50: float
    precision: float
    recall: float
    f1: float
    tp_count: int
    fp_count: int
    fn_count: int
    fps: float
    latency_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ3TrackingExperiment:
    """
    RQ3: ByteTrack Temporal Tracking
    
    Tests:
    - H3.1: ByteTrack reduces FP
    - H3.2: Frame-gap 3-5 maintains ≥95% accuracy
    - H3.3: Track buffer optimization
    - H3.4: Temporal consistency
    """
    
    FRAME_GAPS = [1, 2, 3, 5, 10]
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq3_tracking'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, TrackingResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ3] GPU: {torch.cuda.get_device_name(0)}")
    
    def _save_config(self, config: dict, name: str) -> str:
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
        return str(temp_path)
    
    def _prepare_config(self, config: dict, use_tracker: bool, frame_gap: int) -> dict:
        """Prepare config with safe defaults to avoid runtime errors."""
        # Ensure stage_2 exists
        if 'stage_2' not in config:
            config['stage_2'] = {}
        
        # Set tracking parameters
        config['stage_2']['use_tracker'] = use_tracker
        
        # CRITICAL: Ensure frame_gap is at least 1 to avoid division by zero
        config['stage_2']['frame_gap'] = max(1, int(frame_gap))
        
        # Ensure evaluation section exists
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        # CRITICAL: Disable FLOPs computation to avoid thop hook conflicts
        config['evaluation']['compute_flops'] = False
        
        return config
    
    def _run_tracking_config(self, use_tracker: bool, frame_gap: int, name: str) -> TrackingResult:
        print(f"\n{'='*60}")
        print(f"[RQ3] Running: {name} (tracker={use_tracker}, gap={frame_gap})")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        
        # Prepare config with safe defaults
        config = self._prepare_config(config, use_tracker, frame_gap)
        
        temp_config_path = self._save_config(config, name)
        
        # Create output directory for this experiment
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=max(1, frame_gap),  # Safety check
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        return TrackingResult(
            config_name=name,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            mAP50=results.get('pipeline_map50', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            tp_count=results.get('pipeline_tp', 0),
            fp_count=results.get('pipeline_fp', 0),
            fn_count=results.get('pipeline_fn', 0),
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0)
        )
    
    def run_baseline_comparison(self) -> Dict[str, TrackingResult]:
        """Compare tracking vs no tracking."""
        print("\n[RQ3/H3.1] Tracking vs No Tracking...")
        
        result = self._run_tracking_config(False, 1, 'no_tracking')
        self.results['no_tracking'] = result
        torch.cuda.empty_cache()
        
        result = self._run_tracking_config(True, 1, 'with_tracking')
        self.results['with_tracking'] = result
        
        self._save_results()
        return self.results
    
    def run_frame_gap_sweep(self) -> Dict[str, TrackingResult]:
        """Sweep frame gaps (H3.2)."""
        print("\n[RQ3/H3.2] Frame Gap Sweep...")
        
        for gap in self.FRAME_GAPS:
            name = f"gap_{gap}"
            result = self._run_tracking_config(True, gap, name)
            self.results[name] = result
            self._save_results()
            torch.cuda.empty_cache()
        
        return self.results
    
    def run_full_experiment(self):
        """Run all tracking experiments."""
        self.run_baseline_comparison()
        self.run_frame_gap_sweep()
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        validation = {}
        
        no_track = self.results.get('no_tracking')
        with_track = self.results.get('with_tracking')
        
        # H3.1: Tracking reduces FP
        if no_track and with_track:
            if no_track.fp_count > 0:
                fp_reduction = ((no_track.fp_count - with_track.fp_count) / no_track.fp_count) * 100
            else:
                fp_reduction = 0
            
            validation['H3.1'] = {
                'description': 'ByteTrack reduces false positives',
                'no_tracking_fp': no_track.fp_count,
                'with_tracking_fp': with_track.fp_count,
                'reduction_percent': fp_reduction,
                'no_tracking_mAP50': no_track.mAP50,
                'with_tracking_mAP50': with_track.mAP50,
                'hypothesis_supported': fp_reduction > 0
            }
        
        # H3.2: Frame gap efficiency
        gap1 = self.results.get('gap_1')
        gap_results = {k: v for k, v in self.results.items() if k.startswith('gap_')}
        
        if gap1 and len(gap_results) > 1:
            best_efficiency = None
            for name, result in gap_results.items():
                gap = int(name.split('_')[-1])
                if 3 <= gap <= 5 and gap1.mAP50 > 0:
                    accuracy_retention = result.mAP50 / gap1.mAP50
                    if accuracy_retention >= 0.95:
                        best_efficiency = gap
            
            validation['H3.2'] = {
                'description': 'Frame gap 3-5 maintains ≥95% accuracy',
                'optimal_gap': best_efficiency,
                'gap_results': {k: {'mAP50': v.mAP50, 'fps': v.fps} for k, v in gap_results.items()},
                'hypothesis_supported': best_efficiency is not None
            }
        
        return validation
    
    def _save_results(self):
        output_path = self.output_dir / 'tracking_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ3] Results saved: {output_path}")
    
    def generate_latex_table(self):
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ3: Tracking Comparison}
\label{tab:rq3_tracking}
\begin{tabular}{lcccccc}
\toprule
Configuration & mAP50 & Precision & Recall & F1 & FPS & Latency (ms) \\
\midrule
"""
        for name, result in self.results.items():
            display_name = name.replace('_', ' ').title()
            latex += f"{display_name} & {result.mAP50:.3f} & {result.precision:.3f} & "
            latex += f"{result.recall:.3f} & {result.f1:.3f} & "
            latex += f"{result.fps:.1f} & {result.latency_ms:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq3_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ3] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        gap_results = {k: v for k, v in self.results.items() if k.startswith('gap_')}
        if len(gap_results) > 1:
            gaps = []
            mAPs = []
            fpss = []
            
            for name in sorted(gap_results.keys(), key=lambda x: int(x.split('_')[-1])):
                gap = int(name.split('_')[-1])
                gaps.append(gap)
                mAPs.append(gap_results[name].mAP50)
                fpss.append(gap_results[name].fps)
            
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.set_xlabel('Frame Gap', fontsize=12)
            ax1.set_ylabel('mAP50', color='blue', fontsize=12)
            ax1.plot(gaps, mAPs, 'b-o', linewidth=2, markersize=8)
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('FPS', color='red', fontsize=12)
            ax2.plot(gaps, fpss, 'r--s', linewidth=2, markersize=8)
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title('H3.2: Frame Gap Trade-off', fontsize=14)
            fig.tight_layout()
            plt.savefig(figures_dir / 'frame_gap_tradeoff.png', dpi=300)
            plt.close()
            
            print(f"[RQ3] Figure saved: {figures_dir / 'frame_gap_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ3: Tracking Experiment')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='Results/rq3_tracking')
    parser.add_argument('--experiment', type=str, default='full',
                        choices=['full', 'baseline', 'frame_gap'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ3: TRACKING EXPERIMENT (INTEGRATED)")
    print("=" * 70)
    
    experiment = RQ3TrackingExperiment(config_path=args.config, output_dir=args.output)
    
    if args.experiment == 'full':
        experiment.run_full_experiment()
    elif args.experiment == 'baseline':
        experiment.run_baseline_comparison()
    elif args.experiment == 'frame_gap':
        experiment.run_frame_gap_sweep()
    
    validation = experiment.validate_hypotheses()
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    for h_id, h_result in validation.items():
        status = "✓ SUPPORTED" if h_result.get('hypothesis_supported') else "✗ NOT SUPPORTED"
        print(f"\n{h_id}: {status}")
        print(f"  {h_result.get('description', '')}")
    
    experiment.generate_latex_table()
    experiment.generate_figures()
    
    validation_path = Path(args.output) / 'hypothesis_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()