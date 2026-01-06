#!/usr/bin/env python3
"""
RQ1: Ablation Experiment Runner (FULLY INTEGRATED)
===================================================
Uses your actual SingleExperiment class from main_perclass.py

Usage:
    python run_rq1_ablation.py --config config.yaml
    python run_rq1_ablation.py --config config.yaml --experiment crop_scale
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import your actual pipeline
from main_perclass import SingleExperiment


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    config_name: str
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    tp_count: int
    fp_count: int
    fn_count: int
    gflops: float
    fps: float
    latency_ms: float
    handgun_mAP50: float = 0.0
    knife_mAP50: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ1AblationExperiment:
    """
    RQ1: Ablation Study for Modular Architecture
    
    Tests:
    - H1.1: Person-centric cropping improves mAP50 by ≥10%
    - H1.2: Hierarchical approach reduces FP by ≥20%
    - H1.3: Optimal crop_scale between 1.2-2.0
    """
    
    CROP_SCALES = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq1_ablation'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, ExperimentResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ1] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[RQ1] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _save_config(self, config: dict, name: str) -> str:
        """Save modified config to temp file."""
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(temp_path)
    
    def _prepare_config(self, config: dict) -> dict:
        """Prepare config with safe defaults to avoid runtime errors."""
        # Ensure stage_2 exists
        if 'stage_2' not in config:
            config['stage_2'] = {}
        
        # CRITICAL: Ensure frame_gap is at least 1 to avoid division by zero
        frame_gap = config['stage_2'].get('frame_gap', 1)
        config['stage_2']['frame_gap'] = max(1, int(frame_gap))
        
        # Ensure evaluation section exists
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        # CRITICAL: Disable FLOPs computation to avoid thop hook conflicts
        config['evaluation']['compute_flops'] = False
        
        return config
    
    def _run_single_experiment(self, config: dict, name: str) -> ExperimentResult:
        """Run single experiment using YOUR SingleExperiment class."""
        print(f"\n{'='*60}")
        print(f"[RQ1] Running: {name}")
        print(f"{'='*60}")
        
        # Prepare config with safe defaults
        config = self._prepare_config(config)
        
        # Save modified config
        temp_config_path = self._save_config(config, name)
        
        # Get tracker settings from config
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        
        # Create output directory for this experiment
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Create experiment using YOUR class signature
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        # Run and get results
        results = exp.run()
        
        # Extract per-class metrics
        per_class_map = results.get('pipeline_map50_by_class', {})
        handgun_map = per_class_map.get('handgun', 0.0)
        knife_map = per_class_map.get('knife', 0.0)
        
        return ExperimentResult(
            config_name=name,
            mAP50=results.get('pipeline_map50', 0.0),
            mAP50_95=results.get('pipeline_map50', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            tp_count=results.get('pipeline_tp', 0),
            fp_count=results.get('pipeline_fp', 0),
            fn_count=results.get('pipeline_fn', 0),
            gflops=results.get('gflops', 0.0),
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0),
            handgun_mAP50=handgun_map,
            knife_mAP50=knife_map
        )
    
    def run_baseline(self) -> ExperimentResult:
        """Run baseline (full pipeline with default settings)."""
        print("\n[RQ1] Running BASELINE experiment...")
        
        config = deepcopy(self.base_config)
        result = self._run_single_experiment(config, 'baseline')
        self.results['baseline'] = result
        
        self._save_results()
        return result
    
    def run_no_person_crop(self) -> ExperimentResult:
        """Run WITHOUT person-centric cropping (H1.1)."""
        print("\n[RQ1/H1.1] Running NO PERSON CROP experiment...")
        
        config = deepcopy(self.base_config)
        config['stage_2'] = config.get('stage_2', {})
        config['stage_2']['crop_scale'] = 100.0  # Effectively full frame
        
        result = self._run_single_experiment(config, 'no_person_crop')
        self.results['no_person_crop'] = result
        
        self._save_results()
        return result
    
    def run_crop_scale_sweep(self) -> Dict[str, ExperimentResult]:
        """Sweep crop scales for H1.3."""
        print("\n[RQ1/H1.3] Running CROP SCALE SWEEP...")
        
        results = {}
        
        for scale in self.CROP_SCALES:
            name = f"crop_scale_{scale}"
            print(f"\n[H1.3] Testing crop_scale = {scale}")
            
            config = deepcopy(self.base_config)
            config['stage_2'] = config.get('stage_2', {})
            config['stage_2']['crop_scale'] = scale
            
            result = self._run_single_experiment(config, name)
            results[name] = result
            self.results[name] = result
            
            self._save_results()
            torch.cuda.empty_cache()
        
        return results
    
    def run_no_nms(self) -> ExperimentResult:
        """Run without NMS to test H1.2."""
        print("\n[RQ1/H1.2] Running NO NMS experiment...")
        
        config = deepcopy(self.base_config)
        config['stage_3'] = config.get('stage_3', {})
        config['stage_3']['nms_iou_threshold'] = 1.0
        config['stage_3']['global_nms_threshold'] = 1.0
        
        result = self._run_single_experiment(config, 'no_nms')
        self.results['no_nms'] = result
        
        self._save_results()
        return result
    
    def run_full_ablation(self) -> Dict[str, ExperimentResult]:
        """Run complete ablation study."""
        print("\n" + "=" * 70)
        print("RQ1: FULL ABLATION STUDY")
        print("=" * 70)
        
        self.run_baseline()
        torch.cuda.empty_cache()
        
        self.run_no_person_crop()
        torch.cuda.empty_cache()
        
        self.run_no_nms()
        torch.cuda.empty_cache()
        
        self.run_crop_scale_sweep()
        
        self._save_results()
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        """Validate H1.1, H1.2, H1.3."""
        validation = {}
        
        baseline = self.results.get('baseline')
        no_crop = self.results.get('no_person_crop')
        no_nms = self.results.get('no_nms')
        
        if baseline and no_crop:
            if no_crop.mAP50 > 0:
                improvement = ((baseline.mAP50 - no_crop.mAP50) / no_crop.mAP50) * 100
            else:
                improvement = 100.0 if baseline.mAP50 > 0 else 0.0
            
            validation['H1.1'] = {
                'description': 'Person-centric cropping improves mAP50 by ≥10%',
                'baseline_mAP50': baseline.mAP50,
                'no_crop_mAP50': no_crop.mAP50,
                'improvement_percent': improvement,
                'threshold': 10.0,
                'hypothesis_supported': improvement >= 10.0
            }
        
        if baseline and no_nms:
            if no_nms.fp_count > 0:
                fp_reduction = ((no_nms.fp_count - baseline.fp_count) / no_nms.fp_count) * 100
            else:
                fp_reduction = 0.0
            
            validation['H1.2'] = {
                'description': 'NMS reduces FP by ≥20%',
                'baseline_fp': baseline.fp_count,
                'no_nms_fp': no_nms.fp_count,
                'reduction_percent': fp_reduction,
                'threshold': 20.0,
                'hypothesis_supported': fp_reduction >= 20.0
            }
        
        scale_results = {k: v for k, v in self.results.items() if k.startswith('crop_scale')}
        if scale_results:
            best_name = max(scale_results, key=lambda k: scale_results[k].mAP50)
            best_scale = float(best_name.split('_')[-1])
            
            validation['H1.3'] = {
                'description': 'Optimal crop_scale between 1.2-2.0',
                'optimal_scale': best_scale,
                'optimal_mAP50': scale_results[best_name].mAP50,
                'all_scales': {k: v.mAP50 for k, v in scale_results.items()},
                'expected_range': [1.2, 2.0],
                'hypothesis_supported': 1.2 <= best_scale <= 2.0
            }
        
        return validation
    
    def _save_results(self):
        """Save results to JSON."""
        output_path = self.output_dir / 'ablation_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ1] Results saved: {output_path}")
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table."""
        baseline = self.results.get('baseline')
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ1: Ablation Study Results}
\label{tab:rq1_ablation}
\begin{tabular}{lcccccc}
\toprule
Configuration & mAP50 & Precision & Recall & F1 & FP & $\Delta$mAP \\
\midrule
"""
        for name, result in self.results.items():
            if name.startswith('crop_scale'):
                continue
            
            delta = result.mAP50 - baseline.mAP50 if baseline else 0
            display_name = name.replace('_', ' ').title()
            
            latex += f"{display_name} & {result.mAP50:.3f} & {result.precision:.3f} & "
            latex += f"{result.recall:.3f} & {result.f1:.3f} & {result.fp_count} & "
            latex += f"{delta:+.3f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq1_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ1] LaTeX saved: {output_path}")
        return latex
    
    def generate_figures(self):
        """Generate figures."""
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        scale_results = {k: v for k, v in self.results.items() if k.startswith('crop_scale')}
        if scale_results:
            scales = []
            mAP50s = []
            
            for name in sorted(scale_results.keys(), key=lambda x: float(x.split('_')[-1])):
                scale = float(name.split('_')[-1])
                scales.append(scale)
                mAP50s.append(scale_results[name].mAP50)
            
            plt.figure(figsize=(8, 5))
            plt.plot(scales, mAP50s, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Crop Scale', fontsize=12)
            plt.ylabel('mAP50', fontsize=12)
            plt.title('H1.3: Crop Scale Sensitivity', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            best_idx = np.argmax(mAP50s)
            plt.scatter([scales[best_idx]], [mAP50s[best_idx]], 
                       c='red', s=150, zorder=5, label=f'Optimal: {scales[best_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'crop_scale_sensitivity.png', dpi=300)
            plt.close()
            print(f"[RQ1] Figure saved: {figures_dir / 'crop_scale_sensitivity.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ1: Ablation Experiment')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='Results/rq1_ablation')
    parser.add_argument('--experiment', type=str, default='full',
                        choices=['full', 'baseline', 'crop_scale', 'no_crop', 'no_nms'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ1: ABLATION EXPERIMENT (INTEGRATED)")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Experiment: {args.experiment}")
    
    experiment = RQ1AblationExperiment(config_path=args.config, output_dir=args.output)
    
    if args.experiment == 'full':
        experiment.run_full_ablation()
    elif args.experiment == 'baseline':
        experiment.run_baseline()
    elif args.experiment == 'crop_scale':
        experiment.run_baseline()
        experiment.run_crop_scale_sweep()
    elif args.experiment == 'no_crop':
        experiment.run_baseline()
        experiment.run_no_person_crop()
    elif args.experiment == 'no_nms':
        experiment.run_baseline()
        experiment.run_no_nms()
    
    validation = experiment.validate_hypotheses()
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    for h_id, h_result in validation.items():
        status = "✓ SUPPORTED" if h_result.get('hypothesis_supported') else "✗ NOT SUPPORTED"
        print(f"\n{h_id}: {status}")
        print(f"  {h_result.get('description', '')}")
        for key, value in h_result.items():
            if key not in ['description', 'hypothesis_supported', 'all_scales']:
                print(f"    {key}: {value}")
    
    experiment.generate_latex_table()
    experiment.generate_figures()
    
    validation_path = Path(args.output) / 'hypothesis_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()