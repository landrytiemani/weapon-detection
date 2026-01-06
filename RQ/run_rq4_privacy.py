#!/usr/bin/env python3
"""
RQ4: Privacy Preservation (FULLY INTEGRATED)
=============================================
Tests privacy-preserving configurations using your SingleExperiment class.

Usage:
    python run_rq4_privacy.py --config config.yaml
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
class PrivacyResult:
    config_name: str
    privacy_enabled: bool
    privacy_scope: str
    privacy_method: str
    mAP50: float
    precision: float
    recall: float
    f1: float
    fp_count: int
    fps: float
    latency_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ4PrivacyExperiment:
    """
    RQ4: Privacy-Preserving Weapon Detection
    
    Tests:
    - H4.1: Selective face blurring ≤10% overhead
    - H4.2: Privacy maintains mAP within 2%
    - H4.3: Pixelation faster than Gaussian
    - H4.4: Selective better than blanket
    """
    
    CONFIGURATIONS = [
        {'enabled': False, 'scope': 'none', 'method': 'none', 'name': 'no_privacy'},
        {'enabled': True, 'scope': 'non_targets', 'method': 'pixelate', 'name': 'selective_pixelate'},
        {'enabled': True, 'scope': 'non_targets', 'method': 'gaussian', 'name': 'selective_gaussian'},
        {'enabled': True, 'scope': 'everyone', 'method': 'pixelate', 'name': 'blanket_pixelate'},
    ]
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq4_privacy'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, PrivacyResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ4] GPU: {torch.cuda.get_device_name(0)}")
    
    def _save_config(self, config: dict, name: str) -> str:
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
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
    
    def _run_privacy_config(self, privacy_cfg: dict) -> PrivacyResult:
        name = privacy_cfg['name']
        print(f"\n{'='*60}")
        print(f"[RQ4] Running: {name}")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        
        # Set privacy config
        config['privacy'] = {
            'enabled': privacy_cfg['enabled'],
            'scope': privacy_cfg['scope'],
            'face_blur': {
                'enabled': privacy_cfg['enabled'],
                'method': privacy_cfg['method'],
                'pixel_block': 15,
                'gaussian_ksize': 31,
                'detector': 'none'
            },
            'silhouette': {'enabled': False}
        }
        
        # Prepare config with safe defaults
        config = self._prepare_config(config)
        
        temp_config_path = self._save_config(config, name)
        
        # Create output directory for this experiment
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Get settings
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        return PrivacyResult(
            config_name=name,
            privacy_enabled=privacy_cfg['enabled'],
            privacy_scope=privacy_cfg['scope'],
            privacy_method=privacy_cfg['method'],
            mAP50=results.get('pipeline_map50', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            fp_count=results.get('pipeline_fp', 0),
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0)
        )
    
    def run_all_configurations(self) -> Dict[str, PrivacyResult]:
        print("\n" + "=" * 70)
        print("RQ4: PRIVACY CONFIGURATION COMPARISON")
        print("=" * 70)
        
        for cfg in self.CONFIGURATIONS:
            result = self._run_privacy_config(cfg)
            self.results[cfg['name']] = result
            self._save_results()
            torch.cuda.empty_cache()
        
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        validation = {}
        
        baseline = self.results.get('no_privacy')
        selective_pix = self.results.get('selective_pixelate')
        selective_gauss = self.results.get('selective_gaussian')
        blanket = self.results.get('blanket_pixelate')
        
        # H4.1: Overhead ≤10%
        if baseline and selective_pix and baseline.latency_ms > 0:
            overhead = ((selective_pix.latency_ms - baseline.latency_ms) / baseline.latency_ms) * 100
            validation['H4.1'] = {
                'description': 'Selective face blurring ≤10% overhead',
                'baseline_latency_ms': baseline.latency_ms,
                'privacy_latency_ms': selective_pix.latency_ms,
                'overhead_percent': overhead,
                'threshold': 10.0,
                'hypothesis_supported': overhead <= 10.0
            }
        
        # H4.2: mAP within 2%
        if baseline and selective_pix:
            mAP_delta = abs(baseline.mAP50 - selective_pix.mAP50) * 100
            validation['H4.2'] = {
                'description': 'Privacy maintains mAP within 2%',
                'baseline_mAP50': baseline.mAP50,
                'privacy_mAP50': selective_pix.mAP50,
                'delta_points': mAP_delta,
                'threshold': 2.0,
                'hypothesis_supported': mAP_delta <= 2.0
            }
        
        # H4.3: Pixelate faster than Gaussian
        if selective_pix and selective_gauss:
            if selective_gauss.latency_ms > 0:
                speedup = ((selective_gauss.latency_ms - selective_pix.latency_ms) / 
                          selective_gauss.latency_ms) * 100
            else:
                speedup = 0
            
            validation['H4.3'] = {
                'description': 'Pixelation faster than Gaussian',
                'pixelate_latency_ms': selective_pix.latency_ms,
                'gaussian_latency_ms': selective_gauss.latency_ms,
                'speedup_percent': speedup,
                'hypothesis_supported': speedup > 0
            }
        
        # H4.4: Selective better than blanket
        if selective_pix and blanket and baseline:
            selective_utility = selective_pix.mAP50 / baseline.mAP50 if baseline.mAP50 > 0 else 0
            blanket_utility = blanket.mAP50 / baseline.mAP50 if baseline.mAP50 > 0 else 0
            
            validation['H4.4'] = {
                'description': 'Selective better utility than blanket',
                'selective_utility': selective_utility,
                'blanket_utility': blanket_utility,
                'hypothesis_supported': selective_utility > blanket_utility
            }
        
        return validation
    
    def _save_results(self):
        output_path = self.output_dir / 'privacy_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ4] Results saved: {output_path}")
    
    def generate_latex_table(self):
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ4: Privacy Configuration Comparison}
\label{tab:rq4_privacy}
\begin{tabular}{llccccc}
\toprule
Scope & Method & mAP50 & Precision & Recall & FPS & Latency (ms) \\
\midrule
"""
        for name, result in self.results.items():
            latex += f"{result.privacy_scope} & {result.privacy_method} & {result.mAP50:.3f} & "
            latex += f"{result.precision:.3f} & {result.recall:.3f} & "
            latex += f"{result.fps:.1f} & {result.latency_ms:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq4_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ4] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        if len(self.results) < 2:
            return
        
        # Privacy-utility trade-off
        configs = []
        mAPs = []
        latencies = []
        
        for name, result in self.results.items():
            configs.append(name.replace('_', '\n'))
            mAPs.append(result.mAP50)
            latencies.append(result.latency_ms)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(configs))
        
        ax1.bar(x, mAPs, color='steelblue', alpha=0.7, label='mAP50')
        ax1.set_ylabel('mAP50', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        
        ax2 = ax1.twinx()
        ax2.plot(x, latencies, 'ro-', linewidth=2, markersize=8, label='Latency')
        ax2.set_ylabel('Latency (ms)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('RQ4: Privacy-Utility Trade-off', fontsize=14)
        fig.tight_layout()
        plt.savefig(figures_dir / 'privacy_utility_tradeoff.png', dpi=300)
        plt.close()
        
        print(f"[RQ4] Figure saved: {figures_dir / 'privacy_utility_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ4: Privacy Experiment')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='Results/rq4_privacy')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ4: PRIVACY EXPERIMENT (INTEGRATED)")
    print("=" * 70)
    
    experiment = RQ4PrivacyExperiment(config_path=args.config, output_dir=args.output)
    experiment.run_all_configurations()
    
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