#!/usr/bin/env python3
"""
RQ2: Architecture Comparison (FULLY INTEGRATED)
================================================
Compares RT-DETR vs YOLOv8-EfficientViT using your SingleExperiment class.

Usage:
    python run_rq2_architecture.py --config config.yaml
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
class ArchitectureMetrics:
    name: str
    mAP50: float
    precision: float
    recall: float
    f1: float
    tp_count: int
    fp_count: int
    fn_count: int
    handgun_mAP50: float
    knife_mAP50: float
    handgun_precision: float
    knife_precision: float
    gflops: float
    fps: float
    latency_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ2ArchitectureComparison:
    """
    RQ2: RT-DETR vs YOLOv8-EfficientViT Comparison
    
    Tests:
    - H2.1: RT-DETR achieves 3-8% higher mAP50
    - H2.2: EfficientViT ≥90% accuracy at ≤50% cost
    - H2.3: Larger gap on knives than handguns
    - H2.4: Both ≥10 FPS on GPU
    """
    
    ARCHITECTURES = ['rt_detr', 'yolov8_efficientvit']
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq2_architecture'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, ArchitectureMetrics] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ2] GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    def _run_architecture(self, arch_name: str) -> ArchitectureMetrics:
        print(f"\n{'='*60}")
        print(f"[RQ2] Testing Architecture: {arch_name}")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        config['stage_3']['approach'] = arch_name
        
        # Prepare config with safe defaults
        config = self._prepare_config(config)
        
        temp_config_path = self._save_config(config, arch_name)
        
        # Create output directory for this experiment
        exp_output_dir = str(self.output_dir / 'runs' / arch_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Get settings
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=f"rq2_{arch_name}",
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        per_class_map = results.get('pipeline_map50_by_class', {})
        per_class_prec = results.get('pipeline_precision_by_class', {})
        
        return ArchitectureMetrics(
            name=arch_name,
            mAP50=results.get('pipeline_map50', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            tp_count=results.get('pipeline_tp', 0),
            fp_count=results.get('pipeline_fp', 0),
            fn_count=results.get('pipeline_fn', 0),
            handgun_mAP50=per_class_map.get('handgun', 0.0),
            knife_mAP50=per_class_map.get('knife', 0.0),
            handgun_precision=per_class_prec.get('handgun', 0.0),
            knife_precision=per_class_prec.get('knife', 0.0),
            gflops=results.get('gflops', 0.0),
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0)
        )
    
    def run_comparison(self) -> Dict[str, ArchitectureMetrics]:
        print("\n" + "=" * 70)
        print("RQ2: ARCHITECTURE COMPARISON")
        print("RT-DETR (Transformer) vs YOLOv8-EfficientViT (CNN)")
        print("=" * 70)
        
        for arch_name in self.ARCHITECTURES:
            try:
                result = self._run_architecture(arch_name)
                self.results[arch_name] = result
                self._save_results()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[RQ2] ERROR with {arch_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        validation = {}
        
        rt_detr = self.results.get('rt_detr')
        efficientvit = self.results.get('yolov8_efficientvit')
        
        if not rt_detr or not efficientvit:
            print("[RQ2] Missing results for validation")
            return validation
        
        # H2.1
        mAP_diff = (rt_detr.mAP50 - efficientvit.mAP50) * 100
        validation['H2.1'] = {
            'description': 'RT-DETR achieves 3-8% higher mAP50',
            'rt_detr_mAP50': rt_detr.mAP50,
            'efficientvit_mAP50': efficientvit.mAP50,
            'difference_points': mAP_diff,
            'expected_range': [3.0, 8.0],
            'hypothesis_supported': 3.0 <= mAP_diff <= 8.0
        }
        
        # H2.2
        accuracy_ratio = (efficientvit.mAP50 / rt_detr.mAP50) * 100 if rt_detr.mAP50 > 0 else 100
        cost_ratio = (efficientvit.gflops / rt_detr.gflops) * 100 if rt_detr.gflops > 0 else 100
        
        validation['H2.2'] = {
            'description': 'EfficientViT ≥90% accuracy at ≤50% cost',
            'accuracy_ratio_percent': accuracy_ratio,
            'cost_ratio_percent': cost_ratio,
            'hypothesis_supported': accuracy_ratio >= 90 and cost_ratio <= 50
        }
        
        # H2.3
        knife_gap = (rt_detr.knife_mAP50 - efficientvit.knife_mAP50) * 100
        handgun_gap = (rt_detr.handgun_mAP50 - efficientvit.handgun_mAP50) * 100
        
        validation['H2.3'] = {
            'description': 'Larger gap on knives than handguns',
            'knife_gap_points': knife_gap,
            'handgun_gap_points': handgun_gap,
            'hypothesis_supported': knife_gap >= 8.0 and handgun_gap <= 4.0
        }
        
        # H2.4
        validation['H2.4'] = {
            'description': 'Both ≥10 FPS',
            'rt_detr_fps': rt_detr.fps,
            'efficientvit_fps': efficientvit.fps,
            'hypothesis_supported': rt_detr.fps >= 10 and efficientvit.fps >= 10
        }
        
        return validation
    
    def _save_results(self):
        output_path = self.output_dir / 'architecture_comparison.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ2] Results saved: {output_path}")
    
    def generate_latex_table(self):
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ2: Architecture Comparison}
\label{tab:rq2_architecture}
\begin{tabular}{lcccccc}
\toprule
Architecture & mAP50 & Precision & Recall & F1 & GFLOPs & FPS \\
\midrule
"""
        for name, result in self.results.items():
            display_name = name.replace('_', '-').upper()
            latex += f"{display_name} & {result.mAP50:.3f} & {result.precision:.3f} & "
            latex += f"{result.recall:.3f} & {result.f1:.3f} & "
            latex += f"{result.gflops:.1f} & {result.fps:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq2_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ2] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        if len(self.results) < 2:
            return
        
        rt_detr = self.results.get('rt_detr')
        efficientvit = self.results.get('yolov8_efficientvit')
        
        if not rt_detr or not efficientvit:
            return
        
        # Pareto frontier
        plt.figure(figsize=(8, 6))
        plt.scatter([rt_detr.gflops], [rt_detr.mAP50], s=150, c='blue', label='RT-DETR', marker='o')
        plt.scatter([efficientvit.gflops], [efficientvit.mAP50], s=150, c='red', label='YOLOv8-EfficientViT', marker='s')
        plt.xlabel('GFLOPs', fontsize=12)
        plt.ylabel('mAP50', fontsize=12)
        plt.title('H2.2: Accuracy vs Efficiency', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'pareto_frontier.png', dpi=300)
        plt.close()
        
        # Per-class comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        
        rt_vals = [rt_detr.handgun_mAP50, rt_detr.knife_mAP50]
        eff_vals = [efficientvit.handgun_mAP50, efficientvit.knife_mAP50]
        
        ax.bar(x - width/2, rt_vals, width, label='RT-DETR', color='blue')
        ax.bar(x + width/2, eff_vals, width, label='YOLOv8-EfficientViT', color='red')
        ax.set_ylabel('mAP50', fontsize=12)
        ax.set_title('H2.3: Per-Class Performance', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Handgun', 'Knife'], fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'per_class_comparison.png', dpi=300)
        plt.close()
        
        print(f"[RQ2] Figures saved to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='RQ2: Architecture Comparison')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='Results/rq2_architecture')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ2: ARCHITECTURE COMPARISON (INTEGRATED)")
    print("=" * 70)
    
    experiment = RQ2ArchitectureComparison(config_path=args.config, output_dir=args.output)
    experiment.run_comparison()
    
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