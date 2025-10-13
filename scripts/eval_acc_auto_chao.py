import os
import sys
import re
import subprocess
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class QuantizationBatchEvaluator:
    def __init__(self, mamba_file_path, eval_script_path, results_dir="quant_results"):
        """
        Initialize the batch evaluator
        
        Args:
            mamba_file_path: Path to the mamba.py file
            eval_script_path: Path to the eval_acc_chao.py file
            results_dir: Directory to save results
        """
        self.mamba_file_path = mamba_file_path
        self.eval_script_path = eval_script_path
        self.results_dir = results_dir
        self.original_mamba_content = None
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Backup original file
        self._backup_original_file()
    
    def _backup_original_file(self):
        """Backup the original mamba.py file"""
        with open(self.mamba_file_path, 'r', encoding='utf-8') as f:
            self.original_mamba_content = f.read()
        
        backup_path = f"{self.mamba_file_path}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(self.original_mamba_content)
        logging.info(f"Backed up original file to: {backup_path}")
    
    def _modify_quantizer(self, e_bits, m_bits):
        """
        Modify the quantizer configuration in mamba.py
        
        Args:
            e_bits: Number of exponent bits
            m_bits: Number of mantissa bits
        """
        # Read file content
        with open(self.mamba_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to replace quantizer configuration
        # Pattern: self.quantizer = FloatQuantizer(e_bits=X, m_bits=Y)
        pattern = r'self\.quantizer\s*=\s*FloatQuantizer\(e_bits=\d+,\s*m_bits=\d+\)'
        replacement = f'self.quantizer = FloatQuantizer(e_bits={e_bits}, m_bits={m_bits})'
        
        modified_content = re.sub(pattern, replacement, content)
        
        # Write back to file
        with open(self.mamba_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        logging.info(f"Modified quantizer config: e_bits={e_bits}, m_bits={m_bits}")
    
    def _restore_original_file(self):
        """Restore the original mamba.py file"""
        if self.original_mamba_content:
            with open(self.mamba_file_path, 'w', encoding='utf-8') as f:
                f.write(self.original_mamba_content)
            logging.info("Restored original file")
    
    def _run_evaluation(self):
        """
        Run the evaluation script and capture output
        
        Returns:
            dict: Dictionary containing top1 and top5 accuracy, or None if failed
        """
        try:
            # Run evaluation script
            result = subprocess.run(
                [sys.executable, self.eval_script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            output = result.stdout + result.stderr
            logging.info(f"Evaluation script output:\n{output}")
            
            # Extract accuracy from output
            top1_match = re.search(r'Top-1 accuracy:\s*([\d.]+)%', output)
            top5_match = re.search(r'Top-5 accuracy:\s*([\d.]+)%', output)
            
            if top1_match and top5_match:
                return {
                    'top1': float(top1_match.group(1)),
                    'top5': float(top5_match.group(1))
                }
            else:
                logging.error("Could not extract accuracy from output")
                return None
                
        except subprocess.TimeoutExpired:
            logging.error("Evaluation script timed out")
            return None
        except Exception as e:
            logging.error(f"Error running evaluation script: {e}")
            return None
    
    def run_batch_evaluation(self, quant_configs):
        """
        Run batch evaluation
        
        Args:
            quant_configs: List of quantization configs, each config is (e_bits, m_bits) tuple
        
        Returns:
            dict: Evaluation results for all configurations
        """
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (e_bits, m_bits) in enumerate(quant_configs, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Progress: {i}/{len(quant_configs)}")
            logging.info(f"Testing config: e_bits={e_bits}, m_bits={m_bits}")
            logging.info(f"{'='*60}\n")
            
            try:
                # Modify quantizer configuration
                self._modify_quantizer(e_bits, m_bits)
                
                # Run evaluation
                eval_result = self._run_evaluation()
                
                if eval_result:
                    config_key = f"e{e_bits}_m{m_bits}"
                    results[config_key] = {
                        'e_bits': e_bits,
                        'm_bits': m_bits,
                        'top1_acc': eval_result['top1'],
                        'top5_acc': eval_result['top5']
                    }
                    logging.info(f"✓ Config {config_key}: Top-1={eval_result['top1']:.2f}%, Top-5={eval_result['top5']:.2f}%")
                else:
                    logging.error(f"✗ Config e_bits={e_bits}, m_bits={m_bits} evaluation failed")
                
            except Exception as e:
                logging.error(f"Error processing config e_bits={e_bits}, m_bits={m_bits}: {e}")
            
            # Save intermediate results
            self._save_results(results, timestamp, partial=True)
        
        # Restore original file
        self._restore_original_file()
        
        # Save final results
        self._save_results(results, timestamp, partial=False)
        
        return results
    
    def _save_results(self, results, timestamp, partial=False):
        """Save results to JSON file"""
        suffix = "_partial" if partial else ""
        results_file = os.path.join(
            self.results_dir, 
            f"quant_eval_results_{timestamp}{suffix}.json"
        )
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to: {results_file}")
    
    def print_summary(self, results):
        """Print results summary"""
        if not results:
            logging.warning("No results available")
            return
        
        print("\n" + "="*80)
        print("Quantization Precision Evaluation Results Summary")
        print("="*80)
        print(f"{'Config':<15} {'e_bits':<10} {'m_bits':<10} {'Top-1 Acc':<15} {'Top-5 Acc':<15}")
        print("-"*80)
        
        # Sort by Top-1 accuracy
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]['top1_acc'], 
            reverse=True
        )
        
        for config_key, result in sorted_results:
            print(f"{config_key:<15} {result['e_bits']:<10} {result['m_bits']:<10} "
                  f"{result['top1_acc']:<15.2f} {result['top5_acc']:<15.2f}")
        
        print("="*80 + "\n")


def main():
    # Configure file paths (modify according to your actual paths)
    MAMBA_FILE = "/volume1/users/cfang/simba/simba/mamba_simple.py"  # Modify to your mamba.py path
    EVAL_SCRIPT = "/volume1/users/cfang/simba/scripts/eval_acc_chao.py"  # Modify to your eval_acc_chao.py path
    
    # Define quantization configurations to test
    # Format: (e_bits, m_bits)
    # Total bits = 1 (sign bit) + e_bits + m_bits
    
    quant_configs = [
        # ===== 16-bit (15 effective bits) =====
        (8, 7),   # E8M7 - BF16 standard
        (7, 8),   # E7M8
        (6, 9),   # E6M9
        (5, 10),  # E5M10 - FP16 standard
        
        # ===== 12-bit (11 effective bits) =====
        (8, 3),   # E8M3
        (7, 4),   # E7M4
        (6, 5),   # E6M5
        (5, 6),   # E5M6
        (4, 7),   # E4M7
        (3, 8),   # E3M8
        
        # ===== 10-bit (9 effective bits) =====
        (7, 2),   # E7M2
        (6, 3),   # E6M3
        (5, 4),   # E5M4
        (4, 5),   # E4M5
        (3, 6),   # E3M6
        (2, 7),   # E2M7
        
        # ===== 8-bit (7 effective bits) =====
        (6, 1),   # E6M1
        (5, 2),   # E5M2 - FP8 standard
        (4, 3),   # E4M3 - FP8 standard
        (3, 4),   # E3M4
        (2, 5),   # E2M5
        (1, 6),   # E1M6
        
        # ===== 6-bit (5 effective bits) =====
        (4, 1),   # E4M1
        (3, 2),   # E3M2 - Current configuration
        (2, 3),   # E2M3
        (1, 4),   # E1M4
        
        # ===== 5-bit (4 effective bits) =====
        (3, 1),   # E3M1
        (2, 2),   # E2M2
        (1, 3),   # E1M3
        
        # ===== 4-bit (3 effective bits) =====
        (2, 1),   # E2M1
        (1, 2),   # E1M2
    ]
    
    print(f"Total configurations to test: {len(quant_configs)}")
    print("Configuration range: From 16-bit (E10M5) to 4-bit (E1M2)")
    print(f"Estimated total time: ~{len(quant_configs) * 30} minutes (assuming 30 min per config)\n")
    
    # Create evaluator
    evaluator = QuantizationBatchEvaluator(MAMBA_FILE, EVAL_SCRIPT)
    
    try:
        # Run batch evaluation
        results = evaluator.run_batch_evaluation(quant_configs)
        
        # Print results summary
        evaluator.print_summary(results)
        
    except KeyboardInterrupt:
        logging.warning("\nUser interrupted execution")
        evaluator._restore_original_file()
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        evaluator._restore_original_file()
        raise


if __name__ == "__main__":
    main()