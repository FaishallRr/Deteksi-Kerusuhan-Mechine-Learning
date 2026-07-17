\"\"\"Model evaluation: comprehensive metrics and visualizations.\"\"\"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.fast_eval import main as evaluate
print(\"Evaluation module loaded. Call evaluate() to run full evaluation.\")
