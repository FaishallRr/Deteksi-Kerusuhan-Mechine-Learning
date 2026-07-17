\"\"\"Model training: AttentionMIL for riot detection.\"\"\"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mil_attention import AttentionMILModel
from core.train_mil_final import train_attention_mil

print(\"Training module loaded. Run train_attention_mil() to train the model.\")
