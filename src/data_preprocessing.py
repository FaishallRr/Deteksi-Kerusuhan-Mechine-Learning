\"\"\"Data preprocessing: feature extraction from video using S3D.\"\"\"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_extractor import TemporalFeatureExtractor
print(\"Data preprocessing module loaded. Use TemporalFeatureExtractor for feature extraction.\")
