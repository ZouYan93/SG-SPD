import os
metrics_dir = os.environ.get('EVAL_METRICS_DIR', '').strip() or None
print(f"Metrics directory: {metrics_dir}")