import os


METRIC_HEADERS = [
    'experiment_name',
    'pretrained',
    'encoder',
    'train_input',
    'step1_time_sec',
    'step2_time_sec',
    'total_time_sec',
    'step1_train_samples_per_sec',
    'step2_train_samples_per_sec',
    'step1_best_eval_loss',
    'step2_best_eval_loss',
    'final_best_eval_loss',
    'total_params',
    'trainable_params',
    'best_checkpoint_size_mb',
    'last_checkpoint_size_mb',
]


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_file_size_mb(file_path):
    if not os.path.exists(file_path):
        return None
    return os.path.getsize(file_path) / (1024 * 1024)


def append_metrics_to_excel(excel_path, metric_row):
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError as exc:
        raise ImportError(
            'openpyxl 库未安装，请运行 "pip install openpyxl" 来安装该库，以使用 Excel 相关功能。'
        ) from exc

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    if os.path.exists(excel_path):
        workbook = load_workbook(excel_path)
        sheet = workbook.active
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'experiment_metrics'
        sheet.append(METRIC_HEADERS)

    if sheet.max_row == 0:
        sheet.append(METRIC_HEADERS)

    row_values = [metric_row.get(header, None) for header in METRIC_HEADERS]
    sheet.append(row_values)
    workbook.save(excel_path)
