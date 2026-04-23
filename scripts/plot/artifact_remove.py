import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False


# EGI HydroCel-129 -> standard labels (from Automagic mapping used for ZuCo-style pipelines).
EGI129_TO_STD = {
	"E36": "C3", "E104": "C4", "E129": "CZ", "E24": "F3", "E124": "F4", "E33": "F7", "E122": "F8",
	"E22": "FP1", "E9": "FP2", "E15": "FPZ", "E11": "FZ", "E70": "O1", "E83": "O2", "E52": "P3",
	"E92": "P4", "E58": "P7", "E96": "P8", "E23": "AF3", "E3": "AF4", "E26": "AF7", "E2": "AF8",
	"E16": "AFZ", "E30": "C1", "E105": "C2", "E41": "C5", "E103": "C6", "E37": "CP1", "E87": "CP2",
	"E42": "CP3", "E93": "CP4", "E47": "CP5", "E98": "CP6", "E55": "CPZ", "E19": "F1", "E1": "F10",
	"E4": "F2", "E27": "F5", "E123": "F6", "E32": "F9", "E13": "FC1", "E112": "FC2", "E29": "FC3",
	"E111": "FC4", "E28": "FC5", "E117": "FC6", "E6": "FCZ", "E12": "FT10", "E34": "FT7", "E116": "FT8",
	"E38": "FT9", "E75": "OZ", "E60": "P1", "E64": "P9", "E95": "P10", "E85": "P2", "E51": "P5",
	"E97": "P6", "E67": "PO3", "E77": "PO4", "E65": "PO7", "E90": "PO8", "E72": "POZ", "E62": "PZ",
	"E114": "T10", "E44": "T9", "E100": "TP10", "E46": "TP7", "E102": "TP8", "E57": "TP9",
}

# Paper-style display aliases often used in older nomenclature.
DISPLAY_ALIAS = {
	"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
	"FP1": "FP1", "FP2": "FP2", "FPZ": "FPZ", "FZ": "FZ", "CZ": "CZ", "PZ": "PZ", "OZ": "OZ",
}

# Priority order similar to the figure style you referenced.
PAPER_CHANNEL_ORDER_STD = [
	"F7", "F3", "FP1", "C2", "C4", "P8", "P4", "O2", "O1", "P7", "P3", "T7", "C3", "FPZ", "FZ", "F4", "FP2", "T8", "F8",
]


def load_eeglab_mat(mat_path: Path):
	"""Load EEGLAB .mat and return data(ch, t), srate, channel labels."""
	try:
		mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
		if "EEG" not in mat:
			raise KeyError(f"MAT 文件中未找到 EEG 变量: {mat_path}")

		eeg = mat["EEG"]
		data = np.asarray(eeg.data, dtype=np.float64)
		if data.ndim == 3:
			# Keep first trial for continuous plotting.
			data = data[:, :, 0]
		if data.ndim != 2:
			raise ValueError(f"EEG.data 维度异常: {data.shape}")

		srate = float(eeg.srate)
		chanlocs = np.ravel(eeg.chanlocs)
		labels = [str(ch.labels) if hasattr(ch, "labels") else f"Ch{i+1}" for i, ch in enumerate(chanlocs)]
		if len(labels) != data.shape[0]:
			labels = [f"Ch{i+1}" for i in range(data.shape[0])]
		return data, srate, labels
	except NotImplementedError:
		return load_eeglab_mat_v73(mat_path)


def _decode_h5_char_dataset(ds):
	vals = np.array(ds).flatten()
	return "".join(chr(int(v)) for v in vals if int(v) != 0)


def normalize_channel_label(label: str):
	return str(label).strip().upper()


def map_egi_to_standard_label(label: str):
	lbl = normalize_channel_label(label)
	return EGI129_TO_STD.get(lbl, lbl)


def display_label(std_label: str):
	return DISPLAY_ALIAS.get(std_label, std_label)


def load_eeglab_mat_v73(mat_path: Path):
	"""Load EEGLAB data from MATLAB v7.3 (HDF5) MAT file."""
	try:
		import h5py
	except ImportError as exc:
		raise ImportError("读取 MATLAB v7.3 文件需要 h5py，请先安装: pip install h5py") from exc

	with h5py.File(str(mat_path), "r") as f:
		if "EEG" not in f:
			raise KeyError(f"MAT 文件中未找到 EEG 变量: {mat_path}")
		eeg = f["EEG"]

		data = np.array(eeg["data"], dtype=np.float64)
		srate = float(np.array(eeg["srate"]).squeeze())

		labels = []
		if "chanlocs" in eeg and isinstance(eeg["chanlocs"], h5py.Group):
			chanlocs = eeg["chanlocs"]
			if "labels" in chanlocs:
				label_refs = chanlocs["labels"]
				for ref in np.array(label_refs).flatten():
					if ref:
						labels.append(_decode_h5_char_dataset(f[ref]))

		if data.ndim == 3:
			data = data[:, :, 0]
		if data.ndim != 2:
			raise ValueError(f"EEG.data 维度异常: {data.shape}")

		if labels:
			if data.shape[0] == len(labels):
				pass
			elif data.shape[1] == len(labels):
				data = data.T
			else:
				labels = [f"Ch{i+1}" for i in range(data.shape[0])]
		else:
			labels = [f"Ch{i+1}" for i in range(data.shape[0])]

		return data, srate, labels


def align_by_labels(raw, raw_labels, proc, proc_labels):
	"""Align raw/proc channels by shared labels; return aligned arrays and labels."""
	raw_std = [map_egi_to_standard_label(lbl) for lbl in raw_labels]
	proc_std = [map_egi_to_standard_label(lbl) for lbl in proc_labels]

	raw_map = {lbl: i for i, lbl in enumerate(raw_std)}
	proc_map = {lbl: i for i, lbl in enumerate(proc_std)}
	shared_keys = [lbl for lbl in raw_std if lbl in proc_map]
	if not shared_keys:
		raise ValueError("raw 与 preprocessed 没有可匹配的通道标签")

	raw_idx = np.array([raw_map[k] for k in shared_keys], dtype=int)
	proc_idx = np.array([proc_map[k] for k in shared_keys], dtype=int)
	labels = [display_label(raw_std[i]) for i in raw_idx]

	raw_aligned = raw[raw_idx, :]
	proc_aligned = proc[proc_idx, :]
	return raw_aligned, proc_aligned, labels


def choose_paper_channels(labels):
	"""Choose channels in paper-like order if available."""
	label_to_idx = {normalize_channel_label(lbl): i for i, lbl in enumerate(labels)}
	idx = []
	for std_lbl in PAPER_CHANNEL_ORDER_STD:
		alias = display_label(std_lbl)
		key = normalize_channel_label(alias)
		if key in label_to_idx:
			idx.append(label_to_idx[key])
	return np.array(idx, dtype=int)


def choose_channels(labels, n_select: int = 20):
	"""Pick evenly spaced channels to keep the same before/after comparison."""
	n_ch = len(labels)
	if n_select >= n_ch:
		idx = np.arange(n_ch)
	else:
		idx = np.linspace(0, n_ch - 1, n_select, dtype=int)
	return idx


def parse_channel_indices(text: str, n_ch: int):
	"""Parse comma-separated indices like '0,3,7,10'."""
	parts = [p.strip() for p in text.split(",") if p.strip()]
	idx = np.array([int(p) for p in parts], dtype=int)
	if idx.size == 0:
		raise ValueError("--ch-idx 不能为空")
	if np.any(idx < 0) or np.any(idx >= n_ch):
		raise ValueError(f"--ch-idx 越界，合法范围: 0 ~ {n_ch - 1}")
	return idx


def compute_shared_gain(raw_signals: np.ndarray, proc_signals: np.ndarray, target_half_height: float = 0.38):
	"""Compute one display gain shared by raw/proc so amplitudes are directly comparable."""
	raw_centered = raw_signals - np.median(raw_signals, axis=1, keepdims=True)
	proc_centered = proc_signals - np.median(proc_signals, axis=1, keepdims=True)
	all_centered = np.concatenate([raw_centered, proc_centered], axis=0)
	amp = np.percentile(np.abs(all_centered), 99)
	return target_half_height / (float(amp) + 1e-12)


def plot_stacked(ax, t_ms, signals, labels, title, gain: float):
	"""Display-only vertical offsets with one shared gain for fair before/after comparison."""
	n_ch = signals.shape[0]
	lane_centers = np.arange(n_ch, 0, -1)
	centered = signals - np.median(signals, axis=1, keepdims=True)
	y_all = gain * centered + lane_centers[:, None]

	for i in range(n_ch):
		y_disp = y_all[i]
		ax.plot(t_ms, y_disp, linewidth=1.0)

	ax.set_title(title.replace("EEG", "EEG"))
	ax.set_ylabel("EEG [uV]")
	ax.set_yticks(lane_centers)
	ax.set_yticklabels(labels)
	x_min = float(t_ms.min())
	x_max = float(t_ms.max())
	x_pad = 0 * (x_max - x_min + 1e-12)
	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	y_min = float(np.min(y_all))
	y_max = float(np.max(y_all))
	y_pad = 0.15 * (y_max - y_min + 1e-12)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
	ax.grid(alpha=0.25)


def main():
	parser = argparse.ArgumentParser(description="绘制原始与预处理 EEG 对比图（仅显示偏移）。")
	parser.add_argument(
		"--mat",
		type=str,
		default=r"d:\桌面文件夹\毕设\zuco_project\osfstorage\task1-SR\Raw data\ZAB\ZAB_SR1_EEG.mat",
		help="原始 EEG 的 EEGLAB MAT 文件路径。",
	)
	parser.add_argument(
		"--mat-proc",
		type=str,
		default=r"D:\桌面文件夹\毕设\zuco_project\osfstorage\task1-SR\Preprocessed\ZAB\gip_ZAB_SR1_EEG.mat",
		help="预处理 EEG 的 EEGLAB MAT 文件路径。",
	)
	parser.add_argument("--seconds", type=float, default=4.0, help="从起始位置显示的时间长度（秒）。")
	parser.add_argument("--n-ch", type=int, default=20, help="显示的通道数量。")
	parser.add_argument("--y-zoom", type=float, default=4.0, help="上下两图共同的 Y 轴放大倍数，例如 1.5 或 2.0。")
	parser.add_argument("--paper-order", action="store_true", help="尽量按论文风格的通道名称与顺序显示。")
	parser.add_argument(
		"--ch-idx",
		type=str,
		default="",
		help="逗号分隔的通道索引（从 0 开始），如 0,5,10,15。设置后会覆盖 --n-ch。",
	)
	args = parser.parse_args()

	if not args.mat_proc.strip():
		raise ValueError("请提供 --mat-proc 预处理文件路径。此脚本现在只做双文件读取对比，不在脚本内做预处理。")

	mat_path = Path(args.mat)
	raw, srate, raw_labels = load_eeglab_mat(mat_path)
	proc_path = Path(args.mat_proc)
	proc, srate_proc, proc_labels = load_eeglab_mat(proc_path)
	if abs(srate_proc - srate) > 1e-6:
		raise ValueError("预处理数据与原始数据采样率不一致")
	raw, proc, labels = align_by_labels(raw, raw_labels, proc, proc_labels)
	print(f"[INFO] 原始数据: {mat_path}")
	print(f"[INFO] 预处理数据: {proc_path}")
	print(f"[INFO] 对齐后通道数: {len(labels)}")

	n_samples = min(int(args.seconds * srate), raw.shape[1], proc.shape[1])
	t_ms = np.arange(n_samples) * (1000.0 / srate)

	if args.ch_idx.strip():
		ch_idx = parse_channel_indices(args.ch_idx, len(labels))
	elif args.paper_order:
		paper_idx = choose_paper_channels(labels)
		if paper_idx.size > 0:
			ch_idx = paper_idx[: min(args.n_ch, paper_idx.size)]
		else:
			ch_idx = choose_channels(labels, n_select=args.n_ch)
	else:
		ch_idx = choose_channels(labels, n_select=args.n_ch)
	raw_sel = raw[ch_idx, :n_samples]
	proc_sel = proc[ch_idx, :n_samples]
	label_sel = [labels[i] for i in ch_idx]
	print(f"[INFO] 绘制通道: {', '.join(label_sel)}")
	gain = compute_shared_gain(raw_sel, proc_sel) * max(float(args.y_zoom), 1e-6)
	print(f"[INFO] 共享显示增益: {gain:.6g} (y_zoom={args.y_zoom})")

	fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
	plot_stacked(axes[0], t_ms, raw_sel, label_sel, "原始 EEG", gain)
	plot_stacked(axes[1], t_ms, proc_sel, label_sel, "去噪与伪影剔除后 EEG", gain)
	axes[1].set_xlabel("时间 [ms]")
	axes[0].set_ylabel("EEG [μV]")
	axes[1].set_ylabel("EEG [μV]")
	plt.tight_layout()
	plt.subplots_adjust(left=0.08, right=0.98)
	plt.show()


if __name__ == "__main__":
	main()
