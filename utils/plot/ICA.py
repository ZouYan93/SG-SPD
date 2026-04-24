import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def generate_sources(n_samples: int = 2000, random_state: int = 42):
	"""Generate three independent source signals."""
	rng = np.random.default_rng(random_state)
	t = np.linspace(0, 8, n_samples)

	s1 = np.sin(2 * np.pi * 1.2 * t)  # sine
	s2 = np.sign(np.sin(2 * np.pi * 0.8 * t))  # square-like
	frac = (t * 0.5) % 1.0
	s3 = 2.0 * frac - 1.0  # sawtooth-like

	s1 += 0.05 * rng.standard_normal(n_samples)
	s2 += 0.05 * rng.standard_normal(n_samples)
	s3 += 0.05 * rng.standard_normal(n_samples)

	S = np.c_[s1, s2, s3]
	S -= S.mean(axis=0)
	S /= S.std(axis=0)
	return t, S


def mix_sources(S: np.ndarray):
	"""Mix source signals with a fixed full-rank mixing matrix."""
	A = np.array([
		[1.0, 1.0, 0.5],
		[0.5, 2.0, 1.0],
		[1.5, 1.0, 2.0],
	])
	X = S @ A.T
	return X, A


def run_fastica(
	X: np.ndarray,
	n_components: int = 3,
	max_iter: int = 1000,
	tol: float = 1e-4,
	random_state: int = 0,
):
	"""Use sklearn FastICA to recover independent components."""
	ica = FastICA(
		n_components=n_components,
		algorithm="parallel",
		fun="logcosh",
		max_iter=max_iter,
		tol=tol,
		random_state=random_state,
	)
	S_est = ica.fit_transform(X)
	return S_est, ica.mixing_


def plot_group(ax, t, signals, title):
	"""Plot stacked traces in one panel without modifying original arrays."""
	n_ch = signals.shape[1]
	lane_centers = np.arange(1, n_ch + 1)

	for i in range(n_ch):
		xi = signals[:, i]
		# Display-only scaling: keep original arrays unchanged.
		scale = np.max(np.abs(xi)) + 1e-12
		y_disp = 0.38 * (xi / scale) + lane_centers[i]
		ax.plot(t, y_disp, linewidth=1.0)

	ax.set_title(title)
	ax.set_xlabel("时间（秒）")
	ax.set_ylabel("通道")
	ax.set_yticks(lane_centers)
	ax.set_yticklabels([f"通道{i}" for i in lane_centers])
	ax.set_xlim(float(t.min()), float(t.max()))
	ax.set_ylim(0.5, n_ch + 0.5)
	ax.grid(alpha=0.3)


def main():
	t, S = generate_sources()
	X, _ = mix_sources(S)
	S_est, _ = run_fastica(X, n_components=3, random_state=7)

	fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
	plot_group(axes[0], t, S, "原始信号")
	plot_group(axes[1], t, X, "混合信号")
	plot_group(axes[2], t, S_est, "ICA 分离后信号")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
