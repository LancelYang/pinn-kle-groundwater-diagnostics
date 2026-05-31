"""
KLE (Karhunen-Loeve Expansion) for log-conductivity parameterization.

Purpose
-------
Groundwater hydraulic conductivity K(x,y) exhibits spatial heterogeneity.
KLE expands the random/heterogeneous logK(x,y) as a weighted sum of orthogonal basis functions:

    logK(x, y) = mu(x, y) + sum_i sqrt(lambda_i) phi_i(x, y) xi_i

where:
    mu(x,y)    -- mean field (prior mean, can be constant or spatially varying)
    phi_i(x,y) -- eigenfunctions (KL modes), orthogonal over the domain
    lambda_i   -- eigenvalues, proportional to variance contribution of each mode
    xi_i ~ N(0,1) -- independent standard-normal coefficients (latent parameters to invert)

In the PINN framework, xi_i are treated as learnable network parameters, jointly constrained
by the physics loss and observation data, thereby achieving posterior inference of logK.

Core workflow
-------------
1. Initialize KLE object (given domain size, correlation lengths, variance)
2. Call compute_eigenpairs() to compute eigenvalues/eigenfunctions
3. Call truncate() to truncate to M retained modes
4. Call sample() to generate a logK realization (or bind xi_i to network parameters)

Author: AI Assistant
Date: 2026-05-20
"""

import numpy as np
from numpy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Covariance kernels
# ---------------------------------------------------------------------------

def exponential_cov(x, y, x_prime, y_prime, lx, ly):
    """
    Exponential covariance function (anisotropic 2D).

    C((x,y), (x',y')) = sigma^2 * exp( -|x-x'|/lx - |y-y'|/ly )

    Parameters
    ----------
    x, y, x_prime, y_prime : float or ndarray
        Spatial coordinates (scalar or array)
    lx, ly : float
        Correlation lengths along x/y directions
        Larger lx/ly -> stronger correlation, smoother spatial variation
        Smaller lx/ly -> more rapid spatial variation (high-frequency dominant)

    Returns
    -------
    C : float or ndarray
        Covariance value
    """
    return np.exp(-np.abs(x - x_prime) / lx - np.abs(y - y_prime) / ly)


def gaussian_cov(x, y, x_prime, y_prime, lx, ly):
    """
    Gaussian covariance function (anisotropic 2D).

    C((x,y), (x',y')) = sigma^2 * exp( -[(x-x')/lx]^2 - [(y-y')/ly]^2 )
    """
    return np.exp(-((x - x_prime) / lx) ** 2 - ((y - y_prime) / ly) ** 2)


def matern32_cov(x, y, x_prime, y_prime, lx, ly):
    """
    Matern(3/2) covariance function (smoother than exponential, more flexible than Gaussian).

    C = sigma^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
    where r = sqrt[(x-x')^2/lx^2 + (y-y')^2/ly^2], ell = sqrt(lx*ly)
    """
    r = np.sqrt(((x - x_prime) / lx) ** 2 + ((y - y_prime) / ly) ** 2)
    ell = np.sqrt(lx * ly)
    return (1 + np.sqrt(3) * r / ell) * np.exp(-np.sqrt(3) * r / ell)


# ---------------------------------------------------------------------------
# Main KLE class
# ---------------------------------------------------------------------------

class KLE:
    """
    Karhunen-Loeve Expansion for 2D spatial random fields.

    Parameters
    ----------
    Lx, Ly : float
        Physical domain size (m or km)
    nx, ny : int
        Spatial grid resolution (nx x ny points)
    lx, ly : float
        Correlation lengths (same unit as Lx/Ly). Suggested: lx ~ Lx/3 to Lx/5
    sigma2 : float
        Variance of log-conductivity Var[logK]. Typical: 0.1 ~ 4.0 (log-space)
        Larger variance -> stronger spatial heterogeneity
    mean : float
        Global mean of logK. Default 0 (geometric mean of K = 1 m/d)
    cov_type : str
        Covariance kernel type: 'exponential', 'gaussian', 'matern32'
    seed : int, optional
        Random seed (for reproducible sampling)
    """

    COV_KERNELS = {
        "exponential": exponential_cov,
        "gaussian": gaussian_cov,
        "matern32": matern32_cov,
    }

    def __init__(self, Lx, Ly, nx, ny, lx, ly, sigma2=1.0, mean=0.0,
                 cov_type="exponential", seed=None):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.sigma2 = sigma2
        self.mean = mean
        self.cov_type = cov_type
        self.cov_func = self.COV_KERNELS[cov_type]
        self.seed = seed

        # Spatial grid points (nx x ny)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        # Flatten to 1D vector for matrix operations
        self.x_flat = self.X.ravel()
        self.y_flat = self.Y.ravel()
        self.N = self.nx * self.ny

        # Cache eigenpairs (filled after calling compute_eigenpairs)
        self.eigenvalues = None   # lambda_i, descending order
        self.eigenvectors = None  # phi_i column vectors (shape: N x N)
        self.M = None  # Truncation order

        # Precompute weights (for Simpson/trapezoidal integration, approximate inner product)
        # Simple trapezoidal weights on a uniform grid (analytic integral is exact approximation)
        self._weights = self._compute_weights()

    def _compute_weights(self):
        """Area weight per grid point (for discrete inner product approximation)."""
        dx = self.Lx / (self.nx - 1) if self.nx > 1 else 1.0
        dy = self.Ly / (self.ny - 1) if self.ny > 1 else 1.0
        # Corner weights = 1, edge = 2, interior = 4 (2D trapezoidal integration)
        wx = np.ones(self.nx)
        wx[0] = 0.5
        wx[-1] = 0.5
        wy = np.ones(self.ny)
        wy[0] = 0.5
        wy[-1] = 0.5
        WX, WY = np.meshgrid(wx, wy, indexing="ij")
        w_flat = (WX * WY).ravel()
        return w_flat * dx * dy

    def build_covariance_matrix(self):
        """
        Build covariance matrix C_ij = Cov[logK(x_i), logK(x_j)].

        Returns
        -------
        C : ndarray, shape (N, N)
            Symmetric positive-definite covariance matrix
        """
        print(f"[KLE] Building covariance matrix ({self.nx}x{self.ny} = {self.N} grid points) ...")
        # Vectorized computation avoids Python loops
        # C[i,j] = sigma2 * cov_kernel(x_i, y_i, x_j, y_j)
        x1 = self.x_flat[:, None]  # (N, 1)
        y1 = self.y_flat[:, None]
        x2 = self.x_flat[None, :]   # (1, N)
        y2 = self.y_flat[None, :]

        C = self.sigma2 * self.cov_func(x1, y1, x2, y2, self.lx, self.ly)
        print(f"[KLE] Covariance matrix built, shape={C.shape}, "
              f"cond~{np.linalg.cond(C):.2e}")
        return C

    def compute_eigenpairs(self):
        """
        Compute all eigenvalues and eigenvectors of the covariance matrix.

        Uses scipy.linalg.eigh (symmetric matrix specialist, faster and more stable
        than general eigendecomposition).

        Eigenvalues are returned in ascending order; internally flipped to descending.

        Stores
        ------
        self.eigenvalues  : lambda_i (descending)
        self.eigenvectors : phi_i (one eigenvector per column, normalized)
        """
        C = self.build_covariance_matrix()
        print("[KLE] Solving eigenvalue problem (eigh) ...")
        eigenvalues, eigenvectors = eigh(C)
        # eigh returns ascending order; flip to descending
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        # eigh eigenvectors are already orthonormal (phi_i^T phi_j = delta_ij)
        # No area-weight renormalization, otherwise Var[logK] ~ sigma^2/400 instead of sigma^2
        # Correct relation: Var[logK(x)] = sum_i lambda_i phi_i(x)^2 ~ C(x,x) = sigma^2

        print(f"[KLE] Eigenvalue decomposition complete. max={self.eigenvalues[0]:.4f}, "
              f"min={self.eigenvalues[-1]:.2e}, "
              f"sum~{np.sum(self.eigenvalues):.4f}")
        return self.eigenvalues, self.eigenvectors

    def truncate(self, M=None, var_threshold=0.99):
        """
        Truncate to the first M KL modes.

        Parameters
        ----------
        M : int, optional
            Number of modes to retain. If not specified, auto-determined by var_threshold.
        var_threshold : float
            Cumulative variance fraction threshold (0~1). Default 0.99 retains 99% variance.

        Returns
        -------
        M_actual : int
            Final number of retained modes
        """
        if self.eigenvalues is None:
            raise RuntimeError("Call compute_eigenpairs() first")

        if M is None:
            cumvar = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
            M = np.searchsorted(cumvar, var_threshold) + 1
            print(f"[KLE] Cumulative variance {var_threshold*100:.1f}% → truncation M = {M}")
        else:
            actual_var = np.sum(self.eigenvalues[:M]) / np.sum(self.eigenvalues)
            print(f"[KLE] Fixed truncation M={M}, cumulative var={actual_var*100:.2f}%")

        self.M = M
        self.truncated_eigenvalues = self.eigenvalues[:M]
        self.truncated_eigenvectors = self.eigenvectors[:, :M]
        return M

    def sample(self, xi=None, return_components=False):
        """
        Generate a random realization of logK(x,y).

        logK(x,y) = mean + sum_{i=1}^{M} sqrt(lambda_i) * phi_i(x,y) * xi_i

        Parameters
        ----------
        xi : ndarray, shape (M,), optional
            KL coefficient vector (latent parameters). If not specified, sampled from N(0,1).
            In PINN, xi is typically passed as learnable network parameters.
        return_components : bool
            If True, also return per-mode contributions (for visualization)

        Returns
        -------
        logK : ndarray, shape (nx, ny)
            Log-conductivity field (2D grid)
        xi_actual : ndarray
            Actual coefficients used (random sample if xi=None)
        components : ndarray (optional)
            Per-mode contributions (shape: M x nx x ny)
        """
        if self.eigenvalues is None or self.M is None:
            raise RuntimeError("Call compute_eigenpairs() and truncate() first")

        rng = np.random.default_rng(self.seed)
        if xi is None:
            xi_actual = rng.standard_normal(self.M)
        else:
            xi_actual = np.asarray(xi)

        # logK = mean + Σ √λᵢ · φᵢ · ξᵢ
        logK_flat = self.mean + np.zeros(self.N)
        if return_components:
            components = []

        for i in range(self.M):
            sqrt_lambda = np.sqrt(self.truncated_eigenvalues[i])
            phi = self.truncated_eigenvectors[:, i]
            contribution = sqrt_lambda * phi * xi_actual[i]
            logK_flat += contribution
            if return_components:
                components.append(contribution.reshape(self.nx, self.ny))

        logK = logK_flat.reshape(self.nx, self.ny)

        if return_components:
            return logK, xi_actual, np.array(components)
        return logK, xi_actual

    def compute_spectrum_plot(self, save_path=None):
        """
        Plot eigenvalue spectrum and cumulative variance contribution curves.

        Parameters
        ----------
        save_path : str or Path, optional
            Save path (e.g., 'figures/kle/eigenvalue_spectrum.png')
        """
        if self.eigenvalues is None:
            raise RuntimeError("Call compute_eigenpairs() first")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: eigenvalue spectrum (log scale)
        ax1 = axes[0]
        i_vals = np.arange(1, len(self.eigenvalues) + 1)
        ax1.semilogy(i_vals, self.eigenvalues, "b.-", linewidth=1.5, markersize=4)
        if self.M is not None:
            ax1.axvline(self.M, color="red", linestyle="--", linewidth=1.5,
                        label=f"Truncation M={self.M}")
            ax1.legend(fontsize=11)
        ax1.set_xlabel("Mode index $i$", fontsize=12)
        ax1.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=12)
        ax1.set_title("KL Eigenvalue Spectrum (log scale)", fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Right: cumulative variance contribution
        ax2 = axes[1]
        total_var = np.sum(self.eigenvalues)
        cumvar = np.cumsum(self.eigenvalues) / total_var
        ax2.plot(i_vals, cumvar, "b-", linewidth=2)
        ax2.axhline(0.90, color="orange", linestyle=":", linewidth=1.5, label="90%")
        ax2.axhline(0.95, color="green", linestyle=":", linewidth=1.5, label="95%")
        ax2.axhline(0.99, color="red", linestyle=":", linewidth=1.5, label="99%")
        if self.M is not None:
            ax2.axvline(self.M, color="red", linestyle="--", linewidth=1.5,
                        label=f"$M$={self.M} ({cumvar[self.M-1]*100:.1f}%)")
        ax2.set_xlim([1, min(len(self.eigenvalues), 100)])
        ax2.set_xlabel("Mode index $i$", fontsize=12)
        ax2.set_ylabel("Cumulative variance fraction", fontsize=12)
        ax2.set_title("Cumulative Variance Contribution", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[KLE] Spectrum saved: {save_path}")
        plt.close()
        return fig

    def compute_realization_plot(self, logK, save_path=None,
                                 vmin=None, vmax=None):
        """
        Plot spatial distribution of a logK realization.

        Parameters
        ----------
        logK : ndarray, shape (nx, ny)
            Log-conductivity field (returned by sample())
        save_path : str or Path, optional
        vmin, vmax : float
            Color range. Default: auto.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(self.X, self.Y, logK,
                           cmap="viridis",
                           vmin=vmin or logK.min(),
                           vmax=vmax or logK.max(),
                           shading="auto")
        ax.set_xlabel("$x$ (m)", fontsize=12)
        ax.set_ylabel("$y$ (m)", fontsize=12)
        ax.set_title(f"$\\log K$ Realization\n"
                     f"($\\mu$={self.mean:.2f}, "
                     f"$\\sigma^2$={self.sigma2:.2f}, "
                     f"$l_x$={self.lx:.1f}, $l_y$={self.ly:.1f})",
                     fontsize=12)
        ax.set_aspect("equal")
        cbar = fig.colorbar(pc, ax=ax, label="$\\log K$")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[KLE] Realization plot saved: {save_path}")
        plt.close()
        return fig

    def get_basis_for_pytorch(self):
        """Export KLE basis for PyTorch-based inverse problem training.

        Returns a lightweight dict containing the truncated basis (sqrt(λᵢ),
        φᵢ grid values, and grid coordinates) suitable for differentiable
        logK computation at arbitrary points via bilinear interpolation.

        Returns
        -------
        dict with keys:
            sqrt_lambda : ndarray (M,)
            phi_grid    : ndarray (M, N) — row-major, row i = φᵢ at all grid points
            x_grid      : ndarray (N,)
            y_grid      : ndarray (N,)
            nx, ny      : int
            Lx, Ly      : float
            M           : int
        """
        if self.truncated_eigenvalues is None:
            raise RuntimeError(
                "Must call truncate() before get_basis_for_pytorch()"
            )
        return {
            'sqrt_lambda': np.sqrt(self.truncated_eigenvalues),  # (M,)
            'phi_grid': self.truncated_eigenvectors.T.copy(),    # (M, N) row-major
            'x_grid': self.x_flat.copy(),                        # (N,)
            'y_grid': self.y_flat.copy(),                        # (N,)
            'nx': self.nx,
            'ny': self.ny,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'M': self.M,
        }


# ---------------------------------------------------------------------------
# Quick demo / unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("KLE parameterization -- minimal runnable demo")
    print("=" * 60)

    # Parameters (modifiable)
    kle = KLE(
        Lx=1000.0,       # Domain size 1000 m
        Ly=1000.0,       # 1000 m
        nx=50,           # 50x50 grid
        ny=50,
        lx=200.0,        # Correlation length 200 m
        ly=200.0,
        sigma2=1.0,      # logK variance = 1.0 (moderate heterogeneity)
        mean=0.0,        # logK mean = 0 (geometric mean of K = 1 m/d)
        cov_type="exponential",
        seed=42,
    )

    # Step 1: Compute eigenpairs
    eigenvalues, eigenvectors = kle.compute_eigenpairs()

    # Step 2: Truncate (retain 99% variance)
    M = kle.truncate(var_threshold=0.99)

    # Step 3: Plot spectrum
    kle.compute_spectrum_plot(
        save_path="figures/kle/eigenvalue_spectrum.png"
    )

    # Step 4: Generate 3 random realizations
    for i in range(3):
        kle_i = KLE(
            Lx=1000.0, Ly=1000.0, nx=50, ny=50,
            lx=200.0, ly=200.0, sigma2=1.0, mean=0.0,
            cov_type="exponential", seed=i * 100 + 42
        )
        kle_i.compute_eigenpairs()
        kle_i.truncate(M=M)
        logK, xi = kle_i.sample()
        kle_i.compute_realization_plot(
            logK,
            save_path=f"figures/kle/logK_realization_{i+1}.png"
        )

    print("=" * 60)
    print(f"Demo complete! Output files in figures/kle/")
    print(f"Truncation order M = {M}, retaining {M/len(eigenvalues)*100:.1f}% of modes")
    print("=" * 60)
