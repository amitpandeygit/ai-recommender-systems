"""Listings 24.2 and 24.4 - contextual bandits for the "New for you" slot (sections 24.4 to 24.6).

Every policy chooses one video for slot 4 among a request's candidates (the content
model's 100 best allowed new videos), and learns from the outcomes that have reached the
server. All of them keep their statistics between updates, which come in batches.

  HybridLinear   - Listing 24.2. Li and colleagues' hybrid linear model (WWW 2010): the
                   expected satisfied play is z . beta + theta_v, with beta shared by every
                   video (z: features of the request and the video) and theta_v one number
                   per video. mode "ucb" is LinUCB (mean + alpha x its standard deviation),
                   "ts" is linear Thompson sampling (a draw from the posterior), "greedy" is
                   the mean alone, "epsilon" the mean with a share epsilon of uniform choices.
  LogisticTS     - Chapelle and Li's logistic Thompson sampling (NIPS 2011):
                   P(play) = sigmoid(z . w + w_v), every weight with its own Gaussian
                   (a diagonal Laplace approximation), refitted on each new batch with the
                   last posterior as the prior.
  BetaPerVideo   - one Beta posterior per video, with no context: the simple bandit of
                   section 24.3 run in every language at once.
  propensities() - Listing 24.4. The chance that a policy picks each candidate: exact for
                   epsilon mixing, by Monte Carlo draws of the posterior for Thompson
                   sampling (section 24.7).

Needs Python 3.12 or later. Imported by the other Chapter 24 scripts.
"""
import numpy as np


class HybridLinear:
    """Listing 24.2. State: A0, b0 (shared part, k x k and k) and, per video v, a_v (a
    number), B_v (k) and b_v (a number), as in Li and colleagues' Algorithm 2 with one
    video-specific feature, the constant 1."""
    def __init__(self, k, n_videos, mode="ts", alpha=0.1, lam0=1.0, lam_v=100.0, eps=0.0, kappa=0.0):
        """alpha: how widely the policy explores (LinUCB's alpha; for Thompson sampling, the
        noise scale of the rewards, which multiplies the posterior's spread). kappa: the
        prior's trust in the content model. The model then predicts kappa x z[1] (the
        content gap of contextual.features) plus what it learns, so with no data it ranks exactly
        as the content model does, and it learns corrections to it. lam0: the prior
        precision of each shared weight (one number, or one per feature); lam_v: of each
        video's term."""
        self.mode, self.alpha, self.eps, self.kappa = mode, alpha, eps, kappa
        self.A0, self.b0 = np.diag(np.broadcast_to(np.asarray(lam0, float), (k,))).copy(), np.zeros(k)
        self.a, self.B, self.bv = np.full(n_videos, lam_v), np.zeros((n_videos, k)), np.zeros(n_videos)
        self.refresh()

    def refresh(self):
        """After an update: the shared mean and inverse, and each video's mean."""
        self.A0inv = np.linalg.inv(self.A0)
        self.beta = self.A0inv @ self.b0
        self.L = np.linalg.cholesky(self.A0inv)

    def update(self, Z, v, r, x=None):
        """Add observations (features Z, videos v, satisfied plays r), one at a time. x is
        the video-specific feature of each observation (1 unless given: section 24.5 puts
        the slot's attention weight there)."""
        x = np.ones(len(v)) if x is None else x
        r = r - self.kappa * Z[:, 1]                               # learn what the content model misses
        for z, j, y, xj in zip(Z, v, r, x):
            Ba = self.B[j] / self.a[j]
            self.A0 += np.outer(self.B[j], Ba)
            self.b0 += Ba * self.bv[j]
            self.a[j] += xj * xj
            self.B[j] += xj * z
            self.bv[j] += y * xj
            Ba = self.B[j] / self.a[j]
            self.A0 += np.outer(z, z) - np.outer(self.B[j], Ba)
            self.b0 += y * z - Ba * self.bv[j]
        self.refresh()

    def scores(self, Z, v, rng, scale=1.0):
        """One score per candidate (Z: candidates x k, v: their videos)."""
        a, B, bv = self.a[v], self.B[v], self.bv[v]
        if self.mode == "ts":
            beta = self.beta + scale * self.L @ rng.standard_normal(len(self.beta))
            theta = (bv - B @ beta) / a + scale * rng.standard_normal(len(v)) / np.sqrt(a)
            return self.kappa * Z[:, 1] + Z @ beta + theta
        theta = (bv - B @ self.beta) / a
        mean = self.kappa * Z[:, 1] + Z @ self.beta + theta
        if self.mode == "ucb":
            ZA, BA = Z @ self.A0inv, B @ self.A0inv
            s = (np.einsum("ck,ck->c", ZA, Z) - 2 * np.einsum("ck,ck->c", ZA, B) / a + 1 / a
                 + np.einsum("ck,ck->c", BA, B) / a ** 2)
            return mean + self.alpha * np.sqrt(np.maximum(s, 0.0))
        return mean

    def scores_many(self, Z, v, rng, draws):
        """Thompson scores for `draws` independent posterior draws (draws x candidates)."""
        a, B, bv = self.a[v], self.B[v], self.bv[v]
        beta = self.beta[None, :] + self.alpha * rng.standard_normal((draws, len(self.beta))) @ self.L.T
        theta = (bv[None, :] - beta @ B.T) / a + self.alpha * rng.standard_normal((draws, len(v))) / np.sqrt(a)
        return self.kappa * Z[None, :, 1] + beta @ Z.T + theta

    def choose(self, Z, v, rng):
        """The index of the chosen candidate."""
        s = self.scores(Z, v, rng, self.alpha)
        if self.mode == "epsilon" and rng.random() < self.eps:
            return int(rng.integers(len(v)))
        return int(np.argmax(s))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LogisticTS:
    """Every weight has a Gaussian posterior N(m, 1/q). On each batch, the
    weights are refitted by Newton's method with the last posterior as the prior, and
    each q grows by sum x^2 p (1 - p) (Chapelle and Li, Algorithm 3)."""
    def __init__(self, k, n_videos, q_shared=(0.1, 4.0), q_video=4.0, offset=0.0, scale=1.0, kappa=0.0):
        """q_shared: the prior precision of the constant and of the content gap's weight
        (the gap's prior is firm: when almost every choice is the content model's best
        video, the data alone cannot pin the gap's weight down)."""
        self.m = np.zeros(k + n_videos)
        self.m[0] = offset                                  # the constant starts at the prior's rate
        self.m[1] = kappa                                   # the content gap's weight starts at kappa
        self.q = np.r_[np.resize(np.asarray(q_shared, float), k), np.full(n_videos, q_video)]
        self.k, self.scale = k, scale

    def update(self, Z, v, r, iters=8):
        idx = np.c_[np.tile(np.arange(self.k), (len(v), 1)), self.k + v[:, None]]   # columns used by each row
        X = np.c_[Z, np.ones(len(v))]
        m0, q0 = self.m.copy(), self.q.copy()
        w = m0.copy()
        for _ in range(iters):                              # Newton steps on the diagonal Hessian
            p = sigmoid(np.einsum("nc,nc->n", X, w[idx]))
            g = q0 * (w - m0)
            np.add.at(g, idx.ravel(), (X * (p - r)[:, None]).ravel())
            h = q0.copy()
            np.add.at(h, idx.ravel(), (X ** 2 * (p * (1 - p))[:, None]).ravel())
            w = w - np.clip(g / h, -1.0, 1.0)                # a capped step keeps the diagonal Newton stable
        p = sigmoid(np.einsum("nc,nc->n", X, w[idx]))
        self.m = w
        self.q = q0.copy()
        np.add.at(self.q, idx.ravel(), (X ** 2 * (p * (1 - p))[:, None]).ravel())

    def scores(self, Z, v, rng, scale=None):
        s = self.scale if scale is None else scale
        w = self.m[:self.k] + s * rng.standard_normal(self.k) / np.sqrt(self.q[:self.k])
        wv = self.m[self.k + v] + s * rng.standard_normal(len(v)) / np.sqrt(self.q[self.k + v])
        return Z @ w + wv

    def scores_many(self, Z, v, rng, draws):
        s = self.scale
        w = self.m[:self.k] + s * rng.standard_normal((draws, self.k)) / np.sqrt(self.q[:self.k])
        wv = self.m[self.k + v] + s * rng.standard_normal((draws, len(v))) / np.sqrt(self.q[self.k + v])
        return w @ Z.T + wv

    def choose(self, Z, v, rng):
        return int(np.argmax(self.scores(Z, v, rng)))


class BetaPerVideo:
    """No context: one Beta posterior per video, starting from a prior worth n0 showings."""
    def __init__(self, n_videos, rate, n0=100.0):
        self.a = np.full(n_videos, 1.0 + n0 * rate)
        self.b = np.full(n_videos, 1.0 + n0 * (1 - rate))

    def update(self, Z, v, r):
        np.add.at(self.a, v, r)
        np.add.at(self.b, v, 1.0 - r)

    def scores(self, Z, v, rng, scale=1.0):
        return rng.beta(self.a[v], self.b[v])

    def choose(self, Z, v, rng):
        return int(np.argmax(self.scores(Z, v, rng)))


def propensities(policy, Z, v, rng, floor=0.0, draws=1_000):
    """Listing 24.4. The chance of each candidate under the logged policy: with probability
    floor a uniform choice among the candidates, otherwise the policy's own choice. A
    deterministic policy (greedy, LinUCB) gives its choice 1 - floor + floor / n; Thompson
    sampling's chances have no closed form and are estimated from `draws` independent draws
    of its posterior. Returns the chances and, for Monte Carlo, their standard errors."""
    n = len(v)
    if isinstance(policy, HybridLinear) and policy.mode in ("ucb", "greedy", "epsilon"):
        p = np.zeros(n)
        p[int(np.argmax(policy.scores(Z, v, rng, policy.alpha)))] = 1.0
        eps = floor + (policy.eps if policy.mode == "epsilon" else 0.0) * (1 - floor)
        return (1 - eps) * p + eps / n, np.zeros(n)
    wins = np.bincount(np.argmax(policy.scores_many(Z, v, rng, draws), axis=1), minlength=n)
    p = wins / draws
    se = np.sqrt(p * (1 - p) / draws)
    return (1 - floor) * p + floor / n, (1 - floor) * se
