"""Modelo de Poisson por equipos (Dixon-Coles) con decaimiento temporal.

λ_local     = exp(c + ventaja_local + ataque_local + defensa_visitante)
μ_visitante = exp(c + ataque_visitante + defensa_local)

Los partidos recientes pesan más (peso = exp(-ξ · días)) y cada equipo tiene un prior
normal que encoge hacia la media a los que tienen pocos datos. Con `dixon_coles=True`
se corrige la dependencia de los marcadores bajos (0-0, 1-0, 0-1, 1-1) con ρ.
Se usa igual para goles, goles al descanso, córners y tarjetas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma, nbinom, poisson

from . import config


@dataclass
class PoissonTeamModel:
    xi: float = config.XI
    dixon_coles: bool = True
    prior_sd: float = config.PRIOR_SD
    promoted_prior: float = config.PROMOTED_PRIOR
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.0
    intercept: float = 0.0
    rho: float = 0.0
    shared_var: float = 0.0   # varianza del factor común del partido (p. ej. árbitro / partido caliente)
    indiv_var: float = 0.0    # sobredispersión propia de cada equipo
    n_matches: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, home, away, hg, ag, dates, ref_date=None) -> "PoissonTeamModel":
        home = np.asarray(home, dtype=object)
        away = np.asarray(away, dtype=object)
        hg = np.asarray(hg, dtype=float)
        ag = np.asarray(ag, dtype=float)
        dates = pd.to_datetime(pd.Series(dates)).to_numpy()
        ok = ~(np.isnan(hg) | np.isnan(ag))
        home, away, hg, ag, dates = home[ok], away[ok], hg[ok], ag[ok], dates[ok]
        if len(hg) < 20:
            return self

        ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp(dates.max())
        days = (ref.to_datetime64() - dates) / np.timedelta64(1, "D")
        w = np.exp(-self.xi * np.clip(days, 0, None))

        self.teams = sorted(set(home) | set(away))
        idx = {t: i for i, t in enumerate(self.teams)}
        hi = np.array([idx[t] for t in home])
        ai = np.array([idx[t] for t in away])
        n = len(self.teams)
        self.n_matches = {t: float(np.sum(w[(home == t) | (away == t)] > 0.5)) for t in self.teams}

        # Equipos que no estaban al principio del histórico (ascendidos) → prior algo peor que la media.
        first_cut = pd.Timestamp(dates.min()) + pd.Timedelta(days=200)
        early = set(home[dates <= first_cut.to_datetime64()]) | set(away[dates <= first_cut.to_datetime64()])
        pa = np.array([0.0 if t in early else -self.promoted_prior for t in self.teams])
        pd_ = np.array([0.0 if t in early else self.promoted_prior for t in self.teams])
        prec = 1.0 / self.prior_sd ** 2

        y1, y2 = hg, ag
        z00 = (y1 == 0) & (y2 == 0)
        z01 = (y1 == 0) & (y2 == 1)
        z10 = (y1 == 1) & (y2 == 0)
        z11 = (y1 == 1) & (y2 == 1)
        mean_goals = np.log(max((np.average(y1, weights=w) + np.average(y2, weights=w)) / 2, 0.05))
        dc = self.dixon_coles

        def unpack(x):
            return x[0], x[1], x[2], x[3:3 + n], x[3 + n:3 + 2 * n]

        def nll(x):
            c, h, rho, att, de = unpack(x)
            ll_lam = c + h + att[hi] + de[ai]
            ll_mu = c + att[ai] + de[hi]
            lam, mu = np.exp(ll_lam), np.exp(ll_mu)
            g_lam = y1 - lam
            g_mu = y2 - mu
            ll = y1 * ll_lam - lam + y2 * ll_mu - mu
            g_rho = 0.0
            if dc:
                tau = np.ones_like(lam)
                t00 = np.clip(1 - lam * mu * rho, 1e-6, None)
                t01 = np.clip(1 + lam * rho, 1e-6, None)
                t10 = np.clip(1 + mu * rho, 1e-6, None)
                t11 = max(1 - rho, 1e-6)
                tau = np.where(z00, t00, np.where(z01, t01, np.where(z10, t10, np.where(z11, t11, 1.0))))
                ll = ll + np.log(tau)
                g_lam = g_lam + np.where(z00, -lam * mu * rho / t00, 0) + np.where(z01, lam * rho / t01, 0)
                g_mu = g_mu + np.where(z00, -lam * mu * rho / t00, 0) + np.where(z10, mu * rho / t10, 0)
                dr = np.where(z00, -lam * mu / t00, 0) + np.where(z01, lam / t01, 0) \
                    + np.where(z10, mu / t10, 0) + np.where(z11, -1 / t11, 0)
                g_rho = np.sum(w * dr)
            f = -np.sum(w * ll) + 0.5 * prec * (np.sum((att - pa) ** 2) + np.sum((de - pd_) ** 2))
            wl, wm = w * g_lam, w * g_mu
            g_att = np.bincount(hi, wl, n) + np.bincount(ai, wm, n)
            g_def = np.bincount(ai, wl, n) + np.bincount(hi, wm, n)
            grad = np.concatenate([[np.sum(wl) + np.sum(wm), np.sum(wl), g_rho], g_att, g_def])
            grad = -grad
            grad[3:3 + n] += prec * (att - pa)
            grad[3 + n:] += prec * (de - pd_)
            return f, grad

        x0 = np.concatenate([[mean_goals, 0.2, 0.0], pa.copy(), pd_.copy()])
        bounds = [(None, None), (-1, 1), (-0.2, 0.2) if dc else (0, 0)] + [(-3, 3)] * (2 * n)
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        c, h, rho, att, de = unpack(res.x)
        self.intercept, self.home_adv, self.rho = float(c), float(h), float(rho)
        self.attack, self.defence = att, de
        self._prior = {t: (pa[i], pd_[i]) for i, t in enumerate(self.teams)}
        self.fitted = True
        if not dc:
            lam = np.exp(c + h + att[hi] + de[ai])
            mu = np.exp(c + att[ai] + de[hi])
            sw = np.sum(w)
            s2 = np.sum(w * (y1 - lam) * (y2 - mu)) / np.sum(w * lam * mu)
            self.shared_var = float(np.clip(s2, 0, 0.5))
            ex = (np.sum(w * ((y1 - lam) ** 2 - lam)) + np.sum(w * ((y2 - mu) ** 2 - mu))) \
                / (np.sum(w * lam ** 2) + np.sum(w * mu ** 2))
            self.indiv_var = float(np.clip(ex - self.shared_var, 0, 0.5))
            del sw
        return self

    def _team(self, t: str) -> tuple[float, float]:
        if t in self.teams:
            i = self.teams.index(t)
            return float(self.attack[i]), float(self.defence[i])
        return -self.promoted_prior, self.promoted_prior

    def rates(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self._team(home)
        aa, da = self._team(away)
        lam = np.exp(self.intercept + self.home_adv + ah + da)
        mu = np.exp(self.intercept + aa + dh)
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        lam, mu = self.rates(home, away)
        return self.matrix(lam, mu, max_goals)

    def matrix(self, lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
        if self.dixon_coles:
            return score_matrix(lam, mu, self.rho, max_goals)
        return overdispersed_matrix(lam, mu, self.shared_var, self.indiv_var, max_goals)

    def known(self, team: str) -> bool:
        return team in self.teams

    def strength_table(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        return pd.DataFrame({
            "Equipo": self.teams,
            "Ataque": np.exp(self.attack).round(2),
            "Defensa (debilidad)": np.exp(self.defence).round(2),
            "Partidos recientes": [self.n_matches.get(t, 0) for t in self.teams],
        }).sort_values("Ataque", ascending=False).reset_index(drop=True)


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    if rho:
        m[0, 0] *= max(1 - lam * mu * rho, 0)
        m[0, 1] *= max(1 + lam * rho, 0)
        m[1, 0] *= max(1 + mu * rho, 0)
        m[1, 1] *= max(1 - rho, 0)
    return m / m.sum()


def overdispersed_matrix(lam: float, mu: float, shared_var: float, indiv_var: float,
                         max_goals: int = 25, n_quad: int = 24) -> np.ndarray:
    """Conteos sobredispersos y correlacionados: factor común G~Gamma(media 1, var shared_var) y,
    dado G, binomial negativa por equipo (var extra indiv_var). Con ambas a 0 es Poisson independiente."""
    g = np.arange(max_goals + 1)

    def pmf(m):
        if indiv_var < 1e-6:
            return poisson.pmf(g, m)
        r = 1 / indiv_var
        return nbinom.pmf(g, r, r / (r + m))

    if shared_var < 1e-6:
        mat = np.outer(pmf(lam), pmf(mu))
    else:
        k = 1 / shared_var
        qs = gamma.ppf((np.arange(n_quad) + 0.5) / n_quad, k, scale=1 / k)
        qs = qs / qs.mean()
        mat = sum(np.outer(pmf(lam * q), pmf(mu * q)) for q in qs) / n_quad
    return mat / mat.sum()


def fit_from_df(df: pd.DataFrame, hcol: str, acol: str, ref_date=None, dixon_coles: bool = True,
                min_rows: int = 60) -> PoissonTeamModel | None:
    sub = df.dropna(subset=[hcol, acol]) if {hcol, acol}.issubset(df.columns) else df.iloc[0:0]
    if len(sub) < min_rows:
        return None
    m = PoissonTeamModel(dixon_coles=dixon_coles)
    m.fit(sub["HomeTeam"], sub["AwayTeam"], sub[hcol], sub[acol], sub["Date"], ref_date)
    return m if m.fitted else None
