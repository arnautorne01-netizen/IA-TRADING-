// Motor estadístico: Dixon-Coles con decaimiento temporal + frecuencias históricas.
// Columnas de cada partido: [día, local, visitante, FTHG, FTAG, HTHG, HTAG, HC, AC, HK, AK]
(function (G) {
  const CFG = { xi: 0.0019, priorSd: 0.30, promoted: 0.15, blendM: 100, h2hW: 0.5, nRecent: 10, edge: 0.03 };
  const SCOPES = { ft: [3, 4, true, 10], ht: [5, 6, true, 7], corners: [7, 8, false, 25], cards: [9, 10, false, 14] };
  const SCOPE_NAMES = { ft: "Goles", ht: "1ª parte", corners: "Córners", cards: "Tarjetas" };

  // ---- mercados -------------------------------------------------------------
  const M = [];
  const add = (id, label, family, scope, fn) => M.push({ id, label, family, scope, fn });
  add("1", "Gana el local", "1x2", "ft", (h, a) => h > a);
  add("X", "Empate", "1x2", "ft", (h, a) => h === a);
  add("2", "Gana el visitante", "1x2", "ft", (h, a) => h < a);
  add("1X", "Doble oportunidad 1X", "doble", "ft", (h, a) => h >= a);
  add("X2", "Doble oportunidad X2", "doble", "ft", (h, a) => h <= a);
  add("12", "Doble oportunidad 12 (sin empate)", "doble", "ft", (h, a) => h !== a);
  add("BTTS_SI", "Ambos marcan: Sí", "btts", "ft", (h, a) => h > 0 && a > 0);
  add("BTTS_NO", "Ambos marcan: No", "btts", "ft", (h, a) => h === 0 || a === 0);
  add("L_MARCA", "El local marca", "local_marca", "ft", (h) => h > 0);
  add("L_O1.5", "El local marca 2 o más", "local_marca", "ft", (h) => h >= 2);
  add("V_MARCA", "El visitante marca", "visit_marca", "ft", (h, a) => a > 0);
  add("V_O1.5", "El visitante marca 2 o más", "visit_marca", "ft", (h, a) => a >= 2);
  add("L_U1.5", "El local marca menos de 2", "local_menos", "ft", (h) => h <= 1);
  add("V_U1.5", "El visitante marca menos de 2", "visit_menos", "ft", (h, a) => a <= 1);
  add("1X_O1.5", "1X y más de 1.5 goles", "combo_1x", "ft", (h, a) => h >= a && h + a >= 2);
  add("1X_U3.5", "1X y menos de 3.5 goles", "combo_1x", "ft", (h, a) => h >= a && h + a <= 3);
  add("X2_O1.5", "X2 y más de 1.5 goles", "combo_x2", "ft", (h, a) => h <= a && h + a >= 2);
  add("X2_U3.5", "X2 y menos de 3.5 goles", "combo_x2", "ft", (h, a) => h <= a && h + a <= 3);
  add("1_O1.5", "Gana el local y más de 1.5 goles", "combo_1", "ft", (h, a) => h > a && h + a >= 2);
  add("2_O1.5", "Gana el visitante y más de 1.5 goles", "combo_2", "ft", (h, a) => h < a && h + a >= 2);
  add("HT_O0.5", "Gol en la 1ª parte", "ht_over", "ht", (h, a) => h + a >= 1);
  add("HT_O1.5", "Más de 1.5 goles en la 1ª parte", "ht_over", "ht", (h, a) => h + a >= 2);
  add("HT_U1.5", "Menos de 1.5 goles en la 1ª parte", "ht_under", "ht", (h, a) => h + a <= 1);
  add("HT_1X", "Descanso: 1X", "ht_doble", "ht", (h, a) => h >= a);
  add("HT_X2", "Descanso: X2", "ht_doble", "ht", (h, a) => h <= a);
  add("HT_X", "Descanso: empate", "ht_1x2", "ht", (h, a) => h === a);
  add("C_L_MAS", "Más córners el local", "c_1x2", "corners", (h, a) => h > a);
  add("C_V_MAS", "Más córners el visitante", "c_1x2", "corners", (h, a) => h < a);
  add("K_L_MAS", "Más tarjetas el local", "k_1x2", "cards", (h, a) => h > a);
  add("K_V_MAS", "Más tarjetas el visitante", "k_1x2", "cards", (h, a) => h < a);
  for (const x of [0.5, 1.5, 2.5, 3.5, 4.5]) add("O" + x, `Más de ${x} goles`, "goles_over", "ft", (h, a) => h + a > x);
  for (const x of [1.5, 2.5, 3.5, 4.5]) add("U" + x, `Menos de ${x} goles`, "goles_under", "ft", (h, a) => h + a < x);
  for (const x of [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]) add("C_O" + x, `Más de ${x} córners`, "c_over", "corners", (h, a) => h + a > x);
  for (const x of [8.5, 9.5, 10.5, 11.5, 12.5]) add("C_U" + x, `Menos de ${x} córners`, "c_under", "corners", (h, a) => h + a < x);
  for (const x of [2.5, 3.5, 4.5, 5.5]) {
    add("C_L_O" + x, `Local más de ${x} córners`, "c_local", "corners", (h) => h > x);
    add("C_V_O" + x, `Visitante más de ${x} córners`, "c_visit", "corners", (h, a) => a > x);
  }
  for (const x of [1.5, 2.5, 3.5, 4.5, 5.5]) add("K_O" + x, `Más de ${x} tarjetas`, "k_over", "cards", (h, a) => h + a > x);
  for (const x of [3.5, 4.5, 5.5, 6.5]) add("K_U" + x, `Menos de ${x} tarjetas`, "k_under", "cards", (h, a) => h + a < x);
  for (const x of [0.5, 1.5, 2.5]) {
    add("K_L_O" + x, `Local más de ${x} tarjetas`, "k_local", "cards", (h) => h > x);
    add("K_V_O" + x, `Visitante más de ${x} tarjetas`, "k_visit", "cards", (h, a) => a > x);
  }
  const BOOK = { "1": "H", "X": "D", "2": "A", "O2.5": "O25", "U2.5": "U25" };

  // ---- ajuste del modelo -----------------------------------------------------
  function fit(rows, nTeams, scope, refDay) {
    const [hc, ac, dc] = SCOPES[scope];
    const R = rows.filter((r) => r[hc] >= 0 && r[ac] >= 0 && r[0] < refDay);
    if (R.length < 60) return null;
    const n = nTeams, N = R.length;
    const hi = new Int32Array(N), ai = new Int32Array(N), y1 = new Float64Array(N), y2 = new Float64Array(N), w = new Float64Array(N);
    let minDay = Infinity;
    for (const r of R) minDay = Math.min(minDay, r[0]);
    const early = new Uint8Array(n), seen = new Float64Array(n);
    R.forEach((r, i) => {
      hi[i] = r[1]; ai[i] = r[2]; y1[i] = r[hc]; y2[i] = r[ac];
      w[i] = Math.exp(-CFG.xi * Math.max(0, refDay - r[0]));
      if (r[0] <= minDay + 200) { early[r[1]] = 1; early[r[2]] = 1; }
      if (w[i] > 0.5) { seen[r[1]]++; seen[r[2]]++; }
    });
    const present = new Uint8Array(n);
    R.forEach((r) => { present[r[1]] = 1; present[r[2]] = 1; });
    const pa = new Float64Array(n), pd = new Float64Array(n);
    for (let t = 0; t < n; t++) if (!early[t]) { pa[t] = -CFG.promoted; pd[t] = CFG.promoted; }
    const prec = 1 / CFG.priorSd ** 2;
    let sw = 0, sy = 0;
    for (let i = 0; i < N; i++) { sw += w[i]; sy += w[i] * (y1[i] + y2[i]); }
    const P = 3 + 2 * n;
    const x = new Float64Array(P);
    x[0] = Math.log(Math.max(sy / sw / 2, 0.05)); x[1] = 0.2; x[2] = 0;
    for (let t = 0; t < n; t++) { x[3 + t] = pa[t]; x[3 + n + t] = pd[t]; }

    function grad(x, g) {
      g.fill(0);
      const c = x[0], h = x[1], rho = dc ? x[2] : 0;
      let f = 0;
      for (let i = 0; i < N; i++) {
        const ll = c + h + x[3 + hi[i]] + x[3 + n + ai[i]], lm = c + x[3 + ai[i]] + x[3 + n + hi[i]];
        const lam = Math.exp(ll), mu = Math.exp(lm);
        let gl = y1[i] - lam, gm = y2[i] - mu, li = y1[i] * ll - lam + y2[i] * lm - mu;
        if (dc) {
          const a = y1[i], b = y2[i];
          if (a === 0 && b === 0) { const t = Math.max(1 - lam * mu * rho, 1e-6); li += Math.log(t); gl += -lam * mu * rho / t; gm += -lam * mu * rho / t; g[2] += w[i] * (-lam * mu / t); }
          else if (a === 0 && b === 1) { const t = Math.max(1 + lam * rho, 1e-6); li += Math.log(t); gl += lam * rho / t; g[2] += w[i] * (lam / t); }
          else if (a === 1 && b === 0) { const t = Math.max(1 + mu * rho, 1e-6); li += Math.log(t); gm += mu * rho / t; g[2] += w[i] * (mu / t); }
          else if (a === 1 && b === 1) { const t = Math.max(1 - rho, 1e-6); li += Math.log(t); g[2] += w[i] * (-1 / t); }
        }
        f -= w[i] * li;
        const wl = w[i] * gl, wm = w[i] * gm;
        g[0] += wl + wm; g[1] += wl;
        g[3 + hi[i]] += wl; g[3 + n + ai[i]] += wl; g[3 + ai[i]] += wm; g[3 + n + hi[i]] += wm;
      }
      for (let k = 0; k < 3; k++) g[k] = -g[k];
      if (!dc) g[2] = 0;
      for (let t = 0; t < n; t++) {
        const da = x[3 + t] - pa[t], dd = x[3 + n + t] - pd[t];
        f += 0.5 * prec * (da * da + dd * dd);
        g[3 + t] = -g[3 + t] + prec * da;
        g[3 + n + t] = -g[3 + n + t] + prec * dd;
      }
      return f;
    }
    lbfgs(grad, x, dc ? [[2, -0.2, 0.2]] : [[2, 0, 0]], 400);

    const m = { scope, dc, c: x[0], h: x[1], rho: dc ? x[2] : 0, att: x.slice(3, 3 + n), def: x.slice(3 + n), seen, present, sv: 0, iv: 0 };
    if (!dc) {
      let s1 = 0, s2 = 0, e1 = 0, e2 = 0;
      for (let i = 0; i < N; i++) {
        const [lam, mu] = rates(m, hi[i], ai[i]);
        s1 += w[i] * (y1[i] - lam) * (y2[i] - mu); s2 += w[i] * lam * mu;
        e1 += w[i] * ((y1[i] - lam) ** 2 - lam) + w[i] * ((y2[i] - mu) ** 2 - mu);
        e2 += w[i] * lam * lam + w[i] * mu * mu;
      }
      m.sv = Math.min(Math.max(s1 / s2, 0), 0.5);
      m.iv = Math.min(Math.max(e1 / e2 - m.sv, 0), 0.5);
    }
    return m;
  }

  // L-BFGS con búsqueda lineal de Armijo; `fixed` = [índice, mín, máx] acotados por proyección
  function lbfgs(fg, x, bounds, maxIter) {
    const P = x.length, mem = 8, S = [], Y = [];
    const clamp = (v) => { for (const [k, lo, hi] of bounds) v[k] = Math.min(hi, Math.max(lo, v[k])); };
    let g = new Float64Array(P), f = fg(x, g);
    for (let it = 0; it < maxIter; it++) {
      const q = Float64Array.from(g), al = [];
      for (let k = S.length - 1; k >= 0; k--) {
        const rho = 1 / dot(Y[k], S[k]); const a = rho * dot(S[k], q); al[k] = [a, rho];
        for (let j = 0; j < P; j++) q[j] -= a * Y[k][j];
      }
      if (S.length) { const gm = dot(S[S.length - 1], Y[Y.length - 1]) / dot(Y[Y.length - 1], Y[Y.length - 1]); for (let j = 0; j < P; j++) q[j] *= gm; }
      else { const s = 1 / Math.max(1, Math.sqrt(dot(g, g))); for (let j = 0; j < P; j++) q[j] *= s; }
      for (let k = 0; k < S.length; k++) {
        const b = al[k][1] * dot(Y[k], q);
        for (let j = 0; j < P; j++) q[j] += S[k][j] * (al[k][0] - b);
      }
      let gd = -dot(g, q);
      if (gd >= 0) { S.length = 0; Y.length = 0; for (let j = 0; j < P; j++) q[j] = g[j] / Math.max(1, Math.sqrt(dot(g, g))); gd = -dot(g, q); }
      let step = 1, xn, gn = new Float64Array(P), fn;
      for (let ls = 0; ls < 30; ls++) {
        xn = Float64Array.from(x, (v, j) => v - step * q[j]); clamp(xn);
        fn = fg(xn, gn);
        if (fn <= f + 1e-4 * step * gd) break;
        step *= 0.5;
      }
      const s = xn.map((v, j) => v - x[j]), y = gn.map((v, j) => v - g[j]);
      if (dot(s, y) > 1e-10) { S.push(s); Y.push(y); if (S.length > mem) { S.shift(); Y.shift(); } }
      const df = f - fn;
      x.set(xn); g = gn; f = fn;
      if (Math.abs(df) < 1e-9 * Math.max(1, Math.abs(f)) || Math.sqrt(dot(g, g)) < 1e-5) break;
    }
    return f;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  function teamParams(m, t) {
    if (t != null && t >= 0 && t < m.att.length && m.present[t]) return [m.att[t], m.def[t]];
    return [-CFG.promoted, CFG.promoted];
  }
  function rates(m, h, a) {
    const [ah, dh] = teamParams(m, h), [aa, da] = teamParams(m, a);
    return [Math.exp(m.c + m.h + ah + da), Math.exp(m.c + aa + dh)];
  }

  // ---- matrices de probabilidad -----------------------------------------------
  function pmf(mean, max, iv) {
    const p = new Float64Array(max + 1);
    if (iv < 1e-6) { p[0] = Math.exp(-mean); for (let k = 1; k <= max; k++) p[k] = p[k - 1] * mean / k; }
    else { const r = 1 / iv, q = mean / (r + mean); p[0] = Math.pow(r / (r + mean), r); for (let k = 1; k <= max; k++) p[k] = p[k - 1] * (k - 1 + r) / k * q; }
    return p;
  }
  // Gauss-Hermite (10 nodos) para el factor común lognormal del partido
  const GH_X = [-3.4361591188, -2.5327316742, -1.7566836493, -1.0366108298, -0.3429013272, 0.3429013272, 1.0366108298, 1.7566836493, 2.5327316742, 3.4361591188];
  const GH_W = [7.640432855e-6, 1.343645747e-3, 3.387439446e-2, 0.2401386111, 0.6108626337, 0.6108626337, 0.2401386111, 3.387439446e-2, 1.343645747e-3, 7.640432855e-6];
  function matrix(m, lam, mu, max) {
    const mat = Array.from({ length: max + 1 }, () => new Float64Array(max + 1));
    const addOuter = (a, b, wt) => { for (let i = 0; i <= max; i++) for (let j = 0; j <= max; j++) mat[i][j] += wt * a[i] * b[j]; };
    if (m.dc) {
      addOuter(pmf(lam, max, 0), pmf(mu, max, 0), 1);
      const r = m.rho;
      mat[0][0] *= Math.max(1 - lam * mu * r, 0); mat[0][1] *= Math.max(1 + lam * r, 0);
      mat[1][0] *= Math.max(1 + mu * r, 0); mat[1][1] *= Math.max(1 - r, 0);
    } else if (m.sv < 1e-6) addOuter(pmf(lam, max, m.iv), pmf(mu, max, m.iv), 1);
    else {
      const s = Math.sqrt(Math.log(1 + m.sv));
      for (let k = 0; k < 10; k++) {
        const gq = Math.exp(Math.SQRT2 * s * GH_X[k] - s * s / 2), wt = GH_W[k] / Math.sqrt(Math.PI);
        addOuter(pmf(lam * gq, max, m.iv), pmf(mu * gq, max, m.iv), wt);
      }
    }
    let tot = 0; for (const row of mat) for (const v of row) tot += v;
    for (const row of mat) for (let j = 0; j <= max; j++) row[j] /= tot;
    return mat;
  }

  // ---- liga y pronóstico --------------------------------------------------------
  function League(code, teams, rows, refDay) {
    this.code = code; this.teams = teams; this.rows = rows.slice().sort((a, b) => a[0] - b[0]);
    this.refDay = refDay ?? (this.rows.length ? this.rows[this.rows.length - 1][0] + 1 : 0);
    this.models = {};
    for (const sc of Object.keys(SCOPES)) { const m = fit(this.rows, teams.length, sc, this.refDay); if (m) this.models[sc] = m; }
    const recentFrom = this.refDay - 300, cur = new Set();
    for (const r of this.rows) if (r[0] >= recentFrom) { cur.add(r[1]); cur.add(r[2]); }
    this.current = [...cur].map((i) => teams[i]).sort((a, b) => a.localeCompare(b, "es"));
  }
  League.prototype.idx = function (name) {
    let i = this.teams.indexOf(name);
    if (i >= 0) return i;
    const low = name.toLowerCase();
    i = this.teams.findIndex((t) => t.toLowerCase() === low || t.toLowerCase().includes(low) || low.includes(t.toLowerCase()));
    return i;
  };
  League.prototype.predict = function (homeName, awayName, odds) {
    const h = this.idx(homeName), a = this.idx(awayName);
    const home = h >= 0 ? this.teams[h] : homeName, away = a >= 0 ? this.teams[a] : awayName;
    const warnings = [];
    const ft = this.models.ft;
    for (const [t, i] of [[home, h], [away, a]]) {
      if (i < 0 || (ft && !ft.present[i])) warnings.push(`${t}: sin histórico en esta liga, se trata como recién ascendido.`);
      else if (ft && ft.seen[i] < 10) warnings.push(`${t}: pocos partidos recientes, la estimación es menos fiable.`);
    }
    const homeRows = this.rows.filter((r) => r[1] === h), awayRows = this.rows.filter((r) => r[2] === a);
    const h2hRows = this.rows.filter((r) => (r[1] === h && r[2] === a) || (r[1] === a && r[2] === h));
    const expected = {}, probs = {}, ev = {};
    let topScores = [];
    for (const [sc, m] of Object.entries(this.models)) {
      const [hc, ac, , max] = SCOPES[sc];
      const [lam, mu] = rates(m, h, a);
      expected[sc] = [lam, mu];
      const mat = matrix(m, lam, mu, max);
      for (const mk of M) if (mk.scope === sc) {
        let p = 0; for (let i = 0; i <= max; i++) for (let j = 0; j <= max; j++) if (mk.fn(i, j)) p += mat[i][j];
        probs[mk.id] = p;
      }
      if (sc === "ft") {
        const all = []; for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) all.push([`${i}-${j}`, mat[i][j]]);
        topScores = all.sort((x, y) => y[1] - x[1]).slice(0, 5);
      }
      const pick = (rs, recent) => { const v = rs.filter((r) => r[hc] >= 0 && r[ac] >= 0); return recent ? v.slice(-CFG.nRecent) : v; };
      ev[sc] = { home: pick(homeRows, true), away: pick(awayRows, true), h2h: pick(h2hRows, false), hc, ac };
    }
    const rows = [];
    for (const mk of M) {
      if (!(mk.id in probs)) continue;
      const e = ev[mk.scope], k = {}, n = {};
      for (const src of ["home", "away", "h2h"]) {
        n[src] = e[src].length;
        k[src] = e[src].reduce((s, r) => { const flip = r[1] !== h; return s + (mk.fn(flip ? r[e.ac] : r[e.hc], flip ? r[e.hc] : r[e.ac]) ? 1 : 0); }, 0);
      }
      const K = k.home + k.away + CFG.h2hW * k.h2h, N = n.home + n.away + CFG.h2hW * n.h2h;
      const pm = probs[mk.id], pf = (K + CFG.blendM * pm) / (N + CFG.blendM);
      const book = odds && BOOK[mk.id] && odds[BOOK[mk.id]] > 1 ? odds[BOOK[mk.id]] : null;
      rows.push({ id: mk.id, label: mk.label, family: mk.family, scope: mk.scope, pModel: pm, pHist: N ? K / N : null,
        p: pf, k, n, nHist: N, fair: 1 / pf, min: (1 + CFG.edge) / pf, book, value: book ? pf * book - 1 : null });
    }
    return { league: this.code, home, away, expected, topScores, rows, warnings,
      sample: { home: homeRows.slice(-CFG.nRecent).length, away: awayRows.slice(-CFG.nRecent).length, h2h: h2hRows.length } };
  };

  function selectPicks(rows, minP, minO, maxO, maxPicks = 5) {
    let c = rows.filter((r) => r.p >= minP && r.fair >= minO && r.fair <= maxO);
    c = c.filter((r) => r.nHist < 6 || (r.pHist ?? 0) >= minP - 0.07);
    c = c.filter((r) => r.value == null || r.value > -0.02);
    const best = {};
    for (const r of c) if (!best[r.family] || r.p < best[r.family].p) best[r.family] = r;
    return Object.values(best).map((r) => ({ ...r, conf: r.p >= 0.72 && (r.pHist ?? 0) >= 0.7 ? "Alta" : "Media" }))
      .sort((a, b) => b.p - a.p).slice(0, maxPicks);
  }

  function buildCombo(preds, minOdds = 1.8, maxOdds = 3.2, maxLegs = 4) {
    const legs = preds.filter((p) => p.picks.length).map((p) => ({ match: `${p.home} - ${p.away}`, label: p.picks[0].label, p: p.picks[0].p }))
      .sort((a, b) => b.p - a.p);
    const chosen = []; let prob = 1;
    for (const l of legs) {
      if (chosen.length >= maxLegs) break;
      if (1 / (prob * l.p) > maxOdds) continue;
      chosen.push(l); prob *= l.p;
      if (1 / prob >= minOdds) break;
    }
    return chosen.length >= 2 ? { legs: chosen, prob, fair: 1 / prob } : null;
  }

  // ---- generador de combinadas del día ----------------------------------------------
  // Patas candidatas: una por partido como máximo (si no, las patas dependen entre sí y la
  // probabilidad multiplicada sería falsa). Cuota de cada pata: la de la casa si la hay, si no la justa.
  function legPool(preds, perMatch = 6, minFair = 1.05) {
    return preds.map((p, mi) => {
      const seen = {};
      return p.rows
        .filter((r) => r.p >= 0.5 && r.fair >= minFair && r.fair <= 2.6)
        .filter((r) => r.nHist < 6 || (r.pHist ?? 0) >= r.p - 0.1)
        .filter((r) => r.value == null || r.value > -0.02)
        .sort((a, b) => b.p - a.p)
        .filter((r) => (seen[r.family] = (seen[r.family] || 0) + 1) <= 2)
        .slice(0, perMatch)
        .map((r) => ({ mi, match: `${p.home} - ${p.away}`, label: r.label, p: r.p, fair: r.fair, odds: r.book || r.fair, book: !!r.book, conf: r.p >= 0.72 && (r.pHist ?? 0) >= 0.7 }));
    }).filter((l) => l.length);
  }
  const comboOf = (legs) => {
    const prob = legs.reduce((s, l) => s * l.p, 1), odds = legs.reduce((s, l) => s * l.odds, 1);
    return { legs, prob, odds, fair: 1 / prob, ev: prob * odds - 1, anyBook: legs.some((l) => l.book) };
  };
  const keyOf = (c) => c.legs.map((l) => l.mi + ":" + l.label).sort().join("|");
  function distinct(list, n, maxOverlap = 0.5) {
    const out = [];
    for (const c of list) {
      const ks = new Set(c.legs.map((l) => l.mi + ":" + l.label));
      const clash = out.some((o) => { const shared = o.legs.filter((l) => ks.has(l.mi + ":" + l.label)).length; return shared / Math.min(ks.size, o.legs.length) > maxOverlap; });
      if (!clash) out.push(c);
      if (out.length >= n) break;
    }
    return out;
  }
  // Poda del haz: en cada longitud, los estados que más avanzan hacia la cuota objetivo,
  // premiando el valor frente a la casa y las patas con el histórico de acuerdo
  function keepPerLength(states, k) {
    const by = {};
    for (const s of states) (by[s.length] ||= []).push(s);
    const sc = (s) => s.reduce((x, y) => x + Math.log(y.odds) + 2 * Math.log(y.p * y.odds) + (y.conf ? 0.03 : 0), 0);
    return Object.values(by).flatMap((g) => g.sort((a, b) => sc(b) - sc(a)).slice(0, Math.ceil(k)));
  }
  // Combinadas con cuota >= minOdds, ordenadas por probabilidad (la más alta posible)
  function combosForOdds(preds, minOdds = 3, n = 10, maxLegs = 7, beam = 600) {
    const pool = legPool(preds, 6, 1.2);
    let states = [[]];
    const done = new Map();
    for (const legs of pool) {
      const next = [];
      for (const s of states) {
        next.push(s);
        for (const l of legs) {
          const c = s.concat(l);
          const odds = c.reduce((x, y) => x * y.odds, 1);
          if (odds >= minOdds) { const cb = comboOf(c); done.set(keyOf(cb), cb); }
          else if (c.length < maxLegs) next.push(c);
        }
      }
      states = keepPerLength(next, beam / maxLegs);
    }
    const score = (c) => (c.anyBook ? Math.max(c.prob, c.prob * (1 + c.ev)) : c.prob) - 0.012 * c.legs.length;
    return distinct([...done.values()].sort((a, b) => score(b) - score(a)), n);
  }
  // Escalera: para cada probabilidad objetivo, la combinada con más cuota que no baja de ella
  function comboLadder(preds, targets = [0.9, 0.8, 0.7, 0.6, 0.5], beam = 600) {
    const pool = legPool(preds, 8), used = new Set();
    return targets.map((q) => {
      let states = [[]];
      const cands = [];
      for (const legs of pool) {
        const next = [];
        for (const s of states) {
          next.push(s);
          for (const l of legs) {
            const c = s.concat(l), prob = c.reduce((x, y) => x * y.p, 1);
            if (prob >= q) { next.push(c); cands.push(c); }
          }
        }
        states = next.sort((a, b) => b.reduce((x, y) => x * y.odds, 1) - a.reduce((x, y) => x * y.odds, 1)).slice(0, beam);
      }
      const best = cands.map(comboOf).sort((a, b) => b.odds - a.odds).find((c) => !used.has(keyOf(c)));
      if (best) used.add(keyOf(best));
      return { target: q, combo: best || null };
    });
  }

  G.Engine = { CFG, SCOPES, SCOPE_NAMES, MARKETS: M, League, selectPicks, buildCombo, combosForOdds, comboLadder, fit, rates };
})(typeof window !== "undefined" ? window : globalThis);
