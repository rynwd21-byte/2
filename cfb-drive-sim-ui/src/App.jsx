
import React, { useEffect, useMemo, useState } from "react";

const SUMMARY_SEASONS = [2019, 2021, 2022, 2023, 2024];

function App() {
  // ------------- API URL & health -------------------------------------------
  const [apiUrl, setApiUrl] = useState(() => {
    if (typeof window === "undefined") return "";
    try {
      return localStorage.getItem("cfbApiUrl") || "";
    } catch {
      return "";
    }
  });
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [checkingHealth, setCheckingHealth] = useState(false);

  const baseUrl = useMemo(() => {
    if (!apiUrl) return null;
    try {
      return new URL(apiUrl);
    } catch {
      return null;
    }
  }, [apiUrl]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      if (apiUrl) {
        localStorage.setItem("cfbApiUrl", apiUrl);
      }
    } catch {
      // ignore
    }
  }, [apiUrl]);

  async function checkHealth() {
    if (!baseUrl) return;
    setCheckingHealth(true);
    setHealth(null);
    setHealthError(null);
    try {
      const u = new URL("/health", baseUrl);
      const res = await fetch(u.toString());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      setHealth(j);
    } catch (e) {
      setHealthError(String(e));
    } finally {
      setCheckingHealth(false);
    }
  }

  useEffect(() => {
    if (baseUrl) {
      checkHealth();
    }
  }, [baseUrl]);

  // ------------- Dataset summary (for ingestion/vegas sanity) ---------------
  const [summary, setSummary] = useState({});
  const [busySummary, setBusySummary] = useState(false);

  // ------------- Bootstrap (one-click setup) --------------------------------
  const [bootStart, setBootStart] = useState(2019);
  const [bootEnd, setBootEnd] = useState(2024);
  const [bootIncludeMarket, setBootIncludeMarket] = useState(true);
  const [bootSeedRatings, setBootSeedRatings] = useState(true);
  const [bootCalibrateMulti, setBootCalibrateMulti] = useState(true);
  const [bootForce, setBootForce] = useState(false);
  const [bootBusy, setBootBusy] = useState(false);
  const [bootResult, setBootResult] = useState(null);
  const [bootError, setBootError] = useState(null);

  // ------------- Backtest (shipping gate) ----------------------------------
  const [btSeason, setBtSeason] = useState(2023);
  const [btGames, setBtGames] = useState(200);
  const [btSims, setBtSims] = useState(200);
  const [btBusy, setBtBusy] = useState(false);
  const [btResult, setBtResult] = useState(null);
  const [btBaseline, setBtBaseline] = useState(null);
  const [btError, setBtError] = useState(null);

  async function fetchBaseline() {
    if (!baseUrl) return;
    try {
      const u = new URL("/backtest/baseline", baseUrl);
      u.searchParams.set("test_season", String(btSeason));
      const res = await fetch(u.toString());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      setBtBaseline(j?.baseline || null);
    } catch (e) {
      // Baseline is optional; don't hard-fail the page.
      setBtBaseline(null);
    }
  }

  useEffect(() => {
    if (baseUrl) fetchBaseline();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, btSeason]);

  async function runBacktest(endpoint) {
    if (!baseUrl) return;
    setBtBusy(true);
    setBtError(null);
    setBtResult(null);
    try {
      const u = new URL(endpoint, baseUrl);
      const payload = {
        test_season: Number.parseInt(btSeason, 10) || 2023,
        n_games: Number.parseInt(btGames, 10) || 25,
        n_sims: Number.parseInt(btSims, 10) || 25,
        seed: 1337,
        label: endpoint.includes("set-baseline") ? `baseline ${btSeason}` : "candidate",
      };
      const res = await fetch(u.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await res.text();
      let j;
      try {
        j = text ? JSON.parse(text) : null;
      } catch {
        j = { raw: text };
      }
      if (!res.ok) {
        const msg = j?.detail || j?.error || `HTTP ${res.status}`;
        throw new Error(String(msg));
      }
      setBtResult(j);
      await fetchBaseline();
    } catch (e) {
      setBtError(String(e));
    } finally {
      setBtBusy(false);
    }
  }

  async function runBootstrap() {
    if (!baseUrl) return;
    setBootBusy(true);
    setBootResult(null);
    setBootError(null);
    try {
      const u = new URL("/bootstrap", baseUrl);
      u.searchParams.set("start_season", String(bootStart));
      u.searchParams.set("end_season", String(bootEnd));
      u.searchParams.set("include_market_lines", String(bootIncludeMarket));
      u.searchParams.set("seed_ratings", String(bootSeedRatings));
      u.searchParams.set("calibrate_multi", String(bootCalibrateMulti));
      u.searchParams.set("force", String(bootForce));

      const res = await fetch(u.toString(), { method: "POST" });
      const text = await res.text();
      let j;
      try {
        j = text ? JSON.parse(text) : null;
      } catch {
        j = { raw: text };
      }
      if (!res.ok) {
        const msg = j?.detail || j?.error || `HTTP ${res.status}`;
        throw new Error(String(msg));
      }
      setBootResult(j);

      // Refresh the visible UI state after bootstrap.
      await refreshSummary();
      await loadTeams();
    } catch (e) {
      setBootError(String(e));
    } finally {
      setBootBusy(false);
    }
  }

  async function refreshSummary() {
    if (!baseUrl) return;
    setBusySummary(true);
    const next = {};
    try {
      for (const season of SUMMARY_SEASONS) {
        const u = new URL("/games", baseUrl);
        u.searchParams.set("season", String(season));
        try {
          const res = await fetch(u.toString());
          if (!res.ok) {
            next[season] = {
              count: 0,
              withLines: 0,
              postseason: 0,
              status: `HTTP ${res.status}`,
            };
            continue;
          }
          const j = await res.json();
          const games = Array.isArray(j) ? j : [];
          const count = games.length;
          const withLines = games.filter(
            (g) =>
              g.closing_spread != null ||
              g.closing_total != null
          ).length;
          const postseason = games.filter(
            (g) => g.season_type === "postseason"
          ).length;
          next[season] = { count, withLines, postseason, status: "OK" };
        } catch (e) {
          next[season] = {
            count: 0,
            withLines: 0,
            postseason: 0,
            status: String(e),
          };
        }
      }
    } finally {
      setSummary(next);
      setBusySummary(false);
    }
  }

  useEffect(() => {
    if (baseUrl) {
      refreshSummary();
    }
  }, [baseUrl]);

  // ------------- Teams for future simulations -------------------------------
  const [teams, setTeams] = useState([]);
  const [teamsError, setTeamsError] = useState(null);
  const [loadingTeams, setLoadingTeams] = useState(false);

  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");

  async function loadTeams() {
    if (!baseUrl) return;
    setTeamsError(null);
    setLoadingTeams(true);
    try {
      // First try: use /teams/search to get all known teams
      let res;
      try {
        const u = new URL("/teams/search", baseUrl);
        // empty q = all teams
        u.searchParams.set("q", "");
        res = await fetch(u.toString());
      } catch (e) {
        res = null;
      }

      let arr = [];
      if (res && res.ok) {
        arr = await res.json();
      } else {
        // If that somehow fails, leave arr empty; user can still hit the API manually.
        arr = [];
      }

      if (!Array.isArray(arr)) arr = [];
      arr.sort((a, b) =>
        String(a.name || "").localeCompare(String(b.name || ""))
      );
      setTeams(arr);
      if (arr.length >= 2) {
        setHomeTeam(arr[0].name);
        setAwayTeam(arr[1].name);
      }
    } catch (e) {
      setTeamsError(String(e));
    } finally {
      setLoadingTeams(false);
    }
  }

  useEffect(() => {
    if (baseUrl) {
      loadTeams();
    }
  }, [baseUrl]);

  // ------------- Matchup simulation ----------------------------------------
  const [simBusy, setSimBusy] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simError, setSimError] = useState(null);

  async function runSimulation() {
    if (!baseUrl || !homeTeam || !awayTeam) return;
    setSimBusy(true);
    setSimResult(null);
    setSimError(null);
    try {
      const u = new URL("/simulate-series-by-name", baseUrl);
      const body = {
        home_name: homeTeam,
        away_name: awayTeam,
        n: 1000,
      };
      const res = await fetch(u.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      setSimResult(j);
    } catch (e) {
      setSimError(String(e));
    } finally {
      setSimBusy(false);
    }
  }

  const seriesSummary = useMemo(() => {
    if (!simResult) return null;
    // Support both legacy and newer response shapes.
    const s = simResult.series || simResult.summary || simResult;
    const homeWinPct =
      typeof s.home_win_pct === "number"
        ? s.home_win_pct
        : typeof s.homeWinPct === "number"
          ? s.homeWinPct
          : null;
    const awayWinPct =
      typeof s.away_win_pct === "number"
        ? s.away_win_pct
        : typeof s.awayWinPct === "number"
          ? s.awayWinPct
          : null;
    const otRate =
      typeof s.ot_rate === "number"
        ? s.ot_rate
        : typeof s.otRate === "number"
          ? s.otRate
          : null;
    const meanTotal =
      typeof s.mean_total === "number"
        ? s.mean_total
        : typeof s.meanTotal === "number"
          ? s.meanTotal
          : typeof simResult.mean_total === "number"
            ? simResult.mean_total
            : null;

    if (homeWinPct == null && awayWinPct == null && otRate == null && meanTotal == null) {
      return null;
    }

    return { homeWinPct, awayWinPct, otRate, meanTotal };
  }, [simResult]);

  const marketCompare = useMemo(() => {
    if (!simResult) return null;
    const market = simResult.market || null;
    const blended = simResult.blended || null;
    const edge = simResult.edge || null;
    const modelSpread = typeof simResult.expected_spread === "number" ? simResult.expected_spread : null;
    const modelTotal = typeof simResult.mean_total === "number" ? simResult.mean_total : null;
    return { market, blended, edge, modelSpread, modelTotal };
  }, [simResult]);

  const homeStats = useMemo(() => {
    if (!simResult) return null;
    const h = simResult.home || simResult.home_stats || null;
    if (!h) return null;
    return {
      name: h.name,
      avgPts: h.avg_pts ?? h.avgPts,
      p05: h.p05,
      p50: h.p50,
      p95: h.p95,
    };
  }, [simResult]);

  const awayStats = useMemo(() => {
    if (!simResult) return null;
    const a = simResult.away || simResult.away_stats || null;
    if (!a) return null;
    return {
      name: a.name,
      avgPts: a.avg_pts ?? a.avgPts,
      p05: a.p05,
      p50: a.p50,
      p95: a.p95,
    };
  }, [simResult]);

  // ------------- Render -----------------------------------------------------
  return (
    <div
      style={{
        fontFamily:
          'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        padding: "24px",
        maxWidth: "1100px",
        margin: "0 auto",
        color: "#111827",
      }}
    >
      <h1 style={{ fontSize: "26px", fontWeight: 700, marginBottom: "12px" }}>
        CFB Drive Sim — Monte Carlo UI
      </h1>

      {/* API connection */}
      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: "8px",
          padding: "12px 16px",
          marginBottom: "24px",
          backgroundColor: "#f9fafb",
        }}
      >
        <div style={{ marginBottom: "8px", fontWeight: 600 }}>
          API connection
        </div>
        <div
          style={{
            display: "flex",
            gap: "8px",
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value.trim())}
            placeholder="https://your-api-url.vercel.app"
            style={{
              flex: "1 1 260px",
              padding: "6px 8px",
              borderRadius: "4px",
              border: "1px solid #d1d5db",
              fontSize: "14px",
            }}
          />
          <button
            type="button"
            onClick={checkHealth}
            disabled={!baseUrl || checkingHealth}
            style={{
              padding: "6px 12px",
              borderRadius: "4px",
              border: "none",
              backgroundColor: baseUrl ? "#2563eb" : "#9ca3af",
              color: "white",
              fontSize: "14px",
              cursor: baseUrl ? "pointer" : "default",
            }}
          >
            {checkingHealth ? "Checking…" : "Check API"}
          </button>
          <span style={{ fontSize: "13px" }}>
            {baseUrl == null && apiUrl
              ? "Invalid URL"
              : !apiUrl
              ? "Enter your deployed API URL"
              : health
              ? "Connected"
              : healthError
              ? `Error: ${healthError}`
              : "Idle"}
          </span>
        </div>
      </section>

      {/* Matchup simulator (future games) */}
      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: "8px",
          padding: "16px",
          marginBottom: "24px",
        }}
      >
        <h2
          style={{
            fontSize: "18px",
            fontWeight: 600,
            marginBottom: "12px",
          }}
        >
          Matchup simulator (future games)
        </h2>

        {!baseUrl && (
          <div style={{ fontSize: "13px", color: "#6b7280", marginBottom: 8 }}>
            Enter your API URL above, then click <strong>Check API</strong> to
            load teams.
          </div>
        )}

        {baseUrl && (
          <>
            <div
              style={{
                display: "flex",
                gap: "16px",
                flexWrap: "wrap",
                marginBottom: "12px",
              }}
            >
              <div style={{ minWidth: "260px" }}>
                <label
                  style={{
                    display: "block",
                    fontSize: "13px",
                    marginBottom: "4px",
                  }}
                >
                  Home team
                </label>
                <select
                  value={homeTeam}
                  onChange={(e) => setHomeTeam(e.target.value)}
                  disabled={loadingTeams || teams.length === 0}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    borderRadius: "4px",
                    border: "1px solid #d1d5db",
                    fontSize: "14px",
                  }}
                >
                  {teams.length === 0 && (
                    <option value="">Loading teams…</option>
                  )}
                  {teams.map((t) => (
                    <option key={t.team_id || t.id || t.name} value={t.name}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ minWidth: "260px" }}>
                <label
                  style={{
                    display: "block",
                    fontSize: "13px",
                    marginBottom: "4px",
                  }}
                >
                  Away team
                </label>
                <select
                  value={awayTeam}
                  onChange={(e) => setAwayTeam(e.target.value)}
                  disabled={loadingTeams || teams.length === 0}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    borderRadius: "4px",
                    border: "1px solid #d1d5db",
                    fontSize: "14px",
                  }}
                >
                  {teams.length === 0 && (
                    <option value="">Loading teams…</option>
                  )}
                  {teams.map((t) => (
                    <option key={t.team_id || t.id || t.name} value={t.name}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {teamsError && (
              <div
                style={{
                  color: "#b91c1c",
                  fontSize: "13px",
                  marginBottom: "8px",
                }}
              >
                Error loading teams: {teamsError}
              </div>
            )}

            <button
              type="button"
              disabled={
                !baseUrl ||
                !homeTeam ||
                !awayTeam ||
                simBusy ||
                loadingTeams ||
                teams.length === 0
              }
              onClick={runSimulation}
              style={{
                padding: "8px 14px",
                borderRadius: "4px",
                border: "none",
                backgroundColor:
                  !baseUrl || !homeTeam || !awayTeam || teams.length === 0
                    ? "#9ca3af"
                    : "#16a34a",
                color: "white",
                fontSize: "14px",
                cursor:
                  !baseUrl || !homeTeam || !awayTeam || teams.length === 0
                    ? "default"
                    : "pointer",
              }}
            >
              {simBusy ? "Running simulation…" : "Run 1000-game series"}
            </button>

            {simError && (
              <div
                style={{
                  marginTop: "8px",
                  color: "#b91c1c",
                  fontSize: "13px",
                }}
              >
                Simulation error: {simError}
              </div>
            )}

            {simResult && (
              <div style={{ marginTop: "16px" }}>
                {seriesSummary && (
                  <div
                    style={{
                      padding: "10px 12px",
                      borderRadius: "8px",
                      backgroundColor: "#f3f4f6",
                      marginBottom: "10px",
                      fontSize: "13px",
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      Summary
                    </div>
                    <div>
                      <strong>Home win %:</strong>{" "}
                      {seriesSummary.homeWinPct != null
                        ? (seriesSummary.homeWinPct * 100).toFixed(1) + "%"
                        : "—"}
                    </div>
                    <div>
                      <strong>Away win %:</strong>{" "}
                      {seriesSummary.awayWinPct != null
                        ? (seriesSummary.awayWinPct * 100).toFixed(1) + "%"
                        : "—"}
                    </div>
                    <div>
                      <strong>OT rate:</strong>{" "}
                      {seriesSummary.otRate != null
                        ? (seriesSummary.otRate * 100).toFixed(1) + "%"
                        : "—"}
                    </div>
                    <div>
                      <strong>Mean total points:</strong>{" "}
                      {seriesSummary.meanTotal != null
                        ? seriesSummary.meanTotal.toFixed(1)
                        : "—"}
                    </div>
                  </div>
                )}

                {marketCompare && (
                  <div
                    style={{
                      padding: "10px 12px",
                      borderRadius: "8px",
                      backgroundColor: "#f9fafb",
                      border: "1px solid #e5e7eb",
                      marginBottom: "10px",
                      fontSize: "13px",
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      Model vs Market (closing)
                    </div>

                    {marketCompare.market &&
                    (marketCompare.market.spread_home != null ||
                      marketCompare.market.total != null) ? (
                      <>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
                          <div>
                            <div>
                              <strong>Spread (home, + = favored)</strong>
                            </div>
                            <div>
                              Model: {marketCompare.modelSpread != null ? marketCompare.modelSpread.toFixed(1) : "—"}
                            </div>
                            <div>
                              Market: {marketCompare.market.spread_home != null ? Number(marketCompare.market.spread_home).toFixed(1) : "—"}
                            </div>
                            <div>
                              Edge: {marketCompare.edge?.spread_home != null ? Number(marketCompare.edge.spread_home).toFixed(1) : "—"}
                            </div>
                            <div>
                              Blended: {marketCompare.blended?.spread_home != null ? Number(marketCompare.blended.spread_home).toFixed(1) : "—"}
                            </div>
                          </div>

                          <div>
                            <div>
                              <strong>Total</strong>
                            </div>
                            <div>
                              Model: {marketCompare.modelTotal != null ? marketCompare.modelTotal.toFixed(1) : "—"}
                            </div>
                            <div>
                              Market: {marketCompare.market.total != null ? Number(marketCompare.market.total).toFixed(1) : "—"}
                            </div>
                            <div>
                              Edge: {marketCompare.edge?.total != null ? Number(marketCompare.edge.total).toFixed(1) : "—"}
                            </div>
                            <div>
                              Blended: {marketCompare.blended?.total != null ? Number(marketCompare.blended.total).toFixed(1) : "—"}
                            </div>
                          </div>
                        </div>

                        {(marketCompare.market.season != null || marketCompare.market.week != null) && (
                          <div style={{ marginTop: 6, color: "#6b7280", fontSize: 12 }}>
                            Line found for season {marketCompare.market.season ?? "—"}
                            {marketCompare.market.week != null ? `, week ${marketCompare.market.week}` : ""}.
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={{ color: "#6b7280", fontSize: 12 }}>
                        No closing line found for this exact home/away matchup in the ingested seasons.
                      </div>
                    )}
                  </div>
                )}

                {(homeStats || awayStats) && (
                  <div
                    style={{
                      display: "flex",
                      gap: "12px",
                      flexWrap: "wrap",
                      marginBottom: "10px",
                      fontSize: "13px",
                    }}
                  >
                    {homeStats && (
                      <div
                        style={{
                          flex: "1 1 260px",
                          borderRadius: "8px",
                          padding: "10px 12px",
                          border: "1px solid #e5e7eb",
                        }}
                      >
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>
                          {homeStats.name || homeTeam}
                        </div>
                        <div>Avg pts: {homeStats.avgPts ?? "—"}</div>
                        <div>p05: {homeStats.p05 ?? "—"}</div>
                        <div>p50: {homeStats.p50 ?? "—"}</div>
                        <div>p95: {homeStats.p95 ?? "—"}</div>
                      </div>
                    )}
                    {awayStats && (
                      <div
                        style={{
                          flex: "1 1 260px",
                          borderRadius: "8px",
                          padding: "10px 12px",
                          border: "1px solid #e5e7eb",
                        }}
                      >
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>
                          {awayStats.name || awayTeam}
                        </div>
                        <div>Avg pts: {awayStats.avgPts ?? "—"}</div>
                        <div>p05: {awayStats.p05 ?? "—"}</div>
                        <div>p50: {awayStats.p50 ?? "—"}</div>
                        <div>p95: {awayStats.p95 ?? "—"}</div>
                      </div>
                    )}
                  </div>
                )}

                <div
                  style={{
                    padding: "10px 12px",
                    borderRadius: "8px",
                    backgroundColor: "#f9fafb",
                    fontSize: "12px",
                    maxHeight: "260px",
                    overflow: "auto",
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    Raw simulation output (JSON)
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      fontFamily:
                        "ui-monospace, SFMono-Regular, Menlo, monospace",
                    }}
                  >
                    {JSON.stringify(simResult, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </>
        )}
      </section>

      {/* Dataset & setup (ingestion / vegas sanity) */}
      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: "8px",
          padding: "16px",
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "8px",
            marginBottom: "10px",
          }}
        >
          <h2
            style={{
              fontSize: "18px",
              fontWeight: 600,
              margin: 0,
            }}
          >
            Dataset &amp; setup (historical seasons)
          </h2>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              type="button"
              disabled={!baseUrl || busySummary}
              onClick={refreshSummary}
              style={{
                padding: "6px 10px",
                borderRadius: "4px",
                border: "none",
                backgroundColor: !baseUrl ? "#9ca3af" : "#2563eb",
                color: "white",
                fontSize: "13px",
                cursor: !baseUrl ? "default" : "pointer",
              }}
            >
              {busySummary ? "Refreshing…" : "Refresh summary"}
            </button>
          </div>
        </div>

        {!baseUrl && (
          <div style={{ fontSize: "13px", color: "#6b7280" }}>
            Enter your API URL above to view dataset status.
          </div>
        )}

        {baseUrl && (
          <div
            style={{
              marginBottom: "14px",
              padding: "12px",
              borderRadius: "8px",
              backgroundColor: "#f9fafb",
              border: "1px solid #e5e7eb",
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 8 }}>
              One-click setup (Bootstrap)
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "10px",
                alignItems: "end",
              }}
            >
              <label style={{ fontSize: "13px" }}>
                Start season
                <input
                  type="number"
                  value={bootStart}
                  onChange={(e) => setBootStart(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                />
              </label>

              <label style={{ fontSize: "13px" }}>
                End season
                <input
                  type="number"
                  value={bootEnd}
                  onChange={(e) => setBootEnd(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                />
              </label>

              <label style={{ fontSize: "13px" }}>
                Include market lines
                <select
                  value={String(bootIncludeMarket)}
                  onChange={(e) => setBootIncludeMarket(e.target.value === "true")}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                >
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              </label>

              <label style={{ fontSize: "13px" }}>
                Seed ratings
                <select
                  value={String(bootSeedRatings)}
                  onChange={(e) => setBootSeedRatings(e.target.value === "true")}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>

              <label style={{ fontSize: "13px" }}>
                Calibrate multi
                <select
                  value={String(bootCalibrateMulti)}
                  onChange={(e) => setBootCalibrateMulti(e.target.value === "true")}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>

              <label style={{ fontSize: "13px" }}>
                Force
                <select
                  value={String(bootForce)}
                  onChange={(e) => setBootForce(e.target.value === "true")}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                >
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              </label>

              <button
                type="button"
                disabled={!baseUrl || bootBusy}
                onClick={runBootstrap}
                style={{
                  padding: "8px 12px",
                  borderRadius: 6,
                  border: "none",
                  backgroundColor: !baseUrl ? "#9ca3af" : "#16a34a",
                  color: "white",
                  fontSize: 14,
                  cursor: !baseUrl ? "default" : "pointer",
                }}
              >
                {bootBusy ? "Bootstrapping…" : "Run bootstrap"}
              </button>
            </div>

            <div style={{ fontSize: "12px", color: "#6b7280", marginTop: 8 }}>
              Tip: run this once after deploy. Re-running is safe (idempotent) unless you set
              <strong> Force</strong>.
            </div>

            {bootError && (
              <div style={{ marginTop: 8, color: "#b91c1c", fontSize: 13 }}>
                Bootstrap error: {bootError}
              </div>
            )}

            {bootResult && (
              <details style={{ marginTop: 10 }}>
                <summary style={{ cursor: "pointer", fontSize: 13 }}>
                  View bootstrap response
                </summary>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    backgroundColor: "#0b1220",
                    color: "#e5e7eb",
                    padding: "10px 12px",
                    borderRadius: "8px",
                    overflowX: "auto",
                    fontSize: "12px",
                    marginTop: 8,
                  }}
                >
                  {JSON.stringify(bootResult, null, 2)}
                </pre>
              </details>
            )}
          </div>
        )}

        {baseUrl && (
          <div
            style={{
              marginBottom: "14px",
              padding: "12px",
              borderRadius: "8px",
              backgroundColor: "#f9fafb",
              border: "1px solid #e5e7eb",
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 8 }}>
              Backtest (shipping gate)
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                gap: "10px",
                alignItems: "end",
              }}
            >
              <label style={{ fontSize: "13px" }}>
                Test season
                <input
                  type="number"
                  value={btSeason}
                  onChange={(e) => setBtSeason(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                />
              </label>

              <label style={{ fontSize: "13px" }}>
                Games sampled
                <input
                  type="number"
                  value={btGames}
                  onChange={(e) => setBtGames(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                />
              </label>

              <label style={{ fontSize: "13px" }}>
                Sims per game
                <input
                  type="number"
                  value={btSims}
                  onChange={(e) => setBtSims(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: "6px 8px",
                    borderRadius: 4,
                    border: "1px solid #d1d5db",
                    fontSize: 14,
                  }}
                />
              </label>

              <button
                type="button"
                disabled={!baseUrl || btBusy}
                onClick={() => runBacktest("/backtest/run")}
                style={{
                  padding: "8px 12px",
                  borderRadius: 6,
                  border: "none",
                  backgroundColor: !baseUrl ? "#9ca3af" : "#2563eb",
                  color: "white",
                  fontSize: 14,
                  cursor: !baseUrl ? "default" : "pointer",
                }}
              >
                {btBusy ? "Running…" : "Run backtest"}
              </button>

              <button
                type="button"
                disabled={!baseUrl || btBusy}
                onClick={() => runBacktest("/backtest/set-baseline")}
                style={{
                  padding: "8px 12px",
                  borderRadius: 6,
                  border: "1px solid #d1d5db",
                  backgroundColor: "white",
                  color: "#111827",
                  fontSize: 14,
                  cursor: !baseUrl ? "default" : "pointer",
                }}
              >
                Set baseline
              </button>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
              The backend computes spread MAE, total MAE, and Brier score versus closing lines.
              It returns a verdict (PASS / EVEN / FAIL) relative to your stored baseline.
            </div>

            {btError && (
              <div style={{ marginTop: 8, color: "#b91c1c", fontSize: 13 }}>
                Backtest error: {btError}
              </div>
            )}

            {btResult && (
              <div style={{ marginTop: 10 }}>
                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                  <span
                    style={{
                      display: "inline-block",
                      padding: "4px 10px",
                      borderRadius: 999,
                      fontSize: 12,
                      fontWeight: 700,
                      color: "white",
                      backgroundColor:
                        btResult.verdict === "PASS"
                          ? "#16a34a"
                          : btResult.verdict === "FAIL"
                          ? "#b91c1c"
                          : "#f59e0b",
                    }}
                  >
                    {btResult.verdict}
                  </span>
                  <span style={{ fontSize: 13, color: "#374151" }}>{btResult.reason}</span>
                </div>

                
<div style={{ marginTop: 10, display: "grid", gap: 10 }}>
  <details>
    <summary style={{ cursor: "pointer", fontWeight: 700 }}>Worst spread misses (top 10)</summary>
    <div style={{ overflowX: "auto", marginTop: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
            <th style={{ padding: "6px 8px" }}>Game</th>
            <th style={{ padding: "6px 8px" }}>Model</th>
            <th style={{ padding: "6px 8px" }}>Market</th>
            <th style={{ padding: "6px 8px" }}>Abs err</th>
            <th style={{ padding: "6px 8px" }}>Final</th>
          </tr>
        </thead>
        <tbody>
          {(btResult.worst_spread_misses || []).map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ padding: "6px 8px" }}>{r.away} @ {r.home}</td>
              <td style={{ padding: "6px 8px" }}>{Number(r.model_spread_home).toFixed(1)}</td>
              <td style={{ padding: "6px 8px" }}>{r.market_spread_home == null ? "—" : Number(r.market_spread_home).toFixed(1)}</td>
              <td style={{ padding: "6px 8px" }}>{r.spread_abs_error == null ? "—" : Number(r.spread_abs_error).toFixed(2)}</td>
              <td style={{ padding: "6px 8px" }}>{r.final_away}-{r.final_home}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </details>

  <details>
    <summary style={{ cursor: "pointer", fontWeight: 700 }}>Worst total misses (top 10)</summary>
    <div style={{ overflowX: "auto", marginTop: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
            <th style={{ padding: "6px 8px" }}>Game</th>
            <th style={{ padding: "6px 8px" }}>Model</th>
            <th style={{ padding: "6px 8px" }}>Market</th>
            <th style={{ padding: "6px 8px" }}>Abs err</th>
            <th style={{ padding: "6px 8px" }}>Final</th>
          </tr>
        </thead>
        <tbody>
          {(btResult.worst_total_misses || []).map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ padding: "6px 8px" }}>{r.away} @ {r.home}</td>
              <td style={{ padding: "6px 8px" }}>{Number(r.model_total).toFixed(1)}</td>
              <td style={{ padding: "6px 8px" }}>{r.market_total == null ? "—" : Number(r.market_total).toFixed(1)}</td>
              <td style={{ padding: "6px 8px" }}>{r.total_abs_error == null ? "—" : Number(r.total_abs_error).toFixed(2)}</td>
              <td style={{ padding: "6px 8px" }}>{r.final_away}-{r.final_home}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </details>

  <details>
    <summary style={{ cursor: "pointer", fontWeight: 700 }}>MAE by bucket</summary>
    <div style={{ marginTop: 8, fontSize: 12, color: "#111827" }}>
      <div><strong>Spread MAE</strong></div>
      <div style={{ marginTop: 4 }}>
        Small spreads (|line| ≤ 3): {btResult.bucket_mae?.spread?.small_spreads_abs_le_3 == null ? "—" : btResult.bucket_mae.spread.small_spreads_abs_le_3.toFixed(2)}
      </div>
      <div>
        Large spreads (|line| &gt; 3): {btResult.bucket_mae?.spread?.large_spreads_abs_gt_3 == null ? "—" : btResult.bucket_mae.spread.large_spreads_abs_gt_3.toFixed(2)}
      </div>
      <div style={{ marginTop: 10 }}><strong>Total MAE</strong></div>
      <div style={{ marginTop: 4 }}>
        Low totals (&lt; 45): {btResult.bucket_mae?.total?.low_totals_lt_45 == null ? "—" : btResult.bucket_mae.total.low_totals_lt_45.toFixed(2)}
      </div>
      <div>
        High totals (&gt; 60): {btResult.bucket_mae?.total?.high_totals_gt_60 == null ? "—" : btResult.bucket_mae.total.high_totals_gt_60.toFixed(2)}
      </div>
    </div>
  </details>

  <details>
    <summary style={{ cursor: "pointer", fontWeight: 700 }}>Win probability calibration</summary>
    <div style={{ overflowX: "auto", marginTop: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
            <th style={{ padding: "6px 8px" }}>Bucket</th>
            <th style={{ padding: "6px 8px" }}>Games</th>
            <th style={{ padding: "6px 8px" }}>Actual win rate</th>
          </tr>
        </thead>
        <tbody>
          {(btResult.win_prob_calibration || []).map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ padding: "6px 8px" }}>{r.bucket}</td>
              <td style={{ padding: "6px 8px" }}>{r.games}</td>
              <td style={{ padding: "6px 8px" }}>{Number(r.actual_win_rate).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </details>

  <details>
    <summary style={{ cursor: "pointer", fontWeight: 700 }}>Coverage (actual inside p05–p95)</summary>
    <div style={{ marginTop: 8, fontSize: 12 }}>
      Spread coverage (p90): {btResult.coverage?.spread_p90 == null ? "—" : (btResult.coverage.spread_p90 * 100).toFixed(1)}%
      <br />
      Total coverage (p90): {btResult.coverage?.total_p90 == null ? "—" : (btResult.coverage.total_p90 * 100).toFixed(1)}%
    </div>
  </details>
</div>
<details style={{ marginTop: 8 }}>
                  <summary style={{ cursor: "pointer", fontSize: 13 }}>
                    View backtest details
                  </summary>
                  <pre
                    style={{
                      whiteSpace: "pre-wrap",
                      backgroundColor: "#0b1220",
                      color: "#e5e7eb",
                      padding: "10px 12px",
                      borderRadius: "8px",
                      overflowX: "auto",
                      fontSize: "12px",
                      marginTop: 8,
                    }}
                  >
                    {JSON.stringify(btResult, null, 2)}
                  </pre>
                </details>
              </div>
            )}

            {!btResult && btBaseline && (
              <div style={{ marginTop: 10, fontSize: 13, color: "#374151" }}>
                Baseline loaded: <strong>{btBaseline.label}</strong> (Spread MAE {btBaseline.spread_mae?.toFixed?.(2)}, Total MAE {btBaseline.total_mae?.toFixed?.(2)}, Brier {btBaseline.brier?.toFixed?.(3)})
              </div>
            )}

            {!btResult && !btBaseline && (
              <div style={{ marginTop: 10, fontSize: 13, color: "#6b7280" }}>
                No baseline stored for {btSeason}. Click <strong>Set baseline</strong> once to lock it in.
              </div>
            )}
          </div>
        )}

        {baseUrl && (
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "13px",
            }}
          >
            <thead>
              <tr
                style={{
                  textAlign: "left",
                  borderBottom: "1px solid #e5e7eb",
                }}
              >
                <th style={{ padding: "4px 6px" }}>Season</th>
                <th style={{ padding: "4px 6px" }}>Games in DB</th>
                <th style={{ padding: "4px 6px" }}>With Vegas lines</th>
                <th style={{ padding: "4px 6px" }}>Postseason games</th>
                <th style={{ padding: "4px 6px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {SUMMARY_SEASONS.map((season) => {
                const s =
                  summary[season] || {
                    count: 0,
                    withLines: 0,
                    postseason: 0,
                    status: "—",
                  };
                return (
                  <tr
                    key={season}
                    style={{ borderBottom: "1px solid #f3f4f6" }}
                  >
                    <td style={{ padding: "4px 6px" }}>{season}</td>
                    <td style={{ padding: "4px 6px" }}>
                      {s.count ?? "—"}
                    </td>
                    <td style={{ padding: "4px 6px" }}>
                      {s.withLines ?? "—"}
                    </td>
                    <td style={{ padding: "4px 6px" }}>
                      {s.postseason ?? "—"}
                    </td>
                    <td style={{ padding: "4px 6px" }}>{s.status}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

export default App;
