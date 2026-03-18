import React, { useMemo, useState } from "react";
import "./Upload.css";

const API_BASE =
  process.env.REACT_APP_API_BASE ||
  (window.location.port === "5000" ? "" : "http://localhost:5000");

function pretty(x, digits = 4) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function clamp(n, lo, hi) {
  const v = Number(n);
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function Icon({ name }) {
  if (name === "spark") {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" className="fb-ico" aria-hidden="true">
        <path d="M12 2l1.6 6.2L20 12l-6.4 3.8L12 22l-1.6-6.2L4 12l6.4-3.8L12 2z" />
      </svg>
    );
  }
  if (name === "wave") {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" className="fb-ico" aria-hidden="true">
        <path d="M3 12c3.5 0 3.5-8 7-8s3.5 16 7 16 3.5-8 7-8" fill="none" strokeWidth="2" />
      </svg>
    );
  }
  if (name === "dl") {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" className="fb-ico" aria-hidden="true">
        <path d="M12 3v10m0 0l4-4m-4 4l-4-4M5 17h14v4H5z" fill="none" strokeWidth="2" />
      </svg>
    );
  }
  if (name === "meta") {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" className="fb-ico" aria-hidden="true">
        <path d="M6 2h9l3 3v17H6V2z" fill="none" strokeWidth="2" />
        <path d="M9 10h6M9 14h6M9 18h6" fill="none" strokeWidth="2" />
      </svg>
    );
  }
  return null;
}

export default function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); 
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [lofiEnabled, setLofiEnabled] = useState(true);
  const [lofiGainDb, setLofiGainDb] = useState(0);

  const canRun = useMemo(() => file && status !== "uploading", [file, status]);

  function resetFile() {
    setFile(null);
    setResult(null);
    setError("");
    setStatus("idle");
  }

  async function runPipeline() {
    if (!file) return;

    setStatus("uploading");
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("lofi_enabled", lofiEnabled ? "1" : "0");
      form.append("lofi_gain_db", String(lofiGainDb));

      const resp = await fetch(`${API_BASE}/api/pipeline`, { method: "POST", body: form });

      const contentType = resp.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await resp.text();
        throw new Error(
          `Server returned non-JSON (status ${resp.status}).\nLikely wrong API_BASE or backend crash.\n\n${text.slice(
            0,
            1200
          )}`
        );
      }

      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        const msg = data.details || data.debug || data.error || `Pipeline failed (status ${resp.status})`;
        throw new Error(`Pipeline failed (status ${resp.status}): ${msg}`);
      }

      setResult(data);
      setStatus("done");
    } catch (e) {
      setStatus("error");
      setError(e?.message || String(e));
    }
  }

  const payload = result?.payload || {};
  const meta = payload?.meta || {};
  const knobs = payload?.knobs || {};
  const metrics = payload?.metrics || {};

  const statusText =
    status === "idle" ? "Ready" : status === "uploading" ? "Processing…" : status === "done" ? "Done" : "Error";

  const statusTone =
    status === "done" ? "ok" : status === "error" ? "bad" : status === "uploading" ? "run" : "idle";

  const gainDb = clamp(lofiGainDb, -20, 20);

  const deltaRawToFinal = Number(metrics.focus_after) - Number(metrics.focus_raw);

  const outputAudioUrl = result?.download_url_wav ? `${API_BASE}${result.download_url_wav}` : "";
  const outputMetaUrl = result?.download_url_meta ? `${API_BASE}${result.download_url_meta}` : "";

  const knobItems = [
    { k: "controller_score", v: pretty(knobs.controller_score, 6) },
    { k: "vocal_cut", v: pretty(knobs.vocal_cut, 2) },
    { k: "low_gain_db", v: pretty(knobs.low_gain_db, 1) },
    { k: "mid_gain_db", v: pretty(knobs.mid_gain_db, 1) },
    { k: "high_gain_db", v: pretty(knobs.high_gain_db, 1) },
    { k: "transient_smooth", v: pretty(knobs.transient_smooth, 2) },
    { k: "drc_strength", v: pretty(knobs.drc_strength, 2) },
    { k: "lofi_amount", v: pretty(knobs.lofi_amount, 2) },
  ];

  const advItems = [
    { k: "output.wav", v: String(result?.output_wav || "—"), mono: true },
    { k: "meta.json", v: String(result?.output_meta || "—"), mono: true },
    { k: "lofi_used_layers", v: String(meta.lofi_used_layers || "—"), mono: true },
    {
      k: "normalization",
      v:
        meta.normalization_target_rms_db != null
          ? `RMS ${meta.normalization_target_rms_db} dB · peak ${meta.normalization_peak_limit}`
          : "—",
      mono: true,
    },
  ];

  return (
    <div className="fb-page">
      <div className="fb-shell">
        <header className="fb-top">
          <div className="fb-brand">
            <div className="fb-mark">
              <span className="fb-markA">FocusBeats</span>
              <span className="fb-markC">AI</span>
            </div>
            <div className="fb-tag">Upload a song and transform it for focus.</div>
          </div>

          <div className={`fb-pill ${statusTone}`}>
            <span className="dot" />
            <span>{statusText}</span>
          </div>
        </header>

        <div className="fb-grid">
          <section className="fb-panel">
            <div className="fb-panelHead">
              <div className="fb-panelTitle">
                <span>Upload</span>
              </div>
              <div className="fb-subtle">Pipeline default: lofi = 0.0 dB</div>
            </div>

            <div className={`fb-drop ${file ? "hasFile" : ""}`}>
              {!file ? (
                <label className="fb-choose">
                  <div className="fb-chooseTitle">Choose File</div>
                  <div className="fb-chooseHint">wav · mp3 · flac · m4a · ogg · aac</div>
                  <input
                    className="fb-fileInput"
                    type="file"
                    accept=".wav,.mp3,.flac,.aiff,.aif,.m4a,.ogg,.aac,audio/*"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </label>
              ) : (
                <div className="fb-picked">
                  <div className="fb-pickedName" title={file.name}>
                    {file.name}
                  </div>
                  <button className="fb-change" onClick={resetFile} type="button">
                    Change file
                  </button>
                </div>
              )}
            </div>

            <div className="fb-controls">
              <div className="fb-row">
                <label className="fb-toggle">
                  <input
                    type="checkbox"
                    checked={lofiEnabled}
                    onChange={(e) => setLofiEnabled(e.target.checked)}
                  />
                  <span className="ui" />
                  <span className="txt">Enable lofi</span>
                </label>

                <div className="fb-mini">
                  <span className="fb-miniLabel">Texture</span>
                  <span className="fb-miniVal">{gainDb.toFixed(1)} dB</span>
                </div>
              </div>

              <div className={`fb-slider ${!lofiEnabled ? "disabled" : ""}`}>
                <input
                  type="range"
                  min={-20}
                  max={20}
                  step={0.5}
                  value={gainDb}
                  disabled={!lofiEnabled}
                  onChange={(e) => setLofiGainDb(Number(e.target.value))}
                />
                <div className="fb-sliderTicks">
                  <span>-20</span>
                  <span>0 (default)</span>
                  <span>+20</span>
                </div>
              </div>

              <button className="fb-primary" onClick={runPipeline} disabled={!canRun}>
                {status === "uploading" ? "Processing…" : "Transform"}
              </button>

              <div className="fb-api">
                API: <code>{API_BASE || "(same origin)"}</code>
              </div>

              {error && (
                <div className="fb-error">
                  <div className="fb-errorTitle">Error</div>
                  <pre className="fb-errorText">{error}</pre>
                </div>
              )}
            </div>
          </section>

          <section className="fb-panel fb-panelRight">
            <div className="fb-panelHead">
              <div className="fb-panelTitle">
                <span>Result</span>
              </div>

              <div className="fb-actions">
                {outputAudioUrl && (
                  <a className="fb-action primary" href={outputAudioUrl}>
                    <Icon name="dl" /> <span>WAV</span>
                  </a>
                )}
                {outputMetaUrl && (
                  <a className="fb-action" href={outputMetaUrl}>
                    <Icon name="meta" /> <span>meta</span>
                  </a>
                )}
              </div>
            </div>

            {!result ? (
              <div className="fb-empty">
                <div className="fb-emptyTitle">Nothing yet.</div>
              </div>
            ) : (
              <>
                {outputAudioUrl && (
                  <div className="fb-player">
                    <audio controls preload="none" src={outputAudioUrl} />
                  </div>
                )}

                <div className="fb-hero">
                  <div className="fb-heroHead">
                    <div className="fb-heroTitle">Focus gain</div>
                    <div className="fb-heroPill">final − raw</div>
                  </div>

                  <div className="fb-heroNumber">
                    {Number.isFinite(deltaRawToFinal) ? deltaRawToFinal.toFixed(6) : "—"}
                  </div>

                  <div className="fb-heroNote">Overall improvement from the original input</div>
                </div>

                <div className="fb-stats">
                  <div className="fb-stat">
                    <div className="k">Focus (final)</div>
                    <div className="v">{pretty(metrics.focus_after, 6)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Focus (raw)</div>
                    <div className="v">{pretty(metrics.focus_raw, 6)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Focus (instrumental)</div>
                    <div className="v">{pretty(metrics.focus_before, 6)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Focus (post-DSP)</div>
                    <div className="v">{pretty(metrics.focus_post_dsp, 6)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Similarity (instrumental ↔ final)</div>
                    <div className="v">{pretty(metrics.yamnet_similarity, 6)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Duration (sec)</div>
                    <div className="v">{pretty(metrics.duration_sec_output, 3)}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Vocals removed</div>
                    <div className="v">{String(meta.vocals_removed_method || "—")}</div>
                  </div>

                  <div className="fb-stat">
                    <div className="k">Lofi layers</div>
                    <div className="v">{String(meta.lofi_used_layers_count ?? "—")} used</div>
                  </div>
                </div>

                <details className="fb-advanced">
                  <summary>Advanced</summary>

                  <div className="fb-advCard">
                    <div className="fb-knobs fb-advUniform">
                      <div className="fb-knobsHead">Artifacts</div>
                      <div className="fb-knobsGrid fb-advGridUniform">
                        {advItems.map((it) => (
                          <div className={`fb-knob ${it.mono ? "mono" : ""}`} key={it.k}>
                            <div className="k">{it.k}</div>
                            <div className="v">{it.v}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="fb-knobs">
                      <div className="fb-knobsHead">Knobs</div>
                      <div className="fb-knobsGrid">
                        {knobItems.map((it) => (
                          <div className="fb-knob" key={it.k}>
                            <div className="k">{it.k}</div>
                            <div className="v">{it.v}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <details className="fb-raw">
                      <summary>Raw payload</summary>
                      <div className="fb-rawBox">
                        <pre className="fb-rawPre">{JSON.stringify(payload, null, 2)}</pre>
                      </div>
                    </details>
                  </div>
                </details>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
