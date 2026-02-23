import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { apiGet, apiPost } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function ModelRunPage() {
  const navigate = useNavigate();
  const { data: uploads, loading: loadingUploads } = useApi('/api/uploads');
  const { data: runs, loading: loadingRuns } = useApi('/api/model/runs');

  const [config, setConfig] = useState({
    upload_id: '',
    target: 'revenue',
    adstock_l_max: 8,
    yearly_seasonality: 6,
    chains: 4,
    tune: 1500,
    draws: 1000,
    target_accept: 0.9,
  });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  // Channel analysis state
  const [channelData, setChannelData] = useState(null);
  const [channelLoading, setChannelLoading] = useState(false);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [mergeMap, setMergeMap] = useState({});

  // Fetch channel analysis when upload changes
  useEffect(() => {
    if (!config.upload_id) {
      setChannelData(null);
      setSelectedChannels([]);
      setMergeMap({});
      return;
    }
    setChannelLoading(true);
    apiGet(`/api/uploads/${config.upload_id}/channels`)
      .then((data) => {
        setChannelData(data);
        setSelectedChannels(data.recommended_config.channels);
        setMergeMap(data.recommended_config.merge);
      })
      .catch(() => setChannelData(null))
      .finally(() => setChannelLoading(false));
  }, [config.upload_id]);

  function toggleChannel(ch) {
    setSelectedChannels((prev) =>
      prev.includes(ch) ? prev.filter((c) => c !== ch) : [...prev, ch]
    );
  }

  function resetToRecommended() {
    if (!channelData) return;
    setSelectedChannels(channelData.recommended_config.channels);
    setMergeMap(channelData.recommended_config.merge);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!config.upload_id) return;
    setRunning(true);
    setError(null);

    const channelConfig =
      channelData && selectedChannels.length > 0
        ? { channels: selectedChannels, merge: mergeMap }
        : null;

    try {
      const res = await apiPost('/api/model/run', {
        ...config,
        upload_id: Number(config.upload_id),
        channel_config: channelConfig,
      });
      navigate(`/model/runs/${res.run_id}/results`);
    } catch (err) {
      setError(err.message);
      setRunning(false);
    }
  }

  function handleChange(field, value) {
    setConfig((prev) => ({ ...prev, [field]: value }));
  }

  if (loadingUploads || loadingRuns) return <LoadingSpinner />;

  const successUploads = (uploads || []).filter(
    (u) => u.status === 'success' || u.status === 'partial'
  );

  // Build display list: all channels from analysis (post-merge names)
  const allChannels = channelData
    ? channelData.channels.map((ch) => ch.name)
    : [];
  const droppedSet = new Set(
    channelData ? channelData.recommended_config.dropped : []
  );
  const reasons = channelData ? channelData.recommended_config.reasons : {};

  // Find which source channels merge into each merged name
  const mergeGroups = {};
  if (channelData) {
    for (const [src, dest] of Object.entries(channelData.recommended_config.merge)) {
      if (!mergeGroups[dest]) mergeGroups[dest] = [];
      mergeGroups[dest].push(src);
    }
  }

  return (
    <>
      <h1 className="page-title">Run Model</h1>

      {running && (
        <LoadingSpinner text="Fitting Bayesian MMM — this may take several minutes..." />
      )}

      {!running && (
        <div className="card">
          <div className="card-title">Configure Model Run</div>
          <div className="card-subtitle">
            Select an upload and configure the Bayesian Media Mix Model parameters.
          </div>

          {error && <div className="error-banner">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Dataset</label>
              <select
                value={config.upload_id}
                onChange={(e) => handleChange('upload_id', e.target.value)}
              >
                <option value="">Select an upload...</option>
                {successUploads.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.filename} (ID: {u.id}, {u.row_count} rows)
                  </option>
                ))}
              </select>
            </div>

            {/* Channel selection */}
            {channelLoading && (
              <div className="form-group">
                <label className="form-label">Channels</label>
                <div className="text-sm text-muted">Analyzing channels...</div>
              </div>
            )}

            {channelData && !channelLoading && (
              <div className="form-group">
                <label className="form-label">
                  Channels
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={resetToRecommended}
                    style={{ marginLeft: 12, verticalAlign: 'middle' }}
                  >
                    Use recommended
                  </button>
                </label>
                <div className="form-hint" style={{ marginBottom: 8 }}>
                  Select channels to include in the model. Low-spend channels are
                  deselected by default to improve fit.
                </div>
                <div
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: 8,
                    maxHeight: 280,
                    overflowY: 'auto',
                  }}
                >
                  {allChannels.map((ch) => {
                    const info = channelData.channels.find((c) => c.name === ch);
                    const isDropped = droppedSet.has(ch);
                    const isMerged = !!mergeGroups[ch];
                    const isSelected = selectedChannels.includes(ch);

                    return (
                      <label
                        key={ch}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          padding: '6px 4px',
                          cursor: 'pointer',
                          opacity: isSelected ? 1 : 0.55,
                          borderBottom: '1px solid var(--border)',
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleChannel(ch)}
                        />
                        <span className="mono" style={{ minWidth: 160 }}>
                          {ch}
                          {isMerged && (
                            <span
                              className="text-sm text-muted"
                              style={{ marginLeft: 4 }}
                            >
                              (combined)
                            </span>
                          )}
                        </span>
                        {info && (
                          <>
                            <span className="text-sm" style={{ minWidth: 80 }}>
                              {info.spend_pct}% spend
                            </span>
                            <span className="text-sm text-muted">
                              {info.active_days}/{info.total_days} days
                            </span>
                          </>
                        )}
                        {isDropped && reasons[ch] && (
                          <span className="text-sm" style={{ color: 'var(--amber)' }}>
                            — {reasons[ch]}
                          </span>
                        )}
                      </label>
                    );
                  })}

                  {/* Show merge sub-items */}
                  {Object.entries(mergeGroups).length > 0 && (
                    <div
                      style={{
                        marginTop: 8,
                        padding: '8px 4px',
                        borderTop: '1px solid var(--border)',
                      }}
                    >
                      <div className="text-sm text-muted" style={{ marginBottom: 4 }}>
                        Merged placements:
                      </div>
                      {Object.entries(mergeGroups).map(([dest, sources]) => (
                        <div key={dest} className="text-sm" style={{ paddingLeft: 8 }}>
                          <span className="mono">{dest}</span> ←{' '}
                          {sources.map((s) => (
                            <span key={s} className="mono text-muted" style={{ marginRight: 6 }}>
                              {s}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="form-group">
              <label className="form-label">Target Variable</label>
              <select
                value={config.target}
                onChange={(e) => handleChange('target', e.target.value)}
              >
                <option value="revenue">Revenue</option>
                <option value="orders">Orders</option>
              </select>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Adstock Max Lag</label>
                <input
                  type="number"
                  value={config.adstock_l_max}
                  onChange={(e) => handleChange('adstock_l_max', Number(e.target.value))}
                  min={1}
                  max={30}
                />
                <div className="form-hint">Days of carryover effect</div>
              </div>
              <div className="form-group">
                <label className="form-label">Seasonality Modes</label>
                <input
                  type="number"
                  value={config.yearly_seasonality}
                  onChange={(e) => handleChange('yearly_seasonality', Number(e.target.value))}
                  min={1}
                  max={10}
                />
                <div className="form-hint">Fourier terms for yearly pattern</div>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Chains</label>
                <input
                  type="number"
                  value={config.chains}
                  onChange={(e) => handleChange('chains', Number(e.target.value))}
                  min={1}
                  max={8}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Tune</label>
                <input
                  type="number"
                  value={config.tune}
                  onChange={(e) => handleChange('tune', Number(e.target.value))}
                  min={100}
                  max={5000}
                  step={100}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Draws</label>
                <input
                  type="number"
                  value={config.draws}
                  onChange={(e) => handleChange('draws', Number(e.target.value))}
                  min={100}
                  max={5000}
                  step={100}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Target Accept</label>
                <input
                  type="number"
                  value={config.target_accept}
                  onChange={(e) => handleChange('target_accept', Number(e.target.value))}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary mt-16"
              disabled={!config.upload_id}
            >
              Run Model
            </button>
          </form>
        </div>
      )}

      {runs && runs.length > 0 && !running && (
        <div className="card mt-16">
          <div className="card-title">Previous Runs</div>
          <table className="channel-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Target</th>
                <th>Status</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="mono">{r.id}</td>
                  <td>{r.target_variable}</td>
                  <td>
                    <span
                      className={`confidence-pip ${
                        r.status === 'completed' ? 'high' : r.status === 'failed' ? 'low' : 'moderate'
                      }`}
                    >
                      {r.status}
                    </span>
                  </td>
                  <td className="mono text-sm">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td>
                    {r.status === 'completed' && (
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => navigate(`/model/runs/${r.id}/results`)}
                      >
                        View Results
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
