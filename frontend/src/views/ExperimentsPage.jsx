import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { apiPost, apiPostFile } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import FileDropZone from '../components/FileDropZone';

export default function ExperimentsPage() {
  const { data: uploads } = useApi('/api/uploads');
  const successUploads = (uploads || []).filter(
    (u) => u.status === 'success' || u.status === 'partial'
  );

  return (
    <>
      <h1 className="page-title">Experiments</h1>
      <LagDetection uploads={successUploads} />
      <SpendScaling uploads={successUploads} />
      <ProductExperiment />
    </>
  );
}

function LagDetection({ uploads }) {
  const [uploadId, setUploadId] = useState('');
  const [target, setTarget] = useState('revenue');
  const [maxLag, setMaxLag] = useState(21);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleRun(e) {
    e.preventDefault();
    if (!uploadId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost('/api/experiments/lag-detection', {
        upload_id: Number(uploadId),
        target,
        max_lag: maxLag,
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card experiment-section">
      <div className="experiment-section-title">Lag Detection</div>
      <div className="card-subtitle">
        Detect the optimal lag (days) between each channel's spend and outcome
        using cross-correlation analysis.
      </div>

      <form onSubmit={handleRun}>
        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Dataset</label>
            <select value={uploadId} onChange={(e) => setUploadId(e.target.value)}>
              <option value="">Select upload...</option>
              {uploads.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.filename} (ID: {u.id})
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Target</label>
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="revenue">Revenue</option>
              <option value="orders">Orders</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Max Lag (days)</label>
            <input
              type="number"
              value={maxLag}
              onChange={(e) => setMaxLag(Number(e.target.value))}
              min={1}
              max={60}
            />
          </div>
        </div>
        <button type="submit" className="btn btn-primary" disabled={!uploadId || loading}>
          {loading ? 'Running...' : 'Detect Lags'}
        </button>
      </form>

      {error && <div className="error-banner mt-8">{error}</div>}

      {result && result.channel_lags && (
        <table className="channel-table mt-16">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Optimal Lag</th>
              <th>Peak Correlation</th>
              <th>Significant</th>
            </tr>
          </thead>
          <tbody>
            {result.channel_lags.map((cl) => (
              <tr key={cl.channel}>
                <td className="channel-name">{cl.channel}</td>
                <td className="mono">{cl.optimal_lag_days} days</td>
                <td className="mono">{cl.peak_correlation.toFixed(3)}</td>
                <td>
                  <span
                    className={`confidence-pip ${cl.significant ? 'high' : 'low'}`}
                  >
                    {cl.significant ? 'Yes' : 'No'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function SpendScaling({ uploads }) {
  const [form, setForm] = useState({
    upload_id: '',
    target_channel: '',
    pre_period_start: '',
    pre_period_end: '',
    post_period_start: '',
    post_period_end: '',
    outcome_column: 'revenue',
    alpha: 0.05,
    mmm_predicted_impact: '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleChange(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleRun(e) {
    e.preventDefault();
    if (!form.upload_id || !form.target_channel) return;
    setLoading(true);
    setError(null);
    try {
      const body = {
        ...form,
        upload_id: Number(form.upload_id),
        alpha: Number(form.alpha),
      };
      if (form.mmm_predicted_impact) {
        body.mmm_predicted_impact = Number(form.mmm_predicted_impact);
      } else {
        body.mmm_predicted_impact = null;
      }
      const res = await apiPost('/api/experiments/spend-scaling', body);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card experiment-section">
      <div className="experiment-section-title">Spend-Scaling Experiment</div>
      <div className="card-subtitle">
        Use CausalImpact to estimate the incremental effect of a deliberate spend
        change on a specific channel.
      </div>

      <form onSubmit={handleRun}>
        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Dataset</label>
            <select
              value={form.upload_id}
              onChange={(e) => handleChange('upload_id', e.target.value)}
            >
              <option value="">Select upload...</option>
              {uploads.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.filename} (ID: {u.id})
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Target Channel</label>
            <input
              type="text"
              placeholder="e.g. google_search"
              value={form.target_channel}
              onChange={(e) => handleChange('target_channel', e.target.value)}
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Pre-period Start</label>
            <input
              type="date"
              value={form.pre_period_start}
              onChange={(e) => handleChange('pre_period_start', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Pre-period End</label>
            <input
              type="date"
              value={form.pre_period_end}
              onChange={(e) => handleChange('pre_period_end', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Post-period Start</label>
            <input
              type="date"
              value={form.post_period_start}
              onChange={(e) => handleChange('post_period_start', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Post-period End</label>
            <input
              type="date"
              value={form.post_period_end}
              onChange={(e) => handleChange('post_period_end', e.target.value)}
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Outcome</label>
            <select
              value={form.outcome_column}
              onChange={(e) => handleChange('outcome_column', e.target.value)}
            >
              <option value="revenue">Revenue</option>
              <option value="orders">Orders</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Alpha</label>
            <input
              type="number"
              value={form.alpha}
              onChange={(e) => handleChange('alpha', e.target.value)}
              min={0.01}
              max={0.2}
              step={0.01}
            />
          </div>
          <div className="form-group">
            <label className="form-label">MMM Predicted Impact</label>
            <input
              type="number"
              value={form.mmm_predicted_impact}
              onChange={(e) => handleChange('mmm_predicted_impact', e.target.value)}
              placeholder="Optional"
            />
            <div className="form-hint">Compare with MMM prediction for calibration</div>
          </div>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Running CausalImpact...' : 'Run Experiment'}
        </button>
      </form>

      {error && <div className="error-banner mt-8">{error}</div>}

      {result && (
        <div className="mt-16">
          <div className="quality-grid">
            <div className="quality-item">
              <div className="quality-item-label">Estimated Impact</div>
              <div className="quality-item-value">
                ${result.estimated_impact?.toLocaleString()}
              </div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">CI</div>
              <div className="quality-item-value mono" style={{ fontSize: '0.85rem' }}>
                [${result.impact_ci_lower?.toLocaleString()},{' '}
                ${result.impact_ci_upper?.toLocaleString()}]
              </div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">P-value</div>
              <div className="quality-item-value">{result.p_value?.toFixed(4)}</div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Significant</div>
              <div
                className={`quality-item-value ${result.significant ? 'good' : 'warn'}`}
              >
                {result.significant ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
          {result.recalibration_recommended && (
            <div className="error-banner mt-8">
              Discrepancy vs MMM prediction exceeds 30%. Recalibration recommended.
            </div>
          )}
          {result.summary && (
            <div className="text-sm muted mt-8">{result.summary}</div>
          )}
        </div>
      )}
    </div>
  );
}

function ProductExperiment() {
  const [form, setForm] = useState({
    advertised_products: '',
    campaign_name: '',
    campaign_type: '',
    bidding_strategy: '',
  });
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleChange(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleRun(e) {
    e.preventDefault();
    if (!file || !form.campaign_name) return;
    setLoading(true);
    setError(null);

    const productList = form.advertised_products
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean);

    const formData = new FormData();
    formData.append('file', file);

    // FastAPI expects form fields for the Pydantic model
    const params = new URLSearchParams({
      advertised_products: JSON.stringify(productList),
      campaign_name: form.campaign_name,
      campaign_type: form.campaign_type,
      bidding_strategy: form.bidding_strategy,
    });

    try {
      const res = await fetch(`/api/experiments/product?${params}`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail));
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card experiment-section">
      <div className="experiment-section-title">Product-Level Experiment</div>
      <div className="card-subtitle">
        Google Ads only. Compare advertised vs. non-advertised product performance.
        PMax and auto-bidding campaigns are refused due to selection bias.
      </div>

      <form onSubmit={handleRun}>
        <div className="form-group">
          <label className="form-label">Product Data CSV</label>
          <FileDropZone
            onFile={setFile}
            label="Drop product CSV (product_id, revenue, orders)"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Advertised Products</label>
          <input
            type="text"
            placeholder="product_1, product_2, product_3"
            value={form.advertised_products}
            onChange={(e) => handleChange('advertised_products', e.target.value)}
          />
          <div className="form-hint">Comma-separated product IDs</div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Campaign Name</label>
            <input
              type="text"
              value={form.campaign_name}
              onChange={(e) => handleChange('campaign_name', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Campaign Type</label>
            <input
              type="text"
              value={form.campaign_type}
              onChange={(e) => handleChange('campaign_type', e.target.value)}
              placeholder="e.g. search, shopping"
            />
            <div className="form-hint">PMax / Smart Shopping will be rejected</div>
          </div>
          <div className="form-group">
            <label className="form-label">Bidding Strategy</label>
            <input
              type="text"
              value={form.bidding_strategy}
              onChange={(e) => handleChange('bidding_strategy', e.target.value)}
              placeholder="e.g. manual_cpc"
            />
            <div className="form-hint">Auto-bidding strategies will be rejected</div>
          </div>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading || !file}>
          {loading ? 'Running...' : 'Run Product Experiment'}
        </button>
      </form>

      {error && <div className="error-banner mt-8">{error}</div>}

      {result && (
        <div className="mt-16">
          <div className="quality-grid">
            <div className="quality-item">
              <div className="quality-item-label">Advertised Products</div>
              <div className="quality-item-value">{result.advertised_product_count}</div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Control Products</div>
              <div className="quality-item-value">{result.non_advertised_product_count}</div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Revenue Lift</div>
              <div
                className={`quality-item-value ${
                  result.revenue_lift_pct > 0 ? 'good' : 'warn'
                }`}
              >
                {result.revenue_lift_pct?.toFixed(1)}%
              </div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Orders Lift</div>
              <div
                className={`quality-item-value ${
                  result.orders_lift_pct > 0 ? 'good' : 'warn'
                }`}
              >
                {result.orders_lift_pct?.toFixed(1)}%
              </div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Revenue Significant</div>
              <div
                className={`quality-item-value ${result.revenue_significant ? 'good' : 'warn'}`}
              >
                {result.revenue_significant ? 'Yes' : 'No'} (p={result.p_value_revenue?.toFixed(4)})
              </div>
            </div>
            <div className="quality-item">
              <div className="quality-item-label">Orders Significant</div>
              <div
                className={`quality-item-value ${result.orders_significant ? 'good' : 'warn'}`}
              >
                {result.orders_significant ? 'Yes' : 'No'} (p={result.p_value_orders?.toFixed(4)})
              </div>
            </div>
          </div>
          {result.warnings && result.warnings.length > 0 && (
            <ul className="flag-list mt-8">
              {result.warnings.map((w, i) => (
                <li key={i} className="flag-item">
                  <span className="flag-dot" />
                  {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
