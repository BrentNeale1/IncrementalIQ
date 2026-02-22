import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { apiPost, apiDelete } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';

const PLATFORMS = ['google_ads', 'meta', 'ga4', 'shopify', 'woocommerce'];

export default function ConnectionsPage() {
  const { data: connections, loading, refetch } = useApi('/api/connections');
  const { data: syncs, refetch: refetchSyncs } = useApi('/api/syncs');
  const { data: uploads } = useApi('/api/uploads');
  const [showForm, setShowForm] = useState(false);

  if (loading) return <LoadingSpinner />;

  return (
    <>
      <h1 className="page-title">API Connections</h1>

      <div className="flex items-center justify-between mb-16">
        <div className="text-sm muted">
          Connect advertising platforms and analytics tools to auto-sync data.
        </div>
        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? 'Cancel' : 'Add Connection'}
        </button>
      </div>

      {showForm && (
        <ConnectionForm
          onCreated={() => {
            setShowForm(false);
            refetch();
          }}
        />
      )}

      {connections && connections.length > 0 ? (
        connections.map((conn) => (
          <ConnectionCard
            key={conn.id}
            connection={conn}
            onRefresh={refetch}
            onSyncComplete={refetchSyncs}
          />
        ))
      ) : (
        <div className="empty-state">
          <div className="empty-state-title">No connections</div>
          <div className="empty-state-text">
            Add an API connection to start syncing data from your advertising platforms.
          </div>
        </div>
      )}

      {/* Sync History */}
      {syncs && syncs.length > 0 && (
        <div className="card mt-24">
          <div className="card-title">Sync History</div>
          <table className="channel-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Connection</th>
                <th>Status</th>
                <th>Rows</th>
                <th>Range</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {syncs.map((s) => (
                <tr key={s.id}>
                  <td className="mono">{s.id}</td>
                  <td className="mono">{s.connection_id}</td>
                  <td>
                    <span
                      className={`confidence-pip ${
                        s.status === 'completed' ? 'high' : s.status === 'failed' ? 'low' : 'moderate'
                      }`}
                    >
                      {s.status}
                    </span>
                  </td>
                  <td className="mono">{s.rows_fetched}</td>
                  <td className="mono text-sm">{s.date_range}</td>
                  <td className="mono text-sm">
                    {new Date(s.started_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Merge Section */}
      <MergeSection uploads={uploads || []} />
    </>
  );
}

function ConnectionForm({ onCreated }) {
  const [form, setForm] = useState({
    platform: '',
    display_name: '',
    credentials: '{}',
    config: '{}',
  });
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      let creds, cfg;
      try {
        creds = JSON.parse(form.credentials);
        cfg = JSON.parse(form.config);
      } catch {
        throw new Error('Credentials and Config must be valid JSON.');
      }
      await apiPost('/api/connections', {
        platform: form.platform,
        display_name: form.display_name,
        credentials: creds,
        config: cfg,
      });
      onCreated();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card mb-16">
      <div className="card-title">New Connection</div>
      {error && <div className="error-banner">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Platform</label>
            <select
              value={form.platform}
              onChange={(e) => setForm({ ...form, platform: e.target.value })}
            >
              <option value="">Select...</option>
              {PLATFORMS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Display Name</label>
            <input
              type="text"
              value={form.display_name}
              onChange={(e) => setForm({ ...form, display_name: e.target.value })}
              placeholder="e.g. My Google Ads"
            />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Credentials (JSON)</label>
          <textarea
            rows={3}
            value={form.credentials}
            onChange={(e) => setForm({ ...form, credentials: e.target.value })}
            style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}
          />
        </div>
        <div className="form-group">
          <label className="form-label">Config (JSON)</label>
          <textarea
            rows={3}
            value={form.config}
            onChange={(e) => setForm({ ...form, config: e.target.value })}
            style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}
          />
          <div className="form-hint">
            Account IDs, property IDs, store URLs, etc.
          </div>
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!form.platform || !form.display_name || submitting}
        >
          {submitting ? 'Creating...' : 'Create Connection'}
        </button>
      </form>
    </div>
  );
}

function ConnectionCard({ connection, onRefresh, onSyncComplete }) {
  const [syncing, setSyncing] = useState(false);
  const [syncForm, setSyncForm] = useState({ start_date: '', end_date: '' });
  const [showSync, setShowSync] = useState(false);
  const [error, setError] = useState(null);

  async function handleTest() {
    try {
      const res = await apiPost(`/api/connections/${connection.id}/test`);
      alert(res.success ? 'Connection successful!' : 'Connection failed.');
    } catch (err) {
      alert(`Test failed: ${err.message}`);
    }
  }

  async function handleDeactivate() {
    try {
      await apiDelete(`/api/connections/${connection.id}`);
      onRefresh();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSync(e) {
    e.preventDefault();
    setSyncing(true);
    setError(null);
    try {
      await apiPost('/api/sync', {
        connection_id: connection.id,
        start_date: syncForm.start_date,
        end_date: syncForm.end_date,
      });
      setShowSync(false);
      onSyncComplete();
    } catch (err) {
      setError(err.message);
    } finally {
      setSyncing(false);
    }
  }

  return (
    <div className="connection-card" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
      <div className="flex items-center gap-16">
        <div>
          <div className="connection-platform">{connection.platform}</div>
          <div className="connection-name">{connection.display_name}</div>
        </div>
        <span
          className={`connection-status ${connection.is_active ? 'active' : 'inactive'}`}
        >
          {connection.is_active ? 'Active' : 'Inactive'}
        </span>
        <div className="connection-actions">
          <button className="btn btn-secondary btn-sm" onClick={handleTest}>
            Test
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setShowSync(!showSync)}>
            Sync
          </button>
          {connection.is_active && (
            <button className="btn btn-danger btn-sm" onClick={handleDeactivate}>
              Deactivate
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-banner mt-8">{error}</div>}

      {showSync && (
        <form onSubmit={handleSync} className="mt-8">
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                value={syncForm.start_date}
                onChange={(e) =>
                  setSyncForm({ ...syncForm, start_date: e.target.value })
                }
              />
            </div>
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input
                type="date"
                value={syncForm.end_date}
                onChange={(e) =>
                  setSyncForm({ ...syncForm, end_date: e.target.value })
                }
              />
            </div>
          </div>
          <button type="submit" className="btn btn-primary btn-sm" disabled={syncing}>
            {syncing ? 'Syncing...' : 'Start Sync'}
          </button>
        </form>
      )}
    </div>
  );
}

function MergeSection({ uploads }) {
  const [selected, setSelected] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function toggleUpload(id) {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  }

  async function handleMerge() {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost('/api/merge', { upload_ids: selected });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (uploads.length < 2) return null;

  return (
    <div className="card mt-24">
      <div className="card-title">Merge Data Sources</div>
      <div className="card-subtitle">
        Select 2 or more uploads to merge into a unified dataset.
      </div>

      <div className="mb-8">
        {uploads.map((u) => (
          <label key={u.id} className="flex items-center gap-8 text-sm" style={{ padding: '4px 0' }}>
            <input
              type="checkbox"
              checked={selected.includes(u.id)}
              onChange={() => toggleUpload(u.id)}
            />
            {u.filename} (ID: {u.id}, {u.row_count} rows)
          </label>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button
        className="btn btn-primary"
        disabled={selected.length < 2 || loading}
        onClick={handleMerge}
      >
        {loading ? 'Merging...' : 'Merge Selected'}
      </button>

      {result && (
        <div className="quality-grid mt-16">
          <div className="quality-item">
            <div className="quality-item-label">New Upload ID</div>
            <div className="quality-item-value">{result.upload_id}</div>
          </div>
          <div className="quality-item">
            <div className="quality-item-label">Rows</div>
            <div className="quality-item-value">{result.rows_stored}</div>
          </div>
          <div className="quality-item">
            <div className="quality-item-label">Status</div>
            <div className="quality-item-value">{result.status}</div>
          </div>
        </div>
      )}
    </div>
  );
}
