import { useState } from 'react';
import { apiPostFile } from '../api';
import FileDropZone from '../components/FileDropZone';
import QualityReport from '../components/QualityReport';
import LoadingSpinner from '../components/LoadingSpinner';

export default function UploadPage() {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleFile(file) {
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiPostFile('/api/upload', file);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <>
      <h1 className="page-title">Upload Data</h1>

      <div className="card">
        <div className="card-title">CSV Ingestion</div>
        <div className="card-subtitle">
          Upload a CSV with daily marketing data. Two formats are accepted:
        </div>
        <div className="card-subtitle" style={{ marginTop: '0.5rem' }}>
          <strong>Wide format</strong> (one row per day): date, ga_search_cost,
          ga_search_clicks, ga_search_conv, ga_shopping_*, ga_pmax_*, ga_youtube_*, ga_dgen_*,
          meta_fb_*, meta_insta_*, meta_stories_*, shopify_revenue, shopify_orders,
          ga4_sessions_organic, ga4_sessions_direct, ga4_sessions_email, ga4_sessions_referral.
        </div>
        <div className="card-subtitle" style={{ marginTop: '0.5rem' }}>
          <strong>Long format</strong> (one row per channel per day): date, channel,
          campaign, spend, clicks, in_platform_conversions, revenue,
          orders, sessions_organic, sessions_direct, sessions_email, sessions_referral.
        </div>
        <FileDropZone onFile={handleFile} />
      </div>

      {uploading && <LoadingSpinner text="Uploading and validating..." />}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <>
          <div className="card">
            <div className="card-title">Upload Complete</div>
            <div className="quality-grid">
              <div className="quality-item">
                <div className="quality-item-label">Upload ID</div>
                <div className="quality-item-value">{result.upload_id}</div>
              </div>
              <div className="quality-item">
                <div className="quality-item-label">Status</div>
                <div
                  className={`quality-item-value ${
                    result.status === 'success' ? 'good' : result.status === 'partial' ? 'warn' : 'bad'
                  }`}
                >
                  {result.status}
                </div>
              </div>
              <div className="quality-item">
                <div className="quality-item-label">Rows Stored</div>
                <div className="quality-item-value">{result.rows_stored}</div>
              </div>
            </div>

            {result.warnings && result.warnings.length > 0 && (
              <div className="mt-16">
                <div className="text-sm" style={{ fontWeight: 600 }}>Warnings</div>
                <ul className="flag-list">
                  {result.warnings.map((w, i) => (
                    <li key={i} className="flag-item">
                      <span className="flag-dot" />
                      {w}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <QualityReport quality={result.quality} />
        </>
      )}
    </>
  );
}
