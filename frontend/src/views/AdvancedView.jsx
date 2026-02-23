import { useState } from 'react';
import { useApi, useMutation } from '../hooks/useApi';
import LoadingSpinner from '../components/LoadingSpinner';
import TrustBadge from '../components/TrustBadge';
import ContributionBars from '../components/charts/ContributionBars';

export default function AdvancedView({ runId }) {
  const { data, loading, error, refetch } = useApi(`/api/model/runs/${runId}/advanced`);
  const { mutate, loading: validating, error: valError } = useMutation();
  const [showSensitivity, setShowSensitivity] = useState(false);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-banner">{error}</div>;
  if (!data) return null;

  const trust = data.trust || {};
  const diag = data.diagnostics || {};
  const validation = data.validation;
  const experiments = data.experiments || [];
  const controls = data.controls || [];

  async function handleValidation() {
    try {
      await mutate('/api/model/validate', {
        run_id: Number(runId),
        holdout_fraction: 0.2,
        run_sensitivity: showSensitivity,
      });
      refetch();
    } catch (_) {
      // error is captured in valError
    }
  }

  return (
    <>
      <div className="flex items-center gap-12 mb-16">
        <TrustBadge tier={trust.overall_tier} />
        <span className="mono text-sm">
          Overall: {(trust.overall_score * 100).toFixed(0)}% | Data:{' '}
          {(trust.data_quality_score * 100).toFixed(0)}% | Fit:{' '}
          {(trust.model_fit_score * 100).toFixed(0)}% | Cal:{' '}
          {(trust.calibration_score * 100).toFixed(0)}%
        </span>
      </div>

      {/* Posterior Distribution Summary */}
      <div className="card">
        <div className="card-title">Posterior Distribution Summary</div>
        <table className="posterior-table">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Beta Mean</th>
              <th>Beta SD</th>
              <th>Beta 94% HDI</th>
              <th>Adstock Alpha</th>
              <th>Saturation Lambda</th>
              <th>Contribution %</th>
              <th>Contribution HDI</th>
            </tr>
          </thead>
          <tbody>
            {(data.channels || []).map((ch) => (
              <tr key={ch.channel}>
                <td style={{ fontFamily: 'var(--font-body)', fontWeight: 500 }}>
                  {ch.display_name || ch.channel}
                </td>
                <td>{ch.beta_mean.toFixed(4)}</td>
                <td>{ch.beta_sd.toFixed(4)}</td>
                <td>
                  [{ch.beta_hdi_3.toFixed(4)}, {ch.beta_hdi_97.toFixed(4)}]
                </td>
                <td>
                  {ch.adstock_alpha_mean.toFixed(3)} [{ch.adstock_alpha_hdi_3.toFixed(3)},{' '}
                  {ch.adstock_alpha_hdi_97.toFixed(3)}]
                </td>
                <td>
                  {ch.saturation_lam_mean.toFixed(3)} [{ch.saturation_lam_hdi_3.toFixed(3)},{' '}
                  {ch.saturation_lam_hdi_97.toFixed(3)}]
                </td>
                <td>{ch.contribution_pct.toFixed(1)}%</td>
                <td>
                  [{ch.contribution_hdi_3.toFixed(1)}%, {ch.contribution_hdi_97.toFixed(1)}%]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Control Variable Summary */}
      {controls.length > 0 && (
        <div className="card mt-16">
          <div className="card-title">Control Variable Posteriors</div>
          <table className="posterior-table">
            <thead>
              <tr>
                <th>Control</th>
                <th>Gamma Mean</th>
                <th>Gamma SD</th>
                <th>Gamma 94% HDI</th>
                <th>Contribution %</th>
                <th>Contribution HDI</th>
              </tr>
            </thead>
            <tbody>
              {controls.map((ctrl) => (
                <tr key={ctrl.control}>
                  <td style={{ fontFamily: 'var(--font-body)', fontWeight: 500 }}>
                    {ctrl.display_name || ctrl.control}
                  </td>
                  <td>{ctrl.gamma_mean.toFixed(4)}</td>
                  <td>{ctrl.gamma_sd.toFixed(4)}</td>
                  <td>
                    [{ctrl.gamma_hdi_3.toFixed(4)}, {ctrl.gamma_hdi_97.toFixed(4)}]
                  </td>
                  <td>{ctrl.contribution_pct.toFixed(1)}%</td>
                  <td>
                    [{ctrl.contribution_hdi_3.toFixed(1)}%, {ctrl.contribution_hdi_97.toFixed(1)}%]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Contribution Chart */}
      <div className="card mt-16">
        <ContributionBars channels={data.channels} controls={controls} showErrorBars />
      </div>

      {/* Diagnostics */}
      <div className="card mt-16">
        <div className="card-title">Model Diagnostics</div>
        <div className="quality-grid">
          <div className="quality-item">
            <div className="quality-item-label">R-squared</div>
            <div className="quality-item-value">
              {diag.r_squared_mean?.toFixed(3) ?? '—'}
            </div>
          </div>
          <div className="quality-item">
            <div className="quality-item-label">R-sq HDI</div>
            <div className="quality-item-value mono" style={{ fontSize: '0.85rem' }}>
              [{diag.r_squared_hdi_3?.toFixed(3)}, {diag.r_squared_hdi_97?.toFixed(3)}]
            </div>
          </div>
          <div className="quality-item">
            <div className="quality-item-label">MAPE</div>
            <div
              className={`quality-item-value ${
                diag.mape_mean > 0.15 ? 'warn' : 'good'
              }`}
            >
              {diag.mape_mean != null ? `${(diag.mape_mean * 100).toFixed(1)}%` : '—'}
            </div>
          </div>
          <div className="quality-item">
            <div className="quality-item-label">Divergences</div>
            <div
              className={`quality-item-value ${
                diag.divergences > 0 ? 'warn' : 'good'
              }`}
            >
              {diag.divergences ?? 0}
              {diag.divergence_pct != null && (
                <span className="text-xs muted">
                  {' '}
                  ({(diag.divergence_pct * 100).toFixed(1)}%)
                </span>
              )}
            </div>
          </div>
        </div>

        {diag.warnings && diag.warnings.length > 0 && (
          <ul className="flag-list mt-8">
            {diag.warnings.map((w, i) => (
              <li key={i} className="flag-item">
                <span className="flag-dot" />
                {w}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Data Quality Flags */}
      {data.data_quality_flags && data.data_quality_flags.length > 0 && (
        <div className="card mt-16">
          <div className="card-title">Data Quality Flags</div>
          <ul className="flag-list">
            {data.data_quality_flags.map((f, i) => (
              <li key={i} className="flag-item">
                <span className="flag-dot" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {trust.flags && trust.flags.length > 0 && (
        <div className="card mt-16">
          <div className="card-title">Trust Score Flags</div>
          <ul className="flag-list">
            {trust.flags.map((f, i) => (
              <li key={i} className="flag-item">
                <span className="flag-dot" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Validation */}
      <div className="card mt-16">
        <div className="card-title">Validation</div>
        {validation ? (
          <div>
            {validation.holdout && (
              <div className="quality-grid mb-16">
                <div className="quality-item">
                  <div className="quality-item-label">Holdout MAPE</div>
                  <div
                    className={`quality-item-value ${
                      validation.holdout.mape > 0.15 ? 'warn' : 'good'
                    }`}
                  >
                    {(validation.holdout.mape * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="quality-item">
                  <div className="quality-item-label">Coverage</div>
                  <div
                    className={`quality-item-value ${
                      validation.holdout.coverage < 0.8 ? 'warn' : 'good'
                    }`}
                  >
                    {(validation.holdout.coverage * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
            {validation.sensitivity && (
              <div>
                <div className="text-sm" style={{ fontWeight: 600 }}>
                  Sensitivity Analysis
                </div>
                <table className="posterior-table mt-8">
                  <thead>
                    <tr>
                      <th>Channel</th>
                      <th>Base Beta</th>
                      <th>Perturbed Beta</th>
                      <th>Change %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {validation.sensitivity.channels?.map((ch) => (
                      <tr key={ch.channel}>
                        <td style={{ fontFamily: 'var(--font-body)' }}>{ch.channel}</td>
                        <td>{ch.base_beta?.toFixed(4)}</td>
                        <td>{ch.perturbed_beta?.toFixed(4)}</td>
                        <td>{ch.change_pct?.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <div>
            <p className="text-sm muted mb-8">
              Run holdout validation and posterior predictive checks.
            </p>
            <div className="flex items-center gap-12 mb-8">
              <label className="flex items-center gap-8 text-sm">
                <input
                  type="checkbox"
                  checked={showSensitivity}
                  onChange={(e) => setShowSensitivity(e.target.checked)}
                />
                Include sensitivity analysis
              </label>
            </div>
            {valError && <div className="error-banner">{valError}</div>}
            <button
              className="btn btn-primary"
              onClick={handleValidation}
              disabled={validating}
            >
              {validating ? 'Running...' : 'Run Validation'}
            </button>
          </div>
        )}
      </div>

      {/* Experiments */}
      {experiments.length > 0 && (
        <div className="card mt-16">
          <div className="card-title">Experiment Results</div>
          {experiments.map((exp, i) => (
            <div key={i} className="card" style={{ background: 'var(--bg-cream)' }}>
              <div className="text-sm" style={{ fontWeight: 600 }}>
                {exp.channel || exp.campaign_name || `Experiment ${i + 1}`}
              </div>
              {exp.estimated_impact != null && (
                <div className="quality-grid mt-8">
                  <div className="quality-item" style={{ background: '#fff' }}>
                    <div className="quality-item-label">Impact</div>
                    <div className="quality-item-value">
                      ${exp.estimated_impact?.toLocaleString()}
                    </div>
                  </div>
                  <div className="quality-item" style={{ background: '#fff' }}>
                    <div className="quality-item-label">CI</div>
                    <div className="quality-item-value mono" style={{ fontSize: '0.82rem' }}>
                      [${exp.impact_ci_lower?.toLocaleString()},{' '}
                      ${exp.impact_ci_upper?.toLocaleString()}]
                    </div>
                  </div>
                  <div className="quality-item" style={{ background: '#fff' }}>
                    <div className="quality-item-label">Significant</div>
                    <div
                      className={`quality-item-value ${
                        exp.significant ? 'good' : 'warn'
                      }`}
                    >
                      {exp.significant ? 'Yes' : 'No'}
                    </div>
                  </div>
                </div>
              )}
              {exp.summary && (
                <div className="text-sm muted mt-8">{exp.summary}</div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="text-xs muted mt-16" style={{ fontStyle: 'italic' }}>
        {data.caveat}
      </div>
    </>
  );
}
