import { useApi } from '../hooks/useApi';
import LoadingSpinner from '../components/LoadingSpinner';
import ChannelTable from '../components/ChannelTable';
import TrustBadge from '../components/TrustBadge';
import ContributionBars from '../components/charts/ContributionBars';
import AdstockCurves from '../components/charts/AdstockCurves';
import SaturationCurves from '../components/charts/SaturationCurves';
import ActionCard from '../components/ActionCard';

export default function IntermediateView({ runId }) {
  const { data, loading, error } = useApi(`/api/model/runs/${runId}/intermediate`);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-banner">{error}</div>;
  if (!data) return null;

  const trust = data.trust || {};
  const recs = data.recommendations || [];

  return (
    <>
      <div className="flex items-center gap-12 mb-16">
        <div>
          <span className="text-sm muted">Trust: </span>
          <TrustBadge tier={trust.overall_tier} />
        </div>
        <div className="text-sm muted">
          Data: {(trust.data_quality_score * 100).toFixed(0)}% | Model:{' '}
          {(trust.model_fit_score * 100).toFixed(0)}% | Calibration:{' '}
          {(trust.calibration_score * 100).toFixed(0)}%
        </div>
      </div>

      <div className="card">
        <div className="card-title">Channel Contributions (with 94% HDI)</div>
        <ContributionBars channels={data.channels} showErrorBars />
      </div>

      <div className="card mt-16">
        <div className="card-title">Channel Performance Matrix</div>
        <ChannelTable channels={data.channels} showCI />
      </div>

      <div className="grid-2 mt-16">
        <div className="card">
          <AdstockCurves adstockCurves={data.adstock_curves} />
          {data.adstock_curves && data.adstock_curves.length > 0 && (
            <table className="posterior-table mt-8">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Decay Rate</th>
                  <th>Half-life</th>
                </tr>
              </thead>
              <tbody>
                {data.adstock_curves.map((c) => (
                  <tr key={c.channel}>
                    <td style={{ fontFamily: 'var(--font-body)' }}>
                      {c.display_name || c.channel}
                    </td>
                    <td>
                      {c.alpha_mean.toFixed(3)} [{c.alpha_hdi_3.toFixed(3)},{' '}
                      {c.alpha_hdi_97.toFixed(3)}]
                    </td>
                    <td>{c.half_life_days.toFixed(1)}d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <SaturationCurves saturationCurves={data.saturation_curves} />
          {data.saturation_curves && data.saturation_curves.length > 0 && (
            <table className="posterior-table mt-8">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Saturation (lambda)</th>
                  <th>94% HDI</th>
                </tr>
              </thead>
              <tbody>
                {data.saturation_curves.map((c) => (
                  <tr key={c.channel}>
                    <td style={{ fontFamily: 'var(--font-body)' }}>
                      {c.display_name || c.channel}
                    </td>
                    <td>{c.lam_mean.toFixed(3)}</td>
                    <td>
                      [{c.lam_hdi_3.toFixed(3)}, {c.lam_hdi_97.toFixed(3)}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {recs.length > 0 && (
        <div className="mt-24">
          <h2 className="card-title mb-8">Recommendations</h2>
          <div className="action-cards">
            {recs.map((rec, i) => (
              <ActionCard key={i} recommendation={rec} />
            ))}
          </div>
        </div>
      )}

      <div className="text-xs muted mt-16" style={{ fontStyle: 'italic' }}>
        {data.caveat}
      </div>
    </>
  );
}
