import { useApi } from '../hooks/useApi';
import LoadingSpinner from '../components/LoadingSpinner';
import StatStrip from '../components/StatStrip';
import ActionCard from '../components/ActionCard';
import ChannelTable from '../components/ChannelTable';
import ContributionBars from '../components/charts/ContributionBars';

export default function SimpleView({ runId }) {
  const { data, loading, error } = useApi(`/api/model/runs/${runId}/simple`);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-banner">{error}</div>;
  if (!data) return null;

  const recs = data.recommendations || [];

  return (
    <>
      <StatStrip simpleView={data} />

      {recs.length > 0 && (
        <>
          <h2 className="card-title mb-8">Recommendations</h2>
          <div className="action-cards mb-24">
            {recs.map((rec, i) => (
              <ActionCard key={i} recommendation={rec} />
            ))}
          </div>
        </>
      )}

      <div className="card">
        <ContributionBars channels={data.channels} />
      </div>

      <div className="card mt-16">
        <div className="card-title">Channel Performance</div>
        <ChannelTable channels={data.channels} />
      </div>

      <div className="text-xs muted mt-16" style={{ fontStyle: 'italic' }}>
        {data.caveat}
      </div>
    </>
  );
}
