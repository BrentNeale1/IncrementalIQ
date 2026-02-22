import { useOutletContext, Link } from 'react-router-dom';
import StatStrip from '../components/StatStrip';
import ActionCard from '../components/ActionCard';
import ChannelTable from '../components/ChannelTable';
import ContributionDonut from '../components/charts/ContributionDonut';
import RevenueDecomposition from '../components/charts/RevenueDecomposition';
import LoadingSpinner from '../components/LoadingSpinner';

export default function DashboardPage() {
  const { latestRun, simpleView } = useOutletContext();

  if (!latestRun) {
    return (
      <div className="empty-state">
        <div className="empty-state-title">No model results yet</div>
        <div className="empty-state-text">
          Upload your data and run a model to see your dashboard.
        </div>
        <Link to="/upload" className="btn btn-primary">
          Upload Data
        </Link>
      </div>
    );
  }

  if (latestRun.status !== 'completed') {
    return <LoadingSpinner text="Model is running..." />;
  }

  if (!simpleView) {
    return <LoadingSpinner text="Loading results..." />;
  }

  const recs = simpleView.recommendations || [];

  return (
    <>
      <h1 className="page-title">Dashboard</h1>

      <StatStrip simpleView={simpleView} />

      {recs.length > 0 && (
        <div className="action-cards">
          {recs.map((rec, i) => (
            <ActionCard key={i} recommendation={rec} />
          ))}
        </div>
      )}

      <div className="grid-2 mb-24">
        <div className="card">
          <ContributionDonut
            channels={simpleView.channels}
            baselinePct={simpleView.baseline_pct}
          />
        </div>
        <div className="card">
          <RevenueDecomposition
            channels={simpleView.channels}
            baselinePct={simpleView.baseline_pct}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-title">Channel Performance Matrix</div>
        <ChannelTable channels={simpleView.channels} />
      </div>

      <div className="text-xs muted mt-16" style={{ fontStyle: 'italic' }}>
        {simpleView.caveat}
      </div>
    </>
  );
}
