import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import TrustBadge from './TrustBadge';
import { getChannelColor, getChannelName, CAVEAT_TEXT } from '../constants';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard' },
  { to: '/upload', label: 'Data' },
  { to: '/model/run', label: 'Analysis' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/connections', label: 'Connections' },
];

const SIDEBAR_LINKS = [
  { to: '/', label: 'Dashboard' },
  { to: '/model/run', label: 'Analysis' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/connections', label: 'Connections' },
];

export default function Layout() {
  const location = useLocation();
  const { data: runs } = useApi('/api/model/runs');
  const latestRun = runs && runs.length > 0 ? runs[0] : null;
  const runId = latestRun?.id;

  const { data: simpleView } = useApi(
    runId ? `/api/model/runs/${runId}/simple` : null
  );

  const channels = simpleView?.channels || [];
  const trustTier = simpleView?.trust_tier;

  const { data: uploads } = useApi('/api/uploads');
  const latestUpload = uploads && uploads.length > 0 ? uploads[0] : null;
  const { data: quality } = useApi(
    latestUpload ? `/api/uploads/${latestUpload.id}/quality` : null
  );

  return (
    <>
      <header className="topbar">
        <div className="topbar-brand">IncrementIQ</div>
        <nav className="topbar-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="topbar-right">
          {latestUpload && (
            <span className="topbar-client">{latestUpload.filename}</span>
          )}
          {trustTier && <TrustBadge tier={trustTier} />}
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-section-title">Views</div>
          {SIDEBAR_LINKS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `sidebar-link${isActive ? ' active' : ''}`
              }
            >
              {item.label}
            </NavLink>
          ))}
          {latestRun && latestRun.status === 'completed' && (
            <NavLink
              to={`/model/runs/${latestRun.id}/results`}
              className={({ isActive }) =>
                `sidebar-link${isActive ? ' active' : ''}`
              }
            >
              Results
            </NavLink>
          )}
        </div>

        {channels.length > 0 && (
          <div className="sidebar-section">
            <div className="sidebar-section-title">Channels</div>
            {channels.map((ch) => {
              const isLow = ch.confidence === 'Low confidence — more data needed';
              return (
                <div key={ch.channel} className="sidebar-channel">
                  <span
                    className="sidebar-channel-dot"
                    style={{ background: getChannelColor(ch.channel) }}
                  />
                  <span className="sidebar-channel-name">
                    {getChannelName(ch.channel)}
                  </span>
                  <span
                    className={`sidebar-channel-badge ${isLow ? 'amber' : 'green'}`}
                  >
                    {isLow ? 'low conf.' : `${ch.contribution_pct.toFixed(0)}%`}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        <div className="sidebar-data">
          <div className="sidebar-section-title">Data</div>
          <div className="sidebar-data-row">
            <span className="sidebar-data-label">Days ingested</span>
            <span className="sidebar-data-value">
              {quality?.history_days ?? '—'}
            </span>
          </div>
          <div className="sidebar-data-row">
            <span className="sidebar-data-label">Gaps detected</span>
            <span className="sidebar-data-value">
              {quality?.gap_count ?? '—'}
            </span>
          </div>
          <div className="sidebar-data-row">
            <span className="sidebar-data-label">Last run</span>
            <span className="sidebar-data-value">
              {latestRun
                ? new Date(latestRun.created_at).toLocaleDateString()
                : '—'}
            </span>
          </div>
        </div>

        <div className="sidebar-footer">{CAVEAT_TEXT}</div>
      </aside>

      <main className="main">
        <Outlet context={{ latestRun, simpleView, quality }} />
      </main>
    </>
  );
}
