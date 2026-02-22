import { useState } from 'react';
import { useParams } from 'react-router-dom';
import SimpleView from './SimpleView';
import IntermediateView from './IntermediateView';
import AdvancedView from './AdvancedView';

const TABS = [
  { key: 'simple', label: 'Simple' },
  { key: 'intermediate', label: 'Intermediate' },
  { key: 'advanced', label: 'Advanced' },
];

export default function ResultsPage() {
  const { runId } = useParams();
  const [activeTab, setActiveTab] = useState('simple');

  return (
    <>
      <h1 className="page-title">Model Results</h1>

      <div className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`tab${activeTab === tab.key ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'simple' && <SimpleView runId={runId} />}
      {activeTab === 'intermediate' && <IntermediateView runId={runId} />}
      {activeTab === 'advanced' && <AdvancedView runId={runId} />}
    </>
  );
}
