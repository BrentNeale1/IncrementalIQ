import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import DashboardPage from './views/DashboardPage';
import UploadPage from './views/UploadPage';
import ConnectionsPage from './views/ConnectionsPage';
import ModelRunPage from './views/ModelRunPage';
import ResultsPage from './views/ResultsPage';
import ExperimentsPage from './views/ExperimentsPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/connections" element={<ConnectionsPage />} />
        <Route path="/model/run" element={<ModelRunPage />} />
        <Route path="/model/runs/:runId/results" element={<ResultsPage />} />
        <Route path="/experiments" element={<ExperimentsPage />} />
      </Route>
    </Routes>
  );
}
