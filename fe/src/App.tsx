import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { Cameras } from './pages/Cameras';
import { Search } from './pages/Search';
import { Sessions } from './pages/Sessions';
import { SessionDetail } from './pages/SessionDetail';
import { Configs } from './pages/Configs';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/cameras" element={<Cameras />} />
          <Route path="/sessions" element={<Sessions />} />
          <Route path="/sessions/:id" element={<SessionDetail />} />
          <Route path="/configs" element={<Configs />} />
          <Route path="/search" element={<Search />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
