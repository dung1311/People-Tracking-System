import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout/Layout';
import { Search } from './pages/Search';
import { CameraNetworks } from './pages/CameraNetworks';
import { CameraNetworkDetail } from './pages/CameraNetworkDetail';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/camera_networks" replace />} />
          <Route path="/camera_networks" element={<CameraNetworks />} />
          <Route path="/camera_networks/:id" element={<CameraNetworkDetail />} />
          <Route path="/search" element={<Search />} />
          <Route path="*" element={<Navigate to="/camera_networks" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
