import type { ReactNode } from 'react';
import { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Search, Menu, Film, LogOut, User as UserIcon, AreaChart } from 'lucide-react';
import { Login } from '../../pages/Login';
import '../../styles/components.css';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [userRole, setUserRole] = useState('');
  const [sysStats, setSysStats] = useState<{ cpu: number; ram: number; gpu: number | null }>({ cpu: 0, ram: 0, gpu: null });
  const navigate = useNavigate();

  const checkAuth = () => {
    const token = localStorage.getItem('access_token');
    if (token) {
      setIsLoggedIn(true);
      setUsername(localStorage.getItem('username') || 'Operator');
      setUserRole(localStorage.getItem('user_role') || 'VIEWER');
    } else {
      setIsLoggedIn(false);
    }
  };

  useEffect(() => {
    checkAuth();
  }, []);

  useEffect(() => {
    if (!isLoggedIn) return;
    
    const fetchSysStatus = async () => {
      try {
        const { SystemService } = await import('../../api/services');
        const data = await SystemService.getStatus();
        const cpu = Math.round(data.cpu?.usage_percent || 0);
        const ram = Math.round(data.ram?.usage_percent || 0);
        const gpu = data.gpu && data.gpu.length > 0 ? Math.round(data.gpu[0].utilization_percent) : null;
        setSysStats({ cpu, ram, gpu });
      } catch (err) {
        console.error("Error fetching sidebar system status:", err);
      }
    };

    fetchSysStatus();
    const interval = setInterval(fetchSysStatus, 5000);
    return () => clearInterval(interval);
  }, [isLoggedIn]);

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
    localStorage.removeItem('user_role');
    setIsLoggedIn(false);
    navigate('/');
  };

  if (!isLoggedIn) {
    return <Login onLoginSuccess={checkAuth} />;
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <aside style={{ 
        width: '260px', 
        backgroundColor: 'var(--bg-secondary)', 
        borderRight: '1px solid var(--border-color)',
        padding: 'var(--spacing-lg)',
        display: 'flex',
        flexDirection: 'column',
        position: 'fixed',
        height: '100vh',
        justifyContent: 'space-between',
        zIndex: 100
      }}>
        <div>
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '32px' }}>
              <div style={{ 
                  width: '38px', 
                  height: '38px', 
                  background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 4px 10px rgba(99, 102, 241, 0.3)'
              }}>
                  <Menu color="white" size={18} />
              </div>
              <h1 style={{ fontSize: '1.2rem', fontWeight: 800, letterSpacing: '-0.5px' }}>MCT System</h1>
          </div>

          {/* Navigation Links */}
          <nav style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <NavLink 
              to="/camera_networks" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
               {({ isActive }) => (
                  <>
                      <Film size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Camera Networks</span>
                  </>
               )}
            </NavLink>

            <NavLink 
              to="/roi_analysis" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
              {({ isActive }) => (
                  <>
                      <AreaChart size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Phân tích ROI</span>
                  </>
              )}
            </NavLink>

            <NavLink 
              to="/search" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
              {({ isActive }) => (
                  <>
                      <Search size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Tìm kiếm đối tượng</span>
                  </>
              )}
            </NavLink>


          </nav>
        </div>

        {/* User profile section at the bottom */}
        <div style={{
          borderTop: '1px solid var(--border-color)',
          paddingTop: 'var(--spacing-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          {/* Resource Monitor Widget */}
          {isLoggedIn && (
            <div style={{
              padding: '10px 12px',
              backgroundColor: 'rgba(255, 255, 255, 0.02)',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              fontSize: '0.75rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              color: 'var(--text-secondary)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#a1a1aa', fontWeight: 600, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                <span>Tài nguyên</span>
                <span style={{ color: 'var(--accent-secondary)' }}>LIVE</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>CPU:</span>
                <span style={{ color: 'white', fontWeight: 600 }}>{sysStats.cpu}%</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>RAM:</span>
                <span style={{ color: 'white', fontWeight: 600 }}>{sysStats.ram}%</span>
              </div>
              {sysStats.gpu !== null && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>GPU:</span>
                  <span style={{ color: 'white', fontWeight: 600 }}>{sysStats.gpu}%</span>
                </div>
              )}
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              backgroundColor: 'var(--bg-tertiary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid var(--border-color)'
            }}>
              <UserIcon size={16} color="var(--text-secondary)" />
            </div>
            <div style={{ overflow: 'hidden' }}>
              <span style={{ display: 'block', fontWeight: 600, fontSize: '0.85rem', color: 'white', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
                {username}
              </span>
              <span style={{
                display: 'inline-block',
                fontSize: '0.7rem',
                fontWeight: 700,
                color: userRole === 'ADMIN' ? 'var(--accent-secondary)' : '#a1a1aa',
                backgroundColor: userRole === 'ADMIN' ? 'rgba(236, 72, 153, 0.1)' : 'var(--bg-tertiary)',
                padding: '1px 6px',
                borderRadius: '4px',
                textTransform: 'uppercase'
              }}>
                {userRole}
              </span>
            </div>
          </div>

          <button 
            onClick={handleLogout}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '8px',
              backgroundColor: 'rgba(239, 68, 68, 0.08)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              borderRadius: '8px',
              color: 'var(--error)',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '0.85rem',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.08)';
            }}
          >
            <LogOut size={14} />
            Đăng xuất
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ marginLeft: '260px', width: 'calc(100% - 260px)', padding: 'var(--spacing-xl)', minHeight: '100vh' }}>
        {children}
      </main>
    </div>
  );
}
