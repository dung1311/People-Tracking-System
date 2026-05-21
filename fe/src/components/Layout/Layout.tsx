import type { ReactNode } from 'react';
import { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, Camera, Search, Menu, Film, Settings, LogOut, User as UserIcon } from 'lucide-react';
import { Login } from '../../pages/Login';
import '../../styles/components.css';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [userRole, setUserRole] = useState('');
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
              to="/" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
              {({ isActive }) => (
                  <>
                      <LayoutDashboard size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Tổng quan</span>
                  </>
              )}
            </NavLink>

            <NavLink 
              to="/cameras" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
               {({ isActive }) => (
                  <>
                      <Camera size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Quản lý Camera</span>
                  </>
              )}
            </NavLink>

            <NavLink 
              to="/sessions" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
               {({ isActive }) => (
                  <>
                      <Film size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Phiên bám vết</span>
                  </>
              )}
            </NavLink>

            <NavLink 
              to="/configs" 
              className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
            >
               {({ isActive }) => (
                  <>
                      <Settings size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Hồ sơ cấu hình</span>
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
                      <span style={{ color: isActive ? 'white' : 'var(--text-secondary)', fontSize: '0.95rem' }}>Truy vết Re-ID</span>
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
