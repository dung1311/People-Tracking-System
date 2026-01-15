import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Camera, Search, Menu } from 'lucide-react';
import '../../styles/components.css';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
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
        height: '100vh'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '40px' }}>
            <div style={{ 
                width: '40px', 
                height: '40px', 
                background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                <Menu color="white" size={20} />
            </div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.5px' }}>MCT System</h1>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <NavLink 
            to="/" 
            className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
            style={{ justifyContent: 'flex-start', width: '100%', border: 'none', background: 'transparent', boxShadow: 'none' }}
          >
            {({ isActive }) => (
                <>
                    <LayoutDashboard size={20} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                    <span style={{ color: isActive ? 'white' : 'var(--text-secondary)' }}>Dashboard</span>
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
                    <Camera size={20} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                    <span style={{ color: isActive ? 'white' : 'var(--text-secondary)' }}>Cameras</span>
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
                    <Search size={20} color={isActive ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                    <span style={{ color: isActive ? 'white' : 'var(--text-secondary)' }}>Search</span>
                </>
            )}
          </NavLink>
        </nav>
      </aside>

      {/* Main Content */}
      <main style={{ marginLeft: '260px', width: 'calc(100% - 260px)', padding: 'var(--spacing-xl)' }}>
        {children}
      </main>
    </div>
  );
}
