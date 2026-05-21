import { useState } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { Input } from '../components/Common/Input';
import { AuthService } from '../api/services';
import { LogIn, ShieldAlert } from 'lucide-react';

interface LoginProps {
  onLoginSuccess: () => void;
}

export function Login({ onLoginSuccess }: LoginProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Vui lòng nhập đầy đủ tài khoản và mật khẩu');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await AuthService.login(username, password);
      localStorage.setItem('access_token', response.access_token);
      
      // Get user profile details
      const user = await AuthService.me();
      localStorage.setItem('user_role', user.role);
      localStorage.setItem('username', user.username);
      
      onLoginSuccess();
    } catch (err: any) {
      logger.error('Login failed', err);
      setError('Tài khoản hoặc mật khẩu không chính xác');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: 'var(--bg-primary)',
      backgroundImage: 'radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%), radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.1) 0%, transparent 40%)',
      position: 'fixed',
      inset: 0,
      zIndex: 1000
    }}>
      <Card style={{
        width: '420px',
        padding: '3rem 2.5rem',
        backdropFilter: 'blur(8px)',
        backgroundColor: 'rgba(26, 29, 36, 0.85)',
        border: '1px solid rgba(63, 66, 80, 0.6)',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.5)',
        borderRadius: '16px'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{
            width: '56px',
            height: '56px',
            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem',
            boxShadow: '0 8px 16px -4px rgba(99, 102, 241, 0.5)'
          }}>
            <LogIn color="white" size={26} />
          </div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px', marginBottom: '0.25rem' }}>Đăng nhập</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>MCT Production People Tracking System</p>
        </div>

        {error && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            color: 'var(--error)',
            fontSize: '0.875rem',
            marginBottom: '1.5rem',
          }}>
            <ShieldAlert size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Tên đăng nhập</label>
            <Input
              type="text"
              placeholder="Nhập tên đăng nhập"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={{ width: '100%', height: '44px', fontSize: '0.95rem' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Mật khẩu</label>
            <Input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ width: '100%', height: '44px', fontSize: '0.95rem' }}
            />
          </div>

          <Button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              height: '46px',
              fontWeight: 600,
              fontSize: '1rem',
              marginTop: '1rem',
              background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
              border: 'none',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
            }}
          >
            {loading ? 'Đang xác thực...' : 'Đăng nhập'}
          </Button>
        </form>
      </Card>
    </div>
  );
}

const logger = {
  error: (msg: string, err: any) => console.error(msg, err)
};
