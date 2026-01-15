import type { InputHTMLAttributes } from 'react';
import '../../styles/components.css';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export function Input({ label, className = '', ...props }: InputProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-xs)' }}>
      {label && <label className="text-secondary text-sm">{label}</label>}
      <input className={`input ${className}`} {...props} />
    </div>
  );
}
