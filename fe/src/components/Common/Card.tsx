import type { HTMLAttributes, ReactNode } from 'react';
import '../../styles/components.css';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
}

export function Card({ children, className = '', title, action, ...props }: CardProps) {
  return (
    <div className={`card ${className}`} {...props}>
      {(title || action) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-md)' }}>
           {title && <h3 className="text-lg">{title}</h3>}
           {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
