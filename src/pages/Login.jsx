import { useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { resetDemoData, errorMessage } from '../api';
import { USE_MOCK } from '../api/config';
import Logo from '../components/Logo';
import Icon from '../components/Icon';
import { Alert, Spinner } from '../components/ui';

const ROLES = [
  {
    role: 'teacher',
    title: 'Continue as Teacher',
    who: 'Dr. S. Kulkarni',
    desc: 'Create exams, upload answer keys and syllabus material, review AI-graded scripts.',
    icon: 'clipboard',
  },
  {
    role: 'student',
    title: 'Continue as Student',
    who: 'Rohan Mehta · 1NI22IS041',
    desc: 'Photograph and submit your answer sheet, then view your marks and feedback.',
    icon: 'camera',
  },
];

export default function Login() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState('');
  const [resetDone, setResetDone] = useState(false);

  if (user) return <Navigate to={`/${user.role}`} replace />;

  const handle = async (role) => {
    setBusy(role);
    setError('');
    try {
      await login(role);
      navigate(`/${role}`);
    } catch (err) {
      setError(errorMessage(err));
      setBusy(null);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <div className="flex flex-1 items-center justify-center px-4 py-12">
        <div className="w-full max-w-3xl">
          <div className="mb-10 text-center">
            <Logo size={40} withText={false} />
            <h1 className="mt-4 text-4xl font-semibold sm:text-5xl">PaperMind</h1>
            <p className="mx-auto mt-3 max-w-lg text-ink-soft">
              Handwritten answer sheets, evaluated consistently. OCR, text correction and syllabus-grounded grading —
              with every uncertain mark routed back to the teacher.
            </p>
          </div>

          {error && <Alert className="mb-4">{error}</Alert>}

          <div className="grid gap-4 sm:grid-cols-2">
            {ROLES.map((r) => (
              <button
                key={r.role}
                type="button"
                onClick={() => handle(r.role)}
                disabled={!!busy}
                className="group flex flex-col rounded-lg border border-line bg-card p-6 text-left shadow-sm transition hover:border-accent/50 hover:shadow-md disabled:opacity-60"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-md bg-accent-soft text-accent">
                  <Icon name={r.icon} size={20} />
                </span>
                <span className="mt-4 font-serif text-xl font-semibold text-ink">{r.title}</span>
                <span className="mt-0.5 text-xs font-medium uppercase tracking-wide text-muted">{r.who}</span>
                <span className="mt-3 flex-1 text-sm text-ink-soft">{r.desc}</span>
                <span className="mt-5 inline-flex items-center gap-1.5 text-sm font-medium text-accent">
                  {busy === r.role ? <Spinner size={16} /> : null}
                  {busy === r.role ? 'Signing in…' : 'Enter'}
                  {busy !== r.role && <Icon name="arrowRight" size={16} className="transition-transform group-hover:translate-x-0.5" />}
                </span>
              </button>
            ))}
          </div>

          <p className="mt-8 text-center text-xs text-muted">
            Demo sign-in — no password required.
            {USE_MOCK && (
              <>
                {' '}
                <button
                  type="button"
                  className="underline underline-offset-2 hover:text-ink"
                  onClick={async () => {
                    await resetDemoData();
                    setResetDone(true);
                  }}
                >
                  Reset demo data
                </button>
                {resetDone && <span className="text-ok"> · reset</span>}
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
