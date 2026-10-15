import { Link } from 'react-router-dom';
import { getStudentSubmissions } from '../../api';
import { useAuth } from '../../context/AuthContext';
import useAsync from '../../utils/useAsync';
import { formatDateTime } from '../../utils/grading';
import Icon from '../../components/Icon';
import ReferencePdfLink from '../../components/ReferencePdfLink';
import { Alert, Badge, EmptyState, Loading, PageHeader, StatusBadge } from '../../components/ui';

export default function ResultsList() {
  const { user } = useAuth();
  const { data: subs, loading, error } = useAsync(() => getStudentSubmissions(user.id), [user.id]);

  return (
    <>
      <PageHeader title="Results" subtitle="Your graded answer sheets, with feedback on every question." />
      {loading && <Loading />}
      {error && <Alert>{error}</Alert>}
      {subs && subs.length === 0 && (
        <EmptyState icon="clipboard" title="No results yet">
          Your teacher uploads your answer sheet after the exam. Once it has been graded, your marks and feedback will show up here.
        </EmptyState>
      )}
      {subs && subs.length > 0 && (
        <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-card">
          {subs.map((s) => (
            <li key={s.id}>
              <Link to={`/student/results/${s.id}`} className="flex items-center gap-4 px-4 py-4 hover:bg-paper/70 sm:px-5">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium uppercase tracking-wide text-muted">{s.examSubject}</div>
                  <div className="font-serif text-lg font-semibold text-ink">{s.examTitle}</div>
                  <div className="mt-0.5 text-xs text-ink-soft">Submitted {formatDateTime(s.submittedAt)} · {s.pages} pages</div>
                </div>
                {s.hasReferencePdf && <ReferencePdfLink submissionId={s.id} />}
                <div className="flex flex-col items-end gap-1.5">
                  {s.total != null && (
                    <span className="font-serif text-2xl font-semibold tabular-nums">
                      {s.total}
                      <span className="text-base font-normal text-muted"> / {s.max}</span>
                    </span>
                  )}
                  {s.status === 'needs_review' ? (
                    <Badge tone="warn" icon="clock">{s.pending} pending review</Badge>
                  ) : (
                    <StatusBadge status={s.status} />
                  )}
                </div>
                <Icon name="chevronRight" size={18} className="text-muted" />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
