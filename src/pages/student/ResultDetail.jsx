import { useParams } from 'react-router-dom';
import { getSubmission } from '../../api';
import useAsync from '../../utils/useAsync';
import QuestionBreakdown from '../../components/QuestionBreakdown';
import ProcessingTracker from '../../components/ProcessingTracker';
import ScoreSummary from '../../components/ScoreSummary';
import { Alert, Card, Loading, PageHeader } from '../../components/ui';

export default function ResultDetail() {
  const { submissionId } = useParams();
  const { data, loading, error, reload } = useAsync(() => getSubmission(submissionId), [submissionId]);

  if (loading && !data) return <Loading />;
  if (error) return <Alert>{error}</Alert>;
  const { exam, submission } = data;
  const back = { to: '/student/results', label: 'All results' };

  if (submission.status === 'processing') {
    return (
      <>
        <PageHeader back={back} title={exam.title} subtitle={exam.subject} />
        <Card className="p-5 sm:p-8">
          <ProcessingTracker submissionId={submission.id} onDone={() => reload({ silent: true })} />
          <p className="mt-6 text-center text-sm text-ink-soft">Your answer sheet is being graded. This page will update automatically.</p>
        </Card>
      </>
    );
  }

  if (submission.answers.length === 0) {
    return (
      <>
        <PageHeader back={back} title={exam.title} subtitle={exam.subject} />
        <Card className="p-6 text-sm text-ink-soft">Your answer sheet hasn’t been graded yet. Please check back later.</Card>
      </>
    );
  }

  return (
    <>
      <PageHeader back={back} title={exam.title} subtitle={exam.subject} />
      <div className="space-y-6">
        <ScoreSummary exam={exam} submission={submission} mode="student" />
        <QuestionBreakdown exam={exam} submission={submission} mode="student" />
      </div>
    </>
  );
}
