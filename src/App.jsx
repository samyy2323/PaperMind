import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import TeacherLayout from './layouts/TeacherLayout';
import StudentLayout from './layouts/StudentLayout';
import Login from './pages/Login';
import NotFound from './pages/NotFound';
import ExamsList from './pages/teacher/ExamsList';
import ExamEditor from './pages/teacher/ExamEditor';
import AnswerSheets from './pages/teacher/AnswerSheets';
import KnowledgeBase from './pages/teacher/KnowledgeBase';
import ReviewDashboard from './pages/teacher/ReviewDashboard';
import SubmissionReview from './pages/teacher/SubmissionReview';
import Analytics from './pages/teacher/Analytics';
import ResultsList from './pages/student/ResultsList';
import ResultDetail from './pages/student/ResultDetail';

function RequireRole({ role, children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (user.role !== role) return <Navigate to={`/${user.role}`} replace />;
  return children;
}

function Home() {
  const { user } = useAuth();
  return <Navigate to={user ? `/${user.role}` : '/login'} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />

      <Route
        path="/teacher"
        element={
          <RequireRole role="teacher">
            <TeacherLayout />
          </RequireRole>
        }
      >
        <Route index element={<Navigate to="exams" replace />} />
        <Route path="exams" element={<ExamsList />} />
        <Route path="exams/new" element={<ExamEditor />} />
        <Route path="exams/:examId" element={<ExamEditor />} />
        <Route path="exams/:examId/sheets" element={<AnswerSheets />} />
        <Route path="knowledge-base" element={<KnowledgeBase />} />
        <Route path="review" element={<ReviewDashboard />} />
        <Route path="review/:submissionId" element={<SubmissionReview />} />
        <Route path="analytics" element={<Analytics />} />
      </Route>

      <Route
        path="/student"
        element={
          <RequireRole role="student">
            <StudentLayout />
          </RequireRole>
        }
      >
        <Route index element={<Navigate to="results" replace />} />
        <Route path="results" element={<ResultsList />} />
        <Route path="results/:submissionId" element={<ResultDetail />} />
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
