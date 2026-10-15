import AppShell from './AppShell';

const NAV = [{ to: '/student/results', label: 'My Results', icon: 'clipboard' }];

export default function StudentLayout() {
  return <AppShell nav={NAV} roleLabel="Student" />;
}
