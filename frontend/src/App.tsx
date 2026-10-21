import { useEffect, useState } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { api } from './api';
import { Sidebar } from './components/Sidebar';
import { ToastProvider } from './components/Toast';
import { CallPage } from './pages/CallPage';
import { CallsPage } from './pages/CallsPage';
import { ConfigPage } from './pages/ConfigPage';
import { DashboardPage } from './pages/DashboardPage';
import { KnowledgePage } from './pages/KnowledgePage';
import { PronunciationPage } from './pages/PronunciationPage';
import { TestCallPage } from './pages/TestCallPage';
import type { AgentConfig } from './types';

function Shell() {
  const [liveCalls, setLiveCalls] = useState(0);
  const [backendUp, setBackendUp] = useState(true);
  const [agent, setAgent] = useState<AgentConfig | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const stats = await api.stats();
        if (!cancelled) {
          setLiveCalls(stats.live_calls);
          setBackendUp(true);
        }
      } catch {
        if (!cancelled) setBackendUp(false);
      }
    };
    void tick();
    api.config().then(setAgent).catch(() => undefined);
    const id = window.setInterval(tick, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return (
    <div className="shell">
      <Sidebar
        liveCalls={liveCalls}
        backendUp={backendUp}
        language={agent?.language ?? agent?.default_language ?? '…'}
        agentName={agent?.agent_name ?? 'Agent'}
      />
      <div className="content">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/calls" element={<CallsPage />} />
          <Route path="/calls/:callId" element={<CallPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/pronunciation" element={<PronunciationPage />} />
          <Route path="/test-call" element={<TestCallPage />} />
        </Routes>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Shell />
      </BrowserRouter>
    </ToastProvider>
  );
}
