import { BrowserRouter, Route, Routes } from "react-router-dom";

import Layout from "./components/layout/Layout";
import { SessaoProvider } from "./contexts/SessaoProvider";
import { VideoProvider } from "./contexts/VideoProvider";
import { useSessao } from "./contexts/useSessao";
import Home from "./pages/Home/Home";
import Login from "./pages/Login/Login";
import NotFound from "./pages/NotFound/NotFound";
import Results from "./pages/Results/Results";

function Rotas() {
  const { autenticado, entrou } = useSessao();

  if (!autenticado) {
    return <Login onEntrou={entrou} />;
  }

  return (
    <VideoProvider>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/results/:id" element={<Results />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </VideoProvider>
  );
}

function App() {
  return (
    <BrowserRouter>
      <SessaoProvider>
        <Layout>
          <Rotas />
        </Layout>
      </SessaoProvider>
    </BrowserRouter>
  );
}

export default App;
