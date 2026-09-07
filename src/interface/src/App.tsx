import { BrowserRouter, Route, Routes } from "react-router-dom";

import Layout from "./components/layout/Layout";
import { SessaoProvider } from "./contexts/SessaoProvider";
import { VideoProvider } from "./contexts/VideoProvider";
import { useSessao } from "./contexts/useSessao";
import Account from "./pages/Account/Account";
import Home from "./pages/Home/Home";
import ApiDocs from "./pages/ApiDocs/ApiDocs";
import ForgotPassword from "./pages/ForgotPassword/ForgotPassword";
import Login from "./pages/Login/Login";
import NotFound from "./pages/NotFound/NotFound";
import ResetPassword from "./pages/ResetPassword/ResetPassword";
import Results from "./pages/Results/Results";

function Rotas() {
  const { autenticado, entrou } = useSessao();

  return (
    <Routes>
      <Route path="/api-docs" element={<ApiDocs />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/reset-password" element={<ResetPassword />} />
      {autenticado ? (
        <Route
          path="*"
          element={
            <VideoProvider>
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/account" element={<Account />} />
                <Route path="/results/:id" element={<Results />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </VideoProvider>
          }
        />
      ) : (
        <Route path="*" element={<Login onEntrou={entrou} />} />
      )}
    </Routes>
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
