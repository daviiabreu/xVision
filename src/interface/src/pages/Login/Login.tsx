import { useState } from "react";
import { Link } from "react-router-dom";

import Button from "../../components/ui/Button";
import Icon from "../../components/ui/Icon";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import { autenticar, registrar } from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

interface LoginProps {
  onEntrou: () => void;
}

interface AccessState {
  criandoConta: boolean;
  email: string;
  senha: string;
  erro: string | null;
  enviando: boolean;
  setEmail: (value: string) => void;
  setSenha: (value: string) => void;
  setModo: (creating: boolean) => void;
  submeter: (event: React.FormEvent) => Promise<void>;
}

function mensagemDaFalha(erro: unknown): string {
  if (erro instanceof ErroHttp) {
    if (erro.status === 409) return "Este email já tem conta. Entre em vez de criar.";
    if (erro.status === 401) return "Email ou senha incorretos.";
    if (erro.status === 422) return "Verifique o email e a senha e tente de novo.";
  }
  return erro instanceof Error ? erro.message : ERROR_MESSAGES.SERVER_ERROR;
}

function useAccess(onEntrou: () => void): AccessState {
  const [criandoConta, setCriandoConta] = useState(false);
  const [email, setEmail] = useState("");
  const [senha, setSenha] = useState("");
  const [erro, setErro] = useState<string | null>(null);
  const [enviando, setEnviando] = useState(false);

  async function submeter(event: React.FormEvent) {
    event.preventDefault();
    setErro(null);
    setEnviando(true);
    try {
      if (criandoConta) await registrar(email, senha);
      await autenticar(email, senha);
      onEntrou();
    } catch (failure) {
      console.error(failure);
      setErro(mensagemDaFalha(failure));
    } finally {
      setEnviando(false);
    }
  }

  return {
    criandoConta,
    email,
    senha,
    erro,
    enviando,
    setEmail,
    setSenha,
    setModo: (creating) => {
      setCriandoConta(creating);
      setErro(null);
    },
    submeter,
  };
}

function MarketingPanel() {
  return (
    <section className="hidden max-w-xl lg:block">
      <span className="inline-flex items-center gap-2 rounded-full bg-[#fff8c5] px-3 py-1.5 text-xs font-bold text-amber-800">
        <Icon name="sparkles" className="h-4 w-4" />
        Inteligência para cada finalização
      </span>
      <h1 className="mt-6 text-5xl font-bold leading-[1.08] tracking-tight text-stone-950">
        Do vídeo às métricas, em um só fluxo.
      </h1>
      <p className="mt-6 max-w-lg text-lg leading-8 text-stone-600">
        Acompanhe a execução do pipeline e revise as evidências de cada etapa antes de usar o
        resultado.
      </p>
      <ul className="mt-8 space-y-4 text-sm font-medium text-stone-700">
        {["Progresso por etapa", "Evidências visuais", "Métricas da finalização"].map((item) => (
          <li key={item} className="flex items-center gap-3">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-stone-950 text-white">
              <Icon name="check" className="h-3.5 w-3.5" />
            </span>
            {item}
          </li>
        ))}
      </ul>
      <Link
        to="/api-docs"
        className="mt-10 inline-flex items-center gap-2 rounded-xl border border-stone-300 bg-white px-4 py-3 text-sm font-semibold text-stone-800 shadow-sm transition hover:border-stone-400 hover:bg-stone-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        Integre seus sistemas pela API
        <Icon name="arrow-right" className="h-4 w-4" />
      </Link>
    </section>
  );
}

function ModeSwitcher({ state }: { state: AccessState }) {
  return (
    <div className="mt-6 grid grid-cols-2 rounded-xl bg-stone-100 p-1" aria-label="Modo de acesso">
      {[
        { label: "Entrar", creating: false },
        { label: "Criar conta", creating: true },
      ].map((option) => (
        <button
          key={option.label}
          type="button"
          aria-label={`Usar modo ${option.label}`}
          aria-pressed={state.criandoConta === option.creating}
          onClick={() => state.setModo(option.creating)}
          className={`rounded-lg px-3 py-2 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] ${
            state.criandoConta === option.creating
              ? "bg-white text-stone-950 shadow-sm"
              : "text-stone-500 hover:text-stone-800"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function EmailField({ state }: { state: AccessState }) {
  return (
    <label className="block text-sm font-semibold text-stone-700">
      Email
      <input
        type="email"
        required
        autoComplete="email"
        aria-invalid={state.erro !== null}
        aria-describedby={state.erro ? "erro-login" : undefined}
        value={state.email}
        onChange={(event) => state.setEmail(event.target.value)}
        placeholder="voce@exemplo.com"
        className="mt-2 h-11 w-full rounded-xl border border-stone-300 px-3 text-base outline-none placeholder:text-stone-400 focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
      />
    </label>
  );
}

function PasswordField({ state }: { state: AccessState }) {
  const describedBy = [state.erro ? "erro-login" : undefined, "senha-minima"]
    .filter(Boolean)
    .join(" ");
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor="senha" className="text-sm font-semibold text-stone-700">
          Senha
        </label>
        {!state.criandoConta && (
          <Link
            to="/forgot-password"
            className="text-xs font-semibold text-amber-700 underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
          >
            Esqueceu a senha?
          </Link>
        )}
      </div>
      <label className="block">
        <span className="sr-only">Senha</span>
        <input
          id="senha"
          type="password"
          required
          minLength={12}
          autoComplete={state.criandoConta ? "new-password" : "current-password"}
          aria-invalid={state.erro !== null}
          aria-describedby={describedBy || undefined}
          value={state.senha}
          onChange={(event) => state.setSenha(event.target.value)}
          className="mt-2 h-11 w-full rounded-xl border border-stone-300 px-3 text-base outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
        />
      </label>
      <p id="senha-minima" className="mt-1.5 text-xs text-stone-500">
        Mínimo de 12 caracteres.
      </p>
    </div>
  );
}

function AccessForm({ state }: { state: AccessState }) {
  return (
    <form onSubmit={(event) => void state.submeter(event)} className="mt-6 space-y-5">
      <EmailField state={state} />
      <PasswordField state={state} />
      {state.erro && (
        <p
          id="erro-login"
          role="alert"
          className="rounded-xl bg-red-50 px-3 py-2.5 text-sm font-medium text-red-700"
        >
          {state.erro}
        </p>
      )}
      <Button type="submit" disabled={state.enviando} className="w-full">
        {state.enviando ? "Aguarde..." : state.criandoConta ? "Criar conta e entrar" : "Entrar"}
      </Button>
    </form>
  );
}

function AccessPanel({ state }: { state: AccessState }) {
  return (
    <section className="rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_20px_60px_rgba(28,25,23,0.08)] sm:p-8">
      <p className="text-sm font-semibold text-amber-700">Acesse o XVision</p>
      <h2 className="mt-2 text-3xl font-bold tracking-tight text-stone-950">
        {state.criandoConta ? "Crie sua conta" : "Bem-vindo de volta"}
      </h2>
      <p className="mt-2 text-sm leading-6 text-stone-500">
        {state.criandoConta
          ? "Use um email válido e uma senha com pelo menos 12 caracteres."
          : "Entre para enviar e acompanhar suas análises."}
      </p>
      <ModeSwitcher state={state} />
      <AccessForm state={state} />
      <p className="mt-6 border-t border-stone-200 pt-5 text-center text-sm text-stone-500 lg:hidden">
        Precisa automatizar o envio?{" "}
        <Link
          to="/api-docs"
          className="font-semibold text-stone-900 underline underline-offset-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
        >
          Consulte a API
        </Link>
      </p>
    </section>
  );
}

function Login({ onEntrou }: LoginProps) {
  const state = useAccess(onEntrou);
  useTituloDaPagina(state.criandoConta ? "Criar conta" : "Entrar");
  return (
    <div className="mx-auto grid min-h-[calc(100vh-5rem)] w-full max-w-7xl items-center gap-10 px-5 py-10 sm:px-8 lg:grid-cols-[1fr_28rem] lg:px-12">
      <MarketingPanel />
      <AccessPanel state={state} />
    </div>
  );
}

export default Login;
