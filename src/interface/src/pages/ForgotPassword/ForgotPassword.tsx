import { useState } from "react";
import { Link } from "react-router-dom";

import Button from "../../components/ui/Button";
import Icon from "../../components/ui/Icon";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import { solicitarRedefinicao } from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

function mensagemDaFalha(erro: unknown): string {
  if (erro instanceof ErroHttp && erro.status === 503) {
    return "O envio de email ainda não foi configurado neste ambiente. Fale com quem administra o XVision.";
  }
  if (erro instanceof ErroHttp && erro.status === 429) {
    return "Muitas tentativas seguidas. Aguarde um pouco antes de tentar novamente.";
  }
  return erro instanceof Error ? erro.message : ERROR_MESSAGES.SERVER_ERROR;
}

function usePasswordRecovery() {
  const [email, setEmail] = useState("");
  const [enviando, setEnviando] = useState(false);
  const [mensagem, setMensagem] = useState<string | null>(null);
  const [erro, setErro] = useState<string | null>(null);

  async function submeter(evento: React.FormEvent) {
    evento.preventDefault();
    setEnviando(true);
    setErro(null);
    try {
      setMensagem(await solicitarRedefinicao(email));
    } catch (falha) {
      console.error(falha);
      setErro(mensagemDaFalha(falha));
    } finally {
      setEnviando(false);
    }
  }

  return { email, setEmail, enviando, mensagem, erro, submeter };
}

function RecoverySuccess({ mensagem }: { mensagem: string }) {
  return (
    <div className="mt-7">
      <p role="status" className="rounded-2xl bg-emerald-50 p-4 text-sm text-emerald-800">
        {mensagem}
      </p>
      <BackToLogin className="mt-5 inline-flex" />
    </div>
  );
}

function BackToLogin({ className = "flex" }: { className?: string }) {
  return (
    <Link
      to="/"
      className={`${className} items-center justify-center gap-2 rounded-lg py-2 text-sm font-semibold text-stone-600 hover:text-stone-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]`}
    >
      <Icon name="arrow-left" className="h-4 w-4" />
      Voltar ao login
    </Link>
  );
}

function RecoveryForm({ state }: { state: ReturnType<typeof usePasswordRecovery> }) {
  return (
    <form onSubmit={(evento) => void state.submeter(evento)} className="mt-7 space-y-5">
      <label className="block text-sm font-semibold text-stone-700">
        Email
        <input
          type="email"
          required
          autoComplete="email"
          value={state.email}
          onChange={(evento) => state.setEmail(evento.target.value)}
          aria-invalid={state.erro !== null}
          aria-describedby={state.erro ? "erro-recuperacao" : undefined}
          placeholder="voce@exemplo.com"
          className="mt-2 h-11 w-full rounded-xl border border-stone-300 px-3 text-base outline-none placeholder:text-stone-400 focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
        />
      </label>
      {state.erro && (
        <p
          id="erro-recuperacao"
          role="alert"
          className="rounded-xl bg-red-50 px-3 py-2.5 text-sm font-medium text-red-700"
        >
          {state.erro}
        </p>
      )}
      <Button type="submit" disabled={state.enviando} className="w-full">
        {state.enviando ? "Enviando..." : "Enviar link de recuperação"}
      </Button>
      <BackToLogin />
    </form>
  );
}

function ForgotPassword() {
  const state = usePasswordRecovery();
  useTituloDaPagina("Recuperar senha");

  return (
    <div className="mx-auto flex min-h-[calc(100vh-5rem)] w-full max-w-7xl items-center justify-center px-5 py-10 sm:px-8">
      <section className="w-full max-w-md rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_20px_60px_rgba(28,25,23,0.08)] sm:p-8">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-100 text-amber-800">
          <Icon name="refresh" />
        </span>
        <p className="mt-6 text-sm font-semibold text-amber-700">Recuperação de acesso</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-stone-950">
          Redefina sua senha
        </h1>
        <p className="mt-3 text-sm leading-6 text-stone-600">
          Informe o email da sua conta. Se ele estiver cadastrado, enviaremos um link válido por 30
          minutos.
        </p>

        {state.mensagem ? (
          <RecoverySuccess mensagem={state.mensagem} />
        ) : (
          <RecoveryForm state={state} />
        )}
      </section>
    </div>
  );
}

export default ForgotPassword;
