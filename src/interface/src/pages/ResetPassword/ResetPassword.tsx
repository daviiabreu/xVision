import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

import Button from "../../components/ui/Button";
import Icon from "../../components/ui/Icon";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import { redefinirSenha } from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

function mensagemDaFalha(erro: unknown): string {
  if (erro instanceof ErroHttp && erro.status === 400) {
    return "Este link é inválido, já foi usado ou expirou. Solicite um novo link.";
  }
  if (erro instanceof ErroHttp && erro.status === 422) {
    return "O link parece incompleto. Copie o endereço inteiro do email ou solicite um novo link.";
  }
  return erro instanceof Error ? erro.message : ERROR_MESSAGES.SERVER_ERROR;
}

function usePasswordReset(token: string) {
  const [senha, setSenha] = useState("");
  const [confirmacao, setConfirmacao] = useState("");
  const [enviando, setEnviando] = useState(false);
  const [concluido, setConcluido] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  async function submeter(evento: React.FormEvent) {
    evento.preventDefault();
    setErro(null);
    if (senha !== confirmacao) {
      setErro("As duas senhas precisam ser iguais.");
      return;
    }
    setEnviando(true);
    try {
      await redefinirSenha(token, senha);
      setConcluido(true);
    } catch (falha) {
      console.error(falha);
      setErro(mensagemDaFalha(falha));
    } finally {
      setEnviando(false);
    }
  }

  return { senha, setSenha, confirmacao, setConfirmacao, enviando, concluido, erro, submeter };
}

function InvalidLink() {
  return (
    <div className="mt-7">
      <p role="alert" className="rounded-2xl bg-red-50 p-4 text-sm text-red-700">
        O link não contém um token de recuperação.
      </p>
      <Link
        to="/forgot-password"
        className="mt-5 inline-flex text-sm font-semibold text-stone-800 underline underline-offset-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        Solicitar um novo link
      </Link>
    </div>
  );
}

function ResetSuccess() {
  return (
    <div className="mt-7">
      <p role="status" className="rounded-2xl bg-emerald-50 p-4 text-sm text-emerald-800">
        Senha atualizada. Você já pode entrar com a nova credencial.
      </p>
      <Link
        to="/"
        className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-stone-800 underline underline-offset-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        Ir para o login
        <Icon name="arrow-right" className="h-4 w-4" />
      </Link>
    </div>
  );
}

interface PasswordInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  error: boolean;
}

function PasswordInput({ label, value, onChange, error }: PasswordInputProps) {
  return (
    <label className="block text-sm font-semibold text-stone-700">
      {label}
      <input
        type="password"
        required
        minLength={12}
        autoComplete="new-password"
        value={value}
        onChange={(evento) => onChange(evento.target.value)}
        aria-invalid={error}
        aria-describedby={error ? "erro-nova-senha" : undefined}
        className="mt-2 h-11 w-full rounded-xl border border-stone-300 px-3 text-base outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
      />
    </label>
  );
}

function ResetForm({ state }: { state: ReturnType<typeof usePasswordReset> }) {
  return (
    <form onSubmit={(evento) => void state.submeter(evento)} className="mt-7 space-y-5">
      <PasswordInput
        label="Nova senha"
        value={state.senha}
        onChange={state.setSenha}
        error={state.erro !== null}
      />
      <PasswordInput
        label="Confirme a nova senha"
        value={state.confirmacao}
        onChange={state.setConfirmacao}
        error={state.erro !== null}
      />
      {state.erro && (
        <p
          id="erro-nova-senha"
          role="alert"
          className="rounded-xl bg-red-50 px-3 py-2.5 text-sm font-medium text-red-700"
        >
          {state.erro}
        </p>
      )}
      <Button type="submit" disabled={state.enviando} className="w-full">
        {state.enviando ? "Atualizando..." : "Atualizar senha"}
      </Button>
    </form>
  );
}

function ResetPassword() {
  const token = new URLSearchParams(useLocation().search).get("token") ?? "";
  const state = usePasswordReset(token);
  useTituloDaPagina("Nova senha");

  return (
    <div className="mx-auto flex min-h-[calc(100vh-5rem)] w-full max-w-7xl items-center justify-center px-5 py-10 sm:px-8">
      <section className="w-full max-w-md rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_20px_60px_rgba(28,25,23,0.08)] sm:p-8">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-100 text-amber-800">
          <Icon name="check" />
        </span>
        <p className="mt-6 text-sm font-semibold text-amber-700">Proteja sua conta</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-stone-950">
          Crie uma nova senha
        </h1>
        <p className="mt-3 text-sm leading-6 text-stone-600">
          Use pelo menos 12 caracteres e evite reutilizar a senha de outro serviço.
        </p>

        {!token ? (
          <InvalidLink />
        ) : state.concluido ? (
          <ResetSuccess />
        ) : (
          <ResetForm state={state} />
        )}
      </section>
    </div>
  );
}

export default ResetPassword;
