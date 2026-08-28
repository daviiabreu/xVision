import { useState } from "react";

import Button from "../../components/ui/Button";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import { autenticar, registrar } from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

interface LoginProps {
  onEntrou: () => void;
}

function mensagemDaFalha(erro: unknown): string {
  if (erro instanceof ErroHttp) {
    if (erro.status === 409) {
      return "Este email já tem conta. Entre em vez de criar.";
    }

    if (erro.status === 401) {
      return "Email ou senha incorretos.";
    }

    if (erro.status === 422) {
      return "Verifique o email e a senha e tente de novo.";
    }
  }

  return erro instanceof Error ? erro.message : ERROR_MESSAGES.SERVER_ERROR;
}

function Login({ onEntrou }: LoginProps) {
  const [criandoConta, setCriandoConta] = useState(false);
  const [email, setEmail] = useState("");
  const [senha, setSenha] = useState("");
  const [erro, setErro] = useState<string | null>(null);
  const [enviando, setEnviando] = useState(false);

  useTituloDaPagina(criandoConta ? "Criar conta" : "Entrar");

  async function submeter(evento: React.FormEvent) {
    evento.preventDefault();
    setErro(null);
    setEnviando(true);

    try {
      if (criandoConta) {
        await registrar(email, senha);
      }

      await autenticar(email, senha);
      onEntrou();
    } catch (e) {
      console.error(e);
      setErro(mensagemDaFalha(e));
    } finally {
      setEnviando(false);
    }
  }

  const descricaoDoCampo = erro ? "erro-login" : undefined;

  return (
    <div className="flex w-full justify-center px-6 py-12">
      <form
        onSubmit={(evento) => void submeter(evento)}
        className="flex w-full max-w-sm flex-col gap-4 text-left"
      >
        <h1 className="text-center text-3xl font-semibold text-gray-900">
          {criandoConta ? "Criar conta" : "Entrar"}
        </h1>

        <label className="text-sm font-medium text-gray-700">
          Email
          <input
            type="email"
            required
            autoComplete="email"
            aria-invalid={erro !== null}
            aria-describedby={descricaoDoCampo}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1 w-full rounded-sm border border-gray-300 px-3 py-2 text-base"
          />
        </label>

        <div>
          <label className="text-sm font-medium text-gray-700">
            Senha
            <input
              type="password"
              required
              minLength={12}
              autoComplete={criandoConta ? "new-password" : "current-password"}
              aria-invalid={erro !== null}
              aria-describedby={
                [descricaoDoCampo, "senha-minima"].filter(Boolean).join(" ") || undefined
              }
              value={senha}
              onChange={(e) => setSenha(e.target.value)}
              className="mt-1 w-full rounded-sm border border-gray-300 px-3 py-2 text-base"
            />
          </label>

          <p id="senha-minima" className="mt-1 text-xs text-gray-500">
            Mínimo de 12 caracteres.
          </p>
        </div>

        {erro && (
          <p id="erro-login" role="alert" className="text-sm font-medium text-red-600">
            {erro}
          </p>
        )}

        <Button type="submit" disabled={enviando} className="mt-2">
          {enviando ? "Aguarde..." : criandoConta ? "Criar conta" : "Entrar"}
        </Button>

        <button
          type="button"
          onClick={() => {
            setCriandoConta(!criandoConta);
            setErro(null);
          }}
          className="cursor-pointer text-sm text-gray-500 underline"
        >
          {criandoConta ? "Já tenho conta" : "Criar uma conta"}
        </button>
      </form>
    </div>
  );
}

export default Login;
