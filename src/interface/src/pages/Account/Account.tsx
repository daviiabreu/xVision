import { useEffect, useState } from "react";

import Button from "../../components/ui/Button";
import Icon from "../../components/ui/Icon";
import StatisticCard from "../../components/ui/StatisticCard";
import { useSessao } from "../../contexts/useSessao";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import {
  alterarSenha,
  atualizarEmail,
  buscarEstatisticasConta,
  type AccountStatistics,
} from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

const inputClass =
  "mt-2 h-11 w-full rounded-xl border border-stone-300 px-3 text-base outline-none placeholder:text-stone-400 focus:border-amber-500 focus:ring-2 focus:ring-amber-100";

function mensagemDaConta(error: unknown): string {
  if (error instanceof ErroHttp) {
    if (error.status === 409) return "Este email já está em uso por outra conta.";
    if (error.status === 429)
      return "Muitas tentativas seguidas. Espere alguns minutos e tente novamente.";
    if (error.status === 422) return "Confira os dados preenchidos e tente novamente.";
    if (error.status === 400 && error.detalhe.includes("diferente"))
      return "A nova senha precisa ser diferente da senha atual.";
    if (error.status === 400) return "A senha atual não confere.";
  }
  return error instanceof Error ? error.message : ERROR_MESSAGES.SERVER_ERROR;
}

function useStatistics() {
  const [statistics, setStatistics] = useState<AccountStatistics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    buscarEstatisticasConta().then(
      (response) => {
        if (active) {
          setStatistics(response);
          setError(null);
        }
      },
      (failure: unknown) => {
        if (active)
          setError(failure instanceof Error ? failure.message : ERROR_MESSAGES.SERVER_ERROR);
      },
    );
    return () => {
      active = false;
    };
  }, []);
  return { statistics, error };
}

function useEmailSettings(name: string | null, updateName: (name: string) => void) {
  const [email, setEmail] = useState(name ?? "");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setMessage(null);
    setSaving(true);
    try {
      const principal = await atualizarEmail(email, password);
      updateName(principal.nome);
      setEmail(principal.nome);
      setPassword("");
      setMessage("Email atualizado com sucesso.");
    } catch (failure) {
      setError(mensagemDaConta(failure));
    } finally {
      setSaving(false);
    }
  }
  return { email, setEmail, password, setPassword, message, error, saving, submit };
}

function usePasswordSettings() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setMessage(null);
    if (next !== confirmation) {
      setError("A nova senha e a confirmação precisam ser iguais.");
      return;
    }
    setSaving(true);
    try {
      setMessage(await alterarSenha(current, next));
      setCurrent("");
      setNext("");
      setConfirmation("");
    } catch (failure) {
      setError(mensagemDaConta(failure));
    } finally {
      setSaving(false);
    }
  }
  return {
    current,
    setCurrent,
    next,
    setNext,
    confirmation,
    setConfirmation,
    message,
    error,
    saving,
    submit,
  };
}

function AccountHeader() {
  return (
    <header>
      <p className="text-sm font-semibold text-amber-700">Sua conta</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-stone-950 sm:text-4xl">
        Configurações da conta
      </h1>
      <p className="mt-3 max-w-2xl text-sm leading-6 text-stone-600 sm:text-base">
        Atualize seus dados de acesso e acompanhe o histórico das suas análises.
      </p>
    </header>
  );
}

function StatisticsSection({ state }: { state: ReturnType<typeof useStatistics> }) {
  const inProgress = (state.statistics?.pendentes ?? 0) + (state.statistics?.em_processamento ?? 0);
  return (
    <section aria-labelledby="account-statistics" className="mt-8">
      <div className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#fff8c5] text-amber-800">
          <Icon name="chart" className="h-5 w-5" />
        </span>
        <div>
          <h2 id="account-statistics" className="text-lg font-bold text-stone-950">
            Seu uso do XVision
          </h2>
          <p className="text-sm text-stone-500">Contagem completa de todas as suas análises.</p>
        </div>
      </div>
      {state.error ? (
        <p role="alert" className="mt-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">
          {state.error}
        </p>
      ) : (
        <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatisticCard label="Total" value={state.statistics?.total} tone="stone" />
          <StatisticCard label="Finalizadas" value={state.statistics?.finalizadas} tone="green" />
          <StatisticCard label="Em andamento" value={inProgress} tone="amber" />
          <StatisticCard label="Com falha" value={state.statistics?.falhas} tone="red" />
        </div>
      )}
    </section>
  );
}

function EmailSection({
  name,
  state,
}: {
  name: string | null;
  state: ReturnType<typeof useEmailSettings>;
}) {
  return (
    <section className="rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-7">
      <CardHeader
        icon="mail"
        title="Email de acesso"
        description={`Seu email atual é ${name ?? "não informado"}.`}
      />
      <form onSubmit={(event) => void state.submit(event)} className="mt-6 space-y-5">
        <Field label="Novo email" error={state.error !== null}>
          <input
            type="email"
            required
            autoComplete="email"
            value={state.email}
            onChange={(event) => state.setEmail(event.target.value)}
            aria-invalid={state.error !== null}
            className={inputClass}
          />
        </Field>
        <Field label="Senha atual" error={state.error !== null}>
          <input
            type="password"
            required
            autoComplete="current-password"
            value={state.password}
            onChange={(event) => state.setPassword(event.target.value)}
            aria-invalid={state.error !== null}
            className={inputClass}
          />
        </Field>
        <Feedback error={state.error} message={state.message} />
        <Button type="submit" disabled={state.saving}>
          {state.saving ? "Salvando..." : "Atualizar email"}
        </Button>
      </form>
    </section>
  );
}

function PasswordInput({
  label,
  value,
  setValue,
  error,
  current = false,
}: {
  label: string;
  value: string;
  setValue: (value: string) => void;
  error: boolean;
  current?: boolean;
}) {
  return (
    <Field label={label} error={error}>
      <input
        type="password"
        required
        minLength={current ? undefined : 12}
        autoComplete={current ? "current-password" : "new-password"}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        aria-invalid={error}
        className={inputClass}
      />
    </Field>
  );
}

function PasswordSection({ state }: { state: ReturnType<typeof usePasswordSettings> }) {
  return (
    <section className="rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-7">
      <CardHeader
        icon="lock"
        title="Senha"
        description="Use pelo menos 12 caracteres e não repita sua senha atual."
      />
      <form onSubmit={(event) => void state.submit(event)} className="mt-6 space-y-5">
        <PasswordInput
          label="Senha atual"
          value={state.current}
          setValue={state.setCurrent}
          error={state.error !== null}
          current
        />
        <PasswordInput
          label="Nova senha"
          value={state.next}
          setValue={state.setNext}
          error={state.error !== null}
        />
        <PasswordInput
          label="Confirme a nova senha"
          value={state.confirmation}
          setValue={state.setConfirmation}
          error={state.error !== null}
        />
        <Feedback error={state.error} message={state.message} />
        <Button type="submit" disabled={state.saving}>
          {state.saving ? "Salvando..." : "Atualizar senha"}
        </Button>
      </form>
    </section>
  );
}

function Account() {
  const { nome, atualizarNome } = useSessao();
  const statistics = useStatistics();
  const email = useEmailSettings(nome, atualizarNome);
  const password = usePasswordSettings();
  useTituloDaPagina("Configurações da conta");
  return (
    <div className="mx-auto w-full max-w-6xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <AccountHeader />
      <StatisticsSection state={statistics} />
      <div className="mt-8 grid items-start gap-6 xl:grid-cols-2">
        <EmailSection name={nome} state={email} />
        <PasswordSection state={password} />
      </div>
    </div>
  );
}

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className={`block text-sm font-semibold ${error ? "text-red-700" : "text-stone-700"}`}>
      {label}
      {children}
    </label>
  );
}

function Feedback({ error, message }: { error: string | null; message: string | null }) {
  if (error)
    return (
      <p role="alert" className="rounded-xl bg-red-50 px-3 py-2.5 text-sm font-medium text-red-700">
        {error}
      </p>
    );
  return message ? (
    <p role="status" className="rounded-xl bg-emerald-50 px-3 py-2.5 text-sm text-emerald-800">
      {message}
    </p>
  ) : null;
}

function CardHeader({
  icon,
  title,
  description,
}: {
  icon: "mail" | "lock";
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-stone-100 text-stone-700">
        <Icon name={icon} className="h-5 w-5" />
      </span>
      <div>
        <h2 className="text-lg font-bold text-stone-950">{title}</h2>
        <p className="mt-1 text-sm leading-6 text-stone-500">{description}</p>
      </div>
    </div>
  );
}

export default Account;
