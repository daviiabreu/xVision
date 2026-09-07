import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import logo from "../../assets/logo.svg";
import { useSessao } from "../../contexts/useSessao";
import Icon from "../ui/Icon";

const SIDEBAR_COLLAPSED_KEY = "xvision:sidebar-collapsed";
const links = [
  { to: "/", label: "Análises", icon: "sparkles" as const },
  { to: "/api-docs", label: "Documentação da API", icon: "file-video" as const },
];

function Navigation({
  compact = false,
  currentPath,
  onNavigate,
}: {
  compact?: boolean;
  currentPath: string;
  onNavigate?: () => void;
}) {
  return (
    <nav aria-label="Navegação principal" className="space-y-2">
      {links.map((link) => (
        <Link
          key={link.to}
          to={link.to}
          onClick={onNavigate}
          aria-current={currentPath === link.to ? "page" : undefined}
          aria-label={compact ? link.label : undefined}
          title={compact ? link.label : undefined}
          className={`flex h-12 items-center rounded-xl text-sm font-semibold transition-[background-color,color,padding,gap] duration-300 motion-reduce:transition-none ${compact ? "justify-center gap-0 px-0" : "gap-3 px-4"} ${currentPath === link.to ? "bg-stone-950 text-white" : "text-stone-600 hover:bg-stone-100 hover:text-stone-950"}`}
        >
          <Icon name={link.icon} className="h-5 w-5 shrink-0" />
          <span
            aria-hidden={compact}
            className={`overflow-hidden whitespace-nowrap transition-[max-width,opacity,transform] duration-300 motion-reduce:transition-none ${compact ? "max-w-0 -translate-x-2 opacity-0" : "max-w-48 translate-x-0 opacity-100"}`}
          >
            {link.label}
          </span>
        </Link>
      ))}
    </nav>
  );
}

function usePageFocus(authenticated: boolean, path: string) {
  const mainRef = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    if ("scrollRestoration" in window.history) window.history.scrollRestoration = "manual";
    mainRef.current?.focus({ preventScroll: true });
    const frame = window.requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0 }));
    return () => window.cancelAnimationFrame(frame);
  }, [authenticated, path]);
  return mainRef;
}

function trapTab(event: KeyboardEvent, dialog: HTMLElement | null) {
  if (event.key !== "Tab") return;
  const focusable = Array.from(
    dialog?.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ) ?? [],
  );
  const first = focusable[0];
  const last = focusable.at(-1);
  if (!first || !last) return;
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function useMobileDialog(
  open: boolean,
  close: () => void,
  closeRef: React.RefObject<HTMLButtonElement | null>,
  dialogRef: React.RefObject<HTMLElement | null>,
) {
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
      else trapTab(event, dialogRef.current);
    };
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = previous;
      document.removeEventListener("keydown", onKey);
    };
  }, [close, closeRef, dialogRef, open]);
}

function PublicLayout({
  children,
  mainRef,
}: {
  children: ReactNode;
  mainRef: React.RefObject<HTMLElement | null>;
}) {
  return (
    <div className="min-h-screen bg-[#f7f7f5]">
      <SkipLink />
      <header className="flex h-20 items-center justify-between border-b border-stone-200 bg-white px-5 sm:px-8">
        <Link
          to="/"
          className="rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
        >
          <img src={logo} alt="XVision, ir para o login" className="h-12 w-auto" />
        </Link>
        <Link
          to="/api-docs"
          className="rounded-lg px-3 py-2 text-sm font-semibold text-stone-600 hover:bg-stone-100 hover:text-stone-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
        >
          Documentação da API
        </Link>
      </header>
      <main ref={mainRef} id="conteudo" tabIndex={-1} className="outline-none">
        {children}
      </main>
    </div>
  );
}

function SidebarLogo({ compact }: { compact: boolean }) {
  return (
    <Link
      to="/"
      aria-label="XVision, ir para análises"
      className="relative flex h-12 shrink-0 items-center rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
    >
      <img
        src={logo}
        alt=""
        className={`absolute left-0 h-12 w-auto max-w-52 transition-[opacity,transform] duration-300 ${compact ? "-translate-x-3 opacity-0" : "opacity-100"}`}
      />
      <span
        aria-hidden="true"
        className={`absolute left-1/2 flex h-10 w-10 -translate-x-1/2 items-center justify-center rounded-xl bg-stone-950 text-lg font-black text-[#f8d92e] transition ${compact ? "opacity-100" : "opacity-0"}`}
      >
        X
      </span>
    </Link>
  );
}

function AccountLink({
  compact,
  active,
  onNavigate,
}: {
  compact: boolean;
  active: boolean;
  onNavigate?: () => void;
}) {
  return (
    <Link
      to="/account"
      onClick={onNavigate}
      aria-current={active ? "page" : undefined}
      aria-label={compact ? "Configurações da conta" : undefined}
      title={compact ? "Configurações da conta" : undefined}
      className={`flex h-11 w-full items-center rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] ${compact ? "mt-0 justify-center gap-0" : "mt-3 gap-3 px-3"} ${active ? "bg-stone-950 text-white" : "text-stone-600 hover:bg-stone-100"}`}
    >
      <Icon name="user" className="h-5 w-5 shrink-0" />
      <span
        aria-hidden={compact}
        className={
          compact
            ? "max-w-0 overflow-hidden opacity-0"
            : "max-w-44 overflow-hidden whitespace-nowrap"
        }
      >
        Configurações da conta
      </span>
    </Link>
  );
}

function LogoutButton({ compact, leave }: { compact: boolean; leave: () => void }) {
  return (
    <button
      type="button"
      onClick={leave}
      aria-label={compact ? "Sair da conta" : undefined}
      title={compact ? "Sair da conta" : undefined}
      className={`mt-1 flex h-11 w-full items-center rounded-lg text-sm font-medium text-stone-600 transition-all hover:bg-stone-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] ${compact ? "justify-center gap-0" : "gap-3 px-3"}`}
    >
      <Icon name="log-out" className="h-5 w-5 shrink-0" />
      <span
        aria-hidden={compact}
        className={
          compact
            ? "max-w-0 overflow-hidden opacity-0"
            : "max-w-32 overflow-hidden whitespace-nowrap"
        }
      >
        Sair da conta
      </span>
    </button>
  );
}

function DesktopSidebar({
  compact,
  toggle,
  path,
  name,
  leave,
  inert,
}: {
  compact: boolean;
  toggle: () => void;
  path: string;
  name: string | null;
  leave: () => void;
  inert: boolean;
}) {
  return (
    <aside
      inert={inert}
      className="relative hidden min-h-screen border-r border-stone-200 bg-white lg:flex"
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={!compact}
        aria-controls="desktop-sidebar-content"
        aria-label={compact ? "Expandir menu lateral" : "Recolher menu lateral"}
        title={compact ? "Expandir menu lateral" : "Recolher menu lateral"}
        className="absolute -right-3 top-8 z-30 flex h-7 w-7 items-center justify-center rounded-full border border-stone-200 bg-white text-stone-500 shadow-sm transition hover:bg-stone-950 hover:text-white focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <Icon name={compact ? "arrow-right" : "arrow-left"} className="h-4 w-4" />
      </button>
      <div
        id="desktop-sidebar-content"
        className={`flex min-w-0 flex-1 flex-col overflow-hidden transition-[padding] duration-500 ease-in-out ${compact ? "p-3" : "p-5"}`}
      >
        <SidebarLogo compact={compact} />
        <div className="mt-10">
          <Navigation compact={compact} currentPath={path} />
        </div>
        <DesktopAccount compact={compact} path={path} name={name} leave={leave} />
      </div>
    </aside>
  );
}

function DesktopAccount({
  compact,
  path,
  name,
  leave,
}: {
  compact: boolean;
  path: string;
  name: string | null;
  leave: () => void;
}) {
  return (
    <div className="mt-auto border-t border-stone-200 pt-5">
      <p
        aria-hidden={compact}
        className={`truncate text-sm font-semibold text-stone-900 ${compact ? "max-h-0 opacity-0" : "max-h-8"}`}
      >
        {name ?? "Conta XVision"}
      </p>
      <AccountLink compact={compact} active={path === "/account"} />
      <LogoutButton compact={compact} leave={leave} />
    </div>
  );
}

function MobileHeader({
  open,
  setOpen,
  openRef,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  openRef: React.RefObject<HTMLButtonElement | null>;
}) {
  return (
    <header
      inert={open}
      className="sticky top-0 z-30 flex h-18 items-center justify-between border-b border-stone-200 bg-white/95 px-4 backdrop-blur sm:px-5 lg:hidden"
    >
      <Link to="/" className="rounded-lg focus-visible:ring-2 focus-visible:ring-[#e7bf08]">
        <img src={logo} alt="XVision, ir para análises" className="h-10 w-auto sm:h-11" />
      </Link>
      <button
        ref={openRef}
        type="button"
        onClick={() => setOpen(true)}
        aria-expanded={open}
        aria-controls="mobile-sidebar"
        className="flex h-11 items-center gap-2 rounded-xl border border-stone-200 bg-white px-3 text-sm font-semibold text-stone-700 shadow-sm focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <Icon name="menu" className="h-5 w-5" />
        Menu
      </button>
    </header>
  );
}

function MobileBackdrop({ open, close }: { open: boolean; close: () => void }) {
  return (
    <button
      type="button"
      onClick={close}
      aria-label="Fechar menu pelo fundo"
      aria-hidden={!open}
      tabIndex={-1}
      className={`fixed inset-0 z-40 bg-stone-950/40 transition-opacity duration-500 lg:hidden ${open ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"}`}
    />
  );
}

function MobileDrawerHeader({
  close,
  closeRef,
}: {
  close: () => void;
  closeRef: React.RefObject<HTMLButtonElement | null>;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <Link
        to="/"
        onClick={close}
        className="rounded-lg focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <img src={logo} alt="XVision, ir para análises" className="h-11 w-auto" />
      </Link>
      <button
        ref={closeRef}
        type="button"
        onClick={close}
        aria-label="Fechar menu"
        className="flex h-10 w-10 items-center justify-center rounded-xl text-stone-500 hover:bg-stone-100 focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <Icon name="x" className="h-5 w-5" />
      </button>
    </div>
  );
}

function MobileAccount({
  path,
  name,
  close,
  leave,
}: {
  path: string;
  name: string | null;
  close: () => void;
  leave: () => void;
}) {
  return (
    <div className="mt-auto border-t border-stone-200 pt-5">
      <p className="truncate text-sm font-semibold text-stone-900">{name ?? "Conta XVision"}</p>
      <AccountLink compact={false} active={path === "/account"} onNavigate={close} />
      <button
        type="button"
        onClick={leave}
        className="mt-1 flex h-11 w-full items-center gap-3 rounded-lg px-3 text-sm font-medium text-stone-600 hover:bg-stone-100 focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <Icon name="log-out" className="h-5 w-5" />
        Sair da conta
      </button>
    </div>
  );
}

function MobileSidebar({
  open,
  setOpen,
  close,
  closeRef,
  dialogRef,
  path,
  name,
  leave,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  close: () => void;
  closeRef: React.RefObject<HTMLButtonElement | null>;
  dialogRef: React.RefObject<HTMLElement | null>;
  path: string;
  name: string | null;
  leave: () => void;
}) {
  const leaveAndClose = () => {
    setOpen(false);
    leave();
  };
  return (
    <>
      <MobileBackdrop open={open} close={close} />
      <aside
        ref={dialogRef}
        id="mobile-sidebar"
        role="dialog"
        aria-modal="true"
        aria-label="Menu lateral"
        aria-hidden={!open}
        inert={!open}
        className={`fixed inset-y-0 left-0 z-50 flex w-[min(19rem,calc(100vw-2rem))] flex-col border-r border-stone-200 bg-white p-5 shadow-2xl transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] lg:hidden ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <MobileDrawerHeader close={close} closeRef={closeRef} />
        <div className="mt-10">
          <Navigation currentPath={path} onNavigate={close} />
        </div>
        <MobileAccount path={path} name={name} close={close} leave={leaveAndClose} />
      </aside>
    </>
  );
}

function SkipLink() {
  return (
    <a href="#conteudo" className="skip-link">
      Pular para o conteúdo
    </a>
  );
}

interface AuthenticatedLayoutProps {
  children: ReactNode;
  compact: boolean;
  toggle: () => void;
  path: string;
  name: string | null;
  leave: () => void;
  menuOpen: boolean;
  setMenuOpen: (open: boolean) => void;
  closeMenu: () => void;
  openRef: React.RefObject<HTMLButtonElement | null>;
  closeRef: React.RefObject<HTMLButtonElement | null>;
  dialogRef: React.RefObject<HTMLElement | null>;
  mainRef: React.RefObject<HTMLElement | null>;
}

function AuthenticatedMain({
  children,
  path,
  name,
  leave,
  menuOpen,
  setMenuOpen,
  closeMenu,
  openRef,
  closeRef,
  dialogRef,
  mainRef,
}: Omit<AuthenticatedLayoutProps, "compact" | "toggle">) {
  return (
    <div className="min-w-0">
      <MobileHeader open={menuOpen} setOpen={setMenuOpen} openRef={openRef} />
      <MobileSidebar
        open={menuOpen}
        setOpen={setMenuOpen}
        close={closeMenu}
        closeRef={closeRef}
        dialogRef={dialogRef}
        path={path}
        name={name}
        leave={leave}
      />
      <main
        ref={mainRef}
        id="conteudo"
        tabIndex={-1}
        inert={menuOpen}
        className="min-h-screen outline-none lg:min-h-0"
      >
        {children}
      </main>
    </div>
  );
}

function AuthenticatedLayout({
  children,
  compact,
  toggle,
  path,
  name,
  leave,
  menuOpen,
  setMenuOpen,
  closeMenu,
  openRef,
  closeRef,
  dialogRef,
  mainRef,
}: AuthenticatedLayoutProps) {
  return (
    <div
      className="min-h-screen bg-[#f7f7f5] text-stone-900 transition-[grid-template-columns] duration-500 ease-in-out lg:grid"
      style={{ gridTemplateColumns: compact ? "5.5rem minmax(0, 1fr)" : "17rem minmax(0, 1fr)" }}
    >
      <SkipLink />
      <DesktopSidebar
        compact={compact}
        toggle={toggle}
        path={path}
        name={name}
        leave={leave}
        inert={menuOpen}
      />
      <AuthenticatedMain
        path={path}
        name={name}
        leave={leave}
        menuOpen={menuOpen}
        setMenuOpen={setMenuOpen}
        closeMenu={closeMenu}
        openRef={openRef}
        closeRef={closeRef}
        dialogRef={dialogRef}
        mainRef={mainRef}
      >
        {children}
      </AuthenticatedMain>
    </div>
  );
}

function Layout({ children }: { children: ReactNode }) {
  const { autenticado, nome, sair } = useSessao();
  const path = useLocation().pathname;
  const mainRef = usePageFocus(autenticado, path);
  const [menuOpen, setMenuOpen] = useState(false);
  const openRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closeMenu = useCallback(() => {
    setMenuOpen(false);
    window.requestAnimationFrame(() => openRef.current?.focus());
  }, []);
  useMobileDialog(menuOpen, closeMenu, closeRef, dialogRef);
  const [compact, setCompact] = useState(
    () => localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true",
  );
  const toggle = () =>
    setCompact((current) => {
      const next = !current;
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      return next;
    });
  if (!autenticado) return <PublicLayout mainRef={mainRef}>{children}</PublicLayout>;
  return (
    <AuthenticatedLayout
      compact={compact}
      toggle={toggle}
      path={path}
      name={nome}
      leave={sair}
      menuOpen={menuOpen}
      setMenuOpen={setMenuOpen}
      closeMenu={closeMenu}
      openRef={openRef}
      closeRef={closeRef}
      dialogRef={dialogRef}
      mainRef={mainRef}
    >
      {children}
    </AuthenticatedLayout>
  );
}

export default Layout;
