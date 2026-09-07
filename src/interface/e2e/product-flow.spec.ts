import { expect, test } from "@playwright/test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const PASSWORD = "senha-de-e2e-com-mais-de-doze";
const STATE_PATH = "/work/e2e-state.json";

interface PersistedState {
  email: string;
  password: string;
  videoId: number;
}

test("oferece recuperacao de senha e documentacao publica da API", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("link", { name: "Esqueceu a senha?" })).toBeVisible();
  await page.getByRole("link", { name: "Documentação da API" }).click();

  await expect(page.getByRole("heading", { name: "Automatize o envio de vídeos" })).toBeVisible();
  await expect(page.getByText("API hoje, webhook depois")).toBeVisible();
  await expect(page.getByRole("link", { name: "Baixar coleção Bruno" })).toHaveAttribute(
    "href",
    /\/bruno\.zip$/,
  );
  await expect(page.getByTitle("Referência interativa da API XVision")).toBeVisible();

  await page.goto("/forgot-password");
  await expect(page.getByRole("heading", { name: "Redefina sua senha" })).toBeVisible();
});

test("adapta a navegacao entre desktop e celular", async ({ page }) => {
  const email = `e2e-nav-${Date.now()}@xvision.local`;

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await page.getByRole("button", { name: "Usar modo Criar conta" }).click();
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Senha").fill("senha-de-e2e-com-mais-de-doze");
  await page.getByRole("button", { name: "Criar conta e entrar" }).click();

  const sidebar = page.locator("#desktop-sidebar-content");
  const widthBefore = (await sidebar.boundingBox())?.width ?? 0;
  await page.getByRole("button", { name: "Recolher menu lateral" }).click();
  await expect(page.getByRole("button", { name: "Expandir menu lateral" })).toBeVisible();
  await expect
    .poll(async () => (await sidebar.boundingBox())?.width ?? 0)
    .toBeLessThan(widthBefore);

  await page.reload();
  await expect(page.getByRole("button", { name: "Expandir menu lateral" })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("button", { name: "Menu", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Menu lateral" });
  const close = page.getByRole("button", { name: "Fechar menu", exact: true });
  const logo = dialog.getByRole("link", { name: "XVision, ir para análises" });
  const logout = dialog.getByRole("button", { name: "Sair da conta" });
  await expect(dialog).toBeVisible();
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(logo).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(logout).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(logo).toBeFocused();
  await expect(page.getByRole("link", { name: "Documentação da API" })).toBeVisible();
  await page.getByRole("link", { name: "Configurações da conta" }).click();
  await expect(page.getByRole("heading", { name: "Configurações da conta" })).toBeVisible();

  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Menu lateral" })).toBeHidden();
  await expect(page.getByRole("button", { name: "Menu", exact: true })).toBeFocused();
});

test("atualiza email e senha pelas configuracoes da conta", async ({ page }) => {
  const identifier = Date.now();
  const email = `e2e-account-${identifier}@xvision.local`;
  const updatedEmail = `e2e-account-updated-${identifier}@xvision.local`;
  const password = "senha-de-e2e-com-mais-de-doze";
  const updatedPassword = "nova-senha-de-e2e-com-mais-de-doze";

  await page.goto("/");
  await page.getByRole("button", { name: "Usar modo Criar conta" }).click();
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Senha").fill(password);
  await page.getByRole("button", { name: "Criar conta e entrar" }).click();

  await page.getByRole("link", { name: "Configurações da conta" }).click();
  await expect(page.getByRole("heading", { name: "Configurações da conta" })).toBeVisible();
  await expect(page.getByText("Contagem completa de todas as suas análises.")).toBeVisible();
  await expect(page.getByText("0", { exact: true })).toHaveCount(4);

  const emailForm = page.getByRole("button", { name: "Atualizar email" }).locator("xpath=..");
  await emailForm.getByLabel("Novo email").fill(updatedEmail);
  await emailForm.getByLabel("Senha atual").fill(password);
  await emailForm.getByRole("button", { name: "Atualizar email" }).click();
  await expect(page.getByText("Email atualizado com sucesso.")).toBeVisible();
  await expect(page.locator("#desktop-sidebar-content").getByText(updatedEmail)).toBeVisible();

  const passwordForm = page.getByRole("button", { name: "Atualizar senha" }).locator("xpath=..");
  await passwordForm.getByLabel("Senha atual").fill(password);
  await passwordForm.getByLabel("Nova senha", { exact: true }).fill(updatedPassword);
  await passwordForm.getByLabel("Confirme a nova senha").fill(updatedPassword);
  await passwordForm.getByRole("button", { name: "Atualizar senha" }).click();
  await expect(page.getByText("Senha atualizada com sucesso.")).toBeVisible();

  await page.getByRole("button", { name: "Sair da conta" }).click();
  await page.getByLabel("Email").fill(updatedEmail);
  await page.getByLabel("Senha").fill(updatedPassword);
  await page.getByRole("button", { name: "Entrar", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Configurações da conta" })).toBeVisible();
  await page.getByRole("link", { name: "Análises", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Análises de finalização" })).toBeVisible();
});

test("cria conta, envia um clipe e preserva o resultado completo", async ({ page }) => {
  const email = `e2e-${Date.now()}@xvision.local`;
  const video = process.env.E2E_VIDEO_PATH ?? "/fixtures/resultado.mp4";

  await page.goto("/");
  await page.getByRole("button", { name: "Usar modo Criar conta" }).click();
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Senha").fill(PASSWORD);
  await page.getByRole("button", { name: "Criar conta e entrar" }).click();

  await expect(page.getByRole("heading", { name: "Análises de finalização" })).toBeVisible();

  await page.getByLabel("Selecionar vídeo").setInputFiles(video);
  await page.getByRole("button", { name: "Iniciar análise" }).click();

  const resultado = page.getByRole("link", { name: /resultado\.mp4/ });
  await expect(resultado).toBeVisible({ timeout: 120_000 });
  await resultado.click();

  await expect(page.getByRole("heading", { name: "resultado.mp4" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Etapas do pipeline" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Métricas da finalização" })).toBeVisible();
  await page.getByText("Detecção da bola", { exact: true }).click();
  await expect(page.getByAltText("Quadro de demonstração de Detecção da bola")).toBeVisible();

  const id = Number(page.url().match(/\/results\/(\d+)/)?.[1]);
  expect(id).toBeGreaterThan(0);
  writeFileSync(STATE_PATH, JSON.stringify({ email, password: PASSWORD, videoId: id }));

  await page.reload();
  await expect(page.getByRole("heading", { name: "resultado.mp4" })).toBeVisible();
  await page.getByRole("link", { name: "Voltar para análises" }).click();
  await page.reload();
  await expect(page.getByRole("link", { name: /resultado\.mp4/ })).toBeVisible();
});

test("@after-restart recupera histórico, evidência e render persistidos", async ({ page }) => {
  expect(existsSync(STATE_PATH)).toBe(true);
  const state = JSON.parse(readFileSync(STATE_PATH, "utf8")) as PersistedState;

  await page.goto("/");
  await page.getByLabel("Email").fill(state.email);
  await page.getByLabel("Senha").fill(state.password);
  await page.getByRole("button", { name: "Entrar", exact: true }).click();

  const result = page.getByRole("link", { name: /resultado\.mp4/ });
  await expect(result).toBeVisible();
  await result.click();
  await expect(page.getByRole("heading", { name: "Visão superior" })).toBeVisible();
  await expect(page.locator("video")).toHaveAttribute("src", /^blob:/);
  await page.getByText("Detecção da bola", { exact: true }).click();
  await expect(page.getByAltText("Quadro de demonstração de Detecção da bola")).toBeVisible();
});
