import type { ButtonHTMLAttributes } from "react";

import { classesDeBotao, type Tamanho, type Variante } from "./classesDeBotao";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variante?: Variante;
  tamanho?: Tamanho;
}

function Button({ variante, tamanho, className, type = "button", ...resto }: ButtonProps) {
  return <button type={type} className={classesDeBotao(variante, tamanho, className)} {...resto} />;
}

export default Button;
