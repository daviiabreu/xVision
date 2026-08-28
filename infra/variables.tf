variable "project" {
  type        = string
  default     = "xvision"
  description = "Nome do projeto, usado como prefixo de todo recurso."
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Ambiente declarado. Nada aqui e aplicado nesta sprint."
}

variable "services" {
  description = "A topologia, em termos que nao pertencem a provedor nenhum. Trocar de nuvem troca os recursos em `main.tf`, e nao esta declaracao."

  type = map(object({
    role        = string
    port        = number
    health_path = string
    public      = bool
    needs       = list(string)
    build_args  = optional(map(string), {})
  }))

  default = {
    api = {
      role        = "api"
      port        = 8080
      health_path = "/health"
      public      = true
      needs       = ["database"]
    }
    frontend = {
      role        = "web"
      port        = 3000
      health_path = "/health"
      public      = true
      needs       = ["api"]
      build_args  = { VITE_API_URL = "api" }
    }
  }

  validation {
    condition     = alltrue([for s in var.services : s.port > 0 && s.port < 65536])
    error_message = "Cada servico precisa de uma porta valida."
  }

  validation {
    condition     = alltrue([for nome, s in var.services : alltrue([for d in s.needs : contains(keys(var.services), d) || contains(keys(var.stores), d)])])
    error_message = "Todo `needs` precisa apontar para um servico ou um armazenamento declarado."
  }

  validation {
    condition     = alltrue([for nome, s in var.services : alltrue([for d in values(s.build_args) : contains(s.needs, d)])])
    error_message = "Todo `build_args` precisa apontar para uma dependencia que o servico declara em `needs`."
  }
}

variable "stores" {
  description = "Armazenamento com estado. Fica separado dos servicos porque tem ciclo de vida proprio: servico se recria, banco nao."

  type = map(object({
    engine  = string
    version = string
  }))

  default = {
    database = {
      engine  = "postgres"
      version = "16"
    }
  }
}
