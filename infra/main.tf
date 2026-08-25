locals {
  prefixo = "${var.project}-${var.environment}"

  servicos = {
    for nome, s in var.services : nome => merge(s, {
      name = "${local.prefixo}-${nome}"
    })
  }

  armazenamentos = {
    for nome, a in var.stores : nome => merge(a, {
      name = "${local.prefixo}-${nome}"
    })
  }

  variaveis_de_ambiente = {
    for nome, s in local.servicos : nome => merge(
      { PORT = tostring(s.port), HEALTH_PATH = s.health_path },
      { for d in s.needs : "${upper(d)}_URL" => "<resolvido pelo provedor: ${d}>" }
    )
  }
}

resource "terraform_data" "service" {
  for_each = local.servicos

  input = {
    name        = each.value.name
    role        = each.value.role
    port        = each.value.port
    health_path = each.value.health_path
    public      = each.value.public
    environment = local.variaveis_de_ambiente[each.key]
  }
}

resource "terraform_data" "store" {
  for_each = local.armazenamentos

  input = {
    name    = each.value.name
    engine  = each.value.engine
    version = each.value.version
  }
}
