output "topology" {
  description = "A topologia resolvida. `terraform output -json topology` e o que um passo de implantacao consome."

  value = {
    project     = var.project
    environment = var.environment
    services    = { for nome, r in terraform_data.service : nome => r.input }
    stores      = { for nome, r in terraform_data.store : nome => r.input }
  }
}
