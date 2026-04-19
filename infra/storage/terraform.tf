terraform {
  required_version = ">=1.12.0"

  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = ">= 0.5.196"
    }
  }
}

provider "nebius" {
  domain = "api.eu.nebius.cloud:443"
  service_account = {
    account_id       = var.sa_id
    public_key_id    = var.sa_public_key_id
    private_key_file = "~/.nebius/authkey/private.pem"
  }
}
