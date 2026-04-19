variable "project_id" {
  description = "Nebius project ID"
  type        = string
}

variable "sa_id" {
  description = "Service account ID for Terraform provider auth"
  type        = string
}

variable "sa_public_key_id" {
  description = "Service account public key ID for Terraform provider auth"
  type        = string
}
