# Shared root filesystem mounted on all Slurm nodes
resource "nebius_compute_v1_filesystem" "shared_filesystem" {
  parent_id        = var.project_id
  name             = "poc-shared-filesystem"
  type             = "NETWORK_SSD"
  size_bytes       = 1536 * 1024 * 1024 * 1024
  block_size_bytes = 4 * 1024
}

# Network disk for training data and checkpoints, mounted at /mnt/data
resource "nebius_compute_v1_filesystem" "network_disk" {
  parent_id        = var.project_id
  name             = "poc-network-disk"
  type             = "NETWORK_SSD"
  size_bytes       = 1536 * 1024 * 1024 * 1024
  block_size_bytes = 4 * 1024
}
