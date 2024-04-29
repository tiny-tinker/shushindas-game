variable "project" {
  type        = string
  description = "Enter the name of the project to house the app."
}

variable "region" {
  type       = string
  default    = "us-central1"
  description = "The region for deploying to."
}

variable "destroy_all" {
  type      = bool
  default   = true
  description = "Should we destroy everything?"
}


######
## Providers
######

provider "google" {
  project = var.project
  region  = var.region
}



######
## BigQuery
######


resource "google_bigquery_dataset" "sushindas_stuff" {
  dataset_id                  = "sushindas_stuff"
  friendly_name               = "sushindas_stuff"
  description                 = "Houses all of Sushinda's stuff."
  
  delete_contents_on_destroy  = var.destroy_all

}

resource "google_bigquery_table" "docs" {
  dataset_id = google_bigquery_dataset.sushindas_stuff.dataset_id
  table_id   = "docs"
  deletion_protection = false

  schema = file("${path.module}/docs.schema.json")
}


resource "google_bigquery_table" "chunks" {
  dataset_id = google_bigquery_dataset.sushindas_stuff.dataset_id
  table_id   = "chunks"
  deletion_protection = false

  schema = file("${path.module}/chunks.schema.json")

}
