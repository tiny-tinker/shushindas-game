variable "project" {
  type        = string
  description = "Enter the name of the project to house the app."
}

variable "region" {
  type       = string
  default    = "us-central1"
  description = "The region for deploying to."
}

variable "emb_model" {
  type = string
  default = "textembedding-gecko@002"
  description = "The text embedding model to use for working with embeddings"
}

variable "bq_region" {
  type = string
  default = "US"
  description = "The region for the BQ assets."
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


resource "google_bigquery_dataset" "shushindas_stuff" {
  dataset_id                  = "shushindas_stuff"
  friendly_name               = "shushindas_stuff"
  description                 = "Houses all of Shushinda's stuff."
  
  delete_contents_on_destroy  = var.destroy_all

}

resource "google_bigquery_connection" "connection" {
    connection_id = "vertex_ai"
    location = var.bq_region
    cloud_resource {}
}


resource "google_project_iam_binding" "bq_connection_iam_binding" {
  project = var.project
  role    = "roles/aiplatform.user"

  members = [
    "serviceAccount:${google_bigquery_connection.connection.cloud_resource[0].service_account_id}"
  ]
}

# // Create a BigQuery Job to run a SQL query which includes the Model 
##### unfortunately it looks like this won't work. I got this error:
##### "Cannot set create disposition in jobs with ML DDL statements"
#####
# resource "google_bigquery_job" "query" {
#   job_id     = "create_model_${uuid()}"
#   location   = var.bq_region

#   query {
#     query = <<EOF
#    CREATE OR REPLACE MODEL shushindas_stuff.embedding_model
#    REMOTE WITH CONNECTION `${var.bq_region}.${google_bigquery_connection.connection.connection_id}`
#    OPTIONS (ENDPOINT = '${var.emb_model}')
# EOF
#   }
# }


resource "google_bigquery_table" "docs" {
  dataset_id = google_bigquery_dataset.shushindas_stuff.dataset_id
  table_id   = "docs"
  deletion_protection = false

  schema = file("${path.module}/docs.schema.json")
}


resource "google_bigquery_table" "chunks" {
  dataset_id = google_bigquery_dataset.shushindas_stuff.dataset_id
  table_id   = "chunks"
  deletion_protection = false

  schema = file("${path.module}/chunks.schema.json")

}
