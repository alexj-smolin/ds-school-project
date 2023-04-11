### minio
* console: http://127.0.0.1:9001/

### DVC
* setup s3:
  * `dvc remote add -d dvc_s3 s3://${AWS_S3_BUCKET}`
  * `dvc remote modify dvc_s3 endpointurl http://127.0.0.1:9000`
  * `dvc remote modify dvc_s3 access_key_id ${AWS_ACCESS_KEY_ID}`
  * `dvc remote modify dvc_s3 secret_access_key ${AWS_SECRET_ACCESS_KEY}`
* using:
  * `dvc add <file>`
  * `dvc push` / `dvc pull`



