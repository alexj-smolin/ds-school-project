## prerequisites
* `python 3.9`
* `docker`

## python
* `python -m venv ./venv`
* `source ./venv/bin/activate`
* `pip install -r requirements.txt`

## docker
* `docker-compose up -d` - run s3 (minio), postgres, mlflow containers
* `docker-compose down` - stop containers

## mlflow
* console: http://127.0.0.1:5001/

## minio
* console: http://127.0.0.1:9001/
* add bucket `ds-bucket`

## run samples:
* airplane:
  * `python src/main.py --sample plane.mp4 --oname airplane --owidth 11.5 --oheight 4.8 --cfocal 25 --cwidth 4.92 --cheight 2.77 --ratiodev 0.4  --video --async --smooth 0.7`
* person walk:
  * `python src/main.py --sample fourway.avi --oname person --owidth 0.6 --oheight 1.7 --cfocal 4 --cwidth 7.38 --cheight 4.15 --ratiodev 0.5  --video --async --smooth 0.7`
* carcam1:
  * `python src/main.py --sample carcam1.mp4 --oname car --owidth 2 --oheight 1.6 --cfocal 2.8 --cwidth 5.47 --cheight 3.07 --ratiodev 0.4`

## Unused
  #### DVC
  * setup s3:
    * `dvc remote add -d dvc_s3 s3://${AWS_S3_BUCKET}`
    * `dvc remote modify dvc_s3 endpointurl http://127.0.0.1:9000`
    * `dvc remote modify dvc_s3 access_key_id ${AWS_ACCESS_KEY_ID}`
    * `dvc remote modify dvc_s3 secret_access_key ${AWS_SECRET_ACCESS_KEY}`
  * using:
    * `dvc add <file>`
    * `dvc push` / `dvc pull`
