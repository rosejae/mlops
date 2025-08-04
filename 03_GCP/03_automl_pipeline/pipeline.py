# !pip3 install --upgrade --quiet google-cloud-aiplatform kfp google-cloud-pipeline-components==2.4.1 google-cloud-storage

# !gcloud config list

# PROJECT_ID = "ai-service-467312"
# REGION = "us-central1"
# BUCKET_URI = f"gs://fs-practice-{PROJECT_ID}"

# ! gcloud config set project {PROJECT_ID}
# ! gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}

# shell_output = !gcloud auth list 2>/dev/null
# SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()
# ! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI
# ! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI

# !gsutil ls gs://cloud-samples-data/vision/automl_classification/flowers/all_data_v2.csv

from typing import Any, Dict, List

import google.cloud.aiplatform as aip
import kfp
from kfp.v2 import compiler

import random
import string

PIPELINE_ROOT = "{}/pipeline_root/automl_training".format(BUCKET_URI)

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

@kfp.dsl.pipeline(name="auto-fc-first-flower-clf")
def pipeline(project: str = PROJECT_ID, region: str = REGION):
    
    from google_cloud_pipeline_components.v1.automl.training_job import AutoMLImageTrainingJobRunOp
    from google_cloud_pipeline_components.v1.dataset import ImageDatasetCreateOp
    from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

    ds_op = ImageDatasetCreateOp(
        project=project,
        display_name="flowers_dataset_fc",
        gcs_source="gs://cloud-samples-data/vision/automl_classification/flowers/all_data_v2.csv",
        import_schema_uri=aip.schema.dataset.ioformat.image.single_label_classification,
    )
    
    training_job_run_op = AutoMLImageTrainingJobRunOp(
        project=project,
        display_name="automl-fc-first-flower-clf",
        prediction_type="classification",
        model_type="CLOUD",
        dataset=ds_op.outputs["dataset"],
        model_display_name="automl-fc-first-clf",
        training_fraction_split=0.7,
        validation_fraction_split=0.15,
        test_fraction_split=0.15,
        budget_milli_node_hours=10000,
    )
    
    training_job_run_op.after(ds_op)
    
    endpoint_op = EndpointCreateOp(
        project=project,
        location=region,
        display_name="automl-fc-first-flower-clf",
    ) 
    
    ModelDeployOp(
        model=training_job_run_op.outputs["model"],
        endpoint=endpoint_op.outputs["endpoint"],
        automatic_resources_min_replica_count=1,
        automatic_resources_max_replica_count=1,
    )

compiler.Compiler().compile(
    pipeline_func=pipeline, 
    package_path="fc-first-automl-flower-clf.yaml",
)

job = aip.PipelineJob(
    display_name="first-automl-job",
    template_path="fc-first-automl-flower-clf.yaml",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

job.run()