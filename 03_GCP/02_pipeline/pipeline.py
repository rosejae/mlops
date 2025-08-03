# !pip install --upgrade 'protobuf<4' google-cloud-aiplatform google-cloud-storage 'kfp<2' 'google-cloud-pipeline-components<2'

# !gcloud config list

# PROJECT_ID ="ai-service-467312"
# REGION = "us-central1"
# BUCKET_URI = f"gs://fs-practice-{PROJECT_ID}"

# !gcloud config set project {PROJECT_ID}
# !gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}
# !gsutil -u {PROJECT_ID} cp gs://aju-dev-demos-codelabs/bikes_weather/* {BUCKET_URI}/DATA/

# shell_output = !gcloud auth list 2>/dev/null
# SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()
# !gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI
# !gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI

# !echo $BUCKET_URI

# !gsutil ls {BUCKET_URI}/DATA/

from typing import Any, Dict, List

import google.cloud.aiplatform as aip
import kfp
from kfp.v2 import compiler

import random
import string

PIPELINE_ROOT = "{}/pipeline_root/training".format(BUCKET_URI)

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

hp_dict: str = '{"num_hidden_layers": 1, "hidden_size": 16, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
data_dir: str = f"{BUCKET_URI}/"
TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]

UUID = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
WORKING_DIR = f"{PIPELINE_ROOT}/{UUID}"
MODEL_DISPLAY_NAME = f"train_deploy{UUID}"
print(TRAINER_ARGS, WORKING_DIR, MODEL_DISPLAY_NAME)

DISPLAY_NAME = "fc_first_pipeline_job"

@kfp.dsl.pipeline(name="first-fc-train-endpoint-deploy" + UUID)
def pipeline(
    project: str = PROJECT_ID, 
    model_display_name: str = MODEL_DISPLAY_NAME, 
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
): 
                  
    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
    from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from kfp.v2.components import importer_node
    
    custom_job_task = CustomTrainingJobOp(
        project=project,
        display_name="model-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    "args": TRAINER_ARGS,
                    "env": [{"name": "AIP_MODEL_DIR", "value": WORKING_DIR}],
                    "imageUri": "gcr.io/google-samples/bw-cc-train:latest",
                },
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": "n1-standard-16"
                },
            }
        ],
    )
    
    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=WORKING_DIR,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
            },
        },
    ).after(custom_job_task)

    model_upload_op = ModelUploadOp(
        project=project,
        display_name=model_display_name,
        unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
    )
    
    model_upload_op.after(import_unmanaged_model_task)
    
    endpoint_create_op = EndpointCreateOp(
        project=project,
        display_name="pipelines-created-endpoint",
    )
    
    ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name=model_display_name,
        dedicated_resources_machine_type="n1-standard-16",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="fc_first_model_training_pipeline.json",
)                  

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="fc_first_model_training_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

job.run(service_account=SERVICE_ACCOUNT)

# def get_task_detail(
#     task_details: List[Dict[str, Any]], task_name: str
# ) -> List[Dict[str, Any]]:
#     for task_detail in task_details:
#         if task_detail.task_name == task_name:
#             return task_detail

# pipeline_task_details = (
#     job.gca_resource.job_detail.task_details
# )

# pipeline_task_details

# endpoint_task = get_task_detail(pipeline_task_details, "endpoint-create")
# endpoint_resourceName = (
#     endpoint_task.outputs["endpoint"].artifacts[0].metadata["resourceName"]
# )

# endpoint = aip.Endpoint(endpoint_resourceName)
# endpoint

# endpoint.undeploy_all()
# endpoint.delete()

# model_task = get_task_detail(pipeline_task_details, "model-upload")
# model_resourceName = model_task.outputs["model"].artifacts[0].metadata["resourceName"]
# model = aip.Model(model_resourceName)
# model.delete()