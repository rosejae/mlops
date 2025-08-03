# !pip install --upgrade google-cloud-aiplatform google-cloud-storage kfp google-cloud-pipeline-components

# !gcloud config list

# PROJECT_ID = "ai-service-467312"
# REGION = "us-central1"

# !gcloud config set project {PROJECT_ID}

# BUCKET_URI = f"gs://fs-practice-{PROJECT_ID}"

# !gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}

# shell_output = !gcloud auth list 2>/dev/null
# SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()

# !gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI
# !gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI

from typing import NamedTuple

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import component

PIPELINE_ROOT = "{}/pipeline_root/fc_first_simple_pipeline".format(BUCKET_URI)
aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

@component(base_image="python:3.9")
def hello_world(text: str) -> str:
    print(text)
    return text

compiler.Compiler().compile(hello_world, "hw.yaml")

@component(packages_to_install=["google-cloud-storage"])
def two_outputs(text: str) -> NamedTuple("Outputs", [("output_one", str), ("output_two", str)]):
    
    from google.cloud import storage
    
    text1 = f"output first: {text}"
    text2 = f"output second: {text}"
    
    return (text1, text2)

@component(base_image="python:3.10")
def three_nicemeet_outputs(name: str) -> str:
    result_string = "Nice to meet you! "+name
    print(result_string)
    return result_string

compiler.Compiler().compile(three_nicemeet_outputs, "hw2.yaml")    

@component
def consumer(text1: str, text2: str, text3: str, text4: str) -> str:
    result = f"text1-> {text1}, text2-> {text2}, text3-> {text3}, text4-> {text4}"
    print(result)
    return result

@dsl.pipeline(name="fc-first_pipeline-2524", description="hello pipeline", pipeline_root=PIPELINE_ROOT)
def pipeline(text: str = "hi there", name: str = "fc seoul"):
    hw_task = hello_world(text=text)
    two_outputs_task = two_outputs(text=text)
    three_outputs_task = three_nicemeet_outputs(name=name)
    
    consumer_task = consumer(
        text1=hw_task.output,
        text2=two_outputs_task.outputs["output_one"],
        text3=two_outputs_task.outputs["output_two"],
        text4=three_outputs_task.output
    )
    
compiler.Compiler().compile(pipeline_func=pipeline, package_path="first_pipeline.yaml")

job = aip.PipelineJob(
    display_name = "first_fc_pipeline",
    template_path = "first_pipeline.yaml",
    pipeline_root = PIPELINE_ROOT,
)

job.run()