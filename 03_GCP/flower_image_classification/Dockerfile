FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8

WORKDIR /

COPY train_code /train_code

ENTRYPOINT ["python", "-m", "train_code.task"]
