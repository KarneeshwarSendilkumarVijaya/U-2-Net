# build
#FROM gcr.io/deeplearning-platform-release/tf-gpu.2-11.py310:latest AS build
#FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest As build
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8.py310:latest AS build
ARG COPART_DIR="/opt/copart/"
ARG APP_NAME="u2net-model-train-gcp"
ARG APP_DIR="${COPART_DIR}${APP_NAME}"
ARG REQUIREMENTS=requirements.txt

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    mkdir -p ${APP_DIR}

WORKDIR ${APP_DIR}

COPY ./requirements.txt ${APP_DIR}/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS} --target ${APP_DIR}/libs

# production
#FROM gcr.io/deeplearning-platform-release/tf-gpu.2-11.py310:latest AS production
#FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest As production
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8.py310:latest AS production
ARG COPART_DIR="/opt/copart/"
ARG APP_NAME="u2net-model-train-gcp"
ARG APP_DIR="${COPART_DIR}${APP_NAME}"

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

ENV APP_DIR ${APP_DIR}

# RUN adduser --uid 1002 lidstrainservcie

COPY --from=build ${APP_DIR} ${APP_DIR}
# COPY --from=build --chown=lidstrainservcie ${APP_DIR} ${APP_DIR}
WORKDIR ${APP_DIR}

# Add the additional libraries to the PYTHONPATH
ENV PYTHONPATH="${APP_DIR}/libs"


# copy application
COPY ./model ${APP_DIR}/model
COPY ./saved_models ${APP_DIR}/saved_models
COPY ./service ${APP_DIR}/service
COPY ./test_data ${APP_DIR}/test_data
COPY ./train_data ${APP_DIR}/train_data
COPY ./boot.sh ${APP_DIR}/boot.sh

# COPY --chown=lidstrainservcie ./service ${APP_DIR}/service
# COPY --chown=lidstrainservcie ./commons ${APP_DIR}/commons
# COPY --chown=lidstrainservcie ./utils ${APP_DIR}/utils
# COPY --chown=lidstrainservice ./boot.sh ${APP_DIR}/boot.sh
# USER lidstrainservcie

RUN chmod +x ${APP_DIR}/boot.sh

ENTRYPOINT ["./boot.sh"]
