# Copyright (c) 2023 Parallel Domain, Inc.
# All rights reserved.
#
# Use of this file is only permitted if you have entered into a
# separate written license agreement with Parallel Domain, Inc.

FROM python:3.8

ARG EXTRAS='[data_lab]'

# Non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID user && \
    useradd -r -m -u $USER_ID -g user user
RUN mkdir -p /app && chown user:user /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

USER user

COPY --chown=user:user . /app/
RUN pip install /app/${EXTRAS}

WORKDIR /app
