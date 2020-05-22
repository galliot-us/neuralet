FROM nvcr.io/nvidia/deepstream:5.0-dp-20.04-base

ARG DS_PYBIND_TBZ2='ds_pybind_v0.9.tbz2'
ARG DS_SOURCES_ROOT='/opt/nvidia/deepstream/deepstream/sources'

# defang container
RUN for f in $(find / -perm 4000); do chmod -s "$f"; done;

WORKDIR /tmp

# copy stuff we need at the start of the build
COPY requirements.txt .
# this can't be downloaded directly because a license needs to be accepted,
# and a tarball extracted. This is un-fun:
# https://developer.nvidia.com/deepstream-getting-started#python_bindings
COPY ${DS_PYBIND_TBZ2} .

# extract and install the python bindings
RUN mkdir -p ${DS_SOURCES_ROOT} \
    && tar -xf /tmp/${DS_PYBIND_TBZ2} -C ${DS_SOURCES_ROOT}

# install pip, install requirements, remove pip and deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi \
        python3-gst-1.0 \
        python3-pip \
        python3-opencv \
    && pip3 install --require-hashes -r /tmp/requirements.txt \
    && apt-get purge -y --autoremove \
        python3-pip \
    && rm -rf /var/lib/apt/cache/*

# python3-opencv brings in a *ton* of dependencies so hopefully the new UI will
# dispose of it.

VOLUME /repo
WORKDIR /repo/applications/smart-distancing

EXPOSE 8000

RUN useradd -md /var/smart_distancing -rUs /bin/false smart_distancing
USER smart_distancing:smart_distancing

ENTRYPOINT [ "python3", "-m", "smart_distancing"]
CMD ["--config", "deepstream.ini"]
