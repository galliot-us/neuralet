#!/bin/bash

set -e

readonly SOURCE_DIR=/usr/src/tensorrt/samples/python/uff_ssd/
readonly PLUGIN_SUBDIR=trt
readonly BUILD_DIR=/tmp/smart_distancing_plugin_build_tmp

function prep() {
    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR
    mkdir -p "$1/$PLUGIN_SUBDIR"
}

function build() {
    cd $BUILD_DIR
    cmake $SOURCE_DIR
    make -j$(nproc --all)
}

function install() {
    cd $BUILD_DIR
    cp libflattenconcat.so "$1/$PLUGIN_SUBDIR"
}

function clean() {
    rm -rf $BUILD_DIR
}

function main() {
    if [ "$#" -ne 1 ]; then
        echo "plugin install path required"
    fi
    prep $1
    build
    install $1
    clean
}

main $1
