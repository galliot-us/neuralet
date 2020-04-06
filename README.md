# Neuralet

Neuralet is an open-source platform for edge deep learning models on GPU, TPU, and more. We hope to make it easier to start and evaluate various deep learning models on different edge devices.

We currently provide models for [Coral Dev Board TPU](https://coral.ai/products/dev-board/) and amd64 node with attached usb edge TPU, and soon will release models for other edge devices such as [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) and more.

## How to use Neuralet?

Neuralet is a set of docker containers that are packaged to run directly on the device. A separate docker container is built for each model on each device to make it very simple to get started with any model on any device.

### Currently supported models

#### amd64 + edge TPU
Image Classification
* [efficientnet-edgetpu-L](https://github.com/neuralet/neuralet/tree/master/amd64/efficientnet-edgetpu-L)
* [efficientnet-edgetpu-M](https://github.com/neuralet/neuralet/tree/master/amd64/efficientnet-edgetpu-M)
* [efficientnet-edgetpu-S](https://github.com/neuralet/neuralet/tree/master/amd64/efficientnet-edgetpu-S)
* [inception_v1_224_quant_20181026](https://github.com/neuralet/neuralet/tree/master/amd64/inception_v1_224_quant_20181026)
* [inception_v2_224_quant_20181026](https://github.com/neuralet/neuralet/tree/master/amd64/inception_v2_224_quant_20181026)
* [inception_v3_quant](https://github.com/neuralet/neuralet/tree/master/amd64/inception_v3_quant)
* [inception_v4_299_quant](https://github.com/neuralet/neuralet/tree/master/amd64/inception_v4_299_quant)
* [mobilenet_v1_1.0_224](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_v1_1.0_224)
* [mobilenet_v2_1.0_224](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_v2_1.0_224)
* [mobilenet_v2_1.0_224_inat_bird](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_v2_1.0_224_inat_bird)
* [mobilenet_v2_1.0_224_inat_insect](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_v2_1.0_224_inat_insect)
* [mobilenet_v2_1.0_224_inat_plant](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_v2_1.0_224_inat_plant)

Object Detection
* [mobilenet_ssd_v1](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_ssd_v1)
* [mobilenet_ssd_v2](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_ssd_v2)
* [mobilenet_ssd_v2_face](https://github.com/neuralet/neuralet/tree/master/amd64/mobilenet_ssd_v2_face)

#### Coral Dev Board
Image Classification
* [efficientnet-edgetpu-L](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/efficientnet-edgetpu-L)
* [efficientnet-edgetpu-M](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/efficientnet-edgetpu-M)
* [efficientnet-edgetpu-S](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/efficientnet-edgetpu-S)
* [inception_v1_224_quant_20181026](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/inception_v1_224_quant_20181026)
* [inception_v2_224_quant_20181026](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/inception_v2_224_quant_20181026)
* [inception_v3_quant](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/inception_v3_quant)
* [inception_v4_299_quant](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/inception_v4_299_quant)
* [mobilenet_v1_1.0_224](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_v1_1.0_224)
* [mobilenet_v2_1.0_224](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_v2_1.0_224)
* [mobilenet_v2_1.0_224_inat_bird](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_v2_1.0_224_inat_bird)
* [mobilenet_v2_1.0_224_inat_insect](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_v2_1.0_224_inat_insect)
* [mobilenet_v2_1.0_224_inat_plant](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_v2_1.0_224_inat_plant)

Object Detection
* [mobilenet_ssd_v1](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_ssd_v1)
* [mobilenet_ssd_v2](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_ssd_v2)
* [mobilenet_ssd_v2_face](https://github.com/neuralet/neuralet/tree/master/coral-dev-board/mobilenet_ssd_v2_face)

#### NVIDIA Jetson Nano
Coming soon.

## Contact

You can reach us via these channels:
- [@neuralet](https://twitter.com/neuralet) on Twitter
- [GitHub Issues](https://github.com/neuralet/neuralet/issues)
- [hello@neuralet.com](mailto:hello@neuralet.com?subject=Hello)

## License
[Apache License 2.0](https://github.com/neuralet/neuralet/blob/master/LICENCE)
