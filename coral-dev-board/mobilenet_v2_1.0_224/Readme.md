From this folder:
1 - build the docker image with docker build -t "neuralet-tpu-serving" .
2 - run the docker image, with root of this repo (absolute path for "../../") as your /repo in the docker: 
    docker run  -it --net=host --privileged -v /home/mendel/nkh/RepoCodes/bitbucket/tpu/:/repo neuralet-tpu-serving:latest
3 - now you can send images for inference, an example is: python3 src/client.py PATH_TO_IMAGE
4 - You can also shutdown the serving docker with : python3 src/client.py stop
