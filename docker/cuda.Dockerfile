# Note: This Dockerfile is currently not used for building an image, but
#       for extracting the current version of CUDA to use.
#       By using a Dockerfile, it is possible to automatically upgrade CUDA
#       patch versions through Dependabot.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS cuda11
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS cuda12
