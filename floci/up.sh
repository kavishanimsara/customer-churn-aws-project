#!/usr/bin/bash
#:WARNING do not run!!! this is  selinux specific
podman run -d --name floci \
        --network floci-net \
        -p 4566:4566 \
        -v /run/user/$(id -u)/podman/podman.sock:/var/run/docker.sock:z \
        --security-opt label=disable \
        -e FLOCI_SERVICES_LAMBDA_DOCKER_NETWORK=floci-net \
        -e FLOCI_HOSTNAME=floci \
        floci/floci