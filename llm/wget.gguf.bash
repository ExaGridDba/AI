#!/bin/bash

if (( $# != 1 )) ; then
    echo "$(basename $0) gguf-url"
    exit 1
fi
gguf_url=$1

file_name=${gguf_url%\?*}
file_name=${file_name##*/}

if [[ -f "$file_name" ]] ; then
    echo "Error: $file_name exists"
    exit 1
fi

wget "$gguf_url" -O "$file_name" || exit 1

echo "downloaded from url:"
echo "$gguf_url"

ls -sh "$file_name"

date "+%Y-%m-%dT%H%M%S $(hostname -s) $gguf_url" >> wget.gguf.hist
