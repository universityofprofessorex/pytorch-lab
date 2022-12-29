#!/usr/bin/env bash

pushd ./scratch/datasets/
zip -r twitter_facebook_tiktok.zip twitter_facebook_tiktok -x '**/.DS_Store'
echo ""
unzip -l twitter_facebook_tiktok.zip
popd
